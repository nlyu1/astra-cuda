import torch
import torch.nn as nn
import sys
sys.path.append('./')
from high_low.agent import HighLowTransformerModel
import time
from functools import partial

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision('high')

class IncrementalForwardCompiler:
    """Helper class to compile incremental_forward_with_context with various strategies"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def compile_method(self, mode="reduce-overhead", dynamic=True):
        """
        Compile the incremental_forward_with_context method
        
        Args:
            mode: Compilation mode - "reduce-overhead", "max-autotune", "default"
            dynamic: Whether to use dynamic shapes (recommended for variable context lengths)
        """
        print(f"Compiling incremental_forward_with_context with mode='{mode}', dynamic={dynamic}")
        
        if dynamic:
            # Use dynamic shapes for context dimension
            compiled_fn = torch.compile(
                self.model.incremental_forward_with_context,
                mode=mode,
                dynamic=True,
                fullgraph=False  # Allow graph breaks for better compatibility
            )
        else:
            # Static shapes - faster but less flexible
            compiled_fn = torch.compile(
                self.model.incremental_forward_with_context,
                mode=mode,
                fullgraph=True
            )
        
        return compiled_fn
    
    def create_jit_traced_version(self, batch_size, feature_dim, context_len=5):
        """
        Create a JIT traced version for specific input shapes
        This is more limited but can be faster for fixed input sizes
        """
        print(f"Creating JIT traced version for B={batch_size}, F={feature_dim}, context_len={context_len}")
        
        # Create example inputs
        x = torch.randn(batch_size, feature_dim, device=self.model.device)
        prev_context = torch.randn(context_len, batch_size, self.model.n_embd, device=self.model.device)
        uniform_samples = torch.rand(batch_size, 4, device=self.model.device)
        
        # Trace the function
        with torch.no_grad():
            traced_fn = torch.jit.trace(
                self.model.incremental_forward_with_context,
                (x, prev_context, uniform_samples)
            )
        
        return traced_fn
    
    def benchmark_compilation_modes(self, batch_size=32, feature_dim=128, num_iterations=100):
        """Benchmark different compilation strategies"""
        print(f"\nBenchmarking compilation modes (B={batch_size}, iterations={num_iterations})")
        print("-" * 60)
        
        # Prepare test data
        x = torch.randn(batch_size, feature_dim, device=self.model.device)
        prev_context = torch.randn(5, batch_size, self.model.n_embd, device=self.model.device)
        uniform_samples = torch.rand(batch_size, 4, device=self.model.device)
        
        results = {}
        
        # Test uncompiled
        print("Testing uncompiled...")
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model.incremental_forward_with_context(x, prev_context, uniform_samples)
        torch.cuda.synchronize()
        uncompiled_time = time.time() - start
        results['uncompiled'] = uncompiled_time
        print(f"  Time: {uncompiled_time:.3f}s ({uncompiled_time/num_iterations*1000:.2f}ms per iter)")
        
        # Test different compilation modes
        modes = ["reduce-overhead", "max-autotune", "default"]
        
        for mode in modes:
            try:
                print(f"\nTesting {mode} compilation...")
                compiled_fn = self.compile_method(mode=mode, dynamic=True)
                
                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = compiled_fn(x, prev_context, uniform_samples)
                
                # Benchmark
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    for _ in range(num_iterations):
                        _ = compiled_fn(x, prev_context, uniform_samples)
                torch.cuda.synchronize()
                compiled_time = time.time() - start
                results[mode] = compiled_time
                
                speedup = uncompiled_time / compiled_time
                print(f"  Time: {compiled_time:.3f}s ({compiled_time/num_iterations*1000:.2f}ms per iter)")
                print(f"  Speedup: {speedup:.2f}x")
                
            except Exception as e:
                print(f"  Failed: {e}")
                results[mode] = None
        
        return results

def create_production_ready_model(args, env, compile_mode="reduce-overhead"):
    """
    Create a production-ready model with optional compilation
    
    Args:
        args: Model arguments
        env: Environment
        compile_mode: Compilation mode or None to skip compilation
    
    Returns:
        model: The model with potentially compiled methods
    """
    model = HighLowTransformerModel(args, env, verbose=True)
    model.eval()
    
    if compile_mode:
        print(f"\nCompiling model with mode='{compile_mode}'...")
        compiler = IncrementalForwardCompiler(model)
        
        # Compile the incremental forward method
        model.incremental_forward_with_context = compiler.compile_method(
            mode=compile_mode, 
            dynamic=True
        )
        
        # Optionally compile other methods
        try:
            model.forward = torch.compile(model.forward, mode=compile_mode, dynamic=True)
            print("✓ Also compiled forward method")
        except:
            print("✗ Could not compile forward method")
    
    return model

def main():
    """Example usage and benchmarking"""
    
    # Mock environment
    class MockEnv:
        def observation_shape(self):
            return 32, 128
    
    class Args:
        steps_per_player = 10
        players = 5
        max_contract_value = 100
        max_contracts_per_trade = 10
        device_id = 0
        n_hidden = 256
        n_head = 8
        n_embd = 256
        n_layer = 4
    
    args = Args()
    env = MockEnv()
    
    # Create model
    model = HighLowTransformerModel(args, env, verbose=True)
    model.to(torch.device('cuda:0'))
    
    # Create compiler
    compiler = IncrementalForwardCompiler(model)
    
    # Benchmark different compilation modes
    results = compiler.benchmark_compilation_modes(
        batch_size=256,  # Larger batch for better benchmarking
        num_iterations=100
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    baseline = results['uncompiled']
    for mode, time_taken in results.items():
        if time_taken is not None:
            speedup = baseline / time_taken if mode != 'uncompiled' else 1.0
            print(f"{mode:20s}: {time_taken:.3f}s (speedup: {speedup:.2f}x)")
        else:
            print(f"{mode:20s}: Failed to compile")
    
    # Example: Create a production model
    print("\n" + "=" * 60)
    print("Creating production-ready model...")
    print("=" * 60)
    
    prod_model = create_production_ready_model(args, env, compile_mode="reduce-overhead")
    
    # Test it works
    B, F = env.observation_shape()
    x = torch.randn(B, F, device=prod_model.device)
    
    with torch.no_grad():
        outputs = prod_model.incremental_forward(x, 0)
    
    print(f"\n✓ Production model works! Output action shape: {outputs['action'].shape}")

if __name__ == "__main__":
    main()