import torch
import torch.nn as nn
import time
from contextlib import contextmanager
import sys
sys.path.append('./src')
from high_low.agent import HighLowTransformerModel
from high_low.config import Args
from high_low.env import HighLowTrading

@contextmanager
def cuda_timer(name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")

def test_compilation_recursion():
    print("Testing torch.compile recursion behavior")
    print("-" * 80)
    
    # Setup
    args = Args()
    args.fill_runtime_args()
    game_config = args.get_game_config()
    env = HighLowTrading(game_config)
    device = torch.device(f'cuda:{args.device_id}')
    
    # Create model
    model = HighLowTransformerModel(args, env, verbose=False).to(device)
    model.eval()
    
    # Get observation
    obs_buffer = env.new_observation_buffer()
    env.fill_observation_tensor(obs_buffer)
    
    print("\n1. Check what's actually happening in incremental_forward:")
    
    # The call chain is:
    # incremental_forward (NOT compiled)
    #   -> incremental_forward_with_context (IS compiled)
    #       -> actors.logp_entropy_and_sample (called INSIDE compiled function)
    
    # Let's trace through this
    model.reset_context()
    
    # First, let's see if the actor methods are being compiled
    print("\n2. Test if DiscreteActor methods get compiled when called from compiled function:")
    
    # Create a test function that mimics the structure
    class SimpleActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 16)
            
        def complex_method(self, x):
            # Simulate complex operations like in DiscreteActor
            out = self.linear(x)
            center = torch.sigmoid(out[:, :4])
            half_width = torch.sigmoid(out[:, 4:8]) * 0.5
            eps_fs = torch.sigmoid(out[:, 8:12] / 25.0 - 6.0)
            eps_uniform = torch.sigmoid(out[:, 12:16] / 25.0 - 6.0)
            
            # Simulate distribution operations
            result = center * half_width + eps_fs * eps_uniform
            return result
    
    simple_actor = SimpleActor().to(device)
    
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def compiled_with_actor_call(x, actor):
        # Process x
        processed = x + 1.0
        # Call actor method (should this be compiled?)
        result = actor.complex_method(processed)
        return result
    
    # Test inputs
    x = torch.randn(4096, 256, device=device)
    
    # Warmup
    print("  Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = compiled_with_actor_call(x, simple_actor)
    
    # Time it
    with torch.no_grad():
        with cuda_timer("  Compiled function calling actor method"):
            _ = compiled_with_actor_call(x, simple_actor)
        
        # Compare with direct call
        with cuda_timer("  Direct actor method call"):
            _ = simple_actor.complex_method(x)
    
    print("\n3. Let's check the actual model's compiled function:")
    
    # Look at what's inside incremental_forward_with_context
    print("\n  The compiled function includes:")
    print("  - Encoder processing")
    print("  - Transformer core")  
    print("  - actors.logp_entropy_and_sample call")
    print("  - pinfo model predictions")
    
    # Test the actual compiled function
    model.reset_context()
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            model.reset_context()
            for step in range(8):
                _ = model.incremental_forward(obs_buffer, step)
        
        # Get to step 8
        model.reset_context()
        for step in range(8):
            _ = model.incremental_forward(obs_buffer, step)
        
        # Profile step 8
        uniform_samples = torch.rand(4096, 4, 3, device=device)
        prev_context = model.context.clone()
        
        with cuda_timer("  incremental_forward_with_context (includes actor)"):
            output = model.incremental_forward_with_context(obs_buffer, prev_context, uniform_samples)
    
    print("\n4. Test if torch.compile is actually recursive:")
    
    # Let's create a clearer test
    class OuterModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = InnerModule()
            
        @torch.compile(fullgraph=True)
        def forward(self, x):
            return self.inner.forward(x)
    
    class InnerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
            
        def forward(self, x):
            # Complex operations
            for _ in range(5):
                x = torch.sigmoid(self.linear(x))
            return x
    
    outer = OuterModule().to(device)
    test_x = torch.randn(100, 10, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = outer(test_x)
    
    with torch.no_grad():
        with cuda_timer("  Nested module call (should be compiled recursively)"):
            _ = outer(test_x)
    
    print("\n5. Check torch._dynamo.explain on the actual function:")
    
    # This will show us what's being compiled
    from torch._dynamo import explain
    
    # Create a fresh model
    model_fresh = HighLowTransformerModel(args, env, verbose=False).to(device)
    model_fresh.eval()
    
    with torch.no_grad():
        explanation = explain(model_fresh.incremental_forward_with_context)(
            obs_buffer, torch.zeros(0, device=device), uniform_samples)
        
        print(f"\n  Compilation stats:")
        print(f"  - Graph breaks: {len(explanation.graph_break_reasons)}")
        print(f"  - Ops in graph: {explanation.op_count}")
        print(f"  - Was fully compiled: {len(explanation.graph_break_reasons) == 0}")
        
        if explanation.graph_break_reasons:
            print(f"\n  Graph break reasons:")
            for i, reason in enumerate(explanation.graph_break_reasons[:5]):  # Show first 5
                print(f"    {i+1}. {reason}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    test_compilation_recursion()