import torch
import torch.nn as nn
import time
from contextlib import contextmanager
import sys
sys.path.append('./src')
from discrete_actor import DiscreteActor
from high_low.config import Args

@contextmanager
def cuda_timer(name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")

def debug_actor_forward():
    # Setup
    args = Args()
    args.fill_runtime_args()
    device = torch.device(f'cuda:{args.device_id}')
    
    B = 4096
    n_hidden = 256
    n_actors = 4
    
    print(f"Debugging actor forward with batch_size={B}, n_hidden={n_hidden}, n_actors={n_actors}")
    print("-" * 80)
    
    # Create actor
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([30, 30, 5, 5], device=device)
    
    actor = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor.eval()
    
    # Create test input
    x = torch.randn(B, n_hidden, device=device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = actor(x)
    
    print("\n1. Profile uncompiled actor forward:")
    with torch.no_grad():
        with cuda_timer("  Full actor.forward()"):
            center, half_width, epsilon_fs, epsilon_uniform = actor(x)
    
    # Manual breakdown
    print("\n2. Manual breakdown of actor.forward():")
    with torch.no_grad():
        with cuda_timer("  Linear layer only"):
            output = actor.actor(x)
        
        with cuda_timer("  Slice tensors"):
            c = output[:, :n_actors]
            hw = output[:, n_actors:2*n_actors]
            efs = output[:, 2*n_actors:3*n_actors]
            eu = output[:, 3*n_actors:]
        
        with cuda_timer("  Sigmoid on center"):
            center = torch.sigmoid(c)
        
        with cuda_timer("  Sigmoid + multiply on half_width"):
            half_width = torch.sigmoid(hw) * 0.5
        
        with cuda_timer("  Complex sigmoid on epsilon_fs"):
            epsilon_fs = torch.sigmoid(efs / actor.eps_logic_inv_scale - actor.eps_logic_bias)
        
        with cuda_timer("  Complex sigmoid on epsilon_uniform"):
            epsilon_uniform = torch.sigmoid(eu / actor.eps_logic_inv_scale - actor.eps_logic_bias)
    
    print("\n3. Test simple linear layer for comparison:")
    simple_linear = nn.Linear(n_hidden, n_actors * 4).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(50):
            _ = simple_linear(x)
    
    with torch.no_grad():
        with cuda_timer("  Simple linear layer"):
            _ = simple_linear(x)
        
        with cuda_timer("  Simple linear + 4 sigmoids"):
            out = simple_linear(x)
            _ = torch.sigmoid(out)
    
    print("\n4. Compile and test actor forward:")
    print("  Compiling actor.forward()...")
    
    # Create new actor for compilation
    actor_compiled = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor_compiled.eval()
    actor_compiled.forward = torch.compile(actor_compiled.forward, fullgraph=True, mode="max-autotune")
    
    # Warmup compiled version
    print("  Warming up compiled version...")
    with torch.no_grad():
        for _ in range(10):
            _ = actor_compiled(x)
    
    with torch.no_grad():
        with cuda_timer("  Compiled actor.forward()"):
            _ = actor_compiled(x)
    
    print("\n5. Test with different modes:")
    for mode in ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]:
        actor_test = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
        actor_test.eval()
        actor_test.forward = torch.compile(actor_test.forward, fullgraph=True, mode=mode)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = actor_test(x)
        
        with torch.no_grad():
            with cuda_timer(f"  Compiled with mode={mode}"):
                _ = actor_test(x)
    
    print("\n6. Test different batch sizes:")
    for test_batch in [256, 1024, 2048, 4096, 8192]:
        test_x = torch.randn(test_batch, n_hidden, device=device)
        
        with torch.no_grad():
            with cuda_timer(f"  Batch size {test_batch}"):
                _ = actor(test_x)
    
    print("\n7. Check if there's a warmup issue:")
    times = []
    with torch.no_grad():
        for i in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = actor(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed = (end - start) * 1000
            times.append(elapsed)
    
    print(f"  First 10 runs: {', '.join([f'{t:.2f}ms' for t in times])}")
    print(f"  Mean: {sum(times)/len(times):.2f}ms, Min: {min(times):.2f}ms, Max: {max(times):.2f}ms")
    
    print("\n8. Profile memory allocations:")
    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        start_mem = torch.cuda.memory_allocated(device)
        _ = actor(x)
        end_mem = torch.cuda.memory_allocated(device)
        peak_mem = torch.cuda.max_memory_allocated(device)
    
    print(f"  Memory allocated during forward: {(end_mem - start_mem) / 1024 / 1024:.2f} MB")
    print(f"  Peak memory during forward: {(peak_mem - start_mem) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    debug_actor_forward()