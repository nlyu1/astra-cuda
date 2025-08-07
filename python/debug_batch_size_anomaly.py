import torch
import torch.nn as nn
import time
from contextlib import contextmanager
import sys
sys.path.append('./src')
from discrete_actor import DiscreteActor
from high_low.config import Args

@contextmanager
def cuda_timer(name, return_time=False):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")
    if return_time:
        return elapsed

def investigate_batch_size_anomaly():
    # Setup
    device = torch.device('cuda:0')
    n_hidden = 256
    n_actors = 4
    
    print("Investigating batch size anomaly")
    print("-" * 80)
    
    # Create actor
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([30, 30, 5, 5], device=device)
    
    actor = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor.eval()
    
    # Test multiple runs for each batch size
    batch_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    print("\n1. Multiple runs per batch size (5 runs each after warmup):")
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, n_hidden, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = actor(x)
        
        # Time multiple runs
        times = []
        with torch.no_grad():
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = actor(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        print(f"  Batch {batch_size}: mean={sum(times)/len(times):.2f}ms, "
              f"min={min(times):.2f}ms, max={max(times):.2f}ms, "
              f"times={[f'{t:.2f}' for t in times]}")
    
    print("\n2. Check if it's the first call that's slow:")
    # Create fresh actor
    actor_fresh = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor_fresh.eval()
    
    for batch_size in [256, 1024, 4096]:
        x = torch.randn(batch_size, n_hidden, device=device)
        
        times = []
        with torch.no_grad():
            for i in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = actor_fresh(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        print(f"  Fresh actor, batch {batch_size}, first 5 calls: {[f'{t:.2f}ms' for t in times]}")
    
    print("\n3. Test if it's related to torch.compile autotuning:")
    # Even though we're not using compile, PyTorch might be doing some autotuning
    
    # Disable cudnn autotuning temporarily
    torch.backends.cudnn.benchmark = False
    
    actor_no_autotune = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor_no_autotune.eval()
    
    for batch_size in [256, 1024, 4096]:
        x = torch.randn(batch_size, n_hidden, device=device)
        
        with torch.no_grad():
            with cuda_timer(f"  No autotune, batch {batch_size}"):
                _ = actor_no_autotune(x)
    
    torch.backends.cudnn.benchmark = True
    
    print("\n4. Profile individual operations for batch 256 vs 4096:")
    for batch_size in [256, 4096]:
        print(f"\n  Batch size {batch_size}:")
        x = torch.randn(batch_size, n_hidden, device=device)
        
        # Create fresh components
        linear = nn.Linear(n_hidden, n_actors * 4).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                out = linear(x)
                _ = torch.sigmoid(out)
        
        with torch.no_grad():
            with cuda_timer(f"    Linear layer"):
                out = linear(x)
            
            with cuda_timer(f"    Sigmoid"):
                _ = torch.sigmoid(out)
            
            with cuda_timer(f"    Division by 25"):
                _ = out / 25.0
            
            with cuda_timer(f"    Subtraction by 6"):
                _ = out - 6.0
    
    print("\n5. Check if it's memory allocation related:")
    for batch_size in [256, 4096]:
        x = torch.randn(batch_size, n_hidden, device=device)
        
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated(device)
        
        with torch.no_grad():
            _ = actor(x)
        
        end_mem = torch.cuda.memory_allocated(device)
        peak_mem = torch.cuda.max_memory_allocated(device)
        
        print(f"  Batch {batch_size}: allocated={(end_mem-start_mem)/1024:.1f}KB, "
              f"peak={(peak_mem-start_mem)/1024:.1f}KB")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    investigate_batch_size_anomaly()