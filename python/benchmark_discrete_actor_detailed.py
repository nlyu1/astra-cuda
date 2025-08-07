import torch
import time
from contextlib import contextmanager
import sys
sys.path.append('./src')
from discrete_actor import (DiscreteActor, TriangleActionDistribution, 
                           TriangleVariableWidthDistribution, TriangleFullSupportDistribution,
                           TrapezoidFullSupportDistribution)
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
    return elapsed

def detailed_profile():
    # Setup
    args = Args()
    args.fill_runtime_args()
    device = torch.device(f'cuda:{args.device_id}')
    
    B = 4096
    n_actors = 4
    
    print(f"Detailed profiling with batch_size={B}, n_actors={n_actors}")
    print("-" * 80)
    
    # Generate test parameters
    center = torch.rand(B, n_actors, device=device)
    half_width = torch.rand(B, n_actors, device=device) * 0.5
    epsilon_fs = torch.rand(B, n_actors, device=device)
    epsilon_uniform = torch.rand(B, n_actors, device=device)
    test_x = torch.rand(B, n_actors, device=device)
    
    # Warmup
    for _ in range(10):
        dist = TrapezoidFullSupportDistribution(center[:, 0], epsilon_uniform[:, 0])
        _ = dist.log_prob(test_x[:, 0])
    
    print("\n1. Profile TrapezoidFullSupportDistribution operations:")
    
    # Create single column for testing
    center_1d = center[:, 0]
    epsilon_1d = epsilon_uniform[:, 0]
    test_x_1d = test_x[:, 0]
    
    with cuda_timer("  Create TrapezoidFullSupportDistribution"):
        trapezoid = TrapezoidFullSupportDistribution(center_1d, epsilon_1d)
    
    with cuda_timer("  Triangle component log_prob"):
        triangle_lp = trapezoid.triangle.log_prob(test_x_1d)
    
    with cuda_timer("  logaddexp operation"):
        result = torch.logaddexp(
            triangle_lp + torch.log1p(-epsilon_1d),
            torch.log(epsilon_1d))
    
    with cuda_timer("  Full trapezoid log_prob"):
        _ = trapezoid.log_prob(test_x_1d)
    
    print("\n2. Profile TriangleFullSupportDistribution internals:")
    
    triangle_fs = TriangleFullSupportDistribution(center_1d)
    
    with cuda_timer("  Compute probabilities"):
        prob = torch.where(
            test_x_1d < center_1d, 
            2. / center_1d * test_x_1d, 
            2. / (center_1d - 1) * (test_x_1d - center_1d) + 2)
    
    with cuda_timer("  Log and clamp"):
        _ = torch.log(prob).clamp(min=-100.)
    
    print("\n3. Profile the actual actor workflow:")
    
    # Create actor
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([30, 30, 5, 5], device=device)
    actor = DiscreteActor(256, n_actors, min_values, max_values).to(device)
    
    x = torch.randn(B, 256, device=device)
    uniform_samples = torch.rand(B, n_actors, 3, device=device)
    
    # Step through the process
    with torch.no_grad():
        # Warmup the actor first
        for _ in range(10):
            _ = actor(x)
        
        with cuda_timer("  Actor forward (parameter generation)"):
            center, half_width, epsilon_fs, epsilon_uniform = actor(x)
        
        with cuda_timer("  Create all distributions"):
            dist = TriangleActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)
        
        with cuda_timer("  Sample from distributions"):
            unit_samples = dist.sample(uniform_samples)
        
        with cuda_timer("  Convert to integers"):
            integer_samples = actor._integer_samples_from_unit_samples(unit_samples)
        
        with cuda_timer("  Compute intervals"):
            unit_lb, unit_ub = actor._unit_interval_of_integer_samples(integer_samples)
        
        # Break down logp_interval
        print("\n  Breaking down logp_interval:")
        
        with cuda_timer("    CDF upper bound"):
            cdf_ub = dist.cdf(unit_ub)
        
        with cuda_timer("    CDF lower bound"):
            cdf_lb = dist.cdf(unit_lb)
        
        with cuda_timer("    Log of difference"):
            logp = torch.log(cdf_ub - cdf_lb).clamp(min=-100.)
        
        with cuda_timer("  Full logp_interval"):
            logprobs = dist.logp_interval(unit_lb, unit_ub)
    
    print("\n4. Test vectorization efficiency:")
    
    # Test if operating on all actors at once vs one at a time makes a difference
    total_time_vectorized = 0
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            dist_vec = TriangleActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)
            _ = dist_vec.cdf(test_x)
        torch.cuda.synchronize()
        total_time_vectorized = (time.perf_counter() - start) * 1000
    
    total_time_loop = 0
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            for i in range(n_actors):
                dist_single = TriangleActionDistribution(
                    center[:, i:i+1], half_width[:, i:i+1], 
                    epsilon_fs[:, i:i+1], epsilon_uniform[:, i:i+1])
                _ = dist_single.cdf(test_x[:, i:i+1])
        torch.cuda.synchronize()
        total_time_loop = (time.perf_counter() - start) * 1000
    
    print(f"\n  Vectorized (all actors): {total_time_vectorized:.2f}ms")
    print(f"  Loop (one actor at a time): {total_time_loop:.2f}ms")
    print(f"  Speedup from vectorization: {total_time_loop/total_time_vectorized:.2f}x")
    
    print("\n5. Memory and compute pattern analysis:")
    print(f"  Batch size: {B}")
    print(f"  Number of actors: {n_actors}")
    print(f"  Total elements per operation: {B * n_actors}")
    print(f"  Memory per batch (approx): {B * n_actors * 4 * 10 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    detailed_profile()