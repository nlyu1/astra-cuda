import torch
import torch.nn as nn
import time
from contextlib import contextmanager
import sys
sys.path.append('./src')
from discrete_actor import DiscreteActor, TriangleActionDistribution
from high_low.config import Args

@contextmanager
def cuda_timer(name, results_dict=None):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = (end - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")
    if results_dict is not None:
        results_dict[name] = elapsed

def profile_discrete_actor():
    # Setup
    args = Args()
    args.fill_runtime_args()
    device = torch.device(f'cuda:{args.device_id}')
    
    B = args.num_envs  # 4096
    n_hidden = args.n_embd  # 256
    n_actors = 4  # From your agent.py
    
    print(f"Profiling DiscreteActor with batch_size={B}, n_hidden={n_hidden}, n_actors={n_actors}")
    print("-" * 80)
    
    # Create actor
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([args.max_contract_value, args.max_contract_value, 
                              args.max_contracts_per_trade, args.max_contracts_per_trade], device=device)
    
    actor = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor.eval()
    
    # Generate test data
    x = torch.randn(B, n_hidden, device=device)
    uniform_samples = torch.rand(B, n_actors, 3, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(50):
        with torch.no_grad():
            _ = actor.logp_entropy_and_sample(x, uniform_samples)
    
    results = {}
    
    print("\n1. Profile complete forward pass:")
    with torch.no_grad():
        with cuda_timer("  Total logp_entropy_and_sample", results):
            output = actor.logp_entropy_and_sample(x, uniform_samples)
    
    print("\n2. Profile individual components:")
    
    # Linear layer and parameter computation
    with torch.no_grad():
        with cuda_timer("  Linear layer (4*n_actors output)", results):
            raw_output = actor.actor(x)
        
        with cuda_timer("  Split and sigmoid operations", results):
            center = torch.sigmoid(raw_output[:, :n_actors])
            half_width = torch.sigmoid(raw_output[:, n_actors:2*n_actors]) * 0.5
            epsilon_fs = torch.sigmoid(raw_output[:, 2*n_actors:3*n_actors] / actor.eps_logic_inv_scale - actor.eps_logic_bias)
            epsilon_uniform = torch.sigmoid(raw_output[:, 3*n_actors:] / actor.eps_logic_inv_scale - actor.eps_logic_bias)
    
    print("\n3. Profile distribution operations:")
    
    # Create distribution
    with torch.no_grad():
        with cuda_timer("  Create TriangleActionDistribution", results):
            dist = TriangleActionDistribution(center, half_width, epsilon_fs, epsilon_uniform)
    
    # Sample from distribution
    with torch.no_grad():
        with cuda_timer("  Distribution sampling", results):
            unit_samples = dist.sample(uniform_samples)
    
    # Convert to integer samples
    with torch.no_grad():
        with cuda_timer("  Convert to integer samples", results):
            integer_samples = actor._integer_samples_from_unit_samples(unit_samples)
    
    # Compute intervals
    with torch.no_grad():
        with cuda_timer("  Compute unit intervals", results):
            unit_lb, unit_ub = actor._unit_interval_of_integer_samples(integer_samples)
    
    # Compute log probabilities
    with torch.no_grad():
        with cuda_timer("  Compute log probabilities (logp_interval)", results):
            logprobs = dist.logp_interval(unit_lb, unit_ub) - actor.rangeP1.log()
    
    # Compute entropy
    with torch.no_grad():
        with cuda_timer("  Compute entropy", results):
            entropy = dist.entropy() + actor.rangeP1.log()
    
    print("\n4. Profile distribution internals:")
    
    # Test individual distribution methods
    test_x = torch.rand(B, n_actors, device=device)
    
    with torch.no_grad():
        # CDF computation (used in logp_interval)
        with cuda_timer("  Main distribution CDF", results):
            _ = dist.main.cdf(test_x)
        
        with cuda_timer("  Support distribution CDF", results):
            _ = dist.support.cdf(test_x)
        
        with cuda_timer("  Combined CDF", results):
            _ = dist.cdf(test_x)
        
        # Log prob computation
        with cuda_timer("  Main distribution log_prob", results):
            _ = dist.main.log_prob(test_x)
        
        with cuda_timer("  Support distribution log_prob", results):
            _ = dist.support.log_prob(test_x)
    
    print("\n5. Test with different batch sizes:")
    for test_batch in [256, 1024, 4096, 8192]:
        test_x = torch.randn(test_batch, n_hidden, device=device)
        test_uniform = torch.rand(test_batch, n_actors, 3, device=device)
        
        with torch.no_grad():
            with cuda_timer(f"  Batch size {test_batch}"):
                _ = actor.logp_entropy_and_sample(test_x, test_uniform)
    
    print("\n6. Profile with torch.compile:")
    print("  Compiling actor forward method...")
    actor_compiled = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    actor_compiled.eval()
    actor_compiled.forward = torch.compile(actor_compiled.forward, fullgraph=True, mode="max-autotune")
    
    # Warmup compiled version
    with torch.no_grad():
        for _ in range(10):
            _ = actor_compiled.logp_entropy_and_sample(x, uniform_samples)
    
    with torch.no_grad():
        with cuda_timer("  Compiled logp_entropy_and_sample", results):
            _ = actor_compiled.logp_entropy_and_sample(x, uniform_samples)
    
    print("\n7. Summary and bottleneck analysis:")
    total_time = results.get("  Total logp_entropy_and_sample", 0)
    if total_time > 0:
        print(f"\nTime breakdown (% of total {total_time:.2f}ms):")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for name, time_ms in sorted_results[:10]:
            if "Total" not in name and "Batch size" not in name:
                percentage = (time_ms / total_time) * 100
                print(f"  {name}: {percentage:.1f}%")
    
    # Identify bottlenecks
    print("\nBottleneck analysis:")
    linear_time = results.get("  Linear layer (4*n_actors output)", 0)
    logp_time = results.get("  Compute log probabilities (logp_interval)", 0)
    cdf_time = results.get("  Combined CDF", 0)
    
    if logp_time > linear_time * 2:
        print("  ⚠️  Log probability computation is the main bottleneck")
        print(f"     - Takes {logp_time:.2f}ms vs {linear_time:.2f}ms for linear layer")
        print("     - This involves multiple CDF computations which are expensive")
    
    if cdf_time > 5:
        print("  ⚠️  CDF computation is expensive")
        print("     - Consider caching or approximating CDF values")
    
    print("\nOptimization suggestions:")
    print("  1. The logp_interval computation (involving CDFs) is likely the bottleneck")
    print("  2. Consider approximating or simplifying the triangular distributions")
    print("  3. torch.compile shows limited improvement, suggesting the bottleneck is in complex operations")
    print("  4. Batch size scaling is roughly linear, indicating good parallelization")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    profile_discrete_actor()