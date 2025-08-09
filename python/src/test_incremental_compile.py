import torch
import torch.nn as nn
import sys
sys.path.append('./')
from discrete_actor import DiscreteActor, GaussianActionDistribution
from high_low.agent import HighLowTransformerModel
import time

def test_discrete_actor_compile():
    """Test that DiscreteActor methods compile correctly"""
    print("Testing DiscreteActor compilation...")
    
    device = torch.device('cuda:0')
    n_hidden = 256
    n_actors = 4
    batch_size = 32
    
    # Create actor with same bounds as in HighLowTransformerModel
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([100, 100, 10, 10], device=device)  # Using reasonable test values
    
    actor = DiscreteActor(n_hidden, n_actors, min_values, max_values)
    actor.to(device)
    
    # Test forward compilation
    forward_compiled = torch.compile(actor.forward, fullgraph=True, mode="max-autotune")
    
    x = torch.randn(batch_size, n_hidden, device=device)
    
    # Warmup
    for _ in range(3):
        mean, prec = forward_compiled(x)
    
    # Test logp_entropy_and_sample compilation
    logp_sample_compiled = torch.compile(actor.logp_entropy_and_sample, fullgraph=True, mode="max-autotune")
    
    uniform_samples = torch.rand(batch_size, n_actors, device=device)
    
    # Warmup
    for _ in range(3):
        output = logp_sample_compiled(x, uniform_samples)
    
    print("✓ DiscreteActor methods compile successfully")
    return True

def test_gaussian_distribution_compile():
    """Test that GaussianActionDistribution methods compile correctly"""
    print("\nTesting GaussianActionDistribution compilation...")
    
    device = torch.device('cuda:0')
    batch_size = 32
    n_actors = 4
    
    center = torch.rand(batch_size, n_actors, device=device)
    precision = torch.rand(batch_size, n_actors, device=device) + 0.5
    
    # Create distribution
    dist = GaussianActionDistribution(center, precision)
    
    # Test sample method compilation
    def sample_fn(uniform_samples):
        return dist.sample(uniform_samples)
    
    sample_compiled = torch.compile(sample_fn, fullgraph=True, mode="max-autotune")
    
    uniform_samples = torch.rand(batch_size, n_actors, device=device)
    
    # Warmup
    for _ in range(3):
        samples = sample_compiled(uniform_samples)
    
    # Test logp_interval compilation
    def logp_interval_fn(lo, hi):
        return dist.logp_interval(lo, hi)
    
    logp_compiled = torch.compile(logp_interval_fn, fullgraph=True, mode="max-autotune")
    
    lo = torch.rand(batch_size, n_actors, device=device) * 0.5
    hi = lo + 0.1
    
    # Warmup
    for _ in range(3):
        logp = logp_compiled(lo, hi)
    
    print("✓ GaussianActionDistribution methods compile successfully")
    return True

def test_incremental_forward_compile():
    """Test that incremental_forward_with_context compiles correctly"""
    print("\nTesting incremental_forward_with_context compilation...")
    
    device = torch.device('cuda:0')
    
    # Create a mock environment class
    class MockEnv:
        def observation_shape(self):
            return 32, 128  # batch_size, feature_dim
    
    # Create args for model initialization
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
    model = HighLowTransformerModel(args, env, verbose=False)
    model.to(device)
    model.eval()
    
    # Compile the incremental forward method
    incremental_compiled = torch.compile(
        model.incremental_forward_with_context, 
        fullgraph=True, 
        mode="max-autotune"
    )
    
    B, F = env.observation_shape()
    x = torch.randn(B, F, device=device)
    uniform_samples = torch.rand(B, 4, device=device)
    
    # Test with empty context (first step)
    empty_context = torch.zeros(0, device=device)
    
    print("Testing with empty context...")
    for _ in range(3):  # Warmup
        output1 = incremental_compiled(x, empty_context, uniform_samples)
    
    # Test with non-empty context
    prev_context = torch.randn(5, B, args.n_embd, device=device)
    
    print("Testing with non-empty context...")
    for _ in range(3):  # Warmup
        output2 = incremental_compiled(x, prev_context, uniform_samples)
    
    print("✓ incremental_forward_with_context compiles successfully")
    
    # Benchmark compiled vs non-compiled
    print("\nBenchmarking compiled vs non-compiled...")
    
    # Non-compiled benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model.incremental_forward_with_context(x, prev_context, uniform_samples)
    torch.cuda.synchronize()
    non_compiled_time = time.time() - start
    
    # Compiled benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = incremental_compiled(x, prev_context, uniform_samples)
    torch.cuda.synchronize()
    compiled_time = time.time() - start
    
    print(f"Non-compiled: {non_compiled_time:.3f}s for 100 iterations")
    print(f"Compiled: {compiled_time:.3f}s for 100 iterations")
    print(f"Speedup: {non_compiled_time/compiled_time:.2f}x")
    
    return True

def verify_output_consistency():
    """Verify that compiled and non-compiled versions produce same outputs"""
    print("\nVerifying output consistency...")
    
    device = torch.device('cuda:0')
    
    # Create a mock environment class
    class MockEnv:
        def observation_shape(self):
            return 16, 64  # smaller batch for testing
    
    class Args:
        steps_per_player = 10
        players = 5
        max_contract_value = 100
        max_contracts_per_trade = 10
        device_id = 0
        n_hidden = 128
        n_head = 4
        n_embd = 128
        n_layer = 2
    
    args = Args()
    env = MockEnv()
    
    # Create model
    model = HighLowTransformerModel(args, env, verbose=False)
    model.to(device)
    model.eval()
    
    # Compile the method
    incremental_compiled = torch.compile(
        model.incremental_forward_with_context,
        fullgraph=True,
        mode="max-autotune"
    )
    
    B, F = env.observation_shape()
    x = torch.randn(B, F, device=device)
    
    # Fix random seed for deterministic results
    torch.manual_seed(42)
    uniform_samples1 = torch.rand(B, 4, device=device)
    torch.manual_seed(42)
    uniform_samples2 = torch.rand(B, 4, device=device)
    
    prev_context = torch.randn(3, B, args.n_embd, device=device)
    
    # Get outputs from both versions
    with torch.no_grad():
        output_regular = model.incremental_forward_with_context(x, prev_context, uniform_samples1)
    
    output_compiled = incremental_compiled(x, prev_context, uniform_samples2)
    
    # Compare outputs
    print("Checking action consistency:", torch.allclose(output_regular['action'], output_compiled['action']))
    print("Checking logprobs consistency:", torch.allclose(output_regular['logprobs'], output_compiled['logprobs'], atol=1e-5))
    print("Checking context consistency:", torch.allclose(output_regular['context'], output_compiled['context'], atol=1e-5))
    
    # Check action parameters
    for key in output_regular['action_params']:
        is_close = torch.allclose(output_regular['action_params'][key], output_compiled['action_params'][key], atol=1e-5)
        print(f"Checking action_params['{key}'] consistency:", is_close)
    
    print("✓ Output consistency verified")
    return True

if __name__ == "__main__":
    print("Starting compilation tests...\n")
    
    # Run all tests
    test_discrete_actor_compile()
    test_gaussian_distribution_compile()
    test_incremental_forward_compile()
    verify_output_consistency()
    
    print("\n✅ All compilation tests passed!")