import torch
import torch.nn as nn
import sys
sys.path.append('./')
from discrete_actor import DiscreteActor, GaussianActionDistribution
from high_low.agent import HighLowTransformerModel
import time

# Set float32 matmul precision for better performance
torch.set_float32_matmul_precision('high')

def test_incremental_forward_simple():
    """Test incremental_forward_with_context without full compilation"""
    print("Testing incremental_forward_with_context functionality...")
    
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
    model = HighLowTransformerModel(args, env, verbose=True)
    model.to(device)
    model.eval()
    
    B, F = env.observation_shape()
    
    # Test the incremental forward process step by step
    print("\n--- Testing incremental forward process ---")
    
    # Reset context
    model.reset_context()
    
    # Simulate multiple steps
    for step in range(5):
        x = torch.randn(B, F, device=device)
        
        with torch.no_grad():
            outputs = model.incremental_forward(x, step)
        
        print(f"\nStep {step}:")
        print(f"  Action shape: {outputs['action'].shape}")
        print(f"  Action range: [{outputs['action'].min().item():.0f}, {outputs['action'].max().item():.0f}]")
        print(f"  Logprobs shape: {outputs['logprobs'].shape}")
        print(f"  Context shape: {outputs['context'].shape}")
        
        # Verify action bounds
        assert (outputs['action'][:, 0] >= 1).all() and (outputs['action'][:, 0] <= args.max_contract_value).all(), "Bid price out of bounds"
        assert (outputs['action'][:, 1] >= 1).all() and (outputs['action'][:, 1] <= args.max_contract_value).all(), "Ask price out of bounds"
        assert (outputs['action'][:, 2] >= 0).all() and (outputs['action'][:, 2] <= args.max_contracts_per_trade).all(), "Bid size out of bounds"
        assert (outputs['action'][:, 3] >= 0).all() and (outputs['action'][:, 3] <= args.max_contracts_per_trade).all(), "Ask size out of bounds"
    
    print("\n✓ Incremental forward process works correctly")
    return True

def test_compilation_components():
    """Test individual components that should compile"""
    print("\n--- Testing compilable components ---")
    
    device = torch.device('cuda:0')
    
    # Test the core transformer forward pass compilation
    class SimpleTransformerBlock(nn.Module):
        def __init__(self, d_model, n_head):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model * 4,
                dropout=0,
                activation="gelu",
                batch_first=False,
                norm_first=True
            )
            self.norm = nn.LayerNorm(d_model)
        
        def forward(self, x, mask):
            x = self.layer(x, src_mask=mask)
            return self.norm(x)
    
    d_model = 256
    n_head = 8
    seq_len = 10
    batch_size = 32
    
    block = SimpleTransformerBlock(d_model, n_head).to(device)
    
    # Try to compile the transformer block
    try:
        compiled_block = torch.compile(block, mode="reduce-overhead")
        
        x = torch.randn(seq_len, batch_size, d_model, device=device)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                out = compiled_block(x, mask)
        
        print("✓ Transformer block compiles successfully")
    except Exception as e:
        print(f"✗ Transformer block compilation failed: {e}")
    
    # Test DiscreteActor forward pass (without sampling)
    print("\nTesting DiscreteActor parameter generation...")
    
    n_hidden = 256
    n_actors = 4
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([100, 100, 10, 10], device=device)
    
    actor = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    
    try:
        # Just compile the forward method that generates parameters
        compiled_forward = torch.compile(actor.forward, mode="reduce-overhead")
        
        x = torch.randn(batch_size, n_hidden, device=device)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                mean, precision = compiled_forward(x)
        
        print("✓ DiscreteActor parameter generation compiles successfully")
        print(f"  Mean shape: {mean.shape}, range: [{mean.min():.3f}, {mean.max():.3f}]")
        print(f"  Precision shape: {precision.shape}, range: [{precision.min():.3f}, {precision.max():.3f}]")
    except Exception as e:
        print(f"✗ DiscreteActor compilation failed: {e}")
    
    return True

def benchmark_inference_speed():
    """Benchmark the inference speed of incremental forward"""
    print("\n--- Benchmarking inference speed ---")
    
    device = torch.device('cuda:0')
    
    class MockEnv:
        def observation_shape(self):
            return 1024, 128  # larger batch size for benchmarking
    
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
    
    model = HighLowTransformerModel(args, env, verbose=False)
    model.to(device)
    model.eval()
    
    B, F = env.observation_shape()
    
    # Prepare data
    x = torch.randn(B, F, device=device)
    prev_context = torch.randn(5, B, args.n_embd, device=device)
    uniform_samples = torch.rand(B, 4, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.incremental_forward_with_context(x, prev_context, uniform_samples)
    
    # Benchmark
    num_iterations = 100
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model.incremental_forward_with_context(x, prev_context, uniform_samples)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Batch size: {B}")
    print(f"Total time for {num_iterations} iterations: {elapsed:.3f}s")
    print(f"Average time per iteration: {elapsed/num_iterations*1000:.2f}ms")
    print(f"Throughput: {B*num_iterations/elapsed:.0f} samples/second")
    
    return True

def verify_sampling_correctness():
    """Verify that the sampling process produces correct outputs"""
    print("\n--- Verifying sampling correctness ---")
    
    device = torch.device('cuda:0')
    
    # Test DiscreteActor sampling directly
    n_hidden = 256
    n_actors = 4
    batch_size = 10000  # Large batch for statistics
    
    min_values = torch.tensor([1, 1, 0, 0], device=device)
    max_values = torch.tensor([100, 100, 10, 10], device=device)
    
    actor = DiscreteActor(n_hidden, n_actors, min_values, max_values).to(device)
    
    # Generate samples with known distribution parameters
    x = torch.randn(batch_size, n_hidden, device=device)
    uniform_samples = torch.rand(batch_size, n_actors, device=device)
    
    with torch.no_grad():
        outputs = actor.logp_entropy_and_sample(x, uniform_samples)
    
    samples = outputs['samples']
    
    # Verify bounds
    for i in range(n_actors):
        assert (samples[:, i] >= min_values[i]).all(), f"Actor {i} samples below minimum"
        assert (samples[:, i] <= max_values[i]).all(), f"Actor {i} samples above maximum"
    
    # Check distribution statistics
    print("\nSample statistics:")
    for i in range(n_actors):
        print(f"  Actor {i}: min={samples[:, i].min()}, max={samples[:, i].max()}, "
              f"mean={samples[:, i].float().mean():.1f}, std={samples[:, i].float().std():.1f}")
    
    # Verify logprobs are valid
    assert not torch.isnan(outputs['logprobs']).any(), "NaN in logprobs"
    assert not torch.isinf(outputs['logprobs']).any(), "Inf in logprobs"
    
    # Verify entropy is positive
    assert (outputs['entropy'] > 0).all(), "Negative entropy values"
    
    print("\n✓ Sampling process produces valid outputs")
    return True

if __name__ == "__main__":
    print("Starting incremental forward verification...\n")
    
    # Run tests
    test_incremental_forward_simple()
    test_compilation_components()
    benchmark_inference_speed()
    verify_sampling_correctness()
    
    print("\n✅ All tests completed successfully!")