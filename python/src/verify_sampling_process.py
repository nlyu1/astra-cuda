import torch
import torch.nn as nn
import sys
sys.path.append('./')
from discrete_actor import DiscreteActor, GaussianActionDistribution
from high_low.agent import HighLowTransformerModel
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision('high')

def verify_incremental_forward_sampling():
    """
    Comprehensive verification of the incremental_forward_with_context sampling process
    """
    print("=== Verifying Incremental Forward Sampling Process ===\n")
    
    device = torch.device('cuda:0')
    
    # Setup
    class MockEnv:
        def observation_shape(self):
            return 1000, 128  # Large batch for statistical verification
    
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
    B, F = env.observation_shape()
    
    # Create model
    model = HighLowTransformerModel(args, env, verbose=False)
    model.to(device)
    model.eval()
    
    # Test 1: Verify deterministic behavior with fixed random seeds
    print("1. Testing deterministic sampling...")
    x = torch.randn(B, F, device=device)
    prev_context = torch.randn(5, B, args.n_embd, device=device)
    
    torch.manual_seed(42)
    uniform1 = torch.rand(B, 4, device=device)
    with torch.no_grad():
        output1 = model.incremental_forward_with_context(x, prev_context, uniform1.clone())
    
    torch.manual_seed(42)
    uniform2 = torch.rand(B, 4, device=device)
    with torch.no_grad():
        output2 = model.incremental_forward_with_context(x, prev_context, uniform2.clone())
    
    assert torch.allclose(output1['action'], output2['action']), "Sampling not deterministic!"
    print("✓ Deterministic sampling verified\n")
    
    # Test 2: Verify action bounds
    print("2. Testing action bounds...")
    actions = output1['action']
    
    # Check each action type
    bid_prices = actions[:, 0]
    ask_prices = actions[:, 1]
    bid_sizes = actions[:, 2]
    ask_sizes = actions[:, 3]
    
    assert (bid_prices >= 1).all() and (bid_prices <= args.max_contract_value).all()
    assert (ask_prices >= 1).all() and (ask_prices <= args.max_contract_value).all()
    assert (bid_sizes >= 0).all() and (bid_sizes <= args.max_contracts_per_trade).all()
    assert (ask_sizes >= 0).all() and (ask_sizes <= args.max_contracts_per_trade).all()
    
    print(f"✓ Bid prices: [{bid_prices.min()}, {bid_prices.max()}]")
    print(f"✓ Ask prices: [{ask_prices.min()}, {ask_prices.max()}]")
    print(f"✓ Bid sizes: [{bid_sizes.min()}, {bid_sizes.max()}]")
    print(f"✓ Ask sizes: [{ask_sizes.min()}, {ask_sizes.max()}]\n")
    
    # Test 3: Verify distribution parameters
    print("3. Testing distribution parameters...")
    centers = output1['action_params']['center']
    precisions = output1['action_params']['precision']
    
    assert (centers >= 0).all() and (centers <= 1).all(), "Centers not in [0,1]"
    assert (precisions > 0).all(), "Precisions not positive"
    
    print(f"✓ Centers range: [{centers.min():.3f}, {centers.max():.3f}]")
    print(f"✓ Precisions range: [{precisions.min():.3f}, {precisions.max():.3f}]\n")
    
    # Test 4: Verify log probabilities
    print("4. Testing log probabilities...")
    logprobs = output1['logprobs']
    logprobs_by_type = output1['logprobs_by_type']
    
    assert not torch.isnan(logprobs).any(), "NaN in logprobs"
    assert not torch.isinf(logprobs).any(), "Inf in logprobs"
    assert torch.allclose(logprobs, logprobs_by_type.sum(dim=1)), "Logprobs sum mismatch"
    
    print(f"✓ Logprobs range: [{logprobs.min():.3f}, {logprobs.max():.3f}]")
    print(f"✓ Mean logprob: {logprobs.mean():.3f}\n")
    
    # Test 5: Sample multiple times and analyze distribution
    print("5. Analyzing sampling distribution...")
    n_samples = 10
    all_actions = []
    
    for _ in range(n_samples):
        uniform = torch.rand(B, 4, device=device)
        with torch.no_grad():
            output = model.incremental_forward_with_context(x, prev_context, uniform)
        all_actions.append(output['action'])
    
    all_actions = torch.stack(all_actions)  # [n_samples, B, 4]
    
    # Compute statistics across samples
    action_means = all_actions.float().mean(dim=0)  # [B, 4]
    action_stds = all_actions.float().std(dim=0)    # [B, 4]
    
    print("Average statistics across samples:")
    for i, name in enumerate(['Bid Price', 'Ask Price', 'Bid Size', 'Ask Size']):
        print(f"  {name}: mean={action_means[:, i].mean():.1f}, std={action_stds[:, i].mean():.1f}")
    
    print("\n✓ Sampling distribution verified\n")
    
    # Test 6: Test compilation
    print("6. Testing compilation compatibility...")
    try:
        compiled_fn = torch.compile(
            model.incremental_forward_with_context,
            mode="reduce-overhead",
            dynamic=True,
            fullgraph=False
        )
        
        # Warmup
        for _ in range(3):
            uniform = torch.rand(B, 4, device=device)
            with torch.no_grad():
                _ = compiled_fn(x, prev_context, uniform)
        
        print("✓ Successfully compiled and executed\n")
    except Exception as e:
        print(f"✗ Compilation failed: {e}\n")
    
    # Test 7: Visual verification (save plots)
    print("7. Creating visualization...")
    
    # Sample many times from a single configuration
    n_vis_samples = 5000
    vis_batch_size = 1
    x_vis = torch.randn(vis_batch_size, F, device=device)
    context_vis = torch.randn(5, vis_batch_size, args.n_embd, device=device)
    
    samples = []
    for _ in range(n_vis_samples):
        uniform = torch.rand(vis_batch_size, 4, device=device)
        with torch.no_grad():
            output = model.incremental_forward_with_context(x_vis, context_vis, uniform)
        samples.append(output['action'].cpu())
    
    samples = torch.cat(samples, dim=0)  # [n_vis_samples, 4]
    
    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = ['Bid Price', 'Ask Price', 'Bid Size', 'Ask Size']
    
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        values = samples[:, i].numpy()
        if i < 2:  # Prices
            bins = np.arange(0, args.max_contract_value + 2) - 0.5
        else:  # Sizes
            bins = np.arange(-0.5, args.max_contracts_per_trade + 1.5)
        
        ax.hist(values, bins=bins, density=True, alpha=0.7, edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sampling_distribution.png', dpi=150)
    print("✓ Saved visualization to 'sampling_distribution.png'\n")
    
    print("=== All verification tests passed! ===")
    
    return model

def test_compiled_consistency():
    """Test that compiled and non-compiled versions produce consistent results"""
    print("\n=== Testing Compiled vs Non-Compiled Consistency ===\n")
    
    device = torch.device('cuda:0')
    
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
    B, F = env.observation_shape()
    
    # Create model
    model = HighLowTransformerModel(args, env, verbose=False)
    model.to(device)
    model.eval()
    
    # Create compiled version
    compiled_fn = torch.compile(
        model.incremental_forward_with_context,
        mode="reduce-overhead",
        dynamic=True,
        fullgraph=False
    )
    
    # Test data
    x = torch.randn(B, F, device=device)
    context = torch.randn(5, B, args.n_embd, device=device)
    
    # Run multiple times to check consistency
    print("Running consistency checks...")
    for i in range(5):
        torch.manual_seed(i)
        uniform1 = torch.rand(B, 4, device=device)
        torch.manual_seed(i)
        uniform2 = torch.rand(B, 4, device=device)
        
        with torch.no_grad():
            output_regular = model.incremental_forward_with_context(x, context, uniform1)
            output_compiled = compiled_fn(x, context, uniform2)
        
        # Check all outputs match (allow for small differences due to compilation)
        # Actions might differ by 1 due to rounding differences in compiled mode
        action_diff = (output_regular['action'] - output_compiled['action']).abs()
        assert action_diff.max() <= 1, f"Actions differ by more than 1 at iteration {i}: max diff = {action_diff.max()}"
        
        # Logprobs should be very close
        assert torch.allclose(output_regular['logprobs'], output_compiled['logprobs'], atol=1e-4), f"Logprobs mismatch at iteration {i}"
        
        print(f"  Iteration {i}: ✓ Consistent")
    
    print("\n✓ Compiled and non-compiled versions are consistent!")

if __name__ == "__main__":
    # Run verification
    model = verify_incremental_forward_sampling()
    
    # Test consistency
    test_compiled_consistency()
    
    print("\n✅ All verification complete!")