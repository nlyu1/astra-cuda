# Compare performance between original and optimized transformer
import time
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

import sys
sys.path.append('./src')
import numpy as np
from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.transformer_agent import HighLowTransformerModel
from high_low.optimized_transformer_agent import OptimizedTransformerModel
import astra_cuda as astra
astra.register_games()

# Setup
args = Args()
args.n_hidden = 256
args.n_embd = 128
args.n_head = 2
args.n_layer = 1
args.pre_encoder_blocks = 2
args.post_decoder_blocks = 1
args.device_id = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

args.exp_name = 'performance_comparison'
args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')

game_config = args.get_game_config()
env = HighLowTrading(game_config)

# Create both models
print("Creating models...")
original_agent = HighLowTransformerModel(args, env).to(device)
optimized_agent = OptimizedTransformerModel(args, env).to(device)

# Compile both
print("Compiling models...")
original_agent = original_agent.compile(mode="reduce-overhead")
optimized_agent = optimized_agent.compile(mode="reduce-overhead")

B, F = env.observation_shape()
T = args.steps_per_player
print(f"\nBatch size: {B}, Sequence length: {T}, Feature dim: {F}")

# Count parameters
original_params = sum(p.numel() for p in original_agent.parameters())
optimized_params = sum(p.numel() for p in optimized_agent.parameters())
print(f"Original model parameters: {original_params:,}")
print(f"Optimized model parameters: {optimized_params:,}")

# Create test data
mock_input = torch.randn(B, T, F).to(device)
actions = torch.randint(1, 4, (B, T, 4)).to(device)

# Warmup
print("\nWarming up...")
for _ in range(10):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _ = original_agent(mock_input, actions)
        _ = optimized_agent(mock_input, actions)
torch.cuda.synchronize()

# Benchmark function
def benchmark_model(model, name, mock_input, actions, iterations=100):
    times = []
    
    # Create optimizer for backward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(mock_input, actions)
            loss = outputs['values'].pow(2).mean() + outputs['logprobs'].mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times[10:])  # Skip first 10 for stability
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95)
    }

# Benchmark both models
print("\nBenchmarking original model...")
original_stats = benchmark_model(original_agent, "Original", mock_input, actions)

print("Benchmarking optimized model...")
optimized_stats = benchmark_model(optimized_agent, "Optimized", mock_input, actions)

# Print results
print("\n" + "="*60)
print("PERFORMANCE COMPARISON (Training Step Time)")
print("="*60)

print(f"\nOriginal Transformer:")
print(f"  Mean: {original_stats['mean']:.2f}ms")
print(f"  Std:  {original_stats['std']:.2f}ms")
print(f"  Min:  {original_stats['min']:.2f}ms")
print(f"  P50:  {original_stats['p50']:.2f}ms")
print(f"  P95:  {original_stats['p95']:.2f}ms")

print(f"\nOptimized Transformer:")
print(f"  Mean: {optimized_stats['mean']:.2f}ms")
print(f"  Std:  {optimized_stats['std']:.2f}ms")
print(f"  Min:  {optimized_stats['min']:.2f}ms")
print(f"  P50:  {optimized_stats['p50']:.2f}ms")
print(f"  P95:  {optimized_stats['p95']:.2f}ms")

print(f"\nSpeedup: {original_stats['mean'] / optimized_stats['mean']:.2f}x")
print(f"Time saved per step: {original_stats['mean'] - optimized_stats['mean']:.2f}ms")

# Test inference speed
print("\n" + "="*60)
print("INFERENCE COMPARISON (Action Sampling)")
print("="*60)

def benchmark_inference(model, name, mock_input, iterations=1000):
    times = []
    hidden = model.initial_belief()
    
    for t in range(min(T, 10)):  # Test first 10 timesteps
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(iterations):
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model.sample_actions(mock_input[:, t], hidden)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        hidden = outputs['hidden']
        
        times.append((end - start) / iterations * 1000)  # ms per sample
    
    return np.mean(times)

print("\nBenchmarking inference...")
original_inference = benchmark_inference(original_agent, "Original", mock_input)
optimized_inference = benchmark_inference(optimized_agent, "Optimized", mock_input)

print(f"\nOriginal inference time: {original_inference:.3f}ms per action")
print(f"Optimized inference time: {optimized_inference:.3f}ms per action")
print(f"Inference speedup: {original_inference / optimized_inference:.2f}x")

print("\n" + "="*60)
print("KEY OPTIMIZATIONS APPLIED:")
print("="*60)
print("1. Fused operations with JIT compilation")
print("2. Pre-computed position embeddings and masks")
print("3. Shared actor computations")
print("4. Reduced tensor reshaping")
print("5. Optimized entropy computation")
print("6. Single decoder projection")
print("7. Vectorized log probability calculation")
print("="*60)