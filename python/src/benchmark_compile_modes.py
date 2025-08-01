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

args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')

game_config = args.get_game_config()
env = HighLowTrading(game_config)

B, F = env.observation_shape()
T = args.steps_per_player
print(f"Testing with B={B}, T={T}, F={F}")

# Create test data
mock_input = torch.randn(B, T, F).to(device)
actions = torch.randint(1, 4, (B, T, 4)).to(device)

def benchmark_compile_mode(mode, iterations=100):
    """Benchmark a specific compile mode."""
    print(f"\n{'='*50}")
    print(f"Testing compile mode: {mode}")
    print('='*50)
    
    # Create fresh model
    model = OptimizedTransformerModel(args, env).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Time compilation
    compile_start = time.time()
    model = model.compile(mode=mode, fullgraph=True)
    compile_time = time.time() - compile_start
    print(f"Compilation time: {compile_time:.2f}s")
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(mock_input, actions)
            loss = outputs['values'].pow(2).mean() + outputs['logprobs'].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {iterations} iterations...")
    times = []
    
    for i in range(iterations):
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
        times.append((end - start) * 1000)  # ms
        
        if i == 0:
            print(f"First iteration: {times[0]:.2f}ms")
    
    # Calculate statistics (exclude first 20 iterations)
    times = np.array(times[20:])
    
    results = {
        'mode': mode,
        'compile_time': compile_time,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'p50': np.percentile(times, 50),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
    }
    
    print(f"\nResults for {mode}:")
    print(f"  Mean: {results['mean']:.2f}ms ± {results['std']:.2f}ms")
    print(f"  Min:  {results['min']:.2f}ms")
    print(f"  P50:  {results['p50']:.2f}ms")
    print(f"  P95:  {results['p95']:.2f}ms")
    print(f"  P99:  {results['p99']:.2f}ms")
    
    return results

# Test different modes
modes = ['reduce-overhead', 'max-autotune', 'default']
results = {}

for mode in modes:
    try:
        results[mode] = benchmark_compile_mode(mode)
    except Exception as e:
        print(f"Error with mode {mode}: {e}")

# Summary
print(f"\n{'='*60}")
print("SUMMARY COMPARISON")
print('='*60)
print(f"{'Mode':<20} {'Compile(s)':<12} {'Mean(ms)':<10} {'P95(ms)':<10} {'Std(ms)':<10}")
print('-'*60)

for mode in modes:
    if mode in results:
        r = results[mode]
        print(f"{mode:<20} {r['compile_time']:<12.1f} {r['mean']:<10.2f} {r['p95']:<10.2f} {r['std']:<10.2f}")

# Recommendation
if 'reduce-overhead' in results and 'max-autotune' in results:
    overhead_mean = results['reduce-overhead']['mean']
    autotune_mean = results['max-autotune']['mean']
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print('='*60)
    
    if overhead_mean < 100 and overhead_mean < autotune_mean:
        print("✓ Use 'reduce-overhead' - achieves <100ms with lower latency")
    elif autotune_mean < 100 and autotune_mean < overhead_mean:
        print("✓ Use 'max-autotune' - achieves <100ms with better performance")
    else:
        print("✓ Use 'reduce-overhead' - better for consistent low latency")
    
    print(f"\nLatency difference: {abs(overhead_mean - autotune_mean):.2f}ms")
    print(f"Reduce-overhead is {autotune_mean/overhead_mean:.2f}x the speed" if overhead_mean < autotune_mean 
          else f"Max-autotune is {overhead_mean/autotune_mean:.2f}x the speed")