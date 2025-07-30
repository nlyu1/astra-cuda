#!/usr/bin/env python3
"""Speed benchmark for vectorized GPU market - Python version"""

import time
import argparse
import torch
import astra_cuda
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    num_markets_per_block: int  # threads per block
    num_blocks: int
    total_markets: int
    latency_ms: float
    max_latency_ms: float
    operations_per_second: float  # operations (batches) per second
    markets_per_second: float     # total markets processed per second
    speedup: float                # Speedup vs single-market baseline
    efficiency: float             # Parallel efficiency (speedup / total_markets)
    total_time_seconds: float
    total_steps: int


def test_market_throughput(num_markets_per_block: int, num_blocks: int, device_id: int) -> BenchmarkResult:
    """Test market throughput for a specific configuration"""
    # Calculate total markets
    total_markets = num_markets_per_block * num_blocks
    
    # Market parameters
    max_price_levels = 128
    max_active_orders_per_market = 1024
    max_active_fills_per_market = 1024
    num_customers = 16  # Number of customers per market
    
    # Create VecMarket instance
    market = astra_cuda.order_matching.VecMarket(
        num_markets=total_markets,
        max_price_levels=max_price_levels,
        max_active_orders_per_market=max_active_orders_per_market,
        max_active_fills_per_market=max_active_fills_per_market,
        num_customers=num_customers,
        device_id=device_id,
        threads_per_block=num_markets_per_block
    )
    
    # Create FillBatch and BBOBatch for reuse
    fills = market.new_fill_batch()
    bbos = market.new_bbo_batch()
    
    # Prepare tensors on GPU
    device = torch.device(f'cuda:{device_id}')
    
    # Benchmark parameters
    warmup_steps = 100
    benchmark_steps = 10000
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Warmup phase
    for step in range(warmup_steps):
        # Generate random values directly on GPU
        # Bid prices: 40-49, Ask prices: 51-60
        bid_px = torch.randint(40, 50, (total_markets,), dtype=torch.uint32, device=device)
        ask_px = torch.randint(51, 61, (total_markets,), dtype=torch.uint32, device=device)
        
        # Sizes: 1-100
        bid_sz = torch.randint(1, 101, (total_markets,), dtype=torch.uint32, device=device)
        ask_sz = torch.randint(1, 101, (total_markets,), dtype=torch.uint32, device=device)
        
        # Customer IDs: 0-(num_customers-1)
        customer_ids = torch.randint(0, num_customers, (total_markets,), dtype=torch.uint32, device=device)
        
        market.add_two_sided_quotes(bid_px, bid_sz, ask_px, ask_sz, customer_ids, fills)
    
    # Ensure GPU operations are complete before starting benchmark
    torch.cuda.synchronize()
    
    # Benchmark phase
    start_time = time.perf_counter()
    max_step_time_ms = 0.0
    
    for step in range(benchmark_steps):
        # Generate random values directly on GPU
        # Bid prices: 40-49, Ask prices: 51-60
        bid_px = torch.randint(40, 50, (total_markets,), dtype=torch.uint32, device=device)
        ask_px = torch.randint(51, 61, (total_markets,), dtype=torch.uint32, device=device)
        
        # Sizes: 1-100
        bid_sz = torch.randint(1, 101, (total_markets,), dtype=torch.uint32, device=device)
        ask_sz = torch.randint(1, 101, (total_markets,), dtype=torch.uint32, device=device)
        
        # Customer IDs: 0-(num_customers-1)
        customer_ids = torch.randint(0, num_customers, (total_markets,), dtype=torch.uint32, device=device)
        
        step_start = time.perf_counter()
        
        # Add quotes and match orders
        market.add_two_sided_quotes(bid_px, bid_sz, ask_px, ask_sz, customer_ids, fills)
        
        # Ensure kernel completion for accurate timing
        torch.cuda.synchronize()
        
        step_end = time.perf_counter()
        
        step_time_ms = (step_end - step_start) * 1000
        max_step_time_ms = max(max_step_time_ms, step_time_ms)
        
        # Optionally get BBOs and customer portfolios to ensure full pipeline
        if step % 100 == 0:
            market.get_bbos(bbos)
            # Get customer portfolios to test the new functionality
            portfolios = market.get_customer_portfolios()
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time_seconds = end_time - start_time
    latency_ms = (total_time_seconds * 1000.0) / benchmark_steps
    operations_per_second = benchmark_steps / total_time_seconds
    markets_per_second = (benchmark_steps * total_markets) / total_time_seconds
    
    return BenchmarkResult(
        num_markets_per_block=num_markets_per_block,
        num_blocks=num_blocks,
        total_markets=total_markets,
        latency_ms=latency_ms,
        max_latency_ms=max_step_time_ms,
        operations_per_second=operations_per_second,
        markets_per_second=markets_per_second,
        speedup=0.0,  # will be calculated later
        efficiency=0.0,  # will be calculated later
        total_time_seconds=total_time_seconds,
        total_steps=benchmark_steps
    )


def print_benchmark_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table"""
    # Print header
    print(f"{'Markets/Block':>15} {'Blocks':>12} {'Total Markets':>15} {'Lat(ms)':>12} {'Markets/sec':>18} {'Speedup':>12} {'Efficiency':>12}")
    print('-' * 106)
    
    # Print results
    for result in results:
        print(f"{result.num_markets_per_block:>15} {result.num_blocks:>12} {result.total_markets:>15} "
              f"{result.latency_ms:>12.3f} {result.markets_per_second:>18.0f} "
              f"{result.speedup:>11.1f}x {result.efficiency * 100:>11.1f}%")


def main():
    parser = argparse.ArgumentParser(description='GPU Market Order Matching Benchmark')
    parser.add_argument('-i', '--gpu_id', type=int, default=0,
                        help='Specify GPU device ID (default: 0)')
    args = parser.parse_args()
    
    device_id = args.gpu_id
    
    print("=== GPU Market Order Matching Benchmark ===")
    print("Configuration:")
    print("- Max price levels: 128")
    print("- Max orders per market: 1024")
    print("- Max fills per market: 1024")
    print("- Number of customers per market: 16")
    print("- Random bid/ask orders with crossing prices")
    print()
    
    # Check CUDA device
    if not torch.cuda.is_available():
        print("No CUDA devices found!")
        return 1
    
    device_count = torch.cuda.device_count()
    
    # Validate GPU ID
    if device_id < 0 or device_id >= device_count:
        print(f"Error: Invalid GPU ID {device_id}. Available GPUs: 0-{device_count - 1}")
        return 1
    
    # Get device properties
    device_props = torch.cuda.get_device_properties(device_id)
    print(f"Using GPU {device_id}: {device_props.name}")
    print(f"- Multi-processor count: {device_props.multi_processor_count}")
    print(f"- Total memory: {device_props.total_memory / (1024**3):.1f} GB")
    print()
    
    # Test configurations
    configs = []
    
    # Varying threads per block (num_markets_per_block)
    threads_per_block = [64, 128, 256, 512, 1024]
    num_blocks = [1, 64, 128, 256, 512, 1024]
    
    # Generate all combinations that don't exceed MAX_MARKETS (106496)
    for tpb in threads_per_block:
        for nb in num_blocks:
            total = tpb * nb
            if total <= 106496:  # MAX_MARKETS constraint
                configs.append((tpb, nb))
    
    results = []
    
    print("Running benchmarks...")
    print()
    
    for markets_per_block, blocks in configs:
        print(f"Testing {markets_per_block} markets/block × {blocks} blocks "
              f"(total: {markets_per_block * blocks} markets)...", end='', flush=True)
        
        try:
            result = test_market_throughput(markets_per_block, blocks, device_id)
            results.append(result)
            
            print(f" Done (Markets/sec: {result.markets_per_second:.0f}, "
                  f"Latency: {result.latency_ms:.2f}ms)")
        except Exception as e:
            print(f" Failed: {e}")
    
    # Calculate speedup and efficiency compared to smallest configuration
    if results:
        # Use the smallest configuration as baseline (64 markets)
        baseline_markets_per_second = 0
        for result in results:
            if result.total_markets == 64:  # Smallest configuration
                baseline_markets_per_second = result.markets_per_second
                break
        
        # Calculate speedup and efficiency
        for result in results:
            result.speedup = result.markets_per_second / baseline_markets_per_second
            result.efficiency = result.speedup / (result.total_markets / 64.0)  # Normalized by market count increase
    
    print()
    print("=== BENCHMARK RESULTS ===")
    print()
    
    print_benchmark_results(results)
    
    print()
    print("=== ANALYSIS ===")
    
    # Find best configurations
    if results:
        best_throughput = max(results, key=lambda r: r.markets_per_second)
        best_latency = min(results, key=lambda r: r.latency_ms)
        
        print(f"Best Throughput: {best_throughput.markets_per_second:.0f} markets/sec "
              f"({best_throughput.num_markets_per_block} markets/block × {best_throughput.num_blocks} blocks = "
              f"{best_throughput.total_markets} total markets, {best_throughput.speedup:.1f}x speedup)")
        
        print(f"Best Latency: {best_latency.latency_ms:.3f}ms "
              f"({best_latency.num_markets_per_block} markets/block × {best_latency.num_blocks} blocks)")
        
        # Find baseline for speedup analysis
        baseline_markets_per_second = 0
        for result in results:
            if result.total_markets == 64:
                baseline_markets_per_second = result.markets_per_second
                break
        
        print()
        print("Speedup Analysis:")
        print(f"- Baseline (64 markets): {baseline_markets_per_second:.0f} markets/sec")
        print(f"- Peak speedup: {best_throughput.speedup:.1f}x with {best_throughput.total_markets} markets")
        print(f"- Peak efficiency: {best_throughput.efficiency * 100:.1f}% parallel efficiency")
        
        # Find best configuration for different block counts
        print()
        print("Best configurations by block count:")
        for nb in num_blocks:
            best_for_blocks = None
            for result in results:
                if result.num_blocks == nb:
                    if best_for_blocks is None or result.markets_per_second > best_for_blocks.markets_per_second:
                        best_for_blocks = result
            
            if best_for_blocks:
                print(f"- {nb} blocks: {best_for_blocks.num_markets_per_block} markets/block "
                      f"(Markets/sec: {best_for_blocks.markets_per_second:.0f})")
    
    return 0


if __name__ == "__main__":
    exit(main())