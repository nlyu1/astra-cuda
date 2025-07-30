#!/usr/bin/env python3
"""Optimized benchmark that pre-generates data to minimize Python overhead"""

import time
import argparse
import torch
import astra_cuda
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    num_markets_per_block: int
    num_blocks: int
    total_markets: int
    latency_ms: float
    max_latency_ms: float
    operations_per_second: float
    markets_per_second: float
    speedup: float
    efficiency: float
    total_time_seconds: float
    total_steps: int


def test_market_throughput_optimized(num_markets_per_block: int, num_blocks: int, device_id: int) -> BenchmarkResult:
    """Test market throughput with pre-generated data"""
    total_markets = num_markets_per_block * num_blocks
    
    # Market parameters
    max_price_levels = 128
    max_active_orders_per_market = 1024
    max_active_fills_per_market = 1024
    num_customers = 16
    
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
    
    # Create FillBatch for reuse
    fills = market.new_fill_batch()
    
    # Prepare tensors on GPU
    device = torch.device(f'cuda:{device_id}')
    
    # Benchmark parameters
    warmup_steps = 100
    benchmark_steps = 10000
    
    # Set random seed
    torch.manual_seed(42)
    
    # Pre-generate all data to minimize Python overhead during benchmark
    print(f"Pre-generating {benchmark_steps} batches of data...", end='', flush=True)
    
    # Pre-allocate all tensors
    all_bid_px = torch.randint(40, 50, (benchmark_steps, total_markets), dtype=torch.uint32, device=device)
    all_ask_px = torch.randint(51, 61, (benchmark_steps, total_markets), dtype=torch.uint32, device=device)
    all_bid_sz = torch.randint(1, 101, (benchmark_steps, total_markets), dtype=torch.uint32, device=device)
    all_ask_sz = torch.randint(1, 101, (benchmark_steps, total_markets), dtype=torch.uint32, device=device)
    all_customer_ids = torch.randint(0, num_customers, (benchmark_steps, total_markets), dtype=torch.uint32, device=device)
    
    print(" Done!")
    
    # Warmup phase with pre-generated data
    for step in range(min(warmup_steps, benchmark_steps)):
        market.add_two_sided_quotes(
            all_bid_px[step], all_bid_sz[step], 
            all_ask_px[step], all_ask_sz[step], 
            all_customer_ids[step], fills
        )
    
    # Ensure GPU operations are complete
    torch.cuda.synchronize()
    
    # Benchmark phase - now with minimal Python overhead
    start_time = time.perf_counter()
    max_step_time_ms = 0.0
    
    for step in range(benchmark_steps):
        step_start = time.perf_counter()
        
        # Add quotes - using pre-generated data
        market.add_two_sided_quotes(
            all_bid_px[step], all_bid_sz[step], 
            all_ask_px[step], all_ask_sz[step], 
            all_customer_ids[step], fills
        )
        
        torch.cuda.synchronize()
        
        step_end = time.perf_counter()
        step_time_ms = (step_end - step_start) * 1000
        max_step_time_ms = max(max_step_time_ms, step_time_ms)
    
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
        speedup=0.0,
        efficiency=0.0,
        total_time_seconds=total_time_seconds,
        total_steps=benchmark_steps
    )


def main():
    parser = argparse.ArgumentParser(description='Optimized GPU Market Benchmark')
    parser.add_argument('-i', '--gpu_id', type=int, default=0,
                        help='Specify GPU device ID (default: 0)')
    args = parser.parse_args()
    
    device_id = args.gpu_id
    
    print("=== OPTIMIZED GPU Market Order Matching Benchmark ===")
    print("Pre-generating all test data to minimize Python overhead")
    print()
    
    if not torch.cuda.is_available():
        print("No CUDA devices found!")
        return 1
    
    device_count = torch.cuda.device_count()
    if device_id < 0 or device_id >= device_count:
        print(f"Error: Invalid GPU ID {device_id}")
        return 1
    
    # Test a few key configurations
    configs = [
        (64, 1),
        (256, 128),   # Best for RTX 5090
        (128, 64),    # Best for RTX 4060 Ti
        (512, 64),
        (1024, 64),
    ]
    
    results = []
    
    for markets_per_block, blocks in configs:
        print(f"\nTesting {markets_per_block} markets/block × {blocks} blocks...", flush=True)
        
        try:
            result = test_market_throughput_optimized(markets_per_block, blocks, device_id)
            results.append(result)
            
            print(f"Result: {result.markets_per_second:.0f} markets/sec, "
                  f"Latency: {result.latency_ms:.3f}ms")
        except Exception as e:
            print(f"Failed: {e}")
    
    # Find baseline
    if results:
        baseline_markets_per_second = results[0].markets_per_second  # First config
        for result in results:
            result.speedup = result.markets_per_second / baseline_markets_per_second
            result.efficiency = result.speedup / (result.total_markets / results[0].total_markets)
    
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Config':>20} {'Markets/sec':>15} {'Latency(ms)':>12} {'Speedup':>10}")
    print("-" * 60)
    for r in results:
        config = f"{r.num_markets_per_block}×{r.num_blocks}"
        print(f"{config:>20} {r.markets_per_second:>15.0f} {r.latency_ms:>12.3f} {r.speedup:>9.1f}x")
    
    return 0


if __name__ == "__main__":
    exit(main())