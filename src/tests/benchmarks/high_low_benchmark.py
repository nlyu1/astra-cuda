#!/usr/bin/env python3
"""
High-Low Trading Game CUDA Environment Benchmark

This script benchmarks the performance of the High-Low Trading Game
with various environment configurations on GPU.
"""

import time
import torch
import argparse
from tqdm import trange, tqdm
from typing import Dict, List
from dataclasses import dataclass

import astra_cuda


@dataclass
class BenchmarkConfig:
    num_blocks: int
    threads_per_block: int
    num_envs: int  # num_blocks * threads_per_block
    device_id: int


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    avg_step_time_ms: float
    max_step_time_ms: float
    total_time_seconds: float
    fps: float  # Frames (env-steps) per second
    total_steps: int


class EnvironmentBenchmark:
    def __init__(self, params: Dict):
        # Register games before loading
        astra_cuda.register_games()
        
        # Create game instance
        self.game = astra_cuda.load_game("high_low_trading", params)
        self.state = self.game.new_initial_state()
        
        # Extract game parameters
        self.num_players = params["players"]
        self.steps_per_player = params["steps_per_player"]
        self.max_contract_value = params["max_contract_value"]
        self.customer_max_size = params["customer_max_size"]
        self.num_envs = params["num_markets"]
        self.device_id = params["device_id"]
        self.threads_per_block = params["threads_per_block"]
        
        # Set device
        self.device = torch.device(f"cuda:{self.device_id}")
        
        # Pre-allocate persistent tensors for actions
        self.candidate_values = torch.zeros(
            (self.num_envs, 2), dtype=torch.int32, device=self.device
        )
        self.high_low_settle = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.permutation = torch.zeros(
            (self.num_envs, self.num_players), dtype=torch.int32, device=self.device
        )
        self.customer_sizes = torch.zeros(
            (self.num_envs, self.num_players - 3), dtype=torch.int32, device=self.device
        )
        self.trading_action = torch.zeros(
            (self.num_envs, 4), dtype=torch.int32, device=self.device
        )
        
        # Pre-allocate reward buffers
        self.immediate_rewards = torch.zeros(
            (self.num_envs, self.num_players), dtype=torch.float32, device=self.device
        )
        self.player_rewards = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.terminal_rewards = torch.zeros(
            (self.num_envs, self.num_players), dtype=torch.float32, device=self.device
        )
        
        # Pre-allocate observation buffer
        obs_shape = self.game.observation_tensor_shape()
        self.observation_buffer = torch.zeros(
            obs_shape, dtype=torch.float32, device=self.device
        )
        
        # Pre-generate random actions for all steps
        total_trading_steps = self.steps_per_player * self.num_players
        self.pre_generated_actions = {
            "trading": self._generate_random_trading_actions(total_trading_steps)
        }
    
    def _generate_random_trading_actions(self, num_steps: int) -> torch.Tensor:
        """Pre-generate random trading actions for all steps."""
        # Shape: [num_steps, num_envs, 4] for [bid_px, ask_px, bid_sz, ask_sz]
        actions = torch.zeros(
            (num_steps, self.num_envs, 4), dtype=torch.int32, device=self.device
        )
        
        # Generate random prices in range [1, max_contract_value]
        actions[:, :, 0] = torch.randint(
            1, self.max_contract_value + 1, 
            (num_steps, self.num_envs), dtype=torch.int32, device=self.device
        )
        actions[:, :, 1] = torch.randint(
            1, self.max_contract_value + 1,
            (num_steps, self.num_envs), dtype=torch.int32, device=self.device
        )
        
        # Generate random sizes in range [0, 1] (max_contracts_per_trade is 1)
        actions[:, :, 2] = torch.randint(
            0, 2, (num_steps, self.num_envs), dtype=torch.int32, device=self.device
        )
        actions[:, :, 3] = torch.randint(
            0, 2, (num_steps, self.num_envs), dtype=torch.int32, device=self.device
        )
        
        return actions
    
    def randomize_actions(self, move_number: int, trading_step: int = 0):
        """Set up actions for the current move."""
        if move_number == 0:
            # First chance move: two candidate contract values [1, max_contract_value]
            self.candidate_values[:, 0].fill_(3)
            self.candidate_values[:, 1].fill_(8)
        elif move_number == 1:
            # Second chance move: high/low settlement (0 or 1)
            self.high_low_settle = torch.arange(
                self.num_envs, dtype=torch.int32, device=self.device
            ) % 2
        elif move_number == 2:
            # Third chance move: permutation
            base_perm = torch.arange(
                self.num_players, dtype=torch.int32, device=self.device
            )
            self.permutation = base_perm.unsqueeze(0).repeat(self.num_envs, 1)
        elif move_number == 3:
            # Fourth chance move: customer sizes
            self.customer_sizes[:, 0].fill_(-2)
            if self.num_players - 3 > 1:
                self.customer_sizes[:, 1].fill_(2)
        else:
            # Player trading actions: use pre-generated actions
            self.trading_action = self.pre_generated_actions["trading"][trading_step]
    
    def get_current_action(self, move_number: int) -> torch.Tensor:
        """Get the action tensor for the current move."""
        if move_number == 0:
            return self.candidate_values
        elif move_number == 1:
            return self.high_low_settle
        elif move_number == 2:
            return self.permutation
        elif move_number == 3:
            return self.customer_sizes
        else:
            return self.trading_action
    
    def run_benchmark(self, num_episodes: int) -> BenchmarkResult:
        """Run the benchmark for specified number of episodes."""
        start_time = time.perf_counter()
        max_step_time_ms = 0.0
        total_step_time_ms = 0.0
        total_steps = 0
        
        for episode in trange(num_episodes):
            self.state.reset()
            
            # Apply chance nodes
            # First chance move: candidate values
            self.randomize_actions(0)
            self.state.apply_action(self.get_current_action(0))
            
            # Second chance move: high/low settlement
            self.randomize_actions(1)
            self.state.apply_action(self.get_current_action(1))
            
            # Third chance move: permutation
            self.randomize_actions(2)
            self.state.apply_action(self.get_current_action(2))
            
            # Fourth chance move: customer sizes
            self.randomize_actions(3)
            self.state.apply_action(self.get_current_action(3))
            
            # Now handle player trading actions
            trading_step = 0
            while not self.state.is_terminal():
                current_player = self.state.current_player()
                
                # Generate random trading action
                self.randomize_actions(4, trading_step)
                trading_step += 1
                
                action = self.get_current_action(4)
                
                step_start = time.perf_counter()
                
                # Apply action
                self.state.fill_observation_tensor(self.observation_buffer)
                self.state.apply_action(action)
                
                # Fill immediate rewards after each step
                self.state.fill_rewards(self.immediate_rewards)
                
                # Get cumulative rewards since last action for this player
                self.state.fill_rewards_since_last_action(
                    self.player_rewards, current_player
                )
                
                step_end = time.perf_counter()
                
                step_time_ms = (step_end - step_start) * 1000
                max_step_time_ms = max(max_step_time_ms, step_time_ms)
                total_step_time_ms += step_time_ms
                
                total_steps += 1
            
            # Get terminal rewards
            self.state.fill_returns(self.terminal_rewards)
        
        end_time = time.perf_counter()
        total_time_seconds = end_time - start_time
        avg_step_time_ms = total_step_time_ms / total_steps
        fps = float(total_steps * self.num_envs) / total_time_seconds
        
        # Compute num_blocks from num_envs and threads_per_block
        threads_per_block = self.threads_per_block
        num_blocks = (self.num_envs + threads_per_block - 1) // threads_per_block
        
        return BenchmarkResult(
            config=BenchmarkConfig(
                num_blocks=num_blocks,
                threads_per_block=threads_per_block,
                num_envs=self.num_envs,
                device_id=self.device_id
            ),
            avg_step_time_ms=avg_step_time_ms,
            max_step_time_ms=max_step_time_ms,
            total_time_seconds=total_time_seconds,
            fps=fps,
            total_steps=total_steps
        )


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print(f"{'Envs':>12} {'Blocks':>12} {'Threads/Block':>15} {'Device':>10} "
          f"{'Avg Step(ms)':>15} {'Max Step(ms)':>15} {'Total Time(s)':>15} {'FPS':>15}")
    print("-" * 115)
    
    for result in results:
        print(f"{result.config.num_envs:>12} "
              f"{result.config.num_blocks:>12} "
              f"{result.config.threads_per_block:>15} "
              f"{result.config.device_id:>10} "
              f"{result.avg_step_time_ms:>15.3f} "
              f"{result.max_step_time_ms:>15.3f} "
              f"{result.total_time_seconds:>15.3f} "
              f"{result.fps:>15.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark High-Low Trading CUDA Environment"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device ID (default: 0)"
    )
    parser.add_argument(
        "--episodes", type=int, default=200, 
        help="Number of episodes per configuration (default: 200)"
    )
    args = parser.parse_args()
    
    print("=== High Low Trading CUDA Environment Benchmark ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        exit(1)
    
    device_count = torch.cuda.device_count()
    if args.device >= device_count:
        print(f"Error: Device {args.device} not found. "
              f"Only {device_count} devices available.")
        exit(1)
    
    # Print device info
    device_name = torch.cuda.get_device_name(args.device)
    print(f"Using device {args.device}: {device_name}")
    print()
    
    # Test configurations
    block_counts = [256, 512, 1024]
    thread_counts = [64, 128, 256]
    
    results = []
    
    print(f"Running benchmarks with {args.episodes} episodes per configuration...")
    print()
    
    for num_blocks in block_counts:
        for threads_per_block in thread_counts:
            num_envs = num_blocks * threads_per_block
            print(f"Testing {num_blocks} blocks with {threads_per_block} threads/block = "
                  f"{num_envs} environments...", end="", flush=True)
            
            # Create game parameters matching the C++ version
            params = {
                "steps_per_player": 8,
                "max_contracts_per_trade": 1,
                "customer_max_size": 2,
                "max_contract_value": 10,
                "players": 5,
                "num_markets": num_envs,
                "threads_per_block": threads_per_block,
                "device_id": args.device
            }
            
            try:
                benchmark = EnvironmentBenchmark(params)
                result = benchmark.run_benchmark(args.episodes)
                torch.cuda.synchronize()
                results.append(result)
                
                print(f" Done (FPS: {result.fps:.0f})")
            except Exception as e:
                print(f" Failed: {e}")
    
    print()
    print("=== BENCHMARK RESULTS ===")
    print()
    
    print_results(results)
    
    print()
    print("=== ANALYSIS ===")
    
    # Find best configuration
    if results:
        best_result = max(results, key=lambda r: r.fps)
        print(f"Best FPS: {best_result.fps:.0f} FPS "
              f"({best_result.config.num_blocks} blocks × "
              f"{best_result.config.threads_per_block} threads/block = "
              f"{best_result.config.num_envs} envs)")
        print(f"Best config latency: Avg {best_result.avg_step_time_ms:.3f}ms, "
              f"Max {best_result.max_step_time_ms:.3f}ms")
