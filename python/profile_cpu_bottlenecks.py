#!/usr/bin/env python3
"""
CPU profiling script for vtrace_baseline.py to identify bottlenecks.
"""
import cProfile
import pstats
import io
import time
from contextlib import contextmanager
import torch
import sys
sys.path.append('./src')

@contextmanager
def profile_section(name):
    """Context manager for profiling specific code sections."""
    pr = cProfile.Profile()
    start_time = time.time()
    pr.enable()
    yield
    pr.disable()
    elapsed = time.time() - start_time
    
    # Print timing
    print(f"\n{'='*60}")
    print(f"Section: {name}")
    print(f"Total time: {elapsed:.3f} seconds")
    print(f"{'='*60}")
    
    # Print top functions by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())


def run_profiling_test():
    """Run a full training loop to profile CPU usage."""
    import random
    import numpy as np
    from high_low.agent import HighLowTransformerModel
    from high_low.config import Args
    from high_low.env import HighLowTrading
    from high_low.logger import HighLowLogger
    from high_low.impala import HighLowImpalaTrainer, HighLowImpalaBuffer
    from arena import Arena
    import tyro
    from tqdm import tqdm
    
    # Parse args
    args = tyro.cli(Args)
    args.run_name = f"CPU_PROFILE__{args.exp_name}__{int(time.time())}"
    args.fill_runtime_args()
    # Use full settings but limit iterations
    args.num_iterations = 300
    
    print(f"Running CPU profiling with {args.num_envs} environments...")
    print(f"Steps per iteration: {args.num_steps}")
    print(f"Total iterations: {args.num_iterations}")
    game_config = args.get_game_config()
    
    # Initialize everything
    env = HighLowTrading(game_config)
    device = torch.device(f'cuda:{args.device_id}')
    
    # Create initial agents
    initial_agents = {}
    for j in range(args.players - 1):
        name = f"Random{j}"
        initial_agents[name] = HighLowTransformerModel(args, env, verbose=False).to(device)
        initial_agents[name].compile()
    
    # Create arena and local agent
    pool = Arena(env, initial_agents, device)
    buffer = HighLowImpalaBuffer(args, env.num_features(), device)
    local_agent = HighLowTransformerModel(args, env).to(device)
    local_agent.compile()
    trainer = HighLowImpalaTrainer(args, local_agent, device=device)
    
    # Create buffers
    observation_buffer = env.new_observation_buffer()
    reward_buffer = env.new_reward_buffer()
    returns_buffer = env.new_reward_buffer()
    player_reward_buffer = env.new_player_reward_buffer()
    
    # Warmup compilation (not profiled)
    print("\nWarming up compilation...")
    for _ in range(2):
        player_offset = 0
        npc_agent_names = pool.select_topk(game_config['players'] - 1)
        npc_agents = [pool.agents[name] for name in npc_agent_names]
        for agent in npc_agents:
            agent.reset_context()
        local_agent.reset_context()
        
        for step in range(2):
            env.fill_observation_tensor(observation_buffer)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.inference_mode():
                    _ = local_agent.incremental_forward(observation_buffer, step)
        env.reset()
    
    print("\n\nStarting profiled training loop...")
    
    # Profile the full training loop
    pr = cProfile.Profile()
    pr.enable()
    
    # Disable garbage collection for performance
    import gc
    gc.disable()
    
    agent_name = "0"
    global_step = 0
    
    for iteration in tqdm(range(args.num_iterations)):
        # Manual GC every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            gc.collect()
        # Select agents
        player_offset = np.random.randint(0, game_config['players'])
        npc_agent_names = pool.select_topk(game_config['players'] - 1)
        npc_agents = [pool.agents[name] for name in npc_agent_names]
        for agent in npc_agents:
            agent.reset_context()
        local_agent.reset_context()
        
        round_agent_names = (
            [npc_agent_names[j] for j in range(player_offset)] 
            + [agent_name]
            + [npc_agent_names[j - 1] for j in range(player_offset + 1, game_config['players'])])
        
        # Full rollout
        settlement_preds, private_role_preds = [], []
        for step in range(args.num_steps):
            global_step += args.num_envs
            
            # NPCs before player
            for npc_id in range(player_offset):
                env.fill_observation_tensor(observation_buffer)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    with torch.inference_mode():
                        npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
                env.step(npc_actions)
            
            # Player action
            if step > 0:
                env.fill_rewards_since_last_action(buffer.rewards[step - 1])
                buffer.update_late_stats(
                    {'dones': torch.zeros(args.num_envs, device=device).float()}, step - 1)
            
            env.fill_observation_tensor(buffer.obs[step])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                with torch.inference_mode():
                    forward_results = local_agent.incremental_forward(buffer.obs[step], step)
            action, log_probs = forward_results['action'], forward_results['logprobs']
            settlement_preds.append(forward_results['pinfo_preds']['settle_price'].clone())
            private_role_preds.append(forward_results['pinfo_preds']['private_roles'].argmax(dim=-1))
            
            buffer.update({
                'actions': action,
                'logprobs': log_probs,
            }, step)
            env.step(action)
            
            # NPCs after player
            for player_id in range(player_offset + 1, game_config['players']):
                npc_id = player_id - 1
                env.fill_observation_tensor(observation_buffer)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    with torch.inference_mode():
                        npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
                env.step(npc_actions)
            
            if env.terminal():
                env.fill_returns(returns_buffer)
                env.fill_rewards_since_last_action(buffer.rewards[step], player_offset)
                buffer.update_late_stats({
                    'rewards': player_reward_buffer,
                    'dones': torch.ones(args.num_envs).to(device).float()}, step)
                
                env_info = env.expose_info()
                env_pinfo_targets = env.get_pinfo_targets()
                pool.register_playout_scores(returns_buffer.mean(0), round_agent_names)
                
                buffer.actual_private_roles.copy_(env_pinfo_targets['pinfo_targets'], non_blocking=True)
                buffer.actual_settlement.copy_(env_pinfo_targets['settlement_values'], non_blocking=True)
                
                env.reset()
        
        # Training update
        update_dictionary = buffer.get_update_dictionary()
        trainer_results = trainer.train(update_dictionary)
        
        # Periodic agent registration
        if (iteration + 1) % 10 == 0:
            pool.register_agent(local_agent, agent_name)
            agent_name = str(iteration + 1)
    
    pr.disable()
    
    # Re-enable garbage collection
    gc.enable()
    
    # Print profiling results
    print("\n" + "="*80)
    print("FULL TRAINING LOOP PROFILING RESULTS")
    print("="*80)
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(50)
    print(s.getvalue())
    
    # Also sort by total time
    print("\n" + "="*80)
    print("TOP FUNCTIONS BY TOTAL TIME")
    print("="*80)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())


def analyze_specific_bottlenecks():
    """Analyze specific suspected bottlenecks."""
    import torch
    from copy import deepcopy
    from high_low.agent import HighLowTransformerModel
    from high_low.config import Args
    from high_low.env import HighLowTrading
    import tyro
    
    args = tyro.cli(Args)
    args.fill_runtime_args()
    game_config = args.get_game_config()
    env = HighLowTrading(game_config)
    device = torch.device(f'cuda:{args.device_id}')
    
    print("\nAnalyzing specific bottlenecks...")
    
    # Test deepcopy performance
    agent = HighLowTransformerModel(args, env, verbose=False).to(device)
    agent.compile()
    
    print("\n1. Deepcopy performance:")
    start = time.time()
    for _ in range(5):
        _ = deepcopy(agent)
    elapsed = time.time() - start
    print(f"   Average deepcopy time: {elapsed/5:.3f} seconds")
    
    # Test state_dict copying (alternative to deepcopy)
    print("\n2. State dict copying (alternative):")
    start = time.time()
    for _ in range(5):
        new_agent = HighLowTransformerModel(args, env, verbose=False).to(device)
        new_agent.load_state_dict(agent.state_dict())
        new_agent.compile()
    elapsed = time.time() - start
    print(f"   Average state_dict copy time: {elapsed/5:.3f} seconds")
    
    # Test CPU-GPU transfers
    print("\n3. CPU-GPU transfer overhead:")
    test_tensor = torch.randn(1024, 4, device=device)
    start = time.time()
    for _ in range(300):
        _ = test_tensor.cpu()
    elapsed = time.time() - start
    print(f"   Average GPU->CPU transfer time (1024x4 tensor): {elapsed/100*1000:.3f} ms")
    
    # Test arena probability calculations
    print("\n4. Arena probability calculations:")
    from arena import Arena
    initial_agents = {f"agent_{i}": agent for i in range(10)}
    arena = Arena(env, initial_agents, device)
    
    # Simulate some scores
    for name in arena.agents:
        arena.playout_scores[name] = [torch.randn(1).item() for _ in range(100)]
    
    start = time.time()
    for _ in range(100):
        _ = arena.select_topk(4)
    elapsed = time.time() - start
    print(f"   Average agent selection time: {elapsed/100*1000:.3f} ms")


if __name__ == "__main__":
    print("Starting CPU profiling analysis...")
    print("This will run a full training loop to identify bottlenecks.")
    
    # Run main profiling
    run_profiling_test()
    
    print("\n\nProfiling complete!")