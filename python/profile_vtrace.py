#!/usr/bin/env python3
"""
CPU profiling script for vtrace.py to identify bottlenecks.
Runs training loop for 100 iterations with profiling enabled.
"""
import cProfile
import pstats
import io
import time
from contextlib import contextmanager
import torch
import sys
import gc

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
    from timer import Timer, OneTickTimer
    import tyro
    from tqdm import tqdm
    from pathlib import Path
    from copy import deepcopy
    
    # Parse args with default config
    args = tyro.cli(Args)
    args.run_name = f"CPU_PROFILE__{args.exp_name}__{int(time.time())}"
    args.fill_runtime_args()
    
    # Override iterations to 100 for profiling
    args.num_iterations = 100
    args.track = False  # Disable wandb tracking for profiling
    
    print(f"Running CPU profiling with {args.num_envs} environments...")
    print(f"Steps per iteration: {args.num_steps}")
    print(f"Total iterations: {args.num_iterations}")
    print(f"Game config: players={args.players}, steps_per_player={args.steps_per_player}")
    
    game_config = args.get_game_config()
    
    # Initialize everything
    env = HighLowTrading(game_config)
    device = torch.device(f'cuda:{args.device_id}')
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create initial agents
    initial_agents = {}
    for j in range(args.players - 1):
        name = f"Random{j}"
        initial_agents[name] = HighLowTransformerModel(args, env, verbose=False).to(device)
    
    # Create arena and local agent
    pool = Arena(env, initial_agents, device)
    num_features = env.num_features()
    buffer = HighLowImpalaBuffer(args, num_features, device)
    local_agent = HighLowTransformerModel(args, env).to(device)
    trainer = HighLowImpalaTrainer(
        args, local_agent,
        checkpoint_interval=args.iterations_per_checkpoint,
        device=device)
    
    # Create NPC agents
    npc_agents = [
        HighLowTransformerModel(args, env, verbose=False).to(device)
        for _ in range(game_config['players'] - 1)]
    
    # Create buffers
    observation_buffer = env.new_observation_buffer()
    reward_buffer = env.new_reward_buffer()
    returns_buffer = env.new_reward_buffer()
    player_reward_buffer = env.new_player_reward_buffer()
    
    # Pre-allocate tensors
    done_zeros = torch.zeros(args.num_envs, device=device).float()
    done_ones = torch.ones(args.num_envs, device=device).float()
    
    # Warmup compilation (not profiled)
    # print("\nWarming up compilation...")
    # for _ in range(2):
    #     player_offset = 0
    #     npc_agent_names = pool.select_topk(game_config['players'] - 1)
    #     for j, agent in enumerate(npc_agents):
    #         agent.load_state_dict(pool.agents[npc_agent_names[j]].state_dict())
    #         agent.reset_context()
    #     local_agent.reset_context()
        
    #     for step in range(2):
    #         env.fill_observation_tensor(observation_buffer)
    #         with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    #             with torch.inference_mode():
    #                 _ = local_agent.incremental_forward(observation_buffer, step)
    #     env.reset()
    
    print("\n\nStarting profiled training loop...")
    
    # Profile the full training loop
    pr = cProfile.Profile()
    pr.enable()
    
    # Disable garbage collection for performance
    gc.disable()
    
    # Initialize logging and timing
    logger = HighLowLogger(args)
    global_step = 0
    total_iterations = 0
    timer = Timer()
    ticker = OneTickTimer()
    agent_name = str(total_iterations)
    
    pbar = tqdm(range(args.num_iterations))
    for iteration in pbar:
        # Manual GC every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            gc.collect()
            
        # Pick npc agents and player offset
        player_offset = np.random.randint(0, game_config['players'])
        npc_agent_names = pool.select_topk(game_config['players'] - 1)
        for j, agent in enumerate(npc_agents):
            agent.load_state_dict(pool.agents[npc_agent_names[j]].state_dict())
            agent.reset_context()
        local_agent.reset_context()
        
        round_agent_names = (
            [npc_agent_names[j] for j in range(player_offset)] 
            + [agent_name]
            + [npc_agent_names[j - 1] for j in range(player_offset + 1, game_config['players'])])
        
        # Full rollout
        settlement_preds, private_role_preds = [], []
        buffer.pinfo_tensor = env.pinfo_tensor()
        
        for step in range(args.num_steps):
            torch.compiler.cudagraph_mark_step_begin()
            global_step += args.num_envs
            
            # NPCs before player
            for npc_id in range(player_offset):
                env.fill_observation_tensor(observation_buffer)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
                    with torch.inference_mode():
                        npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
                env.step(npc_actions)
            
            # Player action
            if step > 0:
                env.fill_rewards_since_last_action(buffer.rewards[step - 1])
                buffer.update_late_stats({'dones': done_zeros}, step - 1)
            
            env.fill_observation_tensor(buffer.obs[step])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
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
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
                    with torch.inference_mode():
                        npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
                env.step(npc_actions)
            
            if env.terminal():
                env.fill_returns(returns_buffer)
                env.fill_rewards_since_last_action(buffer.rewards[step], player_offset)
                buffer.update_late_stats({
                    'rewards': player_reward_buffer,
                    'dones': done_ones}, step)
                
                env_info = env.expose_info()
                env_pinfo_targets = env.get_pinfo_targets()
                settlement_preds_stacked = torch.stack(settlement_preds, dim=0)
                logging_inputs = {
                    'returns': returns_buffer,
                    'offset': player_offset,
                    'settlement_preds': settlement_preds_stacked,
                    'private_role_preds': torch.stack(private_role_preds, dim=0),
                    'infos': env_info | env_pinfo_targets}
                
                heavy_logging_update = (
                    logger.counter - logger.last_heavy_counter > args.iterations_per_heavy_logging 
                    and player_offset == 0)
                pool.register_playout_scores(returns_buffer.mean(0), round_agent_names)
                pbar.set_postfix({'mean_score': f'{returns_buffer.mean(0)[player_offset].item():.2f}'})
                
                if heavy_logging_update:
                    pool.log_stats(global_step)
                logger.update_stats(logging_inputs, global_step, heavy_updates=heavy_logging_update)
                
                buffer.actual_private_roles.copy_(env_pinfo_targets['pinfo_targets'], non_blocking=True)
                buffer.actual_settlement.copy_(env_pinfo_targets['settlement_values'], non_blocking=True)
                
                env.reset()
        
        # Training update
        running_sps = timer.tick(args.num_envs * args.num_steps)
        update_dictionary = buffer.get_update_dictionary()
        trainer_results = trainer.train(update_dictionary)
        
        # Periodic agent registration
        total_iterations += 1
        if total_iterations % args.iterations_per_pool_update == 0 and total_iterations >= args.iterations_to_first_pool_update:
            pool.register_agent(local_agent, agent_name)
            agent_name = str(total_iterations)
    
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
    
    # Test agent state_dict copying performance
    agent = HighLowTransformerModel(args, env, verbose=False).to(device)
    
    print("\n1. State dict loading performance:")
    state_dict = agent.state_dict()
    start = time.time()
    for _ in range(10):
        new_agent = HighLowTransformerModel(args, env, verbose=False).to(device)
        new_agent.load_state_dict(state_dict)
    elapsed = time.time() - start
    print(f"   Average state_dict load time: {elapsed/10:.3f} seconds")
    
    # Test CPU-GPU transfers
    print("\n2. CPU-GPU transfer overhead:")
    test_tensor = torch.randn(args.num_envs, 4, device=device)
    start = time.time()
    for _ in range(100):
        _ = test_tensor.cpu()
    elapsed = time.time() - start
    print(f"   Average GPU->CPU transfer time ({args.num_envs}x4 tensor): {elapsed/100*1000:.3f} ms")
    
    # Test arena probability calculations
    # print("\n3. Arena probability calculations:")
    # from arena import Arena
    # initial_agents = {f"agent_{i}": agent for i in range(10)}
    # arena = Arena(env, initial_agents, device)
    
    # # Simulate some scores
    # for name in arena.agents:
    #     arena.playout_scores[name] = [torch.randn(1).item() for _ in range(100)]
    
    # start = time.time()
    # for _ in range(100):
    #     _ = arena.select_topk(4)
    # elapsed = time.time() - start
    # print(f"   Average agent selection time: {elapsed/100*1000:.3f} ms")
    
    # Test environment operations
    print("\n4. Environment operations:")
    obs_buffer = env.new_observation_buffer()
    
    start = time.time()
    for _ in range(100):
        env.fill_observation_tensor(obs_buffer)
    elapsed = time.time() - start
    print(f"   Average observation fill time: {elapsed/100*1000:.3f} ms")
    
    # Test forward pass
    # print("\n5. Model forward pass:")
    # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    #     with torch.inference_mode():
    #         start = time.time()
    #         for _ in range(10):
    #             _ = agent.incremental_forward(obs_buffer, 0)
    #         elapsed = time.time() - start
    # print(f"   Average forward pass time: {elapsed/10*1000:.3f} ms")


if __name__ == "__main__":
    print("Starting CPU profiling analysis...")
    print("This will run 100 training iterations to identify bottlenecks.")
    
    # Run main profiling
    run_profiling_test()
    
    # Analyze specific bottlenecks
    print("\n\n" + "="*80)
    print("ANALYZING SPECIFIC BOTTLENECKS")
    print("="*80)
    analyze_specific_bottlenecks()
    
    print("\n\nProfiling complete!")