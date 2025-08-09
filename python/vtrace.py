# %% Single-agent play against fixed opponents
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import gc
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torch_compile")
os.environ["TORCH_COMPILE_CACHE_DIR"] = os.path.expanduser("~/.cache/torch_compile")

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 256 # Enable more aggressive caching
import wandb
from pathlib import Path
from tqdm import tqdm
import sys 

sys.path.append('./src')

from high_low.agent import HighLowTransformerModel
from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.logger import HighLowLogger
from high_low.impala import HighLowImpalaTrainer, HighLowImpalaBuffer
from timer import Timer, OneTickTimer
import tyro

from arena import Arena 

args = tyro.cli(Args)

args.run_name = f"HighLowTradingVTrace__{args.exp_name}__{args.seed}__{int(time.time())}"
args.fill_runtime_args()
print(args)
game_config = args.get_game_config()
env = HighLowTrading(game_config)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed);

# %% Argument post-processing and logging 

device = torch.device(f'cuda:{args.device_id}')
initial_agents = {}
if args.checkpoint_name != "":
    path = Path("./checkpoints") / (args.checkpoint_name + ".pt")
    weights = torch.load(path, map_location=device, weights_only=False)['model_state_dict']
    print(f"Loading checkpoint {args.checkpoint_name} from {path}")

initial_agent_name = "Random" if args.checkpoint_name == "" else f"{args.checkpoint_name}"
initial_agents[initial_agent_name] = HighLowTransformerModel(
    args, env, verbose=False).to(device)
if args.checkpoint_name != "":   
    initial_agents[initial_agent_name].load_state_dict(weights, strict=False)

num_features = env.num_features()
pool = Arena(env, initial_agents, device)

buffer = HighLowImpalaBuffer(args, num_features, device)

local_agent = HighLowTransformerModel(args, env).to(device)
if args.checkpoint_name != "":
    local_agent.load_state_dict(weights, strict=False)
trainer = HighLowImpalaTrainer(
    args, local_agent, 
    checkpoint_interval=args.iterations_per_checkpoint,
    device=device)
npc_agents = [
    HighLowTransformerModel(args, env, verbose=False).to(device)
    for _ in range(game_config['players'] - 1)]

# %%

global_step = 0
timer = Timer()

logger = HighLowLogger(args)

global_step, total_iterations = 0, 0
timer, ticker = Timer(), OneTickTimer()
agent_name = str(total_iterations) # This is only updated per pool update 

observation_buffer = env.new_observation_buffer()
reward_buffer = env.new_reward_buffer() # [N, P]
returns_buffer = env.new_reward_buffer() # [N, P]
player_reward_buffer = env.new_player_reward_buffer() # [N]
self_play = True 

# Disable garbage collection for performance
gc.disable()

pbar = tqdm(range(args.num_iterations))
done_zeros, done_ones = torch.zeros(args.num_envs, device=device).float(), torch.ones(args.num_envs, device=device).float()

# Pre-allocate GPU buffers for distribution parameters (outside loop for reuse)
dist_params_buffer = {
    'center': torch.zeros(args.num_steps, 4, device=device),
    'precision': torch.zeros(args.num_steps, 4, device=device)}

for iteration in pbar:
    # Manual GC every 100 iterations
    if iteration > 0 and iteration % 50 == 0:
        gc.collect() 
    self_play = random.random() < args.self_play_prob
    # Pick npc agents and player offset
    player_offset = np.random.randint(0, game_config['players'])
    npc_agent_names = pool.select_topk(game_config['players'] - 1)
    for j, agent in enumerate(npc_agents):
        if self_play:
            agent.load_state_dict(local_agent.state_dict())
        else:
            agent.load_state_dict(pool.agents[npc_agent_names[j]].state_dict())
        agent.reset_context()
    local_agent.reset_context()
    round_agent_names = ( # Used for registering pool result 
        [npc_agent_names[j] for j in range(player_offset)] 
        + [agent_name]
        + [npc_agent_names[j - 1] for j in range(player_offset + 1, game_config['players'])])

    ### Rollout ### 
    settlement_preds, private_role_preds = [], []
    buffer.pinfo_tensor = env.pinfo_tensor()
    for step in range(args.num_steps):
        global_step += args.num_envs 

        for npc_id in range(player_offset):
            assert (env.current_player() == npc_id), f"Environment must be ready for NPC {npc_id}, but {env.current_player()} is acting."
            env.fill_observation_tensor(observation_buffer)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
                with torch.inference_mode():
                    torch.compiler.cudagraph_mark_step_begin()
                    npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
            env.step(npc_actions)
        
        #### Agent action ####
        # Prior to stepping & observing new rewards, flush the cumulated rewards to buffer
        # Environment cannot terminate before player makes action (due to the design of our environment)
        assert env.current_player() == player_offset, f"Environment must be ready for player {player_offset}, but {env.current_player()} is acting."
        if step > 0: # Only update if step > 0, since step 0 is the initial state 
            env.fill_rewards_since_last_action(buffer.rewards[step - 1])
            buffer.update_late_stats(
                {'dones': done_zeros}, step - 1)

        # observation, action, log_probs, value can be calculated immediately 
        # Fill observation tensor outside of inference mode to avoid issues with CUDA graph compilation
        env.fill_observation_tensor(buffer.obs[step])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
            with torch.inference_mode():
                torch.compiler.cudagraph_mark_step_begin()
                forward_results = local_agent.incremental_forward(buffer.obs[step], step)
        action, log_probs = forward_results['action'], forward_results['logprobs']
        settlement_preds.append(forward_results['pinfo_preds']['settle_price'].clone())
        private_role_preds.append(forward_results['pinfo_preds']['private_roles'].argmax(dim=-1))

        # Store distribution parameters in GPU buffers (no CPU transfer)
        for k, v in forward_results['action_params'].items():
            dist_params_buffer[k][step] = v.mean(0)

        buffer.update({
            # Observations are implicitly updated above. 
            'actions': action,
            'logprobs': log_probs,
        }, step)
        # Step the environment 
        assert (env.current_player() == player_offset), "Environment must be ready for player"
        env.step(action)

        # Step remaining players
        for player_id in range(player_offset + 1, game_config['players']):
            npc_id = player_id - 1
            assert (env.current_player() == player_id), f"Environment must be ready for NPC {npc_id}, but {env.current_player()} is acting."
            env.fill_observation_tensor(observation_buffer)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
                with torch.inference_mode():
                    
                    npc_actions = npc_agents[npc_id].incremental_forward(observation_buffer, step)['action']
            env.step(npc_actions)

        if env.terminal():
            # This is the cumulative rewards, for logging. 
            env.fill_returns(returns_buffer)
            env.fill_rewards_since_last_action(buffer.rewards[step], player_offset)

            # Reset environment 
            buffer.update_late_stats({
                'rewards': player_reward_buffer,
                'dones': done_ones,
            }, step)

            # Unfortunately, logging must happen before environment is reset
            env_info = env.expose_info()
            env_pinfo_targets = env.get_pinfo_targets() 
            if not self_play: # Only log when not self-playing
                settlement_preds_stacked = torch.stack(settlement_preds, dim=0)
                logging_inputs = {
                    'returns': returns_buffer,
                    'offset': player_offset,
                    'settlement_preds': settlement_preds_stacked,
                    'private_role_preds': torch.stack(private_role_preds, dim=0),
                    'infos': env_info | env_pinfo_targets,
                    'dist_params': dist_params_buffer}
                # Only incur heavy logging when we're in seat 0 and after a certain interval 
                heavy_logging_update = (
                    logger.counter - logger.last_heavy_counter > args.iterations_per_heavy_logging 
                    and player_offset == 0)
                pool.register_playout_scores(returns_buffer.mean(0), round_agent_names) 
                if heavy_logging_update:
                    pool.log_stats(global_step)
                logger.update_stats(logging_inputs, global_step, heavy_updates=heavy_logging_update)

            # Populate buffer's actual private info. See `env.py` env_pinfo_target method
            buffer.actual_private_roles.copy_(env_pinfo_targets['pinfo_targets'], non_blocking=True)
            buffer.actual_settlement.copy_(env_pinfo_targets['settlement_values'], non_blocking=True)
            
            ### Reset environment ###
            env.reset()
        assert (env.current_player() == 0), "Next environment must be ready for player"

    #### Fill buffer end. Update trainer ####
    running_sps = timer.tick(args.num_envs * args.num_steps)
    wandb.log({"performance/SPS": running_sps}, step=global_step)

    update_dictionary = buffer.get_update_dictionary()
    trainer_results = trainer.train(update_dictionary)
    wandb.log(trainer_results, step=global_step)
    trainer.save_checkpoint(iteration)

    total_iterations += 1
    if total_iterations % args.iterations_per_pool_update == 0 and total_iterations >= args.iterations_to_first_pool_update:
        pool.register_agent(local_agent, agent_name)
        agent_name = str(total_iterations)

# Re-enable garbage collection
gc.enable()