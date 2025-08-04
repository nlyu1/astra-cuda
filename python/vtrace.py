# %% Single-agent play against fixed opponents
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import gc

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
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
if args.checkpoint_name is not None:
    path = Path("./checkpoints") / (args.checkpoint_name + ".pt")
    weights = torch.load(path, map_location=device, weights_only=False)['model_state_dict']
    print(f"Loading checkpoint {args.checkpoint_name} from {path}")
for j in range(args.players - 1):
    if args.checkpoint_name is not None: 
        name = f"{args.checkpoint_name}_copy{j}"
    else:
        name = f"Random{j}"
    initial_agents[name] = HighLowTransformerModel(
        args, env, verbose=False).to(device)
    if args.checkpoint_name is not None:   
        initial_agents[name].load_state_dict(weights)

num_features = env.num_features()
pool = Arena(env, initial_agents, device)

buffer = HighLowImpalaBuffer(args, num_features, device)

local_agent = HighLowTransformerModel(args, env).to(device)
if args.checkpoint_name is not None:
    local_agent.load_state_dict(weights)
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

# Disable garbage collection for performance
gc.disable()

pbar = tqdm(range(args.num_iterations))
done_zeros, done_ones = torch.zeros(args.num_envs, device=device).float(), torch.ones(args.num_envs, device=device).float()
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
    round_agent_names = ( # Used for registering pool result 
        [npc_agent_names[j] for j in range(player_offset)] 
        + [agent_name]
        + [npc_agent_names[j - 1] for j in range(player_offset + 1, game_config['players'])])

    ### Rollout ### 
    settlement_preds, private_role_preds = [], []
    buffer.pinfo_tensor = env.pinfo_tensor()
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin() # Mark iterations to enable cuda-graph in compilation
        global_step += args.num_envs 

        for npc_id in range(player_offset):
            assert (env.current_player() == npc_id), f"Environment must be ready for NPC {npc_id}, but {env.current_player()} is acting."
            env.fill_observation_tensor(observation_buffer)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=True):
                with torch.inference_mode():
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
                forward_results = local_agent.incremental_forward(buffer.obs[step], step)
        action, log_probs = forward_results['action'], forward_results['logprobs']
        settlement_preds.append(forward_results['pinfo_preds']['settle_price'].clone())
        private_role_preds.append(forward_results['pinfo_preds']['private_roles'].argmax(dim=-1))

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
            settlement_preds_stacked = torch.stack(settlement_preds, dim=0)
            logging_inputs = {
                'returns': returns_buffer,
                'offset': player_offset,
                'settlement_preds': settlement_preds_stacked,
                'private_role_preds': torch.stack(private_role_preds, dim=0),
                'infos': env_info | env_pinfo_targets}
            # print('Rewards and returns:', buffer.rewards[:, 0].cpu(), returns_buffer[0, player_offset].cpu())
            # print('Settlement:', env_pinfo_targets['settlement_values'][0].item(), 
            #       'info_role:', env_pinfo_targets['info_roles'][0, player_offset].item(), 
            #       'target position:', env_info['target_positions'][0, player_offset].item())
            # print('Position over time:', env_info['players'][0, player_offset, :, -2:].cpu().numpy())
            # Only incur heavy logging when we're in seat 0 and after a certain interval 
            heavy_logging_update = (
                logger.counter - logger.last_heavy_counter > args.iterations_per_heavy_logging 
                and player_offset == 0)
            pool.register_playout_scores(returns_buffer.mean(0), round_agent_names) 
            pbar.set_postfix({'mean_score': f'{returns_buffer.mean(0)[player_offset].item():.2f}'})
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

# Benchmark notes: 
#    Training: At 64 T/B * 128 B, 5090 goes at ~3it/s for normal game, and 5070ti goes at ~5it/s for small game