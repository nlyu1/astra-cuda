# %% Single-agent play against fixed components 
import random
import time
from collections import deque
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import wandb
from tqdm import trange
import sys 

sys.path.append('./utils')
sys.path.append('./algorithms')

from timer import Timer 
from agent import *
from high_low_wrapper import *
from config import Args
from arena import Arena 
from logger import HighLowLogger
from vtrace_config import Args
from vtrace import VTraceBuffer, HighLowVTraceTrainer

args = Args()
args.meta_steps = 1
args.exp_name = 'vtrace_single'
args.num_iterations = 10000000000000
args.fill_runtime_args()
args.run_name = f"HighLowTradingVTrace__{args.exp_name}__{args.seed}__{int(time.time())}"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# %% Argument post-processing and logging 

logger = HighLowLogger(args)

game_config: Dict[str, int] = {
    'steps_per_player': args.steps_per_player,           # Number of trading steps per player
    'max_contracts_per_trade': args.max_contracts_per_trade,     # Maximum contracts in a single trade
    'customer_max_size': args.customer_max_size,           # Maximum position size for customers
    'max_contract_value': args.max_contract_value,         # Maximum value a contract can have
    'players': args.players                      # Total number of players in the game
}

device = torch.device('cuda')
loads = [
    ('./checkpoints/ppo_bigpoolplay_34.pt', 'starter_33'),
    ('./checkpoints/ppo_bigpoolplay_34.pt', 'starter_34'),
    ('./checkpoints/ppo_bigpoolplay_35.pt', 'starter_35'),
    ('./checkpoints/ppo_bigpoolplay_36.pt', 'starter_36'),
]
initial_agents = {}
for path, name in loads:
    initial_agents[name] = HighLowModel(
        game_config, hidden_size=512, num_residual_blocks=4).to(device)
    initial_agents[name].load_state_dict(
        torch.load(path, map_location='cuda', weights_only=False)['model_state_dict'])

env = HighLowWrapper(args, game_config)
num_features = env.observations().shape[-1]

pool = Arena(env, initial_agents)
checkpoint_interval = 2000 # Checkpoint every meta-step

buffer = VTraceBuffer(args, num_features, device)

trainer = HighLowVTraceTrainer(
    args, 
    HighLowModel(game_config, hidden_size=512, num_residual_blocks=5).to(device), 
    name='single', 
    checkpoint_interval=checkpoint_interval,
    device=device)

global_step = 0
timer = Timer()
num_iterations = args.num_iterations

round_new_agent_name = 'challenger'

for iteration in trange(1, num_iterations + 1):
    player_offset = np.random.randint(0, game_config['players'])

    # Choose agents from the pool
    npc_agent_names = pool.select_topk(game_config['players'] - 1)
    npc_agents = [pool.agents[name].cuda() for name in npc_agent_names]
    round_agent_names = (
        [npc_agent_names[j] for j in range(player_offset)] 
        + [round_new_agent_name] 
        + [npc_agent_names[j - 1] for j in range(player_offset + 1, game_config['players'])])

    # Keeps track of the player's accumulated reward from since its last action to right now. 
    # Used for interim updates
    player_reward_cumulation = None

    for step in range(args.num_steps):
        global_step += args.num_envs 

        for npc_id in range(player_offset):
            assert (env.current_player() == npc_id), f"Environment must be ready for NPC {npc_id}"
            npc_obs = env.player_observations(npc_id)
            npc_actions = npc_agents[npc_id].sample_actions(npc_obs)
            npc_actions = np.stack( # num_envs, 4
                [npc_actions['bid_price'], npc_actions['ask_price'], npc_actions['bid_size'], npc_actions['ask_size']],
                axis=-1)
            env.step(npc_actions)
            
            # Only update after player has made an effective move 
            if player_reward_cumulation is not None: 
                player_reward_cumulation += env.rewards()[:, player_offset]
        
        #### Fill buffer ####
        with torch.no_grad():
            # Prior to stepping & observing new rewards, flush the cumulated rewards 
            # Environment cannot terminate before player makes action (due to the design of our environment)
            if player_reward_cumulation is not None:
                buffer.update_late_stats({
                    'rewards': torch.tensor(player_reward_cumulation).to(device).view(-1),
                    'dones': torch.zeros(args.num_envs).to(device).float()}, step - 1)
                player_reward_cumulation.fill(0)
            else:
                player_reward_cumulation = np.zeros(args.num_envs)

            # observation, action, log_probs, value can be calculated immediately 
            obs = torch.from_numpy(env.observations()).pin_memory().to(device, non_blocking=True) 
            action, log_probs, _entropies, _value = trainer.agent(obs)
            action = torch.stack(
                [action['bid_price'], action['ask_price'], 
                action['bid_size'], action['ask_size']], dim=-1)
            buffer.update({
                'obs': obs,
                'actions': action,
                'logprobs': log_probs,
            }, step)

        # Step the environment 
        assert (env.current_player() == player_offset), "Environment must be ready for player"
        env.step(action.cpu().numpy())
        player_reward_cumulation += env.rewards()[:, player_offset]

        # Step remaining players
        for player_id in range(player_offset + 1, game_config['players']):
            npc_id = player_id - 1
            assert (env.current_player() == player_id), f"Environment must be ready for NPC {npc_id}"
            npc_obs = env.player_observations(player_id)
            npc_actions = npc_agents[npc_id].sample_actions(npc_obs)
            npc_actions = np.stack( # num_envs, 4
                [npc_actions['bid_price'], npc_actions['ask_price'], npc_actions['bid_size'], npc_actions['ask_size']],
                axis=-1)
            env.step(npc_actions)
            player_reward_cumulation += env.rewards()[:, player_offset]

        if env.is_terminal():
            # This is the cumulative rewards, for logging
            env_returns = env.returns()
            # Unfortunately, logging must happen before environment is reset
            logging_inputs = {
                'returns': env_returns,
                'offset': player_offset,
                'infos': env.states.expose_info()}
            # Only incur heavy logging when we're in seat 0 and after a certain interval 
            heavy_logging_update = (
                logger.counter - logger.last_heavy_counter > 500
                and player_offset == 0)
            logger.update_stats(logging_inputs, global_step, heavy_updates=heavy_logging_update)
            # # Update pool 
            pool.register_playout_scores(env_returns.mean(0), round_agent_names)

            tensor_reward_cumulation = torch.from_numpy(player_reward_cumulation).pin_memory().to(device, non_blocking=True)
            # Reset environment 
            buffer.update_late_stats({
                'rewards': tensor_reward_cumulation.view(-1),
                'dones': torch.ones(args.num_envs).to(device).float()}, step)
            player_reward_cumulation = None
            env.reset()
        assert (env.current_player() == 0), "Next environment must be ready for player"

    #### Fill buffer end. Update trainer ####
    assert env.is_initial_move_state(), "Environment must have just been reset"
    update_dictionary = buffer.get_update_dictionary()
    trainer.train(update_dictionary, global_step)
    running_sps = timer.tick(args.num_envs * args.num_steps)
    wandb.log({"charts/SPS": running_sps}, step=global_step)

    if iteration % checkpoint_interval == 0:
        # Register agent so that it's available for play
        # pool.register_agent(trainer.agent, round_new_agent_name)    
        pool.random_maintainance()
        pool.debug_printout()

        # Save checkpoint 
        pool.log_stats(step=global_step)
        trainer.save_checkpoint(iteration)
