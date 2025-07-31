# %% Inspect playthroughs of individual agents. 

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import wandb
from tqdm import trange


sys.path.append('./utils')
from timer import Timer 
from agent import *
from high_low_wrapper import *
from config import Args
from plotting import plot_market_and_players, dual_plot


args = Args()
args.meta_steps = 1000000
args.exp_name = 'playthrough'
args.env_workers = 1
args.num_envs_per_worker = 25
args.fill_runtime_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

game_config: Dict[str, int] = {
    'steps_per_player': args.steps_per_player,           # Number of trading steps per player
    'max_contracts_per_trade': args.max_contracts_per_trade,     # Maximum contracts in a single trade
    'customer_max_size': args.customer_max_size,           # Maximum position size for customers
    'max_contract_value': args.max_contract_value,         # Maximum value a contract can have
    'players': args.players                      # Total number of players in the game
}
env = HighLowWrapper(args, game_config)
# %%

device = torch.device('cuda')
agents = [
    torch.compile(HighLowModel(game_config).to(device), mode='max-autotune')
    for _ in range(game_config['players'])]
# load_paths = [
#     '/home/nlyu/Code/astra/python_scripts/high_low_ppo/checkpoints/ppo_poolplay_step_44.pt',
#     '/home/nlyu/Code/astra/python_scripts/high_low_ppo/checkpoints/ppo_poolplay_step_45.pt',
#     '/home/nlyu/Code/astra/python_scripts/high_low_ppo/checkpoints/ppo_poolplay_step_46.pt',
#     '/home/nlyu/Code/astra/python_scripts/high_low_ppo/checkpoints/ppo_poolplay_step_47.pt',
#     '/home/nlyu/Code/astra/python_scripts/high_low_ppo/checkpoints/ppo_poolplay_step_48.pt',
# ]
# for j, path in enumerate(load_paths):
#     agents[j].load_state_dict(
#         torch.load(path, map_location='cuda', weights_only=False)['model_state_dict'])
# %%
env.reset()
with torch.no_grad():
    obs = torch.Tensor(env.observations()).cuda().half()
    while not env.is_terminal():
        for j in range(5):
            actions = agents[j].sample_actions(obs)
            actions = np.stack( # num_envs, 4
                [actions['bid_price'], actions['ask_price'], actions['bid_size'], actions['ask_size']],
                axis=-1)
            env.step(actions)
rewards = np.array(env.returns())
infos = env.states.expose_info()

# %% Debug plotting info

fig = plot_market_and_players(infos, args, env_idx=125, fig_size=(900, 1200))
fig.show()
# %% Plot players' average missed positions

env.states.clone_at(130)
# %%

offset = 1
player_rewards = env.states.returns()[:, offset]
info_roles = infos['info_roles'][:, offset]
good_value_mask = (info_roles == 0)
bad_value_mask = (info_roles == 1)
high_low_mask = (info_roles == 2)
customer_mask = (info_roles == 3)

# Extract rewards and trade volume for each 
reward_logs = {
    'reward/goodValue': player_rewards[good_value_mask].mean(),
    'reward/badValue': player_rewards[bad_value_mask].mean(),
    'reward/highLow': player_rewards[high_low_mask].mean(),
    'reward/customer': player_rewards[customer_mask].mean(),
}

# Last price - contract price 
last_price = np.array(infos['market'][:, :, 2])
last_price[last_price == 0] = 2 * args.max_contract_value # Explicit penalty
last_price_residual = last_price - infos['contract'][:, 2][:, None] 
last_price_rmse = (last_price_residual**2).mean(0) ** .5
last_price_std = last_price_residual.std(0)

buy_volume = infos['market'][:, :, 3].mean(0) # .cumsum(-1)
sell_volume = infos['market'][:, :, 4].mean(0) # .cumsum(-1)
market_fig = dual_plot(
    {
        'last price rmse': last_price_rmse,
        'last price std': last_price_std,
    },
    {
        'buy volume': buy_volume,
        'sell volume': sell_volume,
    },
    y2min=0, y2max=args.max_contracts_per_trade,
    title='Last price rmse and volume over time',
)
market_logs = {
    'market/last_price_rmse': last_price_rmse,
    'market/trade_volume': sell_volume[-1] + buy_volume[-1],
}
market_fig
# %%
# %%
