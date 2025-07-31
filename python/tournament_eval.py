# %%
import os
import random
import sys
from typing import Dict
sys.path.append('./utils')
sys.path.append('./algorithms')

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
from tqdm import trange, tqdm

import json

from pathlib import Path
import itertools 
from collections import defaultdict
import matplotlib.pyplot as plt

from agent import *
from high_low_wrapper import *
from vtrace_config import Args 

args = Args()
args.exp_name = 'ppo_analysis'
args.fill_runtime_args()

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

game_config: Dict[str, int] = {
    'steps_per_player': args.steps_per_player,           # Number of trading steps per player
    'max_contracts_per_trade': args.max_contracts_per_trade,     # Maximum contracts in a single trade
    'customer_max_size': args.customer_max_size,           # Maximum position size for customers
    'max_contract_value': args.max_contract_value,         # Maximum value a contract can have
    'players': args.players                      # Total number of players in the game
}
env = HighLowWrapper(args, game_config)

def sort_key(checkpoint_name):
    step_number = int(checkpoint_name.split('_')[-1].split('.')[0])
    stage_version = checkpoint_name.split('_')[0] 
    return (-1e12 if stage_version == 'starter' else 0) + step_number
# %%

device = torch.device('cuda')
agents = [
    HighLowPrivateModel(game_config, num_residual_blocks=5).to(device)
    for _ in range(game_config['players'])]

checkpoint_root = Path('/home/nlyu/Code/astra/python_scripts/high_low_ppo/checkpoints')
checkpoint_names = sorted(list(
    filter(lambda x: x.startswith('impala_pinfo_poolplay'), os.listdir(checkpoint_root))), key=sort_key)
num_checkpoints = len(checkpoint_names)
print(f'Evaluating {num_checkpoints} checkpoints')
combs = list(itertools.combinations(range(num_checkpoints), 5))
random.shuffle(combs)
print(len(combs), 'possible combinations')
print(checkpoint_names)
# %% Actual evaluation: extremely time consuming 

if os.path.exists('elo_analysis_results.json'):
    results = json.load(open("elo_analysis_results.json"))
    results['names'] = checkpoint_names
    print('Loaded from checkpoint with', len(results['choices']), 'combinations')
else: 
    results = {
        'names': checkpoint_names,
        'choices': [],
        'returns': [],
    }

for comb in tqdm(combs):
    for i, j in enumerate(comb):
        agents[i].load_state_dict(
            torch.load(checkpoint_root / checkpoint_names[j], weights_only=False, map_location='cuda'
        )['model_state_dict'])

    returns = []
    num_eval_episodes = 30

    # for j in trange(num_eval_episodes, desc='ELO eval'):
    for j in trange(num_eval_episodes):
        agent_ordering = np.arange(5).astype(np.int32)
        np.random.shuffle(agent_ordering)
        inverse_ordering = np.argsort(agent_ordering)

        with torch.no_grad():
            while not env.is_terminal():
                current_player = env.current_player()
                agent_idx = agent_ordering[current_player]
            
                obs = torch.Tensor(env.player_observations(current_player)).cuda().float()
                actions = agents[agent_idx].sample_actions(obs)
                actions = np.stack( # num_envs, 4
                    [actions['bid_price'], actions['ask_price'], actions['bid_size'], actions['ask_size']],
                    axis=-1)
                env.step(actions)
        
        rewards = np.array(env.returns()[:, inverse_ordering])
        returns.append(rewards)
        env.reset()
    returns = np.concatenate(returns, axis=0).astype(np.float32)
    results['choices'].append(comb)
    results['returns'].append(returns.mean(axis=0).tolist())
    json.dump(results, open('elo_analysis_results.json', 'w'), indent=4)

# %% View a single bootstrapped sample 
import matplotlib.pyplot as plt

for j in range(5):
    mean, vec = [], returns[:, j]
    for k in range(300):
        indices = np.random.choice(len(vec), size=len(vec), replace=True)
        mean.append(vec[indices].mean())
    plt.hist(mean, label=f'agent {j}, {returns[:, j].mean():.2f}')
plt.legend()
plt.show()
# %% Analyzing the results 

results = json.load(open("elo_analysis_results.json"))
results['names'] = checkpoint_names

indices  = np.array(results["choices"])
returns  = np.array(results["returns"])
num_results = len(returns)
print(num_results, 'results')

avg_returns = defaultdict(list)
for i in range(num_results):
    for j in range(5):
        print(indices[i, j])
        model_name = results["names"][indices[i, j]]
        avg_returns[model_name].append(returns[i, j])

names   = np.array(sorted(avg_returns.keys(), key=sort_key))
means   = np.array([np.mean(avg_returns[n]) for n in names])
std_err = np.array([np.std (avg_returns[n]) for n in names])   # 1 σ

for i in range(len(names)):
    print(names[i], means[i], std_err[i])

# ---- plot: horizontal bars with error bars ----
fig, ax = plt.subplots(figsize=(12, len(names) * 0.35))   # height scales with #models

y = np.arange(len(names))
ax.barh(y, means, xerr=std_err, align="center",
        alpha=0.8, ecolor="black", capsize=3)

ax.set_yticks(y)
ax.set_yticklabels(names)
ax.invert_yaxis()                       # highest-ranked model at the top (optional)
ax.set_xlabel("Average return")
ax.set_title("Model performance (mean ± 1 σ)")

plt.tight_layout()
plt.show()
# %%
