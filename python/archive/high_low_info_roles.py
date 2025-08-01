# %% Script to help engineer environment's get_pinfo_targets method for logging and pinfo prediction
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import wandb
from tqdm import trange, tqdm
import sys 

sys.path.append('./src')
sys.path.append('./src/high_low')


from agent import HighLowTransformerModel
from config import Args
from env import HighLowTrading
from logger import HighLowLogger
from impala import HighLowImpalaTrainer, HighLowImpalaBuffer
from timer import Timer, OneTickTimer

# %%

args = Args()
args.fill_runtime_args()
game_config = args.get_game_config()
env = HighLowTrading(game_config)

# %%
pinfo_targets = (env.permutation - 1).clamp(0, 2) # [N, P]
max_values = env.candidate_values.max(dim=1).values # [N] 
min_values = env.candidate_values.min(dim=1).values # [N, P]
settlement_values = env.high_low * max_values + (1 - env.high_low) * min_values # [N]


info_roles = env.permutation.clone()
pos_0_mask = (env.permutation == 0)
pos_1_mask = (env.permutation == 1)
# Check which contracts were chosen
first_chosen = (env.candidate_values[:, 0] == settlement_values) # [N]
second_chosen = (env.candidate_values[:, 1] == settlement_values) # [N]
N, P = info_roles.shape
info_roles[pos_0_mask] = torch.where(first_chosen, 0, 1).repeat_interleave(P).reshape(N, P)[pos_0_mask].int()
info_roles[pos_1_mask] = torch.where(second_chosen, 0, 1).repeat_interleave(P).reshape(N, P)[pos_1_mask].int()