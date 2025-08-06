# %%

import torch
import random
import numpy as np 
from tqdm import tqdm
import sys 
sys.path.append("/home/nlyu/Code/astra-cuda/python/src/")
from high_low.env import HighLowTrading
from high_low.config import Args
from high_low.agent import HighLowTransformerModel
from high_low.plotting import plot_market_and_players
from high_low.plotting import plot_action_distributions
from pathlib import Path

# %%
args = Args()
args.num_blocks = 256
args.threads_per_block = 64
args.game_setting = 0
args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')
game_config = args.get_game_config()
print('Game setup:')
for k, v in game_config.items():
    print(f"{k}: {v}")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

env = HighLowTrading(game_config)

models = [HighLowTransformerModel(args, env, verbose=False).to(device) for _ in range(args.players)]
checkpoint_root = Path("/home/nlyu/Code/astra-cuda/python/checkpoints")
checkpoint_path = "normaldecencritic_poolnsp4_10000"
for model in models:
    model.load_state_dict(
    torch.load(checkpoint_root / f"{checkpoint_path}.pt", map_location=device, weights_only=False
        )['model_state_dict'])
    model.compile()


returns_buffer = env.new_reward_buffer()
reward_buffer = env.new_player_reward_buffer()
obs_buffer = env.new_observation_buffer()

# %%

env.reset()
for model in models:
    model.reset_context()

pinfo_targets, returns = [], []
logprobs, entropy = [], []
for step in tqdm(range(args.num_steps)):
    for i in range(args.players):
        env.fill_observation_tensor(obs_buffer)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.inference_mode():
                model_outputs = models[i].incremental_forward(obs_buffer, step)
                if i == 0:
                    logprobs.append(model_outputs['logprobs_by_category'].cpu())
                    entropy.append(model_outputs['entropy_by_category'].cpu())
                model_actions = model_outputs['action']
            env.step(model_actions)

assert env.terminal(), "Environment is not terminal"
env.fill_returns(returns_buffer)
returns.append(returns_buffer.cpu())
pinfo_targets.append({k: v.cpu() for k, v in env.get_pinfo_targets().items()})
# pinfo_targets = {
#     k: torch.stack([v[i] for v in pinfo_targets], dim=0) for k, v in pinfo_targets[0].items()}
logprobs = torch.stack(logprobs, dim=0)
entropy = torch.stack(entropy, dim=0)
# %%
customer_mask = pinfo_targets[0]['info_roles'][:, 0] == 3
import matplotlib.pyplot as plt
for t in range(0, 16, 4):
    plt.hist(entropy[t, customer_mask, 3].cpu().numpy(), label=f"Step {t}")
plt.legend()
plt.show()
# %%
