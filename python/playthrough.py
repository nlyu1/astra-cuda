# %%

import torch
import random
import numpy as np 

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
args.num_blocks = 1
args.threads_per_block = 1
args.game_setting = 0
args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')
game_config = args.get_game_config()
print('Game setup:')
for k, v in game_config.items():
    print(f"{k}: {v}")

env = HighLowTrading(game_config)


models = [HighLowTransformerModel(args, env, verbose=False).to(device) for _ in range(args.players - 1)]
checkpoint_root = Path("/home/nlyu/Code/astra-cuda/python/checkpoints")
checkpoint_path = "normaldecencritic_poolnsp4_5000"
# for model in models:
    # model.load_state_dict(
    # torch.load(checkpoint_root / f"{checkpoint_path}.pt", map_location=device, weights_only=False
    #     )['model_state_dict'])
    # model.compile()
advisor_model = HighLowTransformerModel(args, env, verbose=False).to(device)
# advisor_model.load_state_dict(
#     torch.load(checkpoint_root / f"{checkpoint_path}.pt", map_location=device, weights_only=False
#         )['model_state_dict'])
# advisor_model.compile()
# %%

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

env.reset()
for model in models:
    model.reset_context()
advisor_model.reset_context()
returns_buffer = env.new_reward_buffer()
reward_buffer = env.new_player_reward_buffer()
immediate_reward_buffer = env.new_reward_buffer()
obs_buffer = env.new_observation_buffer()
# print(env.env.to_string(0))
expand_actions = lambda x: torch.tensor(x).unsqueeze(0).to(device).int()

# %%
# Example of triangle distribution visualization
from plotting import analyze_triangle_distribution

# Example parameters for a concentrated distribution
fig = analyze_triangle_distribution(
    center=0.5,  # Center of distribution
    half_width=0.1,  # Width of main triangle
    epsilon_fullsupport=0.2,  # Probability of using full-support distribution
    epsilon_uniform=0.3,  # Uniform mixing in full-support
    min_value=1,
    max_value=30,
    title="Example Triangle Distribution"
)
fig['fig'].show()

# %%

# for step in range(args.num_steps - 1):
step = 0
print(f'Observation at step {step}:')
print(env.observation_string(0))
env.fill_observation_tensor(obs_buffer)
print('Observation tensor:', obs_buffer.cpu().numpy())
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    with torch.inference_mode():
        advisor_output = advisor_model.incremental_forward(obs_buffer, step)
        advisor_actions = advisor_output['action']
        print('Advisor actions:', advisor_actions.cpu().numpy())
pred_roles = torch.softmax(advisor_output['pinfo_preds']['private_roles'][0], dim=-1).float().cpu().numpy()
for p in range(args.players):
    print(f"Player {p} role: (ValueCheater {pred_roles[p][0]:.2f}, HighLowCheater {pred_roles[p][1]:.2f}, Customer {pred_roles[p][2]:.2f})")
print(f"Settle price prediction: {advisor_output['pinfo_preds']['settle_price'].float().item():.2f}")

# %%

advisor_action_params = advisor_output['action_params']
print(advisor_action_params.keys())

# Use the new 2x2 plot function
fig = plot_action_distributions(advisor_action_params, args.max_contract_value, args.max_contracts_per_trade)
fig.show()

# %%

# bid_price = int(input('Bid price: '))
# ask_price = int(input('Ask price: '))
# bid_size = int(input('Bid size: '))
# ask_size = int(input('Ask size: '))
bid_price, ask_price = 30, 30
bid_size, ask_size = 1, 0
env.step(expand_actions([bid_price, ask_price, bid_size, ask_size]))
env.fill_rewards(immediate_reward_buffer)
print(f'Made quote {bid_price} @ {ask_price} [{bid_size} x {ask_size}]. Proceeding...')
print(f'Players immediate rewards: {immediate_reward_buffer.cpu().numpy()}')
# %%
for i in range(args.players - 1):
    env.fill_observation_tensor(obs_buffer)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        with torch.inference_mode():
            npc_outputs = models[i].incremental_forward(obs_buffer, step)
            plot_action_distributions(npc_outputs['action_params'], args.max_contract_value, args.max_contracts_per_trade, title=f"NPC {i} Action Distributions").show()
            npc_actions = npc_outputs['action']
            npc_pred_roles = torch.softmax(npc_outputs['pinfo_preds']['private_roles'][0], dim=-1).float().cpu().numpy()
            print(f"Player 0 role: (ValueCheater {npc_pred_roles[0][0]:.2f}, HighLowCheater {npc_pred_roles[0][1]:.2f}, Customer {npc_pred_roles[0][2]:.2f})")
            print(f"NPC {i} actions: {npc_actions.cpu().numpy()}")
        env.step(npc_actions)
# %%
print(env.env.to_string(0))
# %%
