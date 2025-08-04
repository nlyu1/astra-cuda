# %%

import torch
import torch.nn as nn 

import random
import numpy as np 

import sys 
sys.path.append("/home/nlyu/Code/astra-cuda/python/src/")
from high_low.env import HighLowTrading
from high_low.config import Args
from high_low.agent import HighLowTransformerModel
from high_low.plotting import plot_market_and_players

# %%
args = Args()
args.num_blocks = 1
args.threads_per_block = 1
args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')
args.customer_max_size = 2
args.max_contract_value = 10
args.max_contracts_per_trade = 1
args.steps_per_player = 8
args.players = 5
args.num_steps = 8
game_config = args.get_game_config()
print('Game setup:')
for k, v in game_config.items():
    print(f"{k}: {v}")

env = HighLowTrading(game_config)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed);


models = [HighLowTransformerModel(args, env, verbose=False).to(device) for _ in range(args.players - 1)]
for model in models:
    model.load_state_dict(
    torch.load(f"/home/nlyu/Code/astra-cuda/python/checkpoints/smallgame_pool2_78000.pt", map_location=device, weights_only=False
        )['model_state_dict'])
    model.compile()
advisor_model = HighLowTransformerModel(args, env, verbose=False).to(device)
advisor_model.load_state_dict(
    torch.load(f"/home/nlyu/Code/astra-cuda/python/checkpoints/smallgame_pool2_78000.pt", map_location=device, weights_only=False
        )['model_state_dict'])
advisor_model.compile()
# %%

env.reset()
for model in models:
    model.reset_context()
advisor_model.reset_context()
returns_buffer = env.new_reward_buffer()
reward_buffer = env.new_player_reward_buffer()
obs_buffer = env.new_observation_buffer()
# print(env.env.to_string(0))
expand_actions = lambda x: torch.tensor(x).unsqueeze(0).to(device).int()

# %%

for step in range(args.num_steps - 1):
    print(f'Observation at step {step}:')
    print(env.observation_string(0))
    env.fill_observation_tensor(obs_buffer)
    print(obs_buffer.cpu().numpy())
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        with torch.inference_mode():
            advisor_output = advisor_model.incremental_forward(obs_buffer, step)
            advisor_actions = advisor_output['action']
            print(advisor_actions.cpu().numpy())
    pred_roles = torch.softmax(advisor_output['pinfo_preds']['private_roles'][0], dim=-1).float().cpu().numpy()
    for p in range(args.players):
        print(f"Player {p} role: (ValueCheater {pred_roles[p][0]:.2f}, HighLowCheater {pred_roles[p][1]:.2f}, Customer {pred_roles[p][2]:.2f})")
    print(f"Settle price prediction: {advisor_output['pinfo_preds']['settle_price'].float().item():.2f}")

    bid_price = int(input('Bid price: '))
    ask_price = int(input('Ask price: '))
    bid_size = int(input('Bid size: '))
    ask_size = int(input('Ask size: '))
    env.step(expand_actions([bid_price, ask_price, bid_size, ask_size]))
    print(f'Made quote {bid_price} @ {ask_price} [{bid_size} x {ask_size}]. Proceed?')
    input()

    for i in range(args.players - 1):
        env.fill_observation_tensor(obs_buffer)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.inference_mode():
                npc_outputs = models[i].incremental_forward(obs_buffer, step)
                npc_actions = npc_outputs['action']
                npc_pred_roles = torch.softmax(npc_outputs['pinfo_preds']['private_roles'][0], dim=-1).float().cpu().numpy()
                print(f"Player 0 role: (ValueCheater {npc_pred_roles[0][0]:.2f}, HighLowCheater {npc_pred_roles[0][1]:.2f}, Customer {npc_pred_roles[0][2]:.2f})")
                print(f"NPC {i} actions: {npc_actions.cpu().numpy()}")
        env.step(npc_actions)
# %%
print(env.observation_string(0))
# %%
