# %% Single-agent play against fixed opponents
import time
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import wandb
import astra_cuda as astra
astra.register_games()
from tqdm import trange

import sys
sys.path.append('./src')
import numpy as np
from high_low.config import Args
from high_low.env import HighLowTrading
from python.archive.rnn_agent import HighLowGRUModel
# %%
args = Args()
args.hidden_size = 512
args.rnn_hidden_size = 128
args.rnn_n_layers = 1
args.pre_rnn_blocks = 1
args.post_rnn_blocks = 1

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

args.exp_name = 'vtrace_highlow'
args.device_id = 0
args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')

game_config = args.get_game_config()
env = HighLowTrading(game_config)
agent = HighLowGRUModel(args, env).to(device)

B, F = env.observation_shape() # Batch and hidden dimensions
T = args.steps_per_player
print(f"Batch size: {B}, Sequence length: {T}, Feature dim: {F}")
print(f"GRU config: hidden_size={args.hidden_size}, rnn_hidden={args.rnn_hidden_size}, rnn_layers={args.rnn_n_layers}")

# Count parameters
total_params = sum(p.numel() for p in agent.parameters())
trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (assuming float32)")
mock_input = torch.randn(B, T, F).to(device)
actions = torch.randint(1, 4, (B, T, 4)).to(device)

# %%
%%timeit
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    outputs = agent._batch_forward(mock_input, actions)
    (outputs['logprobs']).sum().backward()

# %%
%%timeit
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    outputs = agent(mock_input, actions)
    (outputs['logprobs']).sum().backward()
# %%
%%timeit
hidden = agent.initial_belief()
for t in range(T):
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = agent.sample_actions(mock_input[:, t], hidden)
            hidden = outputs['hidden']
            # print(outputs['action'].shape)
# %%