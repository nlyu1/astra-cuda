# %% Transformer-based single-agent play against fixed opponents
import time
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
import astra_cuda as astra
astra.register_games()

import sys
sys.path.append('./src')
import numpy as np
from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.agent import HighLowTransformerModel

# %%
args = Args()

# Pre/post processing blocks
args.pre_encoder_blocks = 2  # Residual blocks before transformer
args.post_decoder_blocks = 1 # Residual blocks after transformer

# Standard parameters
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

args.exp_name = 'vtrace_highlow_transformer'
args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')

game_config = args.get_game_config()
env = HighLowTrading(game_config)
agent = HighLowTransformerModel(args, env).to(device)

# Compile for faster execution - critical for reducing latency
agent.compile(mode="default")  # Use reduce-overhead for inference

B, F = env.observation_shape()
T = args.steps_per_player
mock_input = torch.randn(T, B, F).to(device)
actions = torch.randint(1, 4, (T, B, 4)).to(device)
print(mock_input.shape, actions.shape)

# %% Benchmark sampling.
%%timeit

"""
~105ms on 5090 with max-autotune and ~113ms for default. For the following configuration:

Sampling 327680 frames per iteration across 16384 environments
Per-gradient step batch size: 81920. 4 gradient steps for 2 updates
Batch size: 16384, Sequence length: 20, Feature dim: 41
Transformer config: 512d hidden, 2 heads, 2 layers
Total parameters: 600,281
Trainable parameters: 600,281
Model size: 2.29 MB (assuming float32)
"""
context = agent.initial_context()
for t in range(T):
    outputs = agent.sample_actions(mock_input[t], context)
    context = outputs['context']
    # print(context.shape)

# %% Example training loop snippet
optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

# %% Single training step example. ~40ms on 5090
%%timeit 
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    outputs = agent(mock_input, actions)
    
    # Example loss computation (you'd use actual rewards/advantages)
    value_loss = 0.5 * outputs['values'].pow(2).mean()
    policy_loss = -outputs['logprobs'].mean()  # Would multiply by advantages
    entropy_loss = -0.01 * outputs['entropy'].mean()  # Entropy bonus
    
    # Private info prediction losses (if using)
    pinfo_losses = {}
    for key, pred in outputs['pinfo_preds'].items():
        # You'd compare against actual targets here
        pinfo_losses[key] = pred.pow(2).mean()
    
    total_loss = value_loss + policy_loss + entropy_loss + sum(pinfo_losses.values())

optimizer.zero_grad()
total_loss.backward()
optimizer.step()
# %%
