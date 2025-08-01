# %% Transformer-based single-agent play against fixed opponents
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
from high_low.transformer_agent import HighLowTransformerModel

# %%
args = Args()
args.num_blocks = 256
args.device_id = 1
# Transformer-specific hyperparameters
args.n_hidden = 256
args.n_embd = 128
args.n_head = 2    # Number of attention heads
args.n_layer = 1   # Number of transformer blocks

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
agent = agent.compile(mode="reduce-overhead")  # Use reduce-overhead for inference

B, F = env.observation_shape()
T = args.steps_per_player
print(f"Batch size: {B}, Sequence length: {T}, Feature dim: {F}")
print(f"Transformer config: {args.n_hidden}d hidden, {args.n_head} heads, {args.n_layer} layers")

# Count parameters
total_params = sum(p.numel() for p in agent.parameters())
trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (assuming float32)")

mock_input = torch.randn(B, T, F).to(device)
actions = torch.randint(1, 4, (B, T, 4)).to(device)

# %% Benchmark parallel batch forward (should be much faster than GRU)
%%timeit
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    outputs = agent._batch_forward(mock_input, actions)
    (outputs['logprobs']).sum().backward()

# %% Benchmark compiled version
# %%timeit
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    outputs = agent(mock_input, actions)
    (outputs['logprobs']).sum().backward()

# %% Benchmark sequential sampling (autoregressive generation)
# %%timeit
hidden = agent.initial_belief()
for t in range(T):
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = agent.sample_actions(mock_input[:, t], hidden)
            print(hidden.shape)
            hidden = outputs['hidden']
print(hidden.shape)

# %% Example training loop snippet
optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)

# %% 
%%timeit 
# Single training step example
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

# print(f"Value loss: {value_loss.item():.4f}")
# print(f"Policy loss: {policy_loss.item():.4f}")
# print(f"Entropy: {outputs['entropy'].mean().item():.4f}")

# %%
a = torch.randn(B, T, F*2).to('cuda:1')

# %%
%%timeit 
b = a.to('cuda:0').cpu()
# %%
