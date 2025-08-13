# %%

import sys
from pathlib import Path
python_root = Path(__file__).parent
sys.path.append(str(python_root / 'src'))

import torch
import pickle 
import numpy as np
from tqdm import trange, tqdm 
from collections import defaultdict
import plotly.graph_objects as go

from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.agent import HighLowTransformerModel
from high_low.rollouts import RolloutGenerator
from bandit import ThompsonBandit

# %%

checkpoint_root = python_root / 'checkpoints'
print(checkpoint_root)

# Get all checkpoint files matching the pattern
import re
checkpoint_files = []
for f in checkpoint_root.glob('small*.pt'):
    match = re.match(r'small(\d+)_(\d+)\.pt', f.name)
    if match:
        pool_run = int(match.group(1))
        num_steps = int(match.group(2))
        checkpoint_files.append((pool_run, num_steps, f))

# Sort by pool run first, then by num_steps
checkpoint_files.sort(key=lambda x: (x[0], x[1]))

# Filter to only keep checkpoints we want (e.g., every 3000 steps)
filtered_checkpoints = []
for pool_run, num_steps, path in checkpoint_files:
    if num_steps % 3000 == 0 and pool_run == 3:
        filtered_checkpoints.append((pool_run, num_steps, path))
filtered_checkpoints = filtered_checkpoints[:10]

print(f"Found {len(filtered_checkpoints)} checkpoints to evaluate")

# %%
# Load initial args from a checkpoint
initial_checkpoint = torch.load(
    python_root / 'checkpoints' / 'small_seedpool_4000.pt', 
    weights_only=False)
args = initial_checkpoint['args']
args.device_id = 1
args.effective_bandit_memory_size = 1000
args.num_rollouts = 30  # Number of rollouts per comparison

device = torch.device(f'cuda:{args.device_id}')
env = HighLowTrading(args.get_game_config())

# Initialize bandit and rollout generator
bandit = ThompsonBandit(args.effective_bandit_memory_size)
rollout_gen = RolloutGenerator(args)

# Create a single persistent main agent instance
main_agent = HighLowTransformerModel(args, env, verbose=False).to(device)
bandit.register_player('seedpool', initial_checkpoint['model_state_dict'])


# %% 
# Process checkpoints in order
results = []

for pool_idx, (pool_run, num_steps, checkpoint_path) in enumerate(tqdm(filtered_checkpoints)):
    print(f"\nEvaluating checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    current_weights = checkpoint['model_state_dict']
    main_agent.load_state_dict(current_weights, strict=False)
    for rollout_idx in tqdm(range(args.num_rollouts), desc="Rollouts"):
        selection = bandit.sample_batch(4)
        # Set up state dicts for rollout (main agent + 4 sampled opponents)
        state_dicts = [current_weights]  # Main agent weights
        for i in range(4):
            state_dicts.append(selection['object'][i])  # Opponent weights (already state dicts)

        # Randomly shuffle state_dicts according to a randomly generated permutation
        permutation = np.random.permutation(5)
        inverse_permutation = np.argsort(permutation)
        shuffled_state_dicts = [state_dicts[i] for i in permutation]

        payoff_matrix = rollout_gen.generate_rollout(shuffled_state_dicts)

        # Permute back the payoff_matrix to restore original order
        # payoff_matrix shape is [num_players, 5, 2]
        unshuffled_payoff_matrix = payoff_matrix[inverse_permutation]
        
        # Get aggregate returns for all players (row 4, col 0 = mean returns) and provide bandit feedback
        all_returns = unshuffled_payoff_matrix[:, 4, 0]  # Shape: [5] - returns for all 5 players
        main_return = all_returns[0]  # Main agent is player 0
        opponent_returns = all_returns[1:]  # Opponents are players 1-4
        opponent_wins = (opponent_returns > main_return).cpu().numpy().astype(int)
        bandit.update_parameters(selection['index'], opponent_wins)

        # # argsort all_returns and print out player names and returns in descending order
        sorted_indices = torch.argsort(all_returns, descending=True)
        player_names = ["main"] + selection['name']
        # print(f"\nRollout {rollout_idx + 1} rankings:")
        # for rank, idx in enumerate(sorted_indices):
        #     print(f"  {rank + 1}. {player_names[idx]}: {all_returns[idx]:.3f}")
        
        # # Print current bandit statistics from the last snapshot
        # if len(bandit.snapshots) > 0:
        #     print("\nCurrent bandit estimates (mean ± std):")
        #     last_snapshot = bandit.snapshots[-1]
        #     for i, name in enumerate(selection['name']):
        #         player_idx = selection['index'][i]
        #         mean = last_snapshot[player_idx, 0]
        #         std = last_snapshot[player_idx, 1]
        #         print(f"  {name}: {mean:.3f} ± {std:.3f}")
    # for j, player_name in enumerate(bandit.player_names):
    #     print(f"    {player_name}: {bandit.snapshots[-1][j, 0]:.3f} ± {bandit.snapshots[-1][j, 1]:.3f}")
    
    # Store results
    results.append({
        'checkpoint': checkpoint_path.name,
        'pool_run': pool_run,
        'num_steps': num_steps})
    
    # Register this checkpoint as a bandit player for future comparisons
    checkpoint_name = f"pool{pool_run}_step{num_steps}"
    bandit.register_player(checkpoint_name, current_weights.copy())

# %%

plotly_fig = bandit.plot_snapshots()
plotly_fig.show()

# %%
bandit.save(python_root / 'bandit_data.pkl')

# %%

checkpoints = [
    'random',
    'random',
    'random',
    'random',
    'small3_3000',
]
random_weights = HighLowTransformerModel(args, env, verbose=False).state_dict().copy()

payoff_matrix = rollout_gen.generate_rollout([
    random_weights if c == 'random' else torch.load(
        python_root / 'checkpoints' / f'{c}.pt', weights_only=False)['model_state_dict']
    for c in checkpoints 
])
print(payoff_matrix[:, 4].cpu().numpy())
# %%

print(bandit.player_names)
# %%
