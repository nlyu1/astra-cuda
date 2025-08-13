# %%

import sys
from pathlib import Path
python_root = Path(__file__).parent
sys.path.append(str(python_root / 'src'))

import torch
import numpy as np
from tqdm import trange, tqdm 
from collections import defaultdict
import plotly.graph_objects as go

from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.agent import HighLowTransformerModel
from high_low.rollouts import RolloutGenerator

# %%

def beta_std(alpha, beta):
    return np.sqrt((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)))

class ThompsonBandit:
    """
    Non-stationary Thompson sampling (Bernoulli) for multi-armed bandits. 
    Used to select the player which is most probable to win against the running main. 

    We use Beta distribution to model P(have higher ranking than main).
    In each round, thompson-sample the players and update parameters. 
    Since main is non-stationary, we use a decay factor to update the parameters after each step. 
    """
    def __init__(self, effective_bandit_memory_size: int):
        # 1 / (1 - gamma) = effective_memory_size 
        self.decay = 1 - 1 / effective_bandit_memory_size 

        self.parameters = None 
        self.player_names = []
        self.player_objects = {}
        self.names_of_players = {}
        self.snapshots = []
        self.processed_snapshots = defaultdict(lambda: {'mean': [], 'std': []})
        self.first_added = defaultdict(lambda: 1e10) # Step at which players' first play was introduced 

        self.last_processed_index = 0
        self.index = 0 

    def num_players(self):
        return len(self.parameters)

    def register_player(self, name, player_object):
        """
        Inserts a player with uniform prior. 
        """
        if name in self.names_of_players:
            raise ValueError(f"Player {name} already registered")
        self.player_names.append(name)
        self.names_of_players[name] = len(self.player_names) - 1

        if self.parameters is None:
            self.parameters = np.ones((1, 2))
        else:
            self.parameters = np.concatenate([self.parameters, np.ones((1, 2))], axis=0)
        self.player_objects[name] = player_object
        self.first_added[name] = self.index
        
    def sample_batch(self, num_samples: int):
        # Returns the names and stored objects of the sampled players
        # Ticks the decay factor
        if self.parameters is None:
            raise ValueError("Register players before sampling")

        samples = np.random.beta(
            self.parameters[:, 0][:, np.newaxis],
            self.parameters[:, 1][:, np.newaxis],
            size=(self.parameters.shape[0], num_samples))
        selection_indices = np.argmax(samples, axis=0)
        selected_players = [self.player_names[i] for i in selection_indices]
        return {
            'name': selected_players,
            'index': selection_indices,
            'object': [self.player_objects[name] for name in selected_players]
        }
    
    def update_parameters(self, selection_indices, bernoulli_wins):
        assert len(selection_indices) == len(bernoulli_wins)
        assert bernoulli_wins.shape == (len(selection_indices), )
        assert bernoulli_wins.min() >= 0 and bernoulli_wins.max() <= 1
        self.index += 1 

        self.parameters[selection_indices, 0] += bernoulli_wins
        self.parameters[selection_indices, 1] += 1 - bernoulli_wins
        self.parameters = self.parameters * self.decay

        means = self.parameters[:, 0] / (self.parameters[:, 0] + self.parameters[:, 1])
        stds = beta_std(self.parameters[:, 0], self.parameters[:, 1])
        # for i in range(len(selection_indices)):
        #     self.first_added[self.player_names[selection_indices[i]]] = min(
        #         self.first_added[self.player_names[selection_indices[i]]], self.index)
        self.snapshots.append(np.concatenate([means[:, np.newaxis], stds[:, np.newaxis]], axis=1))

    def _process_snapshots(self):
        """
        Only need to add from `self.last_processed_index` to `self.index`
        """
        for j in range(self.last_processed_index, self.index):
            for k in range(self.snapshots[j].shape[0]):
                self.processed_snapshots[self.player_names[k]]['mean'].append(float(self.snapshots[j][k, 0]))
                self.processed_snapshots[self.player_names[k]]['std'].append(float(self.snapshots[j][k, 1]))
        self.last_processed_index = self.index 
        return self.processed_snapshots
    
    def save(self, file_path):
        """
        Save bandit state to a file for later analysis.
        Note: player_objects (state dicts) are excluded to save space.
        """
        save_data = {
            # Scalar values
            'decay': self.decay,
            'last_processed_index': self.last_processed_index,
            'index': self.index,
            
            # Numpy arrays
            'parameters': self.parameters,
            'snapshots': self.snapshots,
            
            # Lists and dicts
            'player_names': self.player_names,
            'names_of_players': self.names_of_players,
            
            # Convert defaultdicts to regular dicts for serialization
            'processed_snapshots': dict(self.processed_snapshots),
            'first_added': dict(self.first_added),
            # Note: player_objects excluded (contains PyTorch state dicts)
        }
        np.savez_compressed(file_path, **save_data) 

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
filtered_checkpoints = filtered_checkpoints[:5]

print(f"Found {len(filtered_checkpoints)} checkpoints to evaluate")

# %%
# Load initial args from a checkpoint
initial_checkpoint = torch.load(
    python_root / 'checkpoints' / 'small_seedpool_4000.pt', 
    weights_only=False)
args = initial_checkpoint['args']
args.device_id = 1
args.effective_bandit_memory_size = 1000
args.num_rollouts = 20  # Number of rollouts per comparison

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

def visualize_snapshots(processed_snapshots, first_added):
    """
    x-axis should be number of steps (index)
    y-axis shows win probability (mean only)
    Note that each players' score should be appropriately offset by the step at which they were added. 
    """
    fig = go.Figure()
    
    # Use a color palette
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    total_steps = max(len(processed_snapshots[player_name]['mean']) for player_name in processed_snapshots)
    x_values = list(range(total_steps))
    for idx, player_name in enumerate(processed_snapshots):
        means = np.array(processed_snapshots[player_name]['mean'])
        first_step = int(first_added[player_name])
        print(player_name, first_step, len(means))
        print(means)
            
        padded_means = np.pad(means, (first_step, 0), mode='constant', constant_values=np.nan)
        color = colors[idx % len(colors)]
        
        # Add the mean line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=padded_means,
            mode='lines',
            name=player_name,
            line=dict(color=color, width=2),
            hovertemplate='%{text}<br>Step: %{x}<br>Win Probability: %{y:.3f}<extra></extra>',
            text=[player_name] * len(x_values)))
            
    fig.update_layout(
        title='Thompson Sampling Win Probability Estimates',
        xaxis_title='Update Step',
        yaxis_title='Win Probability',
        yaxis=dict(range=[0, 1]),
        width=1200,
        height=600,
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode='x')
    return fig

processed_snapshots = bandit._process_snapshots()
plotly_fig = visualize_snapshots(processed_snapshots, bandit.first_added)
plotly_fig.show()

# %%

checkpoints = [
    'random',
    'small3_3000',
    'small3_3000',
    'small3_3000',
    'small3_3000',
]
random_weights = main_agent.state_dict().copy()

payoff_matrix = rollout_gen.generate_rollout([
    random_weights if c == 'random' else torch.load(
        python_root / 'checkpoints' / f'{c}.pt', weights_only=False)['model_state_dict']
    for c in checkpoints 
])
print(payoff_matrix[:, 4].cpu().numpy())
# %%

print(bandit.player_names)
# %%
