import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd

def beta_std(alpha, beta):
    return np.sqrt((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)))

class ThompsonSampler:
    """
    Non-stationary Thompson sampling (Bernoulli) for multi-armed bandits. 
    Used to select the player which is most probable to win against the running main. 

    We use Beta distribution to model P(have higher ranking than main).
    In each round, thompson-sample the players and update parameters. 
    Since main is non-stationary, we use a decay factor to update the parameters after each step. 
    """
    def __init__(self, effective_sampler_memory_size: int):
        # 1 / (1 - gamma) = effective_memory_size 
        self.decay = 1 - 1 / effective_sampler_memory_size 

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

        self.parameters *= self.decay ** len(selection_indices) # More samples -> more decay
        np.add.at(self.parameters[:, 0], selection_indices, bernoulli_wins)
        np.add.at(self.parameters[:, 1], selection_indices, 1.0 - bernoulli_wins)
        self.parameters = np.clip(self.parameters, a_min=1.0, a_max=None)

        means = self.parameters[:, 0] / (self.parameters[:, 0] + self.parameters[:, 1])
        stds = beta_std(self.parameters[:, 0], self.parameters[:, 1])
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
            'snapshots': self.snapshots,  # List of numpy arrays
            
            # Lists and dicts
            'player_names': self.player_names,
            'names_of_players': self.names_of_players,
            
            # Convert defaultdicts to regular dicts for serialization
            'processed_snapshots': dict(self.processed_snapshots),
            'first_added': dict(self.first_added),
            # Note: player_objects excluded (contains PyTorch state dicts)
        }
        
        # Use pickle to handle complex nested structures
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f) 
    

def load_from_file(pool_path):
    """
    Recreate a ThompsonSampler object from a saved directory.
    Does not load player objects
    """
    # Load the saved sampler state 
    with open(pool_path, 'rb') as f:
        save_data = pickle.load(f)
    
    # Recreate the sampler
    effective_memory = int(1 / (1 - save_data['decay']))
    sampler = ThompsonSampler(effective_memory)
    
    # Restore all saved attributes
    sampler.decay = save_data['decay']
    sampler.last_processed_index = save_data['last_processed_index']
    sampler.index = save_data['index']
    sampler.parameters = save_data['parameters']
    sampler.snapshots = save_data['snapshots']
    sampler.player_names = save_data['player_names']
    sampler.names_of_players = save_data['names_of_players']
    
    # Convert regular dicts back to defaultdicts
    sampler.processed_snapshots = defaultdict(lambda: {'mean': [], 'std': []}, save_data['processed_snapshots'])
    sampler.first_added = defaultdict(lambda: 1e10, save_data['first_added'])
    sampler.player_objects = defaultdict(lambda: None)
    
    print(f"Loaded ThompsonSampler with {len(sampler.player_names)} players")
    print(f"Index: {sampler.index}, Parameters shape: {sampler.parameters.shape}")
    
    return sampler

def beta_ab_from_mean_std(mean, std):
    v = std ** 2
    cap = mean * (1 - mean)  # upper bound on variance for a Beta with this mean
    k = cap / v - 1.0
    return mean * k, (1 - mean) * k

def plot_stat(sampler, data_type="mean", log_scale=True):
    """Plot statistics over time for all players."""
    assert data_type in ["mean", "std", "alpha", "beta"], f"Invalid data_type: {data_type}"
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    processed_snapshots = sampler._process_snapshots()
    total_steps = max(len(processed_snapshots[player_name]['mean']) 
                     for player_name in processed_snapshots if processed_snapshots[player_name]['mean'])
    if total_steps == 0:
        print("No data to plot")
        return fig
    x_values = list(range(total_steps))
    
    for idx, player_name in enumerate(processed_snapshots):
        means = np.array(processed_snapshots[player_name]['mean'])
        stds = np.array(processed_snapshots[player_name]['std'])
        alphas, betas = beta_ab_from_mean_std(means, stds)
        
        values = {'alpha': alphas, 'beta': betas, 'mean': means, 'std': stds}[data_type]
        if log_scale:
            values = np.log10(values + 1e-10)
        
        first_step = int(sampler.first_added[player_name])
        padded_values = np.pad(values, (first_step, 0), mode='constant', constant_values=np.nan)
        if len(padded_values) < total_steps:
            padded_values = np.pad(padded_values, (0, total_steps - len(padded_values)), 
                                  mode='constant', constant_values=np.nan)
        
        fig.add_trace(go.Scatter(
            x=x_values, y=padded_values, mode='lines', name=player_name,
            line=dict(color=colors[idx % len(colors)], width=2),
            text=[player_name] * len(x_values),
            hovertemplate='%{text}<br>Step: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ))
    
    title = f"{'Log10(' if log_scale else ''}Thompson Sampling {data_type.capitalize()}{')' if log_scale else ''}"
    fig.update_layout(
        title=title, xaxis_title="Steps", yaxis_title=title,
        width=1200, height=600, template='plotly_white',
        # legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    if data_type == "mean":
        ref_value = np.log10(0.5) if log_scale else 0.5
        fig.add_hline(y=ref_value, line_dash="dash", line_color="gray", 
                     annotation_text="50% win rate", annotation_position="right")
    return fig

def approximate_sampling_probs(sampler, samples_each_round=1e4, num_rounds=10):
    """Analyze sampling probabilities by running multiple sampling rounds."""
    k = int(samples_each_round)
    player_counts = {name: [] for name in sampler.player_names}
    
    for run in range(num_rounds):
        batch = sampler.sample_batch(k)
        unique, counts = np.unique(batch['name'], return_counts=True)
        run_counts = dict(zip(unique, counts))
        for player in sampler.player_names:
            player_counts[player].append(run_counts.get(player, 0))
    
    results = []
    for player, counts in player_counts.items():
        counts_array = np.array(counts)
        results.append({
            'name': player,
            'sample_prob': np.mean(counts_array) / k,
            'sample_std': np.std(counts_array) / k
        })
    return pd.DataFrame(results).sort_values('sample_prob', ascending=False).reset_index(drop=True)