import numpy as np
from collections import defaultdict
import pickle
import plotly.graph_objects as go

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
        self.parameters = (self.parameters * self.decay).clip(min=0.1)

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

    def snapshot_plot(self, info_type='mean'):
        """
        x-axis should be number of steps (index)
        y-axis shows win probability (mean only)
        Note that each players' score should be appropriately offset by the step at which they were added. 
        """
        assert info_type in ['mean', 'std'], "Can only plot mean or std"
        fig = go.Figure()
        processed_snapshots = self._process_snapshots()
        
        # Use a color palette
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        total_steps = max(len(processed_snapshots[player_name][info_type]) for player_name in processed_snapshots)
        x_values = list(range(total_steps))
        for idx, player_name in enumerate(processed_snapshots):
            means = np.array(processed_snapshots[player_name][info_type])
            first_step = int(self.first_added[player_name])
                
            padded_means = np.pad(means, (first_step, 0), mode='constant', constant_values=np.nan)
            color = colors[idx % len(colors)]
            
            # Add the mean line
            fig.add_trace(go.Scatter(
                x=x_values,
                y=padded_means,
                mode='lines',
                name=player_name,
                line=dict(color=color, width=2),
                text=[player_name] * len(x_values)))
                
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            width=1200,
            height=600,
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            hovermode='x',
            annotations=[
                dict(
                    text='Thompson Sampling Win Probability Estimates',
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=-0.15,
                    xanchor='center',
                    yanchor='top',
                    showarrow=False,
                    font=dict(size=12))
            ])
        return fig