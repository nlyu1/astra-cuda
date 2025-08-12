# %%

import sys
from pathlib import Path
python_root = Path(__file__).parent
sys.path.append(str(python_root / 'src'))

import torch
import numpy as np
from collections import defaultdict

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

        self.parameters[selection_indices, 1 - bernoulli_wins] += 1 
        self.parameters = self.parameters * self.decay

        means = self.parameters[:, 0] / (self.parameters[:, 0] + self.parameters[:, 1])
        stds = beta_std(self.parameters[:, 0], self.parameters[:, 1])
        self.snapshots.append(np.concatenate([means[:, np.newaxis], stds[:, np.newaxis]], axis=1))

    def snapshot_plots(self):
        """
        Only need to add from `self.last_processed_index` to `self.index`
        """
        for j in range(self.last_processed_index, self.index):
            for k in range(self.snapshots[j].shape[0]):
                self.processed_snapshots[self.player_names[k]]['mean'].append(self.snapshots[j][k, 0])
                self.processed_snapshots[self.player_names[k]]['std'].append(self.snapshots[j][k, 1])
        self.last_processed_index = self.index 
        return self.processed_snapshots 

# %%

checkpoint_root = python_root / 'checkpoints'
print(checkpoint_root)
# Look at checkpoints inside 'checkpoints' Only focus on small{pool_run}_{num_steps}.pt. Sort first by increasing pool run then num_steps
# Next, register a random player as the baseline, then step in increasing order through the checkpoints. 
# For each step, should thompson-sample 4 opponents 10 times, do rollout, and update thompson parameters. 
#    Note that thompson parameters are bernoulli, so change ranking to 1 if return is later than main. 
#    After each step, move on to the next "main" checkpoint and register the last main checkpoint as a thompson bandit player. 

# %%
checkpoint = torch.load(
    python_root / 'checkpoints' / 'small_seedpool_4000.pt', 
    weights_only=False)
args = checkpoint['args']
args.effective_bandit_memory_size = 1000

bandit = ThompsonBandit(args.effective_bandit_memory_size)

bandit.register_player('a', 1)
bandit.register_player('b', 2)
bandit.register_player('c', 3)
bandit.register_player('d', 4)

bandit.sample_batch(10)