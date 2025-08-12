# %%

import torch 
from pathlib import Path
import sys
python_root = Path(__file__).parent
sys.path.append(str(python_root / 'src'))

from high_low.config import Args
from high_low.env import HighLowTrading
from high_low.agent import HighLowTransformerModel

checkpoint = torch.load(
    python_root / 'checkpoints' / 'small_seedpool_4000.pt', 
    weights_only=False)
args = checkpoint['args']
args.device_id = 1
args.num_blocks = 512
args.fill_runtime_args()
device = torch.device(f'cuda:{args.device_id}')

class RolloutGenerator:
    def __init__(self, args):
        device = torch.device(f'cuda:{args.device_id}')
        self.env = HighLowTrading(args.get_game_config())
        self.agents = [
            HighLowTransformerModel(args, self.env, verbose=False).to(device)
            for _ in range(args.players)]
        self.payoff_matrix = torch.zeros(args.players, 4 + 1, 2)
        self.obs_buffer = self.env.new_observation_buffer()
        self.returns_buffer = self.env.new_reward_buffer()

    def generate_rollout(self, state_dicts):
        for j in range(args.players):
            self.agents[j].load_state_dict(state_dicts[j], strict=False)
            self.agents[j].eval()
            self.agents[j].reset_context()
        self.env.reset()

        for step in range(args.num_steps):
            for j in range(args.players):
                self.env.fill_observation_tensor(self.obs_buffer)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    with torch.inference_mode():
                        model_outputs = self.agents[j].incremental_forward(self.obs_buffer, step)
                        model_actions = model_outputs['action']
            self.env.step(model_actions)

        self.env.fill_returns(self.returns_buffer)
        info_roles = self.env.get_pinfo_targets()['info_roles']  # [N, P] with values 0-3

        for player_idx in range(args.players):
            for role in range(4):  # 0: goodValue, 1: badValue, 2: highLow, 3: customer
                # Mask for when this player has this role
                role_mask = (info_roles[:, player_idx] == role)
                
                if role_mask.any():
                    # Get returns for this player when in this role
                    player_returns = self.returns_buffer[role_mask, player_idx]
                    
                    # Store mean and std
                    self.payoff_matrix[player_idx, role, 0] = player_returns.mean()
                    self.payoff_matrix[player_idx, role, 1] = player_returns.std()
        self.payoff_matrix[:, 4, 0] = self.returns_buffer.mean(0)
        self.payoff_matrix[:, 4, 1] = self.returns_buffer.std(0)
        return self.payoff_matrix 
# %%
evaluator = RolloutGenerator(args)
def get_random_weights():
    return HighLowTransformerModel(args, evaluator.env, verbose=False).to(device).state_dict()
state_dicts = [get_random_weights() for _ in range(args.players)]

# %%

%%timeit
payoff_matrix = evaluator.generate_rollout(state_dicts)
# %%

from collections import defaultdict
from copy import deepcopy
from trueskill import Player, Game 

class Arena:
    def __init__(self, args, save_to=None):
        if save_to is None:
            save_to = Path(python_root / 'checkpoints' / args.exp_name)

        self.weights_cache = {} # Cache for 
        self.saved_to_disk = defaultdict(lambda: False) # Indicator for whether we have saved the weights to disk

        self.agent_pool = {}
        self.agent_players = {}
        self.playouts = []
        self.save_to = save_to
        self._initialize()
        if save_to.exists():
            print(f'Arena loaded from {save_to}')
        else:
            print(f'Arena initialized from scratch at {save_to}')
        self.args = args 
        self.device = torch.device(f'cuda:{args.device_id}')

    def _get_weights(self, name):
        if name in self.weights_cache:
            return self.weights_cache[name]
        else:
            # try to look for {save_to}/checkpoints/
            candidate_path = self.save_to / 'checkpoints' / f'{name}.pt'
            if candidate_path.exists():
                self.weights_cache[name] = torch.load(
                    candidate_path, weights_only=False, 
                    map_location=self.device)['model_state_dict']
                self.saved_to_disk[name] = True
                return self.weights_cache[name]
            else:
                raise FileNotFoundError(f'Weights for {name} not found at {candidate_path}')
    
    def register_player(self, name, copy_from=None):
        """
        This registration only puts the player in the registry without committing the weights 
        """
        if copy_from is None:
            assert copy_from in self.agent_players, f'Player {copy_from} not found in registry'
            self.agent_players[name] = deepcopy(self.agent_players[copy_from])
            return 
        assert name not in self.agent_players, f'Player {name} already registered'
        self.agent_players[name] = Player()

    def register_playout_scores(self, names, scores):
        agent_players = [self.agent_players[name] for name in names]
        

    def _load_from_directory(self):
        if not self.save_to.exists():
            self.save_to.mkdir(parents=True, exist_ok=True)
            (self.save_to / "checkpoints").mkdir(parents=True, exist_ok=True)
            print(f'Initialized arena at {self.save_to}')
            return 
        
        for checkpoint_path in self.save_to.glob("checkpoints/*.pt"):
            name = checkpoint_path.stem
            self.weights_cache[name] = self._get_weights(name)
        print(f"Loaded {len(self.weights_cache)} weights from {self.save_to}")


agent_pool = {
    f'random_{j}': get_random_weights()
    for j in range(args.players)}