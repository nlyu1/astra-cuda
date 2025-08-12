import torch 

from env import HighLowTrading
from agent import HighLowTransformerModel

class RolloutGenerator:
    def __init__(self, args):
        self.device = torch.device(f'cuda:{args.device_id}')
        self.env = HighLowTrading(args.get_game_config())
        self.agents = [
            HighLowTransformerModel(args, self.env, verbose=False).to(self.device)
            for _ in range(args.players)]
        self.payoff_matrix = torch.zeros(args.players, 4 + 1, 2)
        self.obs_buffer = self.env.new_observation_buffer()
        self.returns_buffer = self.env.new_reward_buffer()

    def generate_rollout(self, state_dicts):
        for j in range(self.args.players):
            self.agents[j].load_state_dict(state_dicts[j], strict=False)
            self.agents[j].eval()
            self.agents[j].reset_context()
        self.env.reset()

        for step in range(self.args.num_steps):
            for j in range(self.args.players):
                self.env.fill_observation_tensor(self.obs_buffer)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    with torch.inference_mode():
                        model_outputs = self.agents[j].incremental_forward(self.obs_buffer, step)
                        model_actions = model_outputs['action']
            self.env.step(model_actions)

        self.env.fill_returns(self.returns_buffer)
        info_roles = self.env.get_pinfo_targets()['info_roles']  # [N, P] with values 0-3

        for player_idx in range(self.args.players):
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