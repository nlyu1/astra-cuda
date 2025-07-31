# %%
import torch 
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import numpy as np 
import astra_cuda as astra
astra.register_games()

default_config = {
    'customer_max_size': 5,
    'max_contract_value': 30,
    'max_contracts_per_trade': 5,
    'steps_per_player': 20,
    'players': 5,

    'threads_per_block': 128,
    'num_markets': 128*512,
    'device_id': 0,
}

class HighLowTrading:
    def __init__(self, game_config) -> None:
        self.game_config = game_config
        self.device = torch.device(f'cuda:{game_config["device_id"]}')
        self.game = astra.load_game("high_low_trading", game_config)
        self.env = self.game.new_initial_state()
        self.N = game_config['num_markets']
        self.P = game_config['players']
        self.T = game_config['steps_per_player']
        self.cM = game_config['customer_max_size']
        self.S = game_config['max_contracts_per_trade']
        self.M = game_config['max_contract_value']
        self.reset()

    def reset(self):
        self.env.reset()
        # Randomly initialize new environment
        self.candidate_values = torch.randint(1, self.M + 1, (self.N, 2), device=self.device).int()
        self.env.apply_action(self.candidate_values)
        self.high_low = torch.randint(0, 2, (self.N,), device=self.device).int()
        self.env.apply_action(self.high_low)

        self.permutation = torch.argsort(torch.rand(self.N, self.P, device=self.device), dim=1).int()
        self.env.apply_action(self.permutation)

        self.customer_sizes = torch.randint(-self.cM, self.cM, (self.N, self.P - 3), device=self.device).int()
        self.customer_sizes[self.customer_sizes >= 0] += 1
        self.env.apply_action(self.customer_sizes)

    def step(self, action):
        # Returns the scalar denoting player 
        return self.env.apply_action(action)

    def current_player(self):
        current_player = self.env.current_player()
        assert current_player >= 0 and current_player < self.P, "Current player is out of bounds"
        return current_player
    
    def fill_observation_tensor(self, values, player=None):
        return self.env.fill_observation_tensor(self.current_player(), values)
    
    def observation_shape(self):
        return self.game.observation_tensor_shape()
    
    def new_observation_buffer(self):
        return torch.zeros(tuple(self.observation_shape()), device=self.device).float()
    
    def new_reward_buffer(self):
        return torch.zeros((self.N, self.P), device=self.device).float()
    
    def observation_string(self, player=None, index=0):
        return self.env.observation_string(self.current_player(), index)
    
    def fill_rewards(self, reward_buffer):
        return self.env.fill_rewards(reward_buffer)
    
    def fill_returns(self, returns_buffer):
        return self.env.fill_returns(returns_buffer)
    
    def fill_rewards_since_last_action(self, reward_buffer, player_id):
        return self.env.fill_rewards_since_last_action(reward_buffer, player_id)
    
    def terminal(self):
        return self.env.is_terminal()
    
    def accumulated_rewards(self):
        return self.env.accumulated_rewards()


# if __name__ == "__main__": 
interactive_config = {
    'customer_max_size': 5,
    'max_contract_value': 30,
    'max_contracts_per_trade': 5,
    'steps_per_player': 10,
    'players': 4,

    'threads_per_block': 128,
    'num_markets': 128*512,
    'device_id': 0,
}
env = HighLowTrading(interactive_config)
id = 10 
obs_buffer = env.new_observation_buffer()
immediate_rewards = env.new_reward_buffer()
cumulative_rewards = env.new_reward_buffer()
cumulative_returns = env.new_reward_buffer()
amplify = lambda action: torch.stack([action]*env.N, dim=0).to(env.device).int()

for j in range(2): # Play two rounds
    print('############### Public info ###############')
    print(env.env.to_string(id))

    while not env.terminal():
        count = env.env.move_number()
        max_count = env.game.max_game_length()
        
        current_player = env.current_player()
        env.fill_observation_tensor(obs_buffer)
        print(f'############### Player {current_player} observation. Step {count} / {max_count} ###############')
        print(env.observation_string(id))
        print('Observation:', obs_buffer[id].cpu())

        bid_px = np.random.randint(1, env.M)
        ask_px = np.random.randint(bid_px + 1, env.M + 1)
        bid_sz = np.random.randint(0, env.S + 1)
        ask_sz = np.random.randint(0, env.S + 1)
        action_lst = [bid_px, ask_px, bid_sz, ask_sz]
        action = amplify(torch.tensor(action_lst))
        env.fill_rewards_since_last_action(cumulative_rewards, current_player)
        player_who_just_acted = env.step(action)  # Apply the action!
        env.fill_rewards(immediate_rewards)
        print(f'Immediate rewards: {immediate_rewards[id].cpu()}')
        print(f'Cumulative rewards: {cumulative_rewards[id].cpu()}')
        count += 1 

    env.fill_returns(cumulative_returns)
    print('############### Terminal ###############')
    print(env.env.to_string(id))
    print(f'Cumulative returns: {cumulative_returns[id].cpu()}')

    print('\n\n\n\n\n')