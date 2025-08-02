#!/usr/bin/env python3
"""
Generate a clean playthrough of the high-low trading game showing consistent 
environment data and verifying role assignments.
"""

import torch 
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import numpy as np 
import astra_cuda as astra
astra.register_games()

# Configuration for a small test game
config = {
    'customer_max_size': 5,
    'max_contract_value': 30,
    'max_contracts_per_trade': 5,
    'steps_per_player': 3,  # Short game for clarity
    'players': 4,
    'threads_per_block': 128,
    'num_markets': 1,  # Just one market for clarity
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

    def get_pinfo_targets(self):
        # [0, 1, 2] for [valueCheater, highLowCheater, customer], respectively
        pinfo_targets = (self.permutation - 1).clamp(0, 2) # [N, P]
        max_values = self.candidate_values.max(dim=1).values # [N] 
        min_values = self.candidate_values.min(dim=1).values # [N, P]
        settlement_values = self.high_low * max_values + (1 - self.high_low) * min_values # [N]

        #  info roles [0, 1, 2, 3] for goodValue, badValue, highLow, Customer, respectively
        info_roles = self.permutation.clone().clamp(0, 3) # [N, P]
        pos_0_mask = (self.permutation == 0)
        pos_1_mask = (self.permutation == 1)
        # Check which contracts were chosen
        first_chosen = (self.candidate_values[:, 0] == settlement_values) # [N]
        second_chosen = (self.candidate_values[:, 1] == settlement_values) # [N]
        N, P = info_roles.shape
        info_roles[pos_0_mask] = torch.where(first_chosen, 0, 1).repeat_interleave(P).reshape(N, P)[pos_0_mask].int()
        info_roles[pos_1_mask] = torch.where(second_chosen, 0, 1).repeat_interleave(P).reshape(N, P)[pos_1_mask].int()
        return {'pinfo_targets': pinfo_targets, 
                'settlement_values': settlement_values, 
                'candidate_values': self.candidate_values,
                'info_roles': info_roles}

    def step(self, action):
        return self.env.apply_action(action)

    def current_player(self):
        current_player = self.env.current_player()
        assert current_player >= 0 and current_player < self.P, "Current player is out of bounds"
        return current_player
    
    def fill_observation_tensor(self, values, player=None):
        self.env.fill_observation_tensor(self.current_player(), values)
    
    def observation_shape(self):
        return self.game.observation_tensor_shape()

    def num_features(self):
        # This is fixing a single timestep
        assert len(self.game.observation_tensor_shape()) == 2
        return self.game.observation_tensor_shape()[-1]
    
    def expose_info(self):
        return self.env.expose_info()
    
    def new_observation_buffer(self):
        return torch.zeros(tuple(self.observation_shape()), device=self.device).float()
    
    def new_reward_buffer(self):
        return torch.zeros((self.N, self.P), device=self.device).float()
    
    def new_player_reward_buffer(self):
        return torch.zeros((self.N,), device=self.device).float()
    
    def observation_string(self, player=None, index=0):
        if player is None:
            player = self.current_player()
        return self.env.observation_string(player, index)
    
    def fill_rewards(self, reward_buffer):
        return self.env.fill_rewards(reward_buffer)
    
    def fill_returns(self, returns_buffer):
        return self.env.fill_returns(returns_buffer)
    
    def fill_rewards_since_last_action(self, reward_buffer, player=None):
        if player is None:
            player = self.current_player()
        return self.env.fill_rewards_since_last_action(reward_buffer, player)
    
    def terminal(self):
        return self.env.is_terminal()


def main():
    env = HighLowTrading(config)
    env_id = 0  # Single environment
    
    # Print initial game setup
    print('='*80)
    print('GAME SETUP (Public Information)')
    print('='*80)
    print(env.env.to_string(env_id))
    print()
    
    # Show what each player sees at the start
    print('='*80)
    print('INITIAL PRIVATE INFORMATION FOR EACH PLAYER')
    print('='*80)
    for player_id in range(env.P):
        print(f'\n--- Player {player_id} Private View ---')
        # Extract just the private info part
        obs_str = env.env.observation_string(player_id, env_id)
        private_start = obs_str.find('********** Private Information **********')
        private_end = obs_str.find('******************************************', private_start + 1)
        if private_start >= 0 and private_end >= 0:
            print(obs_str[private_start:private_end + 42])
    
    # Play through the game
    print('\n' + '='*80)
    print('GAME PLAY')
    print('='*80)
    
    obs_buffer = env.new_observation_buffer()
    immediate_rewards = env.new_reward_buffer()
    cumulative_rewards = env.new_player_reward_buffer()
    cumulative_returns = env.new_reward_buffer()
    
    round_num = 0
    while not env.terminal():
        current_player = env.current_player()
        round_num += 1
        
        print(f'\n### Round {round_num} - Player {current_player}\'s turn ###')
        
        # Show player's view
        env.fill_observation_tensor(obs_buffer)
        print(f'Player {current_player} observation tensor (first 6 elements):')
        obs_first_6 = obs_buffer[env_id][:6].cpu()
        print(f'  [is_value, is_highlow, is_customer, sin(id), cos(id), private_info]')
        print(f'  {obs_first_6}')
        
        # Simple random action
        bid_px = np.random.randint(1, env.M)
        ask_px = np.random.randint(bid_px + 1, env.M + 1)
        bid_sz = np.random.randint(0, env.S + 1)
        ask_sz = np.random.randint(0, env.S + 1)
        
        print(f'Player {current_player} quotes: {bid_px} @ {ask_px} [{bid_sz} x {ask_sz}]')
        
        # Apply action
        action = torch.tensor([[bid_px, ask_px, bid_sz, ask_sz]], device=env.device).int()
        env.step(action)
        
        # Show rewards
        env.fill_rewards(immediate_rewards)
        print(f'Immediate rewards: {immediate_rewards[env_id].cpu()}')
    
    # Final state
    print('\n' + '='*80)
    print('FINAL STATE')
    print('='*80)
    print(env.env.to_string(env_id))
    
    env.fill_returns(cumulative_returns)
    print(f'\nFinal returns: {cumulative_returns[env_id].cpu()}')
    
    # Verify returns calculation
    print('\nReturns verification:')
    # Extract final positions from the game state string
    game_str = env.env.to_string(env_id)
    import re
    positions = []
    for i in range(env.P):
        match = re.search(f'Player {i} position: \\[(-?\\d+) contracts, (-?\\d+) cash\\]', game_str)
        if match:
            contracts = int(match.group(1))
            cash = int(match.group(2))
            positions.append((contracts, cash))
            
            # Calculate expected return
            if i == 0:  # Customer with target 4
                penalty = abs(contracts - 4) * 30
                expected = cash + contracts * 25 - penalty
            else:
                expected = cash + contracts * 25
            
            print(f'  Player {i}: {cash} + {contracts}*25' + 
                  (f' - {penalty}' if i == 0 else '') + 
                  f' = {expected}')


if __name__ == '__main__':
    main()