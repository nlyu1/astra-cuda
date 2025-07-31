import torch
import astra
import numpy as np 

import scipy
from dataclasses import dataclass
from typing import Dict, Any, List

astra.register_games()

default_config = {
    "steps_per_player": 8,
    "max_contracts_per_trade": 1,
    "customer_max_size": 2,
    "max_contract_value": 10,
    "players": 5
}

class HighLowWrapper:
    def __init__(self, args, game_config: Dict[str, int] = default_config) -> None:
        self.game_config = game_config
        self.num_envs = args.envs_per_worker * args.env_workers
        self.args = args

        self.init_probe_actions() 
        self.remaining_nenvs = self.num_envs - self.num_probe_envs 
        self.remaining_encoder = astra.encoders.high_low_trading.load_vec_encoder(game_config, self.remaining_nenvs)

        self.init_state = astra.load_game("high_low_trading", game_config).new_initial_state()
        self.encoder = astra.encoders.high_low_trading.load_vec_encoder(game_config, args.num_envs)
        print(f'Environment seed: {args.seed}')
        self.rng = np.random.default_rng(args.seed)
        self.initial_move_state = False 
        self.num_resets = 0 

        self.reset()

    def is_initial_move_state(self):
        return self.initial_move_state
    
    def reset(self):
        self.rng = np.random.default_rng(self.args.seed + self.num_resets)
        self.num_resets += 1 

        # Resets all environments 
        self.states = astra.AsyncVecState(
            self.init_state.clone(), 
            num_sync_copies=self.args.env_workers, 
            num_async_copies=self.args.envs_per_worker)

        def validify_customer_size(arr):
            arr[arr >= 0] += 1
            return arr 

        remaining_actions = {
            'value_0': self.rng.integers(0, self.game_config['max_contract_value'], self.remaining_nenvs),
            'value_1': self.rng.integers(0, self.game_config['max_contract_value'], self.remaining_nenvs),
            'high_low': self.rng.integers(0, 2, self.remaining_nenvs),
            'customer_size1': self.remaining_encoder.encode_customer_sizes(validify_customer_size(
                self.rng.integers(-self.game_config['customer_max_size'], self.game_config['customer_max_size'], self.remaining_nenvs).astype(np.int32))),
            'customer_size2': self.remaining_encoder.encode_customer_sizes(validify_customer_size(
                self.rng.integers(-self.game_config['customer_max_size'], self.game_config['customer_max_size'], self.remaining_nenvs).astype(np.int32))),
            'permutation': self.rng.integers(0, scipy.special.factorial(self.game_config['players']), self.remaining_nenvs)
        }
        # Initialization 
        init_actions = {k: np.concatenate(
            [self.probe_actions[k], remaining_actions[k]]
        ).astype(np.int32) for k in self.probe_actions.keys()}
        self.states.apply_actions(init_actions['value_0'])
        self.states.apply_actions(init_actions['value_1'])
        self.states.apply_actions(init_actions['high_low'])
        self.states.apply_actions(init_actions['permutation'])
        self.states.apply_actions(init_actions['customer_size1'])
        self.states.apply_actions(init_actions['customer_size2'])
        self.initial_move_state = True 

    def step(self, actions: np.ndarray):
        actions = actions.astype(np.int32)
        assert actions.shape == (self.num_envs, 4), "Actions must be of shape (num_envs, 4)"
        action_ids = self.encoder.encode_trading_actions(actions)
        self.states.apply_actions(action_ids)
        self.initial_move_state = False 

    def returns(self):
        return self.states.returns()

    def rewards(self):
        return self.states.rewards()

    def current_player(self):
        return self.states.current_player()

    def player_observations(self, player: int):
        return self.states.observation_tensor(player)
    
    def observations(self):
        return self.player_observations(self.current_player())

    def is_terminal(self):
        return self.states.is_terminal()

    def init_probe_actions(self):
        # Returns environment-initialization actions for hard-coded probe environments 
        probe_config = {
            # 'contract_values': [(1, 30), (10, 20), (15, 15)],
            'contract_values': [(1, 10), (5, 5)],
            'high_low': [1, 0],
            # 'high_low': [1],
            'customer_sizes': [(-1, 1), (1, 1)],
            'player_permutation': [
                [2, 3, 0, 4, 1], # (highLow, Cust0[short], Value0[Bad], Cust1[long], Value1[Good])
                [3, 0, 4, 1, 2], # (Cust0[short], Value0[Bad], Cust1[long], Value1[Good], highLow)
                [0, 4, 1, 2, 3], # (Value0[Bad], Cust1[long], Value1[Good], highLow, Cust0[short])
                [4, 1, 2, 3, 0], # (Cust1[long], Value1[Good], highLow, Cust0[short], Value0[Bad])  
                [1, 2, 3, 0, 4], # (Value1[Good], highLow, Cust0[short], Value0[Bad], Cust1[long])
            ]}

        probe_params = {
            'value_0': [], 'value_1': [], 'high_low': [], 
            'customer_size1': [], 'customer_size2': [], 
            'permutation': []
        }

        for (v0, v1) in probe_config['contract_values']:
            for h in probe_config['high_low']:
                for (c1, c2) in probe_config['customer_sizes']:
                    for p in probe_config['player_permutation']:
                        # Print out relevant environments
                        # if (v0, v1) == (1, 10) and h == 1 and (c1, c2) == (-1, 1) and p == [2, 3, 0, 4, 1]:
                        #     print(len(probe_params['value_0']))
                        # if (v0, v1) == (5, 5) and h == 1 and (c1, c2) == (-1, 1) and p == [2, 3, 0, 4, 1]:
                        #     print(len(probe_params['value_0']))
                        # if (v0, v1) == (1, 10) and h == 0 and (c1, c2) == (-1, 1) and p == [2, 3, 0, 4, 1]:
                        #     print(len(probe_params['value_0']))
                        # if (v0, v1) == (1, 10) and h == 0 and (c1, c2) == (1, 1) and p == [2, 3, 0, 4, 1]:
                        #     print(len(probe_params['value_0']))
                        probe_params['value_0'].append(v0)
                        probe_params['value_1'].append(v1)
                        probe_params['high_low'].append(h)
                        probe_params['customer_size1'].append(c1)
                        probe_params['customer_size2'].append(c2)
                        probe_params['permutation'].append(p)
        self.probe_params = probe_params 
        num_probes = len(probe_params['value_0'])
        self.num_probe_envs = num_probes 
        self.num_remaining_envs = self.num_envs - self.num_probe_envs 
        print(f'Initialized {self.num_probe_envs} probe environments. Randomly sampling parameters for {self.num_remaining_envs} remaining environments')

        probe_encoder = astra.encoders.high_low_trading.load_vec_encoder(self.game_config, num_probes)
        probe_actions = {k: np.array(v).astype(np.int32) for k, v in probe_params.items()}

        probe_actions['value_0'] = probe_encoder.encode_contract_values(probe_actions['value_0'])
        probe_actions['value_1'] = probe_encoder.encode_contract_values(probe_actions['value_1'])
        probe_actions['high_low'] = probe_encoder.encode_high_low_actions(probe_actions['high_low'])
        probe_actions['customer_size1'] = probe_encoder.encode_customer_sizes(probe_actions['customer_size1'])
        probe_actions['customer_size2'] = probe_encoder.encode_customer_sizes(probe_actions['customer_size2'])
        probe_actions['permutation'] = probe_encoder.encode_permutations(probe_actions['permutation'])
        self.probe_actions = probe_actions 
