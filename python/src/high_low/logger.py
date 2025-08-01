import os 

import wandb 
import numpy as np 
from plotting import dual_plot, plot_market_and_players
from wandb.sdk.wandb_settings import Settings

class HighLowLogger:
    def __init__(self, args, wandb_initialized=False):  
        self.args = args 
        if not wandb_initialized: 
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=vars(args),
                name=args.run_name,
                save_code=True,
                dir="/tmp/high_low_ppo_wandb", 
            )
        self.counter = 1 
        self.last_heavy_counter = 0 

    def update_stats(self, logging_inputs, global_step, heavy_updates=False):
        self.counter += 1 

        offset = logging_inputs['offset'] # Offset of current player 
        returns = logging_inputs['returns']
        reward = returns[:, offset]
        infos = logging_inputs['infos']

        info_roles = infos['info_roles'][:, offset]
        good_value_mask = (info_roles == 0)
        bad_value_mask = (info_roles == 1)
        high_low_mask = (info_roles == 2)
        customer_mask = (info_roles == 3)

        last_price = np.array(infos['market'][:, :, 2])
        last_price[last_price == 0] = 2 * self.args.max_contract_value # Explicit penalty
        last_price_residual = last_price - infos['contract'][:, 2][:, None] 
        last_price_diff = np.abs(last_price_residual)

        buy_volume = infos['market'][:, :, 3]
        sell_volume = infos['market'][:, :, 4]
        best_bid = infos['market'][:, :, 0] # Shape [num_envs, num_timesteps]
        best_ask = infos['market'][:, :, 1] # Shape [num_envs, num_timesteps]
        bbo_mid = (best_bid + best_ask) / 2
        spread = np.maximum(best_ask - best_bid, 0)
        contract_value = infos['contract'][:, 2][:, None] # Shape [num_envs]
        bbo_mid_diff = np.abs(bbo_mid - contract_value)
        is_within_bbo = (contract_value >= best_bid) & (contract_value <= best_ask)

        # Stratification symbol based on game's 
        customer_size_sum = infos['target_positions'].sum(-1) 
        unique_sizes = np.unique(customer_size_sum)

        if customer_mask.any():
            position_diff = np.abs(
                (infos['target_positions'][:, offset] - infos['players'][:, offset, -1, -1])
            )[customer_mask].mean()
        else:
            position_diff = 0

        log_data = {
            "reward/avg_returns": reward.mean(),
            'reward/goodValue': reward[good_value_mask].mean(),
            'reward/badValue': reward[bad_value_mask].mean(),
            'reward/highLow': reward[high_low_mask].mean(),
            'reward/customer': reward[customer_mask].mean(),
            'reward/welfare': returns.sum(-1).mean(),
            'reward/missed_positions': position_diff,
        }

        for s in [-4, 0, 4]: 
            size_mask = (customer_size_sum == s)
            log_data[f'last_price_diff/{s}'] = last_price_diff[size_mask, -1].mean()
            log_data[f'bbo_mid_diff/{s}'] = bbo_mid_diff[size_mask, 1].mean()
            log_data[f'total_volume/{s}'] = (
                buy_volume[size_mask].mean(0).sum() 
                + sell_volume[size_mask].mean(0).sum())
            log_data[f'final_spread/{s}'] = spread[size_mask, -1].mean()
            log_data[f'final_capture/{s}'] = is_within_bbo[size_mask, -1].mean()

        T, B = logging_inputs['settlement_preds'].shape
        settlement_preds = logging_inputs['settlement_preds'].cpu().numpy() # [T, B]
        private_role_preds = logging_inputs['private_role_preds'].cpu().numpy() # [T, B, num_players]
        private_role_gt = infos['info_roles'] # [B, num_players]
        private_role_gt[private_role_gt == 0] = 1 # Aggregate good value (0) and bad value (1) into value cheater. 
        # (valueCheater 1, highLow 2, customer 3) -> (valueCheater 0, highLow 1, customer 2)
        private_role_gt = np.stack([private_role_gt] * T, axis=0) - 1
        private_role_acc = (private_role_preds == private_role_gt) # [T, B, num_players]
        non_self_mask = np.ones((self.args.players,), dtype=bool)
        non_self_mask[offset] = 0
        non_self_acc = private_role_acc[:, :, non_self_mask].mean(-1) # [T, B]
        self_acc = private_role_acc[:, :, offset] # [T, B]

        
        settlement_price = infos['contract'][:, 2][None, :] # [1, B]
        settlement_diff = np.abs(settlement_preds - settlement_price).mean(-1) # size-T
        log_data['acc/self_acc'] = self_acc.mean()
        for pivot in [0.25, 0.5, 0.75, 1.0]:
            idx = int(pivot * (settlement_preds.shape[0] - 1))
            log_data[f'settlement_pred_diff/{idx+1}'] = settlement_diff[idx].mean()
            log_data[f'acc/non_self_acc{idx+1}'] = non_self_acc[idx].mean()
        wandb.log(log_data, step=global_step)

        if heavy_updates: 
            customer_size_mask = (customer_size_sum == 0)
            self.last_heavy_counter = self.counter
            market_fig = dual_plot(
                {'last price diff': last_price_diff[customer_size_mask].mean(0)},
                {'cumulative volume': (
                    buy_volume[customer_size_mask].mean(0).cumsum() + 
                    sell_volume[customer_size_mask].mean(0).cumsum())},
                title='Last price diff and volume over time',
            )
            spread_fig = dual_plot(
                {'spread': spread[customer_size_mask].mean(0)},
                {'capture_ratio': is_within_bbo[customer_size_mask].mean(0)},
                y2min=0.2, y2max=1.0, title='Spread and capture ratio over time',
            )
            env_probe_indices = [0, 5, 15, 30, 120, 150, 200, 210, 250, 300, 301, 302, 303, 304, 305]
            
            # Combine all heavy plots into the main log_data dictionary
            log_data["market_fig/market"] = wandb.Plotly(market_fig)
            log_data["market_fig/spread"] = wandb.Plotly(spread_fig)
            
            probe_plots = {
                f"probes/probe{probe}": wandb.Plotly(plot_market_and_players(infos, self.args, env_idx=probe))
                for probe in env_probe_indices}
            log_data.update(probe_plots)
            wandb.log(log_data, step=global_step)
