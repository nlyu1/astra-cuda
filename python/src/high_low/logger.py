import os 
import torch 

import wandb 
import numpy as np 
from src.plotting import dual_plot
from src.high_low.plotting import plot_market_and_players
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
                dir="./checkpoints/logs", 
            )
        self.counter = 0 
        self.last_heavy_counter = 0 

    def update_stats(self, logging_inputs, global_step, heavy_updates=False):
        self.counter += 1 

        offset = logging_inputs['offset'] # Offset of current player 
        returns = logging_inputs['returns'] # from env.fill_returns(), [N, P]
        player_rewards = returns[:, offset].cpu()
        infos = logging_inputs['infos'] # from env.expose_info()
        dist_params = logging_inputs['dist_params'] # Distribution parameters from agent

        # Obtain the status of the current agent 
        info_roles = infos['info_roles'][:, offset].cpu()
        good_value_mask = (info_roles == 0)
        bad_value_mask = (info_roles == 1)
        high_low_mask = (info_roles == 2)
        customer_mask = (info_roles == 3)

        # infos['players']: [N, P, T, 6] int standing for (bid_px, ask_px, bid_sz, ask_sz, contract_position, cash_position)
        # infos['market']: [N, P*T, 2+1] int (best bid px, best ask px, last_price)
        # infos['settlement_values']: [N]
        last_price = infos['market'][:, :, 2]
        last_price[last_price == 0] = 2 * self.args.max_contract_value # Explicit penalty
        contract_value = infos['settlement_values'].unsqueeze(-1) # [N, 1]
        last_price_diff = (last_price - contract_value).abs()

        buy_volume = infos['players'][:, :, :, 2] # [N, P, T]T
        sell_volume = infos['players'][:, :, :, 3] # [N, P, T]
        best_bid = infos['market'][:, :, 0] # [N, P*T]
        best_ask = infos['market'][:, :, 1] # [N, P*T]
        bbo_mid = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid).clamp(0)
        bbo_mid_diff = (bbo_mid - contract_value).abs()
        is_within_bbo = (contract_value >= best_bid) & (contract_value <= best_ask)

        # Stratification symbol based on game's total customer demand 
        customer_size_sum = infos['target_positions'].sum(-1) 
        unique_sizes = torch.unique(customer_size_sum)

        if customer_mask.any():
            position_diff = (
                (infos['target_positions'][:, offset] - infos['players'][:, offset, -1, -2])
            )[customer_mask].abs().float().mean().item() # Position difference at last timestep 
        else:
            position_diff = 0

        log_data = {
            "reward/avg_returns": player_rewards.mean().item(),
            'reward/goodValue': player_rewards[good_value_mask].mean().item(),
            'reward/badValue': player_rewards[bad_value_mask].mean().item(),
            'reward/highLow': player_rewards[high_low_mask].mean().item(),
            'reward/customer': player_rewards[customer_mask].mean().item(),
            'reward/welfare': returns.sum(-1).mean().item(),
            'reward/missed_positions': position_diff}
        log_data = log_data | logging_inputs['segment_timer']
        for k in ['pool_logs', 'benchmark_payoffs']:
            if k in logging_inputs:
                log_data = log_data | logging_inputs[k]

        for s in [-10, -6, 0, 6, 10]: 
            size_mask = (customer_size_sum == s)
            log_data = log_data | ({
                f'last_price_diff/{s}': last_price_diff[size_mask, -1].float().mean().item(),
                f'bbo_mid_diff/{s}': bbo_mid_diff[size_mask, 1].float().mean().item(),
                f'total_volume/{s}': (
                    buy_volume[size_mask].float().mean(0).sum() 
                    + sell_volume[size_mask].float().mean(0).sum()).item(),
                f'final_spread/{s}': spread[size_mask, -1].float().mean().item(),
                f'final_capture/{s}': is_within_bbo[size_mask, -1].float().mean().item()
            })

        T, B = logging_inputs['settlement_preds'].shape
        settlement_preds = logging_inputs['settlement_preds'] # [T, B]
        private_role_preds = logging_inputs['private_role_preds'] # [T, B, num_players]
        private_role_gt = infos['pinfo_targets'].unsqueeze(0) # [1, B, num_players]
        infos['settlement_preds'] = settlement_preds
        private_role_acc = (private_role_preds == private_role_gt) # [T, B, num_players]G
        non_self_mask = torch.ones((self.args.players,), dtype=bool)
        non_self_mask[offset] = 0
        non_self_acc = private_role_acc[:, :, non_self_mask].float().mean(-1) # [T, B]
        self_acc = private_role_acc[:, :, offset].float() # [T, B]
        
        settlement_price = contract_value.view(1, -1) # [1, B]
        settlement_diff = (settlement_preds - settlement_price).abs().mean(-1) # [T]
        log_data['acc/self_acc'] = self_acc.mean().item()
        for pivot in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
            idx = int(pivot * (settlement_preds.shape[0] - 1))
            log_data[f'settlement_pred_diff/{idx+1}'] = settlement_diff[idx].mean().item()
            log_data[f'acc/non_self_acc{idx+1}'] = non_self_acc[idx].mean().item()
        
        # Log distribution parameters (only transfer to CPU here)
        for j, name in enumerate(['bid_px', 'ask_px', 'bid_sz', 'ask_sz']):
            for k, v in dist_params.items():
                log_data[f'{k}/{name}'] = v[:, j].mean().item()

        if heavy_updates: 
            customer_size_mask = (customer_size_sum == 0)
            self.last_heavy_counter = self.counter
            
            # Expand settlement_diff from [T] to [T*P] timeline: pad with NaNs before player offset, repeat each prediction P times
            settlement_diff_numpy = settlement_diff.float().cpu().numpy() # [T]
            extended_settlement_diff = np.concatenate([
                np.full(offset, np.nan),  # NaN padding before player's first turn
                np.repeat(settlement_diff_numpy, self.args.players)  # Repeat each prediction P times
            ])[:self.args.players * self.args.steps_per_player]  # Truncate to T*P length
            
            market_fig = dual_plot(
                {'last price diff': last_price_diff[customer_size_mask].float().mean(0).cpu().numpy(), # [T*P]
                 'spread': spread[customer_size_mask].float().mean(0).cpu().numpy(), 
                 'settlement pred diff': extended_settlement_diff},
                 {'capture_ratio': is_within_bbo[customer_size_mask].float().mean(0).cpu().numpy()},
                title='Last-price diff, spread, and capture ratio over time')
            # Add information for pinfo modeling 
            customer_acc = non_self_acc[:, customer_mask].mean(-1).cpu().numpy() # [T] 
            value_cheater_acc = non_self_acc[:, good_value_mask | bad_value_mask].mean(-1).cpu().numpy() # [T]
            high_low_acc = non_self_acc[:, high_low_mask].mean(-1).cpu().numpy() # [T]
            acc_fig = dual_plot(
                {'customer': customer_acc, 'value_cheater': value_cheater_acc, 'high_low': high_low_acc},
                title='Private role modeling accuracy over time')
            
            # Combine all heavy plots into the main log_data dictionary
            log_data["market_fig/market"] = wandb.Plotly(market_fig)
            log_data["market_fig/pinfo"] = wandb.Plotly(acc_fig)     

            env_probe_indices = [0, 5, 15, 30, 120, 150, 200, 210, 250, 300, 301, 302, 303, 304, 305]
            # Filter out indices that are beyond the number of environments
            num_envs = infos['market'].shape[0]
            env_probe_indices = [idx for idx in env_probe_indices if idx < num_envs]
            probe_plots = {
                f"probes/probe{probe}": wandb.Plotly(plot_market_and_players(infos, self.args, env_idx=probe))
                for probe in env_probe_indices}
            log_data.update(probe_plots)
        wandb.log(log_data, step=global_step)