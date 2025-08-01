from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

ArrayLike = Union[List, npt.NDArray]

def plot_market_and_players(infos, args, env_idx=0, fig_size=(450, 600)):
    """
    Generates a compact 4x2 subplot grid in a specific order for market analysis.

    - Layout: [Market Overview, Trade Volume], [All Positions, Player 0], etc.
    - All axis titles are removed for a minimal look.
    
    Args:
        infos (dict): A dictionary containing the simulation state info.
        args: An object containing configuration parameters.
        env_idx (int): The index of the environment instance to plot.
        fig_size (tuple): A (width, height) tuple for the figure size in pixels.
    """
    # Configurable top margin (adjust as needed for crooked display)
    top_margin = 100
    
    num_players = args.players
    player_role_mapping = {0: 'GoodValue', 1: 'BadValue', 2: 'HighLowCheater', 3: 'Customer'}
    
    # --- 1. Initialize Figure and Subplot Titles for New Layout ---
    subplot_titles = [
        "BBO", "Player Positions"
    ] + [f"p{i}" for i in range(num_players)]
    
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.06
    )
    
    # Market data is [N, P*T, 3] where columns are [best_bid, best_ask, last_price]
    steps_per_player = args.steps_per_player
    total_timesteps = num_players * steps_per_player
    
    # Convert tensors to numpy if needed
    market_tensor = infos['market']
    if isinstance(market_tensor, torch.Tensor):
        market_tensor = market_tensor.cpu().numpy()
    
    player_tensor = infos['players']
    if isinstance(player_tensor, torch.Tensor):
        player_tensor = player_tensor.cpu().numpy()
        
    # --- 2. Add Plots in the New Order ---

    # Plot 1: Market Overview (Prices) -> (1, 1)
    # NULL_INDEX from C++ is 0xFFFFFFFF (interpreted as -1 for signed ints)
    NULL_INDEX = -1 
    
    # Extract market data for the specific environment
    market_data = market_tensor[env_idx]  # [P*T, 3]
    x_steps = np.arange(total_timesteps)
    
    # Extract columns using tensor/array operations and convert to float
    best_bids = market_data[:, 0].astype(float)
    best_asks = market_data[:, 1].astype(float)
    last_prices = market_data[:, 2].astype(float)
    
    # Replace NULL_INDEX with NaN for plotting
    best_bids[best_bids == NULL_INDEX] = np.nan
    best_asks[best_asks == NULL_INDEX] = np.nan
    last_prices[last_prices == NULL_INDEX] = np.nan
    
    # Get settlement value
    settlement_value = infos['settlement_values'][env_idx]
    if isinstance(settlement_value, torch.Tensor):
        settlement_value = settlement_value.item()
    
    market_traces = {
        'bid': best_bids,
        'ask': best_asks,
        'last_price': last_prices,
        'contract_value': np.full(total_timesteps, settlement_value, dtype=float),
    }
    for name, data in market_traces.items():
        color = 'blue' if name == 'bid' else 'red' if name == 'ask' else (
            'mediumseagreen' if name == 'contract_value' else 'black'
        )
        fig.add_trace(go.Scatter(x=x_steps, y=data, name=name, mode="lines", line=dict(color=color)), row=1, col=1)

    # Plot 2: All Player Positions -> (1, 2)
    # Extract player data
    player_data = player_tensor[env_idx]  # [P, T, 6]
    # Columns: [bid_px, ask_px, bid_sz, ask_sz, contract_position, cash_position]
    
    player_positions = {}
    position_x_steps = np.arange(steps_per_player)  # Positions are tracked per round
    for player_idx in range(num_players):
        pos_data = player_data[player_idx, :, 4]  # contract_position column
        player_positions[player_idx] = pos_data
        fig.add_trace(go.Scatter(x=position_x_steps, y=pos_data, name=f'P{player_idx} pos'), row=1, col=2)

    # Plots 3-7: Individual Player Prices -> (2, 1) onwards
    # Reconstruct contract info from available data
    candidate_values = infos['candidate_values'][env_idx]
    if isinstance(candidate_values, torch.Tensor):
        candidate_values = candidate_values.cpu().numpy()
    
    min_value = candidate_values.min()
    max_value = candidate_values.max()
    contract_value = settlement_value
    settlement_string = f"({min_value},{max_value},{contract_value})"
    bad_value = min_value if contract_value == max_value else max_value

    target_positions = infos['target_positions'][env_idx]
    if isinstance(target_positions, torch.Tensor):
        target_positions = target_positions.cpu().numpy()
    target_positions = target_positions.astype(int)
    
    for player_idx in range(num_players):
        plot_idx_0based = player_idx + 2 # Player plots start at the 3rd subplot (index 2)
        row = plot_idx_0based // 2 + 1
        col = plot_idx_0based % 2 + 1

        if player_idx == 0: 
            # Additionally plot settlement_preds on player 0's plot
            settlement_preds = infos['settlement_preds'][:, int(env_idx)].float().cpu().numpy()
            # settlement_preds has shape [T] where T is the number of timesteps
            # We need to plot this across the player's timesteps
            fig.add_trace(
                go.Scatter(x=position_x_steps, y=settlement_preds, 
                          name=f'P{player_idx} settle_pred', 
                          mode="lines", 
                          line=dict(color='green', dash='dash')),
                row=row, col=col
            )
            # Also add actual settlement value as horizontal line
            fig.add_trace(
                go.Scatter(x=position_x_steps, 
                          y=np.full(len(position_x_steps), contract_value), 
                          name=f'P{player_idx} actual_settle', 
                          mode="lines", 
                          line=dict(color='darkgreen', width=1)),
                row=row, col=col
            )

        player_info = player_data[player_idx]  # [T, 6]
        player_bids = player_info[:, 0].copy()
        player_asks = player_info[:, 1].copy()
        player_bidsz = player_info[:, 2]
        player_asksz = player_info[:, 3]
        
        # Set bid/ask to edge values when size is 0
        player_bids[player_bidsz == 0] = 0 
        player_asks[player_asksz == 0] = args.max_contract_value

        for name, data in {'bid': player_bids, 'ask': player_asks}.items():
            color = 'blue' if name == 'bid' else 'red'
            fig.add_trace(go.Scatter(x=position_x_steps, y=data, name=f'P{player_idx} {name}', mode="lines", line=dict(color=color)), row=row, col=col)
        
        # Get player role
        info_roles = infos['info_roles'][env_idx]
        if isinstance(info_roles, torch.Tensor):
            info_roles = info_roles.cpu().numpy()
        player_role = info_roles[player_idx]
        
        # Update subplot title text
        role_name = player_role_mapping.get(player_role, "Unknown")
        if role_name == 'BadValue': 
            appendix = f' (BadValue: {bad_value})'
        elif role_name == 'GoodValue': 
            appendix = f' (GoodValue: {contract_value})'
        elif role_name == 'HighLowCheater': 
            appendix = f' (HighLowCheater: {"high" if contract_value == max_value else "low"})'
        elif role_name == 'Customer': 
            appendix = f' (Customer: {target_positions[player_idx]})'
        else: 
            appendix = ''
        fig.layout.annotations[plot_idx_0based].update(
            text=f'player {player_idx}. {appendix}'
        )
        
    # --- 3. Final Layout and Axis Configuration ---
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        showlegend=False,
        template="plotly_white",
        margin=dict(l=25, r=25, t=top_margin, b=80),
    )
    
    # Add title as annotation at the bottom with smaller font
    fig.add_annotation(
        text=f'Env {env_idx}. Settle {settlement_string}',
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12),
        xanchor="center"
    )
    
    # Configure all axes based on their new positions
    for i in range(1, 8):  # Now we have 7 subplots instead of 8
        row, col = ((i-1) // 2 + 1, ((i-1) % 2) + 1)
        
        # New plot locations
        is_pos_plot = (row == 1 and col == 2)

        fig.update_yaxes(title_text="", row=row, col=col)
        fig.update_xaxes(title_text="", row=row, col=col)
        
        if is_pos_plot:
            pass # Use auto-range for positions
        else: # Market overview and individual player plots
            fig.update_yaxes(range=[0, args.max_contract_value], row=row, col=col)

    fig.update_layout(autosize=False)
    fig.update_layout(xaxis=dict(automargin=True), yaxis=dict(automargin=True))
    return fig