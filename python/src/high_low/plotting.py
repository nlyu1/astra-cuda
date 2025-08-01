import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    
    num_players = 5
    player_role_mapping = {0: 'GoodValue', 1: 'BadValue', 2: 'HighLowCheater', 3: 'Customer'}
    
    # --- 1. Initialize Figure and Subplot Titles for New Layout ---
    subplot_titles = [
        "BBO", "Trade Volume",
        "Player Positions"
    ] + [f"p{i}" for i in range(num_players)]
    
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.06
    )
    
    x_steps = np.arange(infos['market'][env_idx].shape[0])

    # --- 2. Add Plots in the New Order ---

    # Plot 1: Market Overview (Prices) -> (1, 1)
    market_info = infos['market'][env_idx]
    env_params = infos['environment'][env_idx]
    last_price = np.array(market_info[:, 2]) # Don't do in-place operations
    last_price[last_price == 0] = np.nan
    market_traces = {
        'bid': market_info[:, 0],
        'ask': market_info[:, 1],
        'last_price': last_price,
        'contract_value': np.full_like(x_steps, env_params[2], dtype=float),
    }
    for name, data in market_traces.items():
        color = 'blue' if name == 'bid' else 'red' if name == 'ask' else (
            'mediumseagreen' if name == 'contract_value' else 'black'
        )
        fig.add_trace(go.Scatter(x=x_steps, y=data, name=name, mode="lines", line=dict(color=color)), row=1, col=1)

    # Plot 2: Market Trade Volume -> (1, 2)
    trade_traces = {
        'buy_size': infos['market'][env_idx, :, 3], 
        'sell_size': infos['market'][env_idx, :, 4]
    }
    for name, data in trade_traces.items():
        color = 'blue' if name == 'buy_size' else 'red'
        fig.add_trace(go.Scatter(x=x_steps, y=data, name=name, mode='lines', line=dict(color=color)), row=1, col=2)
    
    # Plot 3: All Player Positions -> (2, 1)
    player_positions = {}
    for player_idx in range(num_players):
        pos_data = infos['players'][env_idx, player_idx, :, 4]
        player_positions[player_idx] = pos_data # Store for this plot
        fig.add_trace(go.Scatter(x=x_steps, y=pos_data, name=f'P{player_idx} pos'), row=2, col=1)

    # Plots 4-8: Individual Player Prices -> (2, 2) onwards
    min_value, max_value, contract_value = infos['contract'][env_idx].tolist()
    settlement_string = f"({min_value},{max_value},{contract_value})"
    bad_value = min_value if contract_value == max_value else max_value

    target_positions = infos['target_positions'][env_idx].astype(int).tolist()
    for player_idx in range(num_players):
        plot_idx_0based = player_idx + 3 # Player plots start at the 4th subplot (index 3)
        row = plot_idx_0based // 2 + 1
        col = plot_idx_0based % 2 + 1

        player_info = infos['players'][env_idx, player_idx]
        player_bids, player_asks = np.copy(player_info[:, 0]), np.copy(player_info[:, 1])
        player_bidsz, player_asksz = player_info[:, 2], player_info[:, 3]
        player_bids[player_bidsz == 0] = 0 
        player_asks[player_asksz == 0] = args.max_contract_value

        for name, data in {'bid': player_bids, 'ask': player_asks}.items():
            color = 'blue' if name == 'bid' else 'red'
            fig.add_trace(go.Scatter(x=x_steps, y=data, name=f'P{player_idx} {name}', mode="lines", line=dict(color=color)), row=row, col=col)
        
        player_role = infos['info_roles'][env_idx, player_idx]
        # Update subplot title text, which is stored in fig.layout.annotations
        player_role = player_role_mapping.get(player_role, "Unknown")
        if player_role == 'BadValue': 
            appendix = f' (BadValue: {bad_value})'
        elif player_role == 'GoodValue': 
            appendix = f' (GoodValue: {contract_value})'
        elif player_role == 'HighLowCheater': 
            appendix = f' (HighLowCheater: {"high" if contract_value == max_value else "low"})'
        elif player_role == 'Customer': 
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
    for i in range(1, 9):
        row, col = ((i-1) // 2 + 1, ((i-1) % 2) + 1)
        
        # New plot locations
        is_vol_plot = (row == 1 and col == 2)
        is_pos_plot = (row == 2 and col == 1)

        fig.update_yaxes(title_text="", row=row, col=col)
        fig.update_xaxes(title_text="", row=row, col=col)
        
        if is_vol_plot:
            fig.update_yaxes(range=[0, args.max_contracts_per_trade], row=row, col=col)
        elif is_pos_plot:
            pass # Use auto-range for positions
        else: # Market overview and individual player plots
            fig.update_yaxes(range=[0, args.max_contract_value], row=row, col=col)

    fig.update_layout(autosize=False)
    fig.update_layout(xaxis=dict(automargin=True), yaxis=dict(automargin=True))
    return fig