from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

ArrayLike = Union[List, npt.NDArray]


def dual_plot(
    y1: Dict[str, ArrayLike],
    y2: Optional[Dict[str, ArrayLike]] = None,
    x: Optional[ArrayLike] = None,
    title: Optional[str] = None,
    y1title: Optional[str] = None,
    y2title: Optional[str] = None,
    xtitle: Optional[str] = None,
    y1min: Optional[float] = None,
    y1max: Optional[float] = None,
    y2min: Optional[float] = None,
    y2max: Optional[float] = None,
) -> go.Figure:
    """Create a dual-axis plot using plotly with optional axis range control."""
    # Configurable top margin (adjust as needed for crooked display)
    top_margin = 60
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if x is None:
        # Default x-axis is the index of the first data series
        x = np.arange(len(list(y1.values())[0]))

    # Add primary y-axis traces
    for name, data in y1.items():
        fig.add_trace(
            go.Scatter(x=x, y=data, name=name, mode="lines"), secondary_y=False
        )

    # Add secondary y-axis traces if they exist
    if y2 is not None:
        for name, data in y2.items():
            fig.add_trace(
                go.Scatter(x=x, y=data, name=name, mode="lines"), secondary_y=True
            )

    # Update layout and template
    fig.update_layout(
        margin=dict(l=20, r=20, t=top_margin, b=60),
        showlegend=True,
        template="plotly_white",
    )

    # Add title as annotation at the bottom with smaller font
    if title:
        fig.add_annotation(
            text=title,
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12),
            xanchor="center"
        )

    # Update primary y-axis title and range
    primary_y_updates = {"title_text": y1title if y1title is not None else ""}
    if y1min is not None and y1max is not None:
        primary_y_updates["range"] = [y1min, y1max]
    fig.update_yaxes(**primary_y_updates, secondary_y=False)

    # Update secondary y-axis title and range
    if y2 is not None:
        secondary_y_updates = {"title_text": y2title if y2title is not None else ""}
        if y2min is not None and y2max is not None:
            secondary_y_updates["range"] = [y2min, y2max]
        fig.update_yaxes(**secondary_y_updates, secondary_y=True)

    # Update x-axis title
    if xtitle is not None:
        fig.update_xaxes(title_text=xtitle)

    fig.update_layout(autosize=False)
    fig.update_layout(xaxis=dict(automargin=True), yaxis=dict(automargin=True))
    return fig

import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta as scipy_beta

def analyze_discrete_alpha_beta(alpha, beta, min_value, max_value, title="", m=None, kappa=None):
    """
    Plot the continuous beta distribution PDF and discrete probability bars.
    
    Args:
        alpha: float, alpha parameter of beta distribution (ignored if m and kappa provided)
        beta: float, beta parameter of beta distribution (ignored if m and kappa provided)
        min_value: float, minimum discrete value
        max_value: float, maximum discrete value
        title: str, title for the plot
        m: float, location parameter [0,1] (optional)
        kappa: float, dispersion parameter (optional)
    
    If m and kappa are provided, alpha and beta are computed as:
        alpha = kappa * m
        beta = kappa * (1 - m)
    """
    # If m and kappa are provided, compute alpha and beta from them
    if m is not None and kappa is not None:
        alpha = kappa * m
        beta = kappa * (1 - m)
    # Create x values for continuous PDF
    x_continuous = np.linspace(0, 1, 1000)
    y_continuous = scipy_beta.pdf(x_continuous, alpha, beta)
    entropy_value = scipy_beta.entropy(alpha, beta)
    
    # Transform to actual value range for visualization
    x_continuous_scaled = x_continuous * (max_value - min_value + 1) + (min_value - 0.5)
    
    # Calculate discrete probabilities using straight-through estimator
    discrete_values = np.arange(min_value, max_value + 1)
    discrete_probs = []
    
    for val in discrete_values:
        # Straight-through estimator: treat discrete value as if it were continuous
        # This matches the logp_entropy method in BetaActor
        normalized_x = (val - (min_value - 0.5)) / (max_value - min_value + 1)
        
        # Use PDF value at the normalized point (straight-through approximation)
        # Scale by bin width for probability interpretation
        prob = scipy_beta.pdf(normalized_x, alpha, beta) / (max_value - min_value + 1)
        discrete_probs.append(prob)
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add continuous PDF line
    fig.add_trace(go.Scatter(
        x=x_continuous_scaled,
        y=y_continuous / (max_value - min_value + 1),  # Adjust density for scaled x-axis
        mode='lines',
        name='Beta PDF (scaled)',
        line=dict(color='blue', width=2)
    ))
    
    # Add discrete probability bars
    fig.add_trace(go.Bar(
        x=discrete_values,
        y=discrete_probs,
        name='Discrete Probabilities',
        marker=dict(color='lightblue', opacity=0.7),
        width=0.8
    ))
    
    # Calculate kappa and m from alpha and beta
    # From BetaActor: alpha = kappa * m, beta = kappa * (1 - m)
    # Therefore: kappa = alpha + beta, m = alpha / (alpha + beta)
    kappa = alpha + beta
    m = alpha / (alpha + beta)
    
    # Update layout
    fig.update_layout(
        title=f'{title} (α={alpha:.2f}, β={beta:.2f}, κ={kappa:.2f}, m={m:.3f}, H={entropy_value:.2f})',
        xaxis_title='Value',
        yaxis_title='Probability Density',
        xaxis=dict(
            range=[min_value - 1, max_value + 1],
            dtick=1
        ),
        showlegend=True,
        hovermode='x unified')
    
    return {'fig': fig, 'kappa': kappa, 'm': m, 'entropy': entropy_value, 'discrete_probs': discrete_probs}