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