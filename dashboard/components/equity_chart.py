"""Equity curve line chart component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import FIGURE_LAYOUT, STRATEGY_COLORS


def create_equity_chart(equity_curves: dict[str, pd.Series]) -> go.Figure:
    """Create overlaid equity curve chart for multiple strategies.

    Each series represents cumulative returns starting at 1.0.

    Args:
        equity_curves: {strategy_name: pd.Series of cumulative equity values}

    Returns:
        go.Figure with one line per strategy.
    """
    fig = go.Figure()

    for strat_name, curve in equity_curves.items():
        color_key = strat_name.lower().replace(" ", "_").replace("-", "_")
        color = STRATEGY_COLORS.get(color_key, "#888888")
        fig.add_trace(go.Scatter(
            x=curve.index,
            y=curve.values,
            mode="lines",
            name=strat_name,
            line=dict(color=color, width=1.5),
        ))

    fig.update_layout(
        **FIGURE_LAYOUT,
        title="Strategy Equity Curves (Growth of $1)",
        xaxis_title="Date",
        yaxis_title="Cumulative Value",
        height=450,
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )

    return fig
