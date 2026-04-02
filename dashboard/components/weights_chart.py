"""Allocation weights visualization — bar and pie charts."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import COLORS, FIGURE_LAYOUT, STRATEGY_COLORS


def create_weights_bar_chart(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    """Horizontal bar chart of a single strategy's portfolio weights.

    Sorted descending by weight. Only shows assets with weight > 0.001.
    Color intensity maps to weight magnitude.

    Args:
        weights: pd.Series indexed by asset name, values = allocation weights.
        title: Chart title.

    Returns:
        go.Figure with horizontal bars.
    """
    # Filter near-zero weights and sort descending
    w = weights[weights > 0.001].sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=w.values,
        y=w.index,
        orientation="h",
        marker=dict(
            color=w.values,
            colorscale="Blues",
            cmin=0,
            cmax=max(w.values) if len(w) > 0 else 1,
        ),
        text=[f"{v:.1%}" for v in w.values],
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        **FIGURE_LAYOUT,
        title=title,
        xaxis_title="Weight",
        xaxis=dict(
            tickformat=".0%",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
        ),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        height=max(300, len(w) * 28 + 100),
        showlegend=False,
    )

    return fig


def create_weights_bar(weights_by_strategy: dict[str, pd.Series]) -> go.Figure:
    """Grouped bar chart comparing weights across multiple strategies.

    Args:
        weights_by_strategy: {strategy_name: pd.Series of weights}

    Returns:
        go.Figure with grouped bars (one group per asset).
    """
    if not weights_by_strategy:
        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT, title="No strategy results")
        return fig

    # Collect all assets across strategies, sorted by average weight
    all_assets: set[str] = set()
    for w in weights_by_strategy.values():
        all_assets.update(w.index)

    avg_weights = {}
    for asset in all_assets:
        vals = [w.get(asset, 0.0) for w in weights_by_strategy.values()]
        avg_weights[asset] = sum(vals) / len(vals)

    # Top 15 assets by average weight for readability
    sorted_assets = sorted(avg_weights, key=avg_weights.get, reverse=True)[:15]

    fig = go.Figure()
    for strat_name, w in weights_by_strategy.items():
        color_key = strat_name.lower().replace(" ", "_").replace("-", "_")
        color = STRATEGY_COLORS.get(color_key, "#888888")
        fig.add_trace(go.Bar(
            name=strat_name,
            x=sorted_assets,
            y=[float(w.get(a, 0.0)) for a in sorted_assets],
            marker_color=color,
        ))

    fig.update_layout(
        **FIGURE_LAYOUT,
        title="Strategy Weight Comparison (Top 15 Assets)",
        barmode="group",
        xaxis_title="Asset",
        yaxis_title="Weight",
        yaxis=dict(
            tickformat=".0%",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
        ),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        height=450,
        legend=dict(orientation="h", y=-0.2),
    )

    return fig


