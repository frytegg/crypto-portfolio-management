"""Efficient frontier scatter plot component."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from dashboard.theme import COLORS, FIGURE_LAYOUT


def create_efficient_frontier_figure(
    frontier_data: dict,
    current_portfolio: tuple[float, float] | None = None,
) -> go.Figure:
    """Create efficient frontier plot with asset markers and key portfolios.

    Args:
        frontier_data: Output of compute_efficient_frontier(). Dict with keys
            frontier_risks, frontier_returns, asset_risks, asset_returns,
            max_sharpe_weights, min_vol_weights.
        current_portfolio: Optional (volatility, return) tuple for the current
            portfolio marker.

    Returns:
        go.Figure with frontier curve, asset dots, max Sharpe star,
        min vol diamond, CML dashed line, and optional current portfolio.
    """
    fig = go.Figure()

    risks = frontier_data["frontier_risks"]
    rets = frontier_data["frontier_returns"]
    asset_risks = frontier_data["asset_risks"]
    asset_returns = frontier_data["asset_returns"]

    # 1. Frontier curve
    fig.add_trace(go.Scatter(
        x=risks,
        y=rets,
        mode="lines",
        line=dict(color="#00d4ff", width=2),
        name="Efficient Frontier",
    ))

    # 2. Individual assets
    fig.add_trace(go.Scatter(
        x=asset_risks.values.tolist(),
        y=asset_returns.values.tolist(),
        mode="markers+text",
        text=list(asset_risks.index),
        textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=8, color="#888"),
        name="Assets",
    ))

    # 3. Max Sharpe point
    sharpe_ratios = [
        r / v if v > 0 else 0.0 for r, v in zip(rets, risks)
    ]
    max_sharpe_idx = int(np.argmax(sharpe_ratios))
    max_sharpe_risk = risks[max_sharpe_idx]
    max_sharpe_ret = rets[max_sharpe_idx]

    fig.add_trace(go.Scatter(
        x=[max_sharpe_risk],
        y=[max_sharpe_ret],
        mode="markers",
        marker=dict(symbol="star", size=16, color="gold"),
        name="Max Sharpe",
    ))

    # 4. Min volatility point
    min_vol_idx = int(np.argmin(risks))
    fig.add_trace(go.Scatter(
        x=[risks[min_vol_idx]],
        y=[rets[min_vol_idx]],
        mode="markers",
        marker=dict(symbol="diamond", size=14, color="lime"),
        name="Min Volatility",
    ))

    # 5. Capital Market Line (from rf=0 through max Sharpe, extended)
    rf = 0.0
    if max_sharpe_risk > 0:
        slope = (max_sharpe_ret - rf) / max_sharpe_risk
        cml_x_end = max(risks) * 1.2
        cml_y_end = rf + slope * cml_x_end
        fig.add_trace(go.Scatter(
            x=[0, cml_x_end],
            y=[rf, cml_y_end],
            mode="lines",
            line=dict(dash="dash", color="rgba(255,215,0,0.5)"),
            name="CML",
        ))

    # 6. Current portfolio (optional)
    if current_portfolio is not None:
        fig.add_trace(go.Scatter(
            x=[current_portfolio[0]],
            y=[current_portfolio[1]],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=14,
                color="red",
                line=dict(width=2),
            ),
            name="Current Portfolio",
        ))

    fig.update_layout(
        **FIGURE_LAYOUT,
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Expected Return",
        height=520,
        showlegend=True,
    )

    return fig
