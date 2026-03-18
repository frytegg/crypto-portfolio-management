"""GARCH volatility forecast chart components."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import COLORS, FIGURE_LAYOUT


def create_garch_vol_chart(
    asset: str,
    garch_result: dict,
    prices: pd.Series,
) -> go.Figure:
    """Create GARCH conditional volatility chart with price overlay.

    Two y-axes: left = conditional volatility (line), right = asset price
    (secondary line, dimmed).

    Args:
        asset: Asset ticker name (e.g. "BTC").
        garch_result: Output dict from fit_garch() containing
            "conditional_volatility" (pd.Series).
        prices: Price series for the asset (same date range).

    Returns:
        Plotly Figure with dual y-axes.
    """
    cond_vol = garch_result["conditional_volatility"]

    fig = go.Figure()

    # Conditional volatility (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=cond_vol.index,
            y=cond_vol.values,
            name="Conditional Vol",
            line=dict(color=COLORS["danger"], width=2),
            yaxis="y",
        )
    )

    # Price overlay (right y-axis, dimmed)
    aligned_prices = prices.reindex(cond_vol.index)
    fig.add_trace(
        go.Scatter(
            x=aligned_prices.index,
            y=aligned_prices.values,
            name="Price",
            line=dict(color=COLORS["text_muted"], width=1, dash="dot"),
            opacity=0.5,
            yaxis="y2",
        )
    )

    fig.update_layout(
        **FIGURE_LAYOUT,
        title=f"{asset} — GJR-GARCH(1,1,1) Conditional Volatility",
        height=450,
        yaxis=dict(
            title="Conditional Volatility",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            side="left",
        ),
        yaxis2=dict(
            title="Price (USD)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
    )

    return fig


def create_vol_heatmap(
    garch_results: dict[str, dict],
    n_days: int = 90,
) -> go.Figure:
    """Create a volatility heatmap: assets as rows, last N days as columns.

    Values are conditional_volatility from GARCH fits.
    Colorscale: RdYlGn_r (red = high vol, green = low vol).

    Args:
        garch_results: Output from fit_all_garch(), keyed by asset name.
        n_days: Number of trailing days to show.

    Returns:
        Plotly Figure with go.Heatmap.
    """
    assets = sorted(garch_results.keys())
    if not assets:
        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT, title="No GARCH data available")
        return fig

    # Use the index from the first asset's conditional_volatility
    first_vol = garch_results[assets[0]]["conditional_volatility"]
    dates = first_vol.index[-n_days:]

    z_data = []
    for asset in assets:
        vol = garch_results[asset]["conditional_volatility"]
        aligned = vol.reindex(dates).fillna(0)
        z_data.append(aligned.values)

    z_matrix = np.array(z_data)

    date_labels = [d.strftime("%m/%d") for d in dates]

    fig = go.Figure(
        go.Heatmap(
            z=z_matrix,
            x=date_labels,
            y=assets,
            colorscale="RdYlGn_r",
            colorbar=dict(title="Vol", thickness=15),
            hovertemplate=(
                "Asset: %{y}<br>Date: %{x}<br>Vol: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        **FIGURE_LAYOUT,
        title=f"Conditional Volatility Heatmap (Last {n_days} Days)",
        height=max(300, len(assets) * 25 + 100),
        xaxis=dict(
            title="Date",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
            tickangle=-45,
            dtick=max(1, n_days // 15),
        ),
        yaxis=dict(
            title="",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
        ),
    )

    return fig
