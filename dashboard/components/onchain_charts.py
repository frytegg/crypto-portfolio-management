"""On-chain signal charts — TVL, stablecoin, DEX volume."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import COLORS, FIGURE_LAYOUT


def create_tvl_chart(
    total_tvl: pd.Series,
    eth_tvl: pd.Series,
    sol_tvl: pd.Series,
) -> go.Figure:
    """Stacked area chart: Ethereum, Solana, and Other TVL.

    Args:
        total_tvl: Total DeFi TVL (pd.Series, date index).
        eth_tvl: Ethereum TVL.
        sol_tvl: Solana TVL.

    Returns:
        go.Figure with stacked area chart.
    """
    # Align indices — use total_tvl as the master index
    idx = total_tvl.index
    eth = eth_tvl.reindex(idx).fillna(0)
    sol = sol_tvl.reindex(idx).fillna(0)
    other = total_tvl - eth - sol
    other = other.clip(lower=0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=idx,
        y=eth.values,
        name="Ethereum",
        mode="lines",
        fill="tozeroy",
        line=dict(color="#627eea", width=0.5),
        fillcolor="rgba(98,126,234,0.5)",
        hovertemplate="Ethereum: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=idx,
        y=sol.values,
        name="Solana",
        mode="lines",
        fill="tozeroy",
        line=dict(color="#00ffa3", width=0.5),
        fillcolor="rgba(0,255,163,0.4)",
        hovertemplate="Solana: $%{y:,.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=idx,
        y=other.values,
        name="Other",
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["text_muted"], width=0.5),
        fillcolor="rgba(153,153,153,0.3)",
        hovertemplate="Other: $%{y:,.0f}<extra></extra>",
    ))

    # Total TVL as overlay line
    fig.add_trace(go.Scatter(
        x=idx,
        y=total_tvl.values,
        name="Total TVL",
        mode="lines",
        line=dict(color=COLORS["text"], width=2),
        hovertemplate="Total: $%{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="DeFi Total Value Locked (TVL)",
        xaxis_title="Date",
        yaxis_title="TVL (USD)",
        height=400,
        legend=dict(orientation="h", y=-0.15),
    )

    return fig


def create_stablecoin_chart(stablecoin_mcap: pd.Series) -> go.Figure:
    """Line chart with area fill for stablecoin market cap. Adds 30d rolling average.

    Args:
        stablecoin_mcap: Stablecoin total market cap (pd.Series, date index).

    Returns:
        go.Figure with area chart and 30d MA overlay.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stablecoin_mcap.index,
        y=stablecoin_mcap.values,
        name="Stablecoin Market Cap",
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["info"], width=1.5),
        fillcolor="rgba(52,152,219,0.2)",
        hovertemplate="$%{y:,.0f}<extra>Market Cap</extra>",
    ))

    # 30d rolling average
    if len(stablecoin_mcap) >= 30:
        ma_30 = stablecoin_mcap.rolling(30).mean()
        fig.add_trace(go.Scatter(
            x=ma_30.index,
            y=ma_30.values,
            name="30d MA",
            mode="lines",
            line=dict(color=COLORS["warning"], width=2, dash="dash"),
            hovertemplate="$%{y:,.0f}<extra>30d MA</extra>",
        ))

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="Stablecoin Total Market Cap",
        xaxis_title="Date",
        yaxis_title="Market Cap (USD)",
        height=400,
        legend=dict(orientation="h", y=-0.15),
    )

    return fig


def create_dex_volume_chart(dex_volume: pd.Series) -> go.Figure:
    """Bar chart for daily DEX volume with 7d and 30d MA overlays.

    Args:
        dex_volume: Daily DEX volume (pd.Series, date index).

    Returns:
        go.Figure with bars + moving average lines.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=dex_volume.index,
        y=dex_volume.values,
        name="Daily Volume",
        marker_color=COLORS["primary"],
        opacity=0.6,
        hovertemplate="$%{y:,.0f}<extra>Daily Vol</extra>",
    ))

    # 7d moving average
    if len(dex_volume) >= 7:
        ma_7 = dex_volume.rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=ma_7.index,
            y=ma_7.values,
            name="7d MA",
            mode="lines",
            line=dict(color=COLORS["success"], width=2),
            hovertemplate="$%{y:,.0f}<extra>7d MA</extra>",
        ))

    # 30d moving average
    if len(dex_volume) >= 30:
        ma_30 = dex_volume.rolling(30).mean()
        fig.add_trace(go.Scatter(
            x=ma_30.index,
            y=ma_30.values,
            name="30d MA",
            mode="lines",
            line=dict(color=COLORS["warning"], width=2, dash="dash"),
            hovertemplate="$%{y:,.0f}<extra>30d MA</extra>",
        ))

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="DEX Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume (USD)",
        height=400,
        legend=dict(orientation="h", y=-0.15),
        bargap=0.1,
    )

    return fig
