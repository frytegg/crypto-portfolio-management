"""Regime detection overlay chart component."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import COLORS, FIGURE_LAYOUT

# Regime background colors (semi-transparent)
REGIME_COLORS: dict[str, str] = {
    "Bull": "rgba(0,200,83,0.15)",
    "Bear": "rgba(255,82,82,0.15)",
    "Sideways": "rgba(255,193,7,0.15)",
}

# Solid colors for legend annotations
REGIME_LEGEND_COLORS: dict[str, str] = {
    "Bull": COLORS["success"],
    "Bear": COLORS["danger"],
    "Sideways": COLORS["warning"],
}


def create_regime_chart(
    prices: pd.Series,
    regime_data: dict,
) -> go.Figure:
    """Price chart with regime-colored background rectangles.

    Draws the price as a line, then overlays shaded rectangles for each
    contiguous regime period. Adds a legend annotation for each regime color.

    Args:
        prices: Price series (e.g. BTC prices) with DatetimeIndex.
        regime_data: Output dict from detect_regimes() containing
            "regimes" (pd.Series of state ints) and "regime_names" (dict).

    Returns:
        Plotly Figure with price line and regime shading.
    """
    regimes = regime_data["regimes"]
    name_map = regime_data["regime_names"]

    # Align prices to regime index
    aligned_prices = prices.reindex(regimes.index)

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=aligned_prices.index,
            y=aligned_prices.values,
            name="Price",
            line=dict(color=COLORS["info"], width=2),
            mode="lines",
        )
    )

    # Build regime rectangles for contiguous periods
    shapes = _build_regime_shapes(regimes, name_map, aligned_prices)
    fig.update_layout(shapes=shapes)

    # Add invisible traces for legend entries (one per regime)
    seen_regimes: set[str] = set()
    for state, name in name_map.items():
        if name not in seen_regimes:
            seen_regimes.add(name)
            color = REGIME_LEGEND_COLORS.get(name, COLORS["text_muted"])
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=color, symbol="square"),
                    name=f"{name} Regime",
                    showlegend=True,
                )
            )

    fig.update_layout(
        **FIGURE_LAYOUT,
        title="Market Regime Detection (BTC)",
        height=500,
        yaxis=dict(
            title="Price (USD)",
            gridcolor=COLORS["grid"],
            zerolinecolor=COLORS["grid"],
        ),
        legend=dict(x=0, y=1.12, orientation="h"),
    )

    return fig


def _build_regime_shapes(
    regimes: pd.Series,
    name_map: dict[int, str],
    prices: pd.Series,
) -> list[dict]:
    """Build layout.Shape rectangles for each contiguous regime period.

    Returns a list of shape dicts for fig.update_layout(shapes=...).
    """
    shapes: list[dict] = []
    if len(regimes) == 0:
        return shapes

    dates = regimes.index
    states = regimes.values

    # Walk through and identify contiguous blocks
    block_start = 0
    for i in range(1, len(states)):
        if states[i] != states[block_start]:
            # End of block — emit shape
            _add_shape(shapes, dates, states[block_start], block_start, i - 1, name_map, prices)
            block_start = i

    # Final block
    _add_shape(shapes, dates, states[block_start], block_start, len(states) - 1, name_map, prices)

    return shapes


def _add_shape(
    shapes: list[dict],
    dates: pd.DatetimeIndex,
    state: int,
    start_idx: int,
    end_idx: int,
    name_map: dict[int, str],
    prices: pd.Series,
) -> None:
    """Append a regime rectangle shape to the shapes list."""
    regime_name = name_map.get(int(state), f"Regime_{state}")
    fill_color = REGIME_COLORS.get(regime_name, "rgba(128,128,128,0.1)")

    shapes.append(
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=dates[start_idx],
            x1=dates[end_idx],
            y0=0,
            y1=1,
            fillcolor=fill_color,
            line=dict(width=0),
            layer="below",
        )
    )
