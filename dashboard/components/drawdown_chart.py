"""Drawdown area chart component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import COLORS, FIGURE_LAYOUT, STRATEGY_COLORS, hex_to_rgba


def create_drawdown_chart(drawdowns: dict[str, pd.Series]) -> go.Figure:
    """Create drawdown chart with filled area below zero.

    Args:
        drawdowns: {strategy_name: pd.Series of drawdown values (negative)}.
            If a single series is passed, renders a filled red area with
            a horizontal line at the worst drawdown.

    Returns:
        go.Figure with filled area traces.
    """
    fig = go.Figure()

    for name, dd in drawdowns.items():
        color_key = name.lower().replace(" ", "_").replace("-", "_")
        color = STRATEGY_COLORS.get(color_key, COLORS["danger"])

        is_single = len(drawdowns) == 1

        fig.add_trace(go.Scatter(
            x=dd.index,
            y=dd.values * 100,  # Convert to percentage
            mode="lines",
            name=name,
            line=dict(color=color if not is_single else COLORS["danger"], width=1.5),
            fill="tozeroy" if is_single else None,
            fillcolor=hex_to_rgba(COLORS["danger"], 0.3) if is_single else None,
            hovertemplate="%{y:.2f}%<extra>" + name + "</extra>",
        ))

        # Add worst drawdown horizontal line for single strategy view
        if is_single:
            worst_dd = dd.min() * 100
            fig.add_hline(
                y=worst_dd,
                line_dash="dash",
                line_color=COLORS["warning"],
                line_width=1,
                annotation_text=f"Max DD: {worst_dd:.2f}%",
                annotation_position="bottom left",
                annotation_font_color=COLORS["warning"],
            )

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )

    return fig
