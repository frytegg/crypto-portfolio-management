"""KPI metric card component."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def create_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    color: str = "primary",
) -> dbc.Card:
    """Create a styled metric card for KPI display.

    Args:
        title: Card subtitle/label (e.g. "Total Market Cap").
        value: Main displayed value (e.g. "$1.2T").
        subtitle: Optional secondary text (e.g. "+2.3% 24h").
        color: Bootstrap color name for left border accent.
    """
    body_children = [
        html.H6(title, className="card-subtitle text-muted"),
        html.H3(value, className="card-title"),
    ]
    if subtitle:
        body_children.append(
            html.P(subtitle, className="card-text text-muted small")
        )

    return dbc.Card(
        dbc.CardBody(body_children),
        className="mb-3 shadow-sm",
        style={"borderLeft": f"4px solid var(--bs-{color})"},
    )
