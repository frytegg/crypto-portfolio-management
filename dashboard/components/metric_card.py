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
    """Create a styled metric card."""
    raise NotImplementedError
