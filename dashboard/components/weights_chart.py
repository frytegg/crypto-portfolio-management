"""Allocation weights visualization — bar and pie charts."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_weights_bar(weights: dict[str, pd.Series]) -> go.Figure:
    """Stacked bar chart of portfolio weights across strategies."""
    raise NotImplementedError


def create_weights_pie(weights: pd.Series, title: str = "") -> go.Figure:
    """Pie chart for a single strategy's weights."""
    raise NotImplementedError
