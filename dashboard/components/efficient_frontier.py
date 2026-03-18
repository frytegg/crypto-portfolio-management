"""Efficient frontier scatter plot component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_efficient_frontier_figure(
    frontier_data: pd.DataFrame,
    portfolio_points: list[dict] | None = None,
) -> go.Figure:
    """Create efficient frontier plot with optional portfolio markers."""
    raise NotImplementedError
