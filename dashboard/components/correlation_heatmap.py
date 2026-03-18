"""Correlation matrix heatmap component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap from returns DataFrame."""
    raise NotImplementedError
