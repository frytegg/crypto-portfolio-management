"""Drawdown area chart component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_drawdown_chart(drawdowns: dict[str, pd.Series]) -> go.Figure:
    """Create drawdown chart. Keys = strategy names, values = drawdown series."""
    raise NotImplementedError
