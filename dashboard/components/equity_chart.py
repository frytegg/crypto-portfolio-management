"""Equity curve line chart component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_equity_chart(equity_curves: dict[str, pd.Series]) -> go.Figure:
    """Create equity curve chart. Keys = strategy names, values = equity series."""
    raise NotImplementedError
