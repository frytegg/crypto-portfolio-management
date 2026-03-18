"""GARCH volatility forecast chart component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_garch_chart(
    conditional_vol: pd.Series,
    forecast_vol: float | None = None,
    asset_name: str = "",
) -> go.Figure:
    """Create GARCH conditional volatility + forecast chart."""
    raise NotImplementedError
