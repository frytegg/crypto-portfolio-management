"""Regime detection overlay chart component."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_regime_chart(
    prices: pd.Series,
    regime_history: pd.Series,
) -> go.Figure:
    """Price chart with regime-colored background bands."""
    raise NotImplementedError
