"""On-chain signal charts — TVL, stablecoin, DEX volume."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def create_tvl_chart(tvl_data: pd.DataFrame) -> go.Figure:
    """Total Value Locked over time."""
    raise NotImplementedError


def create_stablecoin_chart(stablecoin_data: pd.DataFrame) -> go.Figure:
    """Stablecoin market cap and dominance."""
    raise NotImplementedError


def create_dex_chart(dex_data: pd.DataFrame) -> go.Figure:
    """DEX volume trend chart."""
    raise NotImplementedError
