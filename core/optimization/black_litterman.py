"""Black-Litterman with on-chain signal views.

CRITICAL: blacklitterman_stats() MUST be called BEFORE optimization(model='BL').
Calling optimization(model='BL') without blacklitterman_stats() gives garbage -- no error raised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from core.data.onchain import OnchainSignals
from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def generate_onchain_views(
    signals: OnchainSignals,
    asset_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert on-chain signals to Black-Litterman P (picking matrix) and Q (view vector)."""
    raise NotImplementedError


def optimize_black_litterman(
    returns: pd.DataFrame,
    signals: OnchainSignals,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """BL optimization with on-chain views."""
    raise NotImplementedError
