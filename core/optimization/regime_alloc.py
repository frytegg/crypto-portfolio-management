"""Regime-aware allocation -- switches strategy based on HMM state."""
from __future__ import annotations

import pandas as pd
import structlog

from core.models.regime import RegimeResult
from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)

REGIME_STRATEGY_MAP: dict[str, str] = {
    "Bull": "markowitz",
    "Bear": "garch_gmv",
    "Sideways": "risk_parity",
}


def optimize_regime_aware(
    returns: pd.DataFrame,
    regime_result: RegimeResult,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Select and run strategy based on current regime."""
    raise NotImplementedError
