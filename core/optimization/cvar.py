"""Mean-CVaR optimization via riskfolio-lib.

Uses rm='CVaR', alpha=0.05 (95% confidence level).
"""
from __future__ import annotations

import pandas as pd
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_cvar(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Mean-CVaR optimization."""
    raise NotImplementedError
