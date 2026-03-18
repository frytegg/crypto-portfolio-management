"""Equal Risk Contribution (Risk Parity) via riskfolio-lib.

CRITICAL: rp_optimization() can fail on highly correlated assets.
Always fall back to rrp_optimization().
"""
from __future__ import annotations

import pandas as pd
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_risk_parity(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """ERC optimization with rrp_optimization() fallback."""
    raise NotImplementedError
