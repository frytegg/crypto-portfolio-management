"""Hierarchical Risk Parity via riskfolio-lib HCPortfolio.

CRITICAL: Always set solvers=['CLARABEL', 'SCS'] as fallback chain.
"""
from __future__ import annotations

import pandas as pd
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_hrp(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """HRP with codependence='pearson', linkage='ward'."""
    raise NotImplementedError


def get_dendrogram_data(returns: pd.DataFrame) -> dict:
    """Extract linkage matrix and labels for dendrogram visualization."""
    raise NotImplementedError
