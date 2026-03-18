"""Markowitz Mean-Variance Optimization + efficient frontier.

Uses riskfolio-lib Portfolio class with rm='MV', obj='Sharpe'.
CRITICAL: efficient_frontier() returns weights DataFrame -- compute risk/return manually for Plotly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_markowitz(
    returns: pd.DataFrame,
    cov_method: str = "ledoit",
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Run MVO with rm='MV', obj='Sharpe'."""
    raise NotImplementedError


def optimize_garch_gmv(
    returns: pd.DataFrame,
    garch_cov: np.ndarray,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """GMV optimization using GARCH-forecasted covariance matrix."""
    raise NotImplementedError


def compute_efficient_frontier(
    returns: pd.DataFrame,
    cov_method: str = "ledoit",
    n_points: int = 50,
    max_weight: float = 0.15,
) -> pd.DataFrame:
    """Compute efficient frontier coordinates for Plotly.

    Returns DataFrame with columns: 'volatility', 'return', 'sharpe',
    plus one column per asset with weights at each frontier point.
    """
    raise NotImplementedError
