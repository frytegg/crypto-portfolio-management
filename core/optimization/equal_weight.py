"""Equal Weight (1/N) benchmark allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.optimization._base import PortfolioResult


def optimize_equal_weight(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """Compute 1/N equal weight portfolio.

    Args:
        returns: T x N DataFrame of daily log returns.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        PortfolioResult with equal weights across all assets.
    """
    n = len(returns.columns)
    weights = pd.Series(1.0 / n, index=returns.columns, name="weights")

    port_returns = (returns * weights).sum(axis=1)
    ann_ret = float(port_returns.mean() * 365)
    ann_vol = float(port_returns.std() * np.sqrt(365))
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    return PortfolioResult(
        name="Equal Weight",
        weights=weights,
        expected_return=ann_ret,
        expected_volatility=ann_vol,
        sharpe_ratio=sharpe,
        metadata={"n_assets": n},
    )
