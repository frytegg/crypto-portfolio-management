"""Mean-CVaR optimization via riskfolio-lib.

Uses rm='CVaR' with configurable alpha (confidence level).
Default alpha=0.05 corresponds to 95% CVaR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_cvar(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    objective: str = "MinRisk",
    method_cov: str = "ledoit",
    method_mu: str = "hist",
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Mean-CVaR optimization.

    Args:
        returns: T x N DataFrame of daily log returns.
        alpha: Significance level for CVaR (0.05 = 95% CVaR).
        objective: "MinRisk" | "Sharpe" | "MaxRet" | "Utility".
        method_cov: Covariance estimator for riskfolio-lib.
        method_mu: Expected returns estimator.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        max_weight: Upper bound per asset weight.

    Returns:
        PortfolioResult with CVaR-optimized weights.

    Raises:
        ValueError: If the solver fails.
    """
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.alpha = alpha
    port.upperlng = max_weight

    w = port.optimization(model="Classic", rm="CVaR", obj=objective, rf=0)

    if w is None or w.isnull().all().all():
        raise ValueError(
            f"CVaR optimization failed — solver returned None. "
            f"alpha={alpha}, objective={objective}"
        )

    weights = w["weights"]
    weights.name = "weights"

    # Compute portfolio stats using sample moments
    mu = port.mu.values.flatten()
    cov = port.cov.values
    w_arr = weights.values

    ann_ret = float(w_arr @ mu) * 365
    ann_vol = float(np.sqrt(w_arr @ cov @ w_arr)) * np.sqrt(365)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    log.info(
        "cvar_optimized",
        alpha=alpha,
        objective=objective,
        ann_return=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 4),
        n_nonzero=int((weights > 1e-6).sum()),
    )

    return PortfolioResult(
        name="Mean-CVaR",
        weights=weights,
        expected_return=ann_ret,
        expected_volatility=ann_vol,
        sharpe_ratio=sharpe,
        metadata={
            "alpha": alpha,
            "objective": objective,
        },
    )
