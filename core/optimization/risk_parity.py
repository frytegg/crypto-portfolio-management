"""Equal Risk Contribution (Risk Parity) via riskfolio-lib.

CRITICAL (WARNING 6): rp_optimization() uses log-barrier formulation and can
fail on highly correlated assets (common in crypto bear markets).
Always fall back to rrp_optimization().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_risk_parity(
    returns: pd.DataFrame,
    risk_measure: str = "MV",
    method_cov: str = "ledoit",
    risk_free_rate: float = 0.0,
    risk_budget: list[float] | None = None,
) -> PortfolioResult:
    """Equal Risk Contribution optimization with automatic fallback.

    Tries rp_optimization() first. On failure (common with highly correlated
    crypto assets), falls back to rrp_optimization() (relaxed risk parity).

    Args:
        returns: T x N DataFrame of daily log returns.
        risk_measure: Risk measure ("MV", "CVaR", "CDaR").
        method_cov: Covariance estimator for riskfolio-lib.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        risk_budget: Optional per-asset risk budget. If None, uses equal
                     budget (1/N). Values are normalized to sum to 1.

    Returns:
        PortfolioResult with risk parity weights.

    Raises:
        ValueError: If both rp and rrp optimization fail.
    """
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov=method_cov)

    b: np.ndarray | None = None
    if risk_budget is not None:
        b_arr = np.array(risk_budget, dtype=float)
        b_arr = b_arr / b_arr.sum()  # normalize
        b = b_arr.reshape(-1, 1)  # riskfolio expects (N, 1) array

    method_used = "rp"
    try:
        w = port.rp_optimization(model="Classic", rm=risk_measure, rf=0, b=b)
        if w is None or w.isnull().all().all():
            raise ValueError("rp_optimization returned None")
    except Exception as e:
        log.warning(
            "rp_optimization_failed_fallback_to_rrp",
            error=str(e),
            risk_measure=risk_measure,
        )
        method_used = "rrp (fallback)"
        w = port.rrp_optimization(model="Classic", version="A", l=0, b=b)
        if w is None or w.isnull().all().all():
            raise ValueError(
                "Risk parity optimization failed — both rp and rrp solvers returned None."
            )

    weights = w["weights"]
    weights.name = "weights"

    # Compute portfolio stats
    mu = port.mu.values.flatten()
    cov = port.cov.values
    w_arr = weights.values

    ann_ret = float(w_arr @ mu) * 365
    ann_vol = float(np.sqrt(w_arr @ cov @ w_arr)) * np.sqrt(365)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    log.info(
        "risk_parity_optimized",
        method=method_used,
        risk_measure=risk_measure,
        ann_return=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 4),
        n_nonzero=int((weights > 1e-6).sum()),
    )

    return PortfolioResult(
        name="Equal Risk Contribution",
        weights=weights,
        expected_return=ann_ret,
        expected_volatility=ann_vol,
        sharpe_ratio=sharpe,
        metadata={
            "method": method_used,
            "risk_measure": risk_measure,
            "method_cov": method_cov,
        },
    )
