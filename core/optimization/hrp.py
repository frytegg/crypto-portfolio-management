"""Hierarchical Risk Parity via riskfolio-lib HCPortfolio.

CRITICAL: Always set solvers=['CLARABEL', 'SCS'] as fallback chain (WARNING 7).
NOTE: HCPortfolio does NOT have upperlng — max_weight is advisory only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp
import structlog
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_hrp(
    returns: pd.DataFrame,
    codependence: str = "pearson",
    covariance: str = "ledoit",
    linkage: str = "ward",
    risk_measure: str = "MV",
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Hierarchical Risk Parity optimization.

    HCPortfolio does NOT support weight constraints (no upperlng).
    If any weight exceeds max_weight * 1.5, a warning is logged.

    Args:
        returns: T x N DataFrame of daily log returns.
        codependence: Codependence measure ("pearson", "spearman", "kendall").
        covariance: Covariance estimator ("ledoit", "hist", "oas", etc.).
        linkage: Hierarchical clustering linkage method ("ward", "single",
                 "complete", "average").
        risk_measure: Risk measure for allocation ("MV", "CVaR", "CDaR").
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        max_weight: Advisory max weight — triggers warning if exceeded by 1.5x.

    Returns:
        PortfolioResult with HRP-allocated weights.

    Raises:
        ValueError: If the solver fails.
    """
    port = rp.HCPortfolio(returns=returns)
    w = port.optimization(
        model="HRP",
        codependence=codependence,
        method_cov=covariance,
        rm=risk_measure,
        linkage=linkage,
        max_k=10,
        leaf_order=True,
    )

    if w is None or w.isnull().all().all():
        raise ValueError("HRP optimization failed — solver returned None.")

    weights = w["weights"]
    weights.name = "weights"

    # Advisory max_weight check
    exceeded = weights[weights > max_weight * 1.5]
    if not exceeded.empty:
        log.warning(
            "hrp_weight_concentration",
            exceeded_assets=dict(exceeded.round(4)),
            advisory_limit=max_weight,
        )

    # Compute portfolio stats
    mu = returns.mean().values
    cov = returns.cov().values
    w_arr = weights.values

    ann_ret = float(w_arr @ mu) * 365
    ann_vol = float(np.sqrt(w_arr @ cov @ w_arr)) * np.sqrt(365)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    log.info(
        "hrp_optimized",
        codependence=codependence,
        linkage=linkage,
        ann_return=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 4),
        n_nonzero=int((weights > 1e-6).sum()),
    )

    return PortfolioResult(
        name="Hierarchical Risk Parity",
        weights=weights,
        expected_return=ann_ret,
        expected_volatility=ann_vol,
        sharpe_ratio=sharpe,
        metadata={
            "codependence": codependence,
            "covariance": covariance,
            "linkage": linkage,
            "risk_measure": risk_measure,
        },
    )


def get_hrp_dendrogram_data(
    returns: pd.DataFrame,
    codependence: str = "pearson",
    linkage: str = "ward",
) -> dict:
    """Extract linkage matrix for custom Plotly dendrogram rendering.

    Computes a correlation-based distance matrix, then applies hierarchical
    clustering to produce a scipy-compatible linkage matrix.

    Args:
        returns: T x N DataFrame of daily log returns.
        codependence: Codependence measure ("pearson", "spearman", "kendall").
        linkage: Clustering linkage method ("ward", "single", "complete", "average").

    Returns:
        dict with keys:
            "linkage_matrix": np.ndarray — scipy linkage matrix (shape (N-1, 4))
            "asset_names": list[str] — asset names in column order
    """
    if codependence == "pearson":
        corr = returns.corr(method="pearson").values
    elif codependence == "spearman":
        corr = returns.corr(method="spearman").values
    elif codependence == "kendall":
        corr = returns.corr(method="kendall").values
    else:
        corr = returns.corr(method="pearson").values

    # Correlation distance: d = sqrt((1 - rho) / 2), range [0, 1]
    dist = np.sqrt((1 - np.clip(corr, -1, 1)) / 2)
    np.fill_diagonal(dist, 0.0)

    dist_condensed = squareform(dist, checks=False)
    Z = scipy_linkage(dist_condensed, method=linkage)

    return {
        "linkage_matrix": Z,
        "asset_names": list(returns.columns),
    }
