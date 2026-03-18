"""Markowitz Mean-Variance Optimization + efficient frontier.

Uses riskfolio-lib Portfolio class.
CRITICAL: efficient_frontier() returns weights DataFrame -- compute risk/return
manually for Plotly (see WARNING 5 in critical-warnings.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def optimize_markowitz(
    returns: pd.DataFrame,
    objective: str = "Sharpe",
    risk_measure: str = "MV",
    risk_free_rate: float = 0.0,
    method_cov: str = "ledoit",
    method_mu: str = "hist",
    max_weight: float = 0.15,
    min_weight: float = 0.0,
) -> PortfolioResult:
    """Run Markowitz MVO with configurable objective and risk measure.

    Args:
        returns: T x N DataFrame of daily log returns.
        objective: "Sharpe" | "MinRisk" | "MaxRet" | "Utility".
        risk_measure: "MV" (variance) | "CVaR" | "CDaR".
        risk_free_rate: Annual risk-free rate.
        method_cov: Covariance estimator for riskfolio-lib ("ledoit", "hist", etc.).
        method_mu: Expected returns estimator ("hist", "ewma1", "JS").
        max_weight: Upper bound per asset weight.
        min_weight: Lower bound per asset weight.

    Returns:
        PortfolioResult with optimized weights.

    Raises:
        ValueError: If the solver fails to find a solution.
    """
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.upperlng = max_weight
    port.lowerlng = min_weight

    w = port.optimization(
        model="Classic",
        rm=risk_measure,
        obj=objective,
        rf=risk_free_rate,
        l=2,
    )

    if w is None or w.isnull().all().all():
        raise ValueError(
            "Markowitz optimization failed — solver returned None. "
            f"objective={objective}, risk_measure={risk_measure}"
        )

    # w is DataFrame with one column "weights", rows = assets
    weights = w["weights"]
    weights.name = "weights"

    ann_ret = float((weights.values @ port.mu.values.flatten())) * 365
    ann_vol = float(np.sqrt(weights.values @ port.cov.values @ weights.values)) * np.sqrt(365)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    log.info(
        "markowitz_optimized",
        objective=objective,
        risk_measure=risk_measure,
        ann_return=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 4),
        n_assets=int((weights > 1e-6).sum()),
    )

    return PortfolioResult(
        name="Markowitz MVO",
        weights=weights,
        expected_return=ann_ret,
        expected_volatility=ann_vol,
        sharpe_ratio=sharpe,
        metadata={
            "objective": objective,
            "risk_measure": risk_measure,
            "method_cov": method_cov,
        },
    )


def compute_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    risk_measure: str = "MV",
    method_cov: str = "ledoit",
    method_mu: str = "hist",
    max_weight: float = 0.15,
) -> dict:
    """Compute efficient frontier coordinates for Plotly.

    IMPORTANT (WARNING 5): frontier is a DataFrame of weights
    (N_assets rows x n_points columns). Risk/return coordinates must be
    computed manually per column.

    Args:
        returns: T x N DataFrame of daily log returns.
        n_points: Number of frontier points.
        risk_measure: Risk measure for optimization ("MV", "CVaR", "CDaR").
        method_cov: Covariance estimator.
        method_mu: Expected returns estimator.
        max_weight: Upper bound per asset weight.

    Returns:
        dict with keys:
            "frontier_returns": list of annualized returns per frontier point
            "frontier_risks": list of annualized vols per frontier point
            "frontier_weights": pd.DataFrame (assets x points)
            "max_sharpe_weights": pd.Series — weights at max Sharpe point
            "min_vol_weights": pd.Series — weights at min vol point
            "asset_returns": pd.Series — individual asset expected returns
            "asset_risks": pd.Series — individual asset vols
    """
    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    port.upperlng = max_weight

    frontier = port.efficient_frontier(
        model="Classic", rm=risk_measure, points=n_points, rf=0
    )

    if frontier is None or frontier.empty:
        raise ValueError("Efficient frontier computation failed — solver returned None.")

    mu = port.mu.values.flatten()  # (N,)
    cov = port.cov.values          # (N, N)

    # Compute risk/return for each frontier point
    frontier_returns: list[float] = []
    frontier_risks: list[float] = []

    for col in frontier.columns:
        w = frontier[col].values  # (N,)
        ret_i = float(w @ mu) * 365
        risk_i = float(np.sqrt(w @ cov @ w)) * np.sqrt(365)
        frontier_returns.append(ret_i)
        frontier_risks.append(risk_i)

    # Find max Sharpe and min vol points
    sharpe_ratios = [
        r / v if v > 0 else 0.0
        for r, v in zip(frontier_returns, frontier_risks)
    ]
    max_sharpe_idx = int(np.argmax(sharpe_ratios))
    min_vol_idx = int(np.argmin(frontier_risks))

    max_sharpe_weights = frontier.iloc[:, max_sharpe_idx]
    max_sharpe_weights.name = "weights"
    min_vol_weights = frontier.iloc[:, min_vol_idx]
    min_vol_weights.name = "weights"

    # Individual asset expected returns and vols
    asset_returns = pd.Series(mu * 365, index=returns.columns, name="expected_return")
    asset_risks = pd.Series(
        np.sqrt(np.diag(cov)) * np.sqrt(365),
        index=returns.columns,
        name="volatility",
    )

    log.info(
        "efficient_frontier_computed",
        n_points=len(frontier_returns),
        min_vol=round(min(frontier_risks), 4),
        max_ret=round(max(frontier_returns), 4),
        max_sharpe=round(max(sharpe_ratios), 4),
    )

    return {
        "frontier_returns": frontier_returns,
        "frontier_risks": frontier_risks,
        "frontier_weights": frontier,
        "max_sharpe_weights": max_sharpe_weights,
        "min_vol_weights": min_vol_weights,
        "asset_returns": asset_returns,
        "asset_risks": asset_risks,
    }


def optimize_garch_gmv(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Global Minimum Variance using GARCH-forecasted covariance.

    Fits GJR-GARCH(1,1,1) to each asset, builds a DCC-like covariance,
    and runs minimum-variance optimization on it.

    Args:
        returns: T x N DataFrame of daily log returns.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        max_weight: Upper bound per asset weight.

    Returns:
        PortfolioResult with GARCH-GMV weights.

    Raises:
        ValueError: If the solver fails.
    """
    from core.models.garch import build_garch_covariance
    from core.models.returns import estimate_returns

    garch_cov = build_garch_covariance(returns)
    mu = estimate_returns(returns, method="historical")

    port = rp.Portfolio(returns=returns)
    # Override stats with GARCH-based estimates
    port.mu = mu.to_frame().T  # riskfolio expects (1 x N) DataFrame
    port.cov = garch_cov
    port.upperlng = max_weight

    w = port.optimization(model="Classic", rm="MV", obj="MinRisk", rf=0)

    if w is None or w.isnull().all().all():
        raise ValueError("GARCH-GMV optimization failed — solver returned None.")

    weights = w["weights"]
    weights.name = "weights"

    ann_ret = float(weights.values @ mu.values) * 365
    ann_vol = float(np.sqrt(weights.values @ garch_cov.values @ weights.values)) * np.sqrt(365)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    log.info(
        "garch_gmv_optimized",
        ann_return=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 4),
        n_assets=int((weights > 1e-6).sum()),
    )

    return PortfolioResult(
        name="GARCH-Enhanced GMV",
        weights=weights,
        expected_return=ann_ret,
        expected_volatility=ann_vol,
        sharpe_ratio=sharpe,
        metadata={"strategy": "GARCH-GMV", "uses_garch_cov": True},
    )
