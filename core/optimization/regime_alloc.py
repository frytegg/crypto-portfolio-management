"""Regime-aware allocation — switches strategy based on HMM state.

Bull  → Max Sharpe (Markowitz)
Bear  → Min Volatility (Markowitz)
Sideways → Risk Parity
"""
from __future__ import annotations

import structlog
import pandas as pd

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)

REGIME_STRATEGY_MAP: dict[str, str] = {
    "Bull": "Max Sharpe (Markowitz)",
    "Bear": "Min Volatility (Markowitz)",
    "Sideways": "Risk Parity",
}


def optimize_regime_aware(
    returns: pd.DataFrame,
    regime_info: dict,
    method_cov: str = "ledoit",
    risk_free_rate: float = 0.0,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Select and run optimization strategy based on current HMM regime.

    Args:
        returns: T x N DataFrame of daily log returns.
        regime_info: Output of detect_regimes() — dict with keys
            "current_regime_name", "regime_means", "transition_matrix", etc.
        method_cov: Covariance estimator forwarded to sub-strategies.
        risk_free_rate: Annual risk-free rate.
        max_weight: Upper bound per asset weight.

    Returns:
        PortfolioResult with regime-adapted weights and metadata.
    """
    current_regime = regime_info["current_regime_name"]

    if current_regime == "Bull":
        from core.optimization.markowitz import optimize_markowitz

        result = optimize_markowitz(
            returns,
            objective="Sharpe",
            max_weight=max_weight,
            method_cov=method_cov,
            risk_free_rate=risk_free_rate,
        )
        strategy_used = "Max Sharpe (Markowitz)"

    elif current_regime == "Bear":
        from core.optimization.markowitz import optimize_markowitz

        result = optimize_markowitz(
            returns,
            objective="MinRisk",
            max_weight=max_weight,
            method_cov=method_cov,
            risk_free_rate=risk_free_rate,
        )
        strategy_used = "Min Volatility (Markowitz)"

    else:  # Sideways or unknown
        from core.optimization.risk_parity import optimize_risk_parity

        result = optimize_risk_parity(
            returns,
            method_cov=method_cov,
            risk_free_rate=risk_free_rate,
        )
        strategy_used = "Risk Parity"

    result.name = f"Regime-Aware ({current_regime})"
    result.metadata.update({
        "current_regime": current_regime,
        "strategy_used": strategy_used,
        "regime_means": regime_info["regime_means"].tolist(),
        "transition_matrix": regime_info["transition_matrix"].tolist(),
    })

    log.info(
        "regime_aware_optimized",
        current_regime=current_regime,
        strategy_used=strategy_used,
        ann_return=round(result.expected_return, 4),
        ann_vol=round(result.expected_volatility, 4),
    )

    return result
