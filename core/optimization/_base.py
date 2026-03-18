"""Shared data structures for all optimization strategies."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PortfolioResult:
    """Standardized output from every optimization strategy."""

    name: str
    weights: pd.Series  # index=asset_names, values sum to 1.0
    expected_return: float  # annualized
    expected_volatility: float  # annualized
    sharpe_ratio: float
    metadata: dict = field(default_factory=dict)


STRATEGY_NAMES: dict[str, str] = {
    "equal_weight": "Equal Weight (1/N)",
    "markowitz": "Markowitz MVO",
    "garch_gmv": "GARCH-Enhanced GMV",
    "hrp": "Hierarchical Risk Parity",
    "risk_parity": "Equal Risk Contribution",
    "cvar": "Mean-CVaR",
    "black_litterman": "Black-Litterman",
    "regime_aware": "Regime-Aware",
}
