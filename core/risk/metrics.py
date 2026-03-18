"""Portfolio risk and performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def compute_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute all risk metrics for a returns series.

    Returns dict with keys:
        'annual_return', 'annual_volatility', 'sharpe_ratio', 'sortino_ratio',
        'calmar_ratio', 'omega_ratio', 'max_drawdown', 'max_drawdown_duration',
        'var_95', 'cvar_95', 'skewness', 'kurtosis', 'win_rate',
        'best_day', 'worst_day'
    """
    raise NotImplementedError


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown time series from returns."""
    raise NotImplementedError
