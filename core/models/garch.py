"""GARCH/GJR-GARCH volatility forecasting via arch library.

CRITICAL: arch_model(y, vol='GARCH', p=1, o=1, q=1) -- NO vol='GJR-GARCH' string.
CRITICAL: Use returns * 100 for numerical stability. Divide output by 100.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    horizon: int = 10,
) -> dict[str, Any]:
    """Fit GJR-GARCH(p,o,q) to a single asset's returns.

    Returns dict with keys:
        'conditional_volatility': pd.Series (annualized, decimal)
        'forecast_volatility': float (annualized, decimal, h-step ahead)
        'params': dict (omega, alpha, gamma, beta)
        'aic': float
        'bic': float
        'converged': bool
        'error': str | None
    """
    raise NotImplementedError


def fit_all_garch(
    returns: pd.DataFrame,
    horizon: int = 10,
) -> dict[str, dict[str, Any]]:
    """Fit GJR-GARCH to all assets. Returns {asset_name: garch_result}."""
    raise NotImplementedError


def build_garch_covariance(
    returns: pd.DataFrame,
    garch_results: dict[str, dict[str, Any]],
    base_cov: np.ndarray,
) -> np.ndarray:
    """Replace diagonal of base_cov with GARCH-forecasted variances."""
    raise NotImplementedError
