"""Expected returns estimation methods via riskfolio-lib.

Supported methods (mapped to riskfolio-lib's method_mu parameter):
    "historical"   → method_mu="hist"   — Historical mean (annualized × 365)
    "ewma"         → method_mu="ewma1"  — Exponentially weighted moving average
    "james_stein"  → method_mu="JS"     — James-Stein shrinkage estimator
"""
from __future__ import annotations

import pandas as pd
import riskfolio as rp
import structlog

log = structlog.get_logger(__name__)

_METHOD_MAP: dict[str, str] = {
    "historical": "hist",
    "ewma": "ewma1",
    "james_stein": "JS",
}

VALID_METHODS = list(_METHOD_MAP.keys())


def estimate_returns(
    returns: pd.DataFrame,
    method: str = "historical",
) -> pd.Series:
    """Estimate expected returns per asset (annualized).

    Uses riskfolio-lib's Portfolio.assets_stats() to compute expected returns
    with the specified estimator, then reads the internal mu vector.

    All returns are assumed to be daily log returns — annualized by the
    factor riskfolio-lib applies internally (252 trading days by default).

    Args:
        returns: T x N DataFrame of daily log returns.
        method: Estimation method. One of "historical", "ewma", "james_stein".

    Returns:
        pd.Series indexed by asset name with annualized expected returns.

    Raises:
        ValueError: If method is invalid.
    """
    if method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown return estimation method '{method}'. "
            f"Valid methods: {VALID_METHODS}"
        )

    rp_method = _METHOD_MAP[method]

    log.info(
        "estimating_returns",
        method=method,
        rp_method=rp_method,
        n_assets=returns.shape[1],
    )

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu=rp_method, method_cov="hist")

    # port.mu is a DataFrame with shape (1, N) — flatten to Series
    mu: pd.Series = port.mu.iloc[0]
    mu.name = "expected_return"

    log.info(
        "returns_estimated",
        method=method,
        min_return=round(float(mu.min()), 6),
        max_return=round(float(mu.max()), 6),
    )

    return mu
