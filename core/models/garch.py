"""GARCH/GJR-GARCH volatility forecasting via arch library.

CRITICAL: arch_model(y, vol='GARCH', p=1, o=1, q=1) — NO vol='GJR-GARCH' string.
The o=1 parameter is what makes it GJR-GARCH (asymmetric leverage term).

CRITICAL: If rescale=True, input returns are multiplied by 100 before fitting
for numerical stability. conditional_volatility is divided by 100 on output,
forecast_variance is divided by 100^2 on output.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from arch import arch_model

log = structlog.get_logger(__name__)


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "StudentsT",
    rescale: bool = True,
) -> dict[str, Any]:
    """Fit a GJR-GARCH(p,o,q) model to a single asset's return series.

    Uses arch_model with vol='GARCH' and the o parameter to control
    asymmetry. When o=1 this is GJR-GARCH; when o=0 it's standard GARCH.

    Args:
        returns: pd.Series of daily log returns (decimal, not percentage).
        p: GARCH lag order (volatility persistence).
        o: Asymmetric (GJR) lag order. 0 = standard GARCH, 1 = GJR-GARCH.
        q: ARCH lag order (shock impact).
        dist: Error distribution. One of "Normal", "StudentsT", "SkewStudent".
              "StudentsT" is strongly recommended for crypto (fat tails).
        rescale: If True, multiply returns by 100 before fitting (arch
                 convention for numerical stability). Output is rescaled back.

    Returns:
        dict with keys:
            "conditional_volatility": pd.Series — fitted daily conditional
                volatility in decimal scale (same index as input returns).
            "forecast_variance": float — 1-step-ahead variance forecast
                (rescaled back to decimal scale).
            "forecast_vol": float — sqrt of forecast_variance.
            "params": dict — fitted model parameters (omega, alpha, gamma, beta, etc.).
            "aic": float — Akaike Information Criterion.
            "bic": float — Bayesian Information Criterion.
            "model_result": arch result object (for advanced use).
    """
    scaled = returns * 100 if rescale else returns

    am = arch_model(
        scaled,
        vol="GARCH",
        p=p,
        o=o,
        q=q,
        dist=dist,
        mean="Constant",
        rescale=False,
    )
    result = am.fit(disp="off", show_warning=False)

    cond_vol = result.conditional_volatility
    if rescale:
        cond_vol = cond_vol / 100

    forecast = result.forecast(horizon=1, reindex=False)
    fcast_var = forecast.variance.iloc[-1, 0]
    if rescale:
        fcast_var = fcast_var / (100**2)

    log.info(
        "garch_fitted",
        asset=returns.name,
        p=p,
        o=o,
        q=q,
        dist=dist,
        aic=round(result.aic, 2),
        forecast_vol=round(np.sqrt(fcast_var), 6),
    )

    return {
        "conditional_volatility": cond_vol,
        "forecast_variance": fcast_var,
        "forecast_vol": np.sqrt(fcast_var),
        "params": dict(result.params),
        "aic": result.aic,
        "bic": result.bic,
        "model_result": result,
    }


def fit_all_garch(
    returns: pd.DataFrame,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "StudentsT",
    rescale: bool = True,
) -> dict[str, dict[str, Any]]:
    """Fit GJR-GARCH to every asset in the DataFrame.

    On convergence failure or any exception, falls back to sample volatility
    so that downstream consumers always get a complete result dict.

    Args:
        returns: T x N DataFrame of daily log returns.
        p, o, q, dist, rescale: forwarded to fit_garch().

    Returns:
        Dict keyed by column name → fit_garch result dict.
        Failed fits include an extra "error" key with the exception message.
    """
    results: dict[str, dict[str, Any]] = {}

    for col in returns.columns:
        try:
            results[col] = fit_garch(
                returns[col], p=p, o=o, q=q, dist=dist, rescale=rescale
            )
        except Exception as e:
            log.warning("garch_fit_failed", asset=col, error=str(e))
            vol = returns[col].std()
            results[col] = {
                "conditional_volatility": pd.Series(vol, index=returns.index),
                "forecast_vol": vol,
                "forecast_variance": vol**2,
                "params": {},
                "aic": np.nan,
                "bic": np.nan,
                "model_result": None,
                "error": str(e),
            }

    return results


def build_garch_covariance(
    returns: pd.DataFrame,
    garch_results: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Build a GARCH-implied covariance matrix using a DCC-like approach.

    Steps:
        1. Fit GJR-GARCH(1,1,1) to each asset → get forecasted vol sigma_i
           (or use pre-computed garch_results).
        2. Compute standardized residuals: resid[col] = returns[col] / cond_vol[col].
        3. Compute constant correlation matrix R from standardized residuals.
        4. D = diag(sigma). Covariance = D @ R @ D.

    This is a simplified DCC (Dynamic Conditional Correlation) approach that
    uses GARCH-forecasted volatilities with a constant correlation structure.

    Args:
        returns: T x N DataFrame of daily log returns.
        garch_results: Optional pre-computed results from fit_all_garch().
                       If None, fit_all_garch(returns) is called internally.

    Returns:
        N x N GARCH-implied covariance matrix as pd.DataFrame with asset
        names as both index and columns.
    """
    if garch_results is None:
        garch_results = fit_all_garch(returns)

    assets = list(returns.columns)
    n = len(assets)

    # Extract forecast_vol per asset → diagonal vector
    sigma = np.array([garch_results[col]["forecast_vol"] for col in assets])

    # Compute standardized residuals
    resid = pd.DataFrame(index=returns.index, columns=assets, dtype=float)
    for col in assets:
        cond_vol = garch_results[col]["conditional_volatility"]
        # Guard against division by zero (e.g. fallback constant vol)
        safe_vol = cond_vol.replace(0, np.nan).fillna(cond_vol.mean())
        if isinstance(safe_vol, float):
            # Fallback case: cond_vol is a scalar broadcast to Series
            safe_vol = pd.Series(safe_vol, index=returns.index)
        resid[col] = returns[col] / safe_vol

    # Drop rows with NaN/inf in residuals before computing correlation
    resid = resid.replace([np.inf, -np.inf], np.nan).dropna()

    # Constant correlation from standardized residuals
    if len(resid) < 2:
        # Not enough data for correlation — use identity
        log.warning("garch_cov_insufficient_residuals", n_valid=len(resid))
        corr = np.eye(n)
    else:
        corr = np.corrcoef(resid.values, rowvar=False)
        # Ensure perfectly symmetric (corrcoef can have tiny asymmetry)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

    # D @ R @ D
    D = np.diag(sigma)
    cov_matrix = D @ corr @ D

    log.info(
        "garch_covariance_built",
        n_assets=n,
        min_sigma=round(float(sigma.min()), 6),
        max_sigma=round(float(sigma.max()), 6),
    )

    return pd.DataFrame(cov_matrix, index=assets, columns=assets)
