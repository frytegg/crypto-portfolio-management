"""Covariance and correlation matrix estimation via riskfolio-lib.

Supported methods (mapped to riskfolio-lib's method_cov parameter):
    "ledoit"   → "ledoit"   — Ledoit-Wolf shrinkage (DEFAULT, recommended for crypto)
    "sample"   → "hist"     — Sample covariance (unreliable with few observations)
    "gerber"   → "gerber1"  — Gerber statistic (robust to outliers)
    "gerber2"  → "gerber2"  — Gerber statistic variant 2
    "oas"      → "oas"      — Oracle Approximating Shrinkage
    "denoised" → "denoise"  — Random Matrix Theory denoising (Marchenko-Pastur)

All methods delegate to riskfolio-lib's Portfolio.assets_stats() which computes
the covariance internally using the specified estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp
import structlog

log = structlog.get_logger(__name__)

# Maps our public method names → riskfolio-lib method_cov strings
_METHOD_MAP: dict[str, str] = {
    "ledoit": "ledoit",
    "sample": "hist",
    "gerber": "gerber1",
    "gerber2": "gerber2",
    "oas": "oas",
    "denoised": "fixed",
}

VALID_METHODS = list(_METHOD_MAP.keys())


def _validate_psd(matrix: np.ndarray, label: str = "covariance") -> None:
    """Assert matrix is symmetric and positive semi-definite.

    Raises ValueError if any eigenvalue is below -1e-8 tolerance.
    """
    # Symmetry check
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError(
            f"{label} matrix is not symmetric. "
            f"Max asymmetry: {np.max(np.abs(matrix - matrix.T)):.2e}"
        )

    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue = eigenvalues.min()
    if min_eigenvalue < -1e-8:
        raise ValueError(
            f"{label} matrix is not positive semi-definite. "
            f"Minimum eigenvalue: {min_eigenvalue:.6e} (tolerance: -1e-8)"
        )


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit",
) -> pd.DataFrame:
    """Estimate NxN covariance matrix from a returns DataFrame.

    Uses riskfolio-lib's Portfolio.assets_stats() to compute the covariance
    matrix with the specified estimation method.

    Args:
        returns: T x N DataFrame of daily log returns.
                 Must have at least 2 assets and 10 observations.
        method: Estimation method. One of:
                "ledoit"   — Ledoit-Wolf shrinkage (default, recommended)
                "sample"   — Sample covariance
                "gerber"   — Gerber statistic (robust to outliers)
                "gerber2"  — Gerber statistic variant 2
                "oas"      — Oracle Approximating Shrinkage
                "denoised" — Random Matrix Theory denoising

    Returns:
        N x N covariance matrix as a pd.DataFrame with asset names as
        both index and columns.

    Raises:
        ValueError: If method is invalid, returns has insufficient data,
                    or the resulting matrix is not PSD.
    """
    if method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown covariance method '{method}'. "
            f"Valid methods: {VALID_METHODS}"
        )

    if returns.shape[0] < 10:
        raise ValueError(
            f"Need at least 10 observations, got {returns.shape[0]}"
        )
    if returns.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 assets, got {returns.shape[1]}"
        )

    rp_method = _METHOD_MAP[method]
    log.info(
        "estimating_covariance",
        method=method,
        rp_method=rp_method,
        n_assets=returns.shape[1],
        n_obs=returns.shape[0],
    )

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_cov=rp_method, method_mu="hist")

    cov: pd.DataFrame = port.cov
    _validate_psd(cov.values, label="covariance")

    log.info("covariance_estimated", shape=cov.shape)
    return cov


def estimate_correlation(
    returns: pd.DataFrame,
    method: str = "ledoit",
) -> pd.DataFrame:
    """Estimate NxN correlation matrix from a returns DataFrame.

    Computes the covariance matrix using estimate_covariance(), then
    normalizes by the outer product of standard deviations:
        corr[i,j] = cov[i,j] / (std[i] * std[j])

    Args:
        returns: T x N DataFrame of daily log returns.
        method: Estimation method (same as estimate_covariance).

    Returns:
        N x N correlation matrix as a pd.DataFrame with asset names as
        both index and columns. Diagonal entries are 1.0.

    Raises:
        ValueError: If the underlying covariance estimation fails.
    """
    cov = estimate_covariance(returns, method=method)

    # Standard deviations from the covariance diagonal
    stds = np.sqrt(np.diag(cov.values))

    # Outer product of stds
    outer_stds = np.outer(stds, stds)

    # Avoid division by zero (shouldn't happen with valid data)
    outer_stds = np.where(outer_stds == 0, 1.0, outer_stds)

    corr_values = cov.values / outer_stds

    # Force exact 1.0 on diagonal (avoid floating point drift)
    np.fill_diagonal(corr_values, 1.0)

    corr = pd.DataFrame(corr_values, index=cov.index, columns=cov.columns)

    log.info("correlation_estimated", method=method, shape=corr.shape)
    return corr
