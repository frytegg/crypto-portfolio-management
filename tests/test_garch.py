"""Tests for core/models/garch.py — GARCH volatility forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.garch import build_garch_covariance, fit_all_garch, fit_garch


REQUIRED_KEYS = {
    "conditional_volatility",
    "forecast_variance",
    "forecast_vol",
    "params",
    "aic",
    "bic",
    "model_result",
}


class TestFitGarch:
    """Tests for fit_garch()."""

    def test_returns_dict_with_all_required_keys(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """fit_garch should return a dict with all required keys."""
        result = fit_garch(sample_returns["BTC"])

        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_conditional_volatility_is_series(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """conditional_volatility should be a pd.Series with same index as input."""
        result = fit_garch(sample_returns["ETH"])

        cond_vol = result["conditional_volatility"]
        assert isinstance(cond_vol, pd.Series)
        assert len(cond_vol) == len(sample_returns)

    def test_rescaling_magnitude(self, sample_returns: pd.DataFrame) -> None:
        """After rescaling back, conditional_volatility should be same order
        of magnitude as the raw returns standard deviation."""
        col = "BTC"
        result = fit_garch(sample_returns[col], rescale=True)

        raw_std = sample_returns[col].std()
        cond_vol_mean = result["conditional_volatility"].mean()

        # Should be within 10x of each other (same order of magnitude)
        ratio = cond_vol_mean / raw_std
        assert 0.1 < ratio < 10.0, (
            f"Rescaling looks wrong: cond_vol mean={cond_vol_mean:.6f}, "
            f"raw std={raw_std:.6f}, ratio={ratio:.2f}"
        )

    def test_forecast_vol_is_sqrt_of_variance(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """forecast_vol should equal sqrt(forecast_variance)."""
        result = fit_garch(sample_returns["SOL"])

        expected = np.sqrt(result["forecast_variance"])
        np.testing.assert_allclose(result["forecast_vol"], expected, rtol=1e-10)

    def test_forecast_variance_positive(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """Forecast variance must be strictly positive."""
        result = fit_garch(sample_returns["BNB"])
        assert result["forecast_variance"] > 0

    def test_aic_bic_are_finite(self, sample_returns: pd.DataFrame) -> None:
        """AIC and BIC should be finite floats."""
        result = fit_garch(sample_returns["ADA"])
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])


class TestFitAllGarch:
    """Tests for fit_all_garch()."""

    def test_returns_result_for_every_asset(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """fit_all_garch should return a dict keyed by every column name."""
        results = fit_all_garch(sample_returns)

        assert set(results.keys()) == set(sample_returns.columns)
        for col in sample_returns.columns:
            assert REQUIRED_KEYS.issubset(results[col].keys())

    def test_fallback_on_constant_series(self) -> None:
        """A constant (zero-variance) series must not raise an exception.
        Whether arch converges or falls back to sample vol, fit_all_garch
        should return a valid result dict for every asset."""
        n = 100
        dates = pd.bdate_range(start="2024-01-02", periods=n)
        # One normal asset, one constant (degenerate for GARCH)
        df = pd.DataFrame(
            {
                "NORMAL": np.random.default_rng(42).normal(0, 0.03, n),
                "CONSTANT": np.zeros(n),
            },
            index=dates,
        )

        # Must not raise
        results = fit_all_garch(df)

        # Both assets should have results with all required keys
        assert "NORMAL" in results
        assert "CONSTANT" in results
        assert REQUIRED_KEYS.issubset(results["NORMAL"].keys())
        assert REQUIRED_KEYS.issubset(results["CONSTANT"].keys())


class TestBuildGarchCovariance:
    """Tests for build_garch_covariance()."""

    def test_returns_valid_symmetric_matrix(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """build_garch_covariance should return a symmetric NxN DataFrame."""
        cov = build_garch_covariance(sample_returns)

        n = sample_returns.shape[1]
        assert cov.shape == (n, n)
        assert list(cov.index) == list(sample_returns.columns)
        assert list(cov.columns) == list(sample_returns.columns)

        # Symmetric
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)

    def test_diagonal_positive(self, sample_returns: pd.DataFrame) -> None:
        """Diagonal (variances) must be strictly positive."""
        cov = build_garch_covariance(sample_returns)
        diagonal = np.diag(cov.values)
        assert np.all(diagonal > 0)

    def test_accepts_precomputed_results(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """Should accept pre-computed garch_results without refitting."""
        garch_results = fit_all_garch(sample_returns)
        cov = build_garch_covariance(sample_returns, garch_results=garch_results)

        n = sample_returns.shape[1]
        assert cov.shape == (n, n)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-12)
