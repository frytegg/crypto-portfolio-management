"""Tests for core/models/covariance.py — covariance and correlation estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.covariance import estimate_correlation, estimate_covariance


class TestEstimateCovariance:
    """Tests for estimate_covariance()."""

    def test_ledoit_returns_valid_psd_matrix(self, sample_returns: pd.DataFrame) -> None:
        """Ledoit-Wolf should return a square, symmetric, PSD NxN matrix."""
        cov = estimate_covariance(sample_returns, method="ledoit")

        n_assets = sample_returns.shape[1]

        # Shape is NxN
        assert cov.shape == (n_assets, n_assets)

        # Is a DataFrame with correct labels
        assert list(cov.index) == list(sample_returns.columns)
        assert list(cov.columns) == list(sample_returns.columns)

        # Symmetric
        assert np.allclose(cov.values, cov.values.T, atol=1e-10)

        # Positive semi-definite (all eigenvalues >= -1e-8)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() >= -1e-8

    def test_sample_cov_diagonal_equals_asset_variances(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """Sample covariance diagonal should match per-asset variance."""
        cov = estimate_covariance(sample_returns, method="sample")

        expected_variances = sample_returns.var(ddof=1).values
        actual_diagonal = np.diag(cov.values)

        # riskfolio uses ddof=1 internally for sample covariance
        np.testing.assert_allclose(actual_diagonal, expected_variances, rtol=1e-4)

    def test_gerber_returns_valid_matrix(self, sample_returns: pd.DataFrame) -> None:
        """Gerber method should return a valid PSD covariance matrix."""
        cov = estimate_covariance(sample_returns, method="gerber")

        n_assets = sample_returns.shape[1]
        assert cov.shape == (n_assets, n_assets)
        assert np.allclose(cov.values, cov.values.T, atol=1e-10)

        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() >= -1e-8

    def test_oas_returns_valid_psd_matrix(self, sample_returns: pd.DataFrame) -> None:
        """OAS (Oracle Approximating Shrinkage) should return a PSD matrix."""
        cov = estimate_covariance(sample_returns, method="oas")

        n_assets = sample_returns.shape[1]
        assert cov.shape == (n_assets, n_assets)
        assert np.allclose(cov.values, cov.values.T, atol=1e-10)

        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() >= -1e-8

    def test_denoised_returns_valid_psd_matrix(self, sample_returns: pd.DataFrame) -> None:
        """Denoised (Random Matrix Theory) should return a PSD matrix."""
        cov = estimate_covariance(sample_returns, method="denoised")

        n_assets = sample_returns.shape[1]
        assert cov.shape == (n_assets, n_assets)
        assert np.allclose(cov.values, cov.values.T, atol=1e-10)

        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert eigenvalues.min() >= -1e-8

    def test_ledoit_shrinkage_reduces_eigenvalue_spread(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """Ledoit-Wolf shrinkage should reduce the spread of eigenvalues
        compared to the sample covariance (that's the whole point of shrinkage)."""
        sample_cov = estimate_covariance(sample_returns, method="sample")
        ledoit_cov = estimate_covariance(sample_returns, method="ledoit")

        sample_eigenvalues = np.linalg.eigvalsh(sample_cov.values)
        ledoit_eigenvalues = np.linalg.eigvalsh(ledoit_cov.values)

        sample_spread = sample_eigenvalues.max() / sample_eigenvalues.min()
        ledoit_spread = ledoit_eigenvalues.max() / ledoit_eigenvalues.min()

        assert ledoit_spread < sample_spread, (
            f"Ledoit eigenvalue spread ({ledoit_spread:.1f}) should be smaller "
            f"than sample ({sample_spread:.1f})"
        )

    def test_invalid_method_raises_valueerror(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """Unknown method string should raise ValueError with valid methods listed."""
        with pytest.raises(ValueError, match="Unknown covariance method"):
            estimate_covariance(sample_returns, method="invalid_method")

    def test_insufficient_observations_raises(self) -> None:
        """Fewer than 10 observations should raise ValueError."""
        tiny = pd.DataFrame(
            np.random.default_rng(0).normal(size=(5, 3)),
            columns=["A", "B", "C"],
        )
        with pytest.raises(ValueError, match="at least 10 observations"):
            estimate_covariance(tiny)

    def test_single_asset_raises(self) -> None:
        """A single asset should raise ValueError (need at least 2)."""
        single = pd.DataFrame(
            np.random.default_rng(0).normal(size=(20, 1)),
            columns=["A"],
        )
        with pytest.raises(ValueError, match="at least 2 assets"):
            estimate_covariance(single)


class TestEstimateCorrelation:
    """Tests for estimate_correlation()."""

    def test_diagonal_is_ones(self, sample_returns: pd.DataFrame) -> None:
        """Correlation matrix diagonal should be exactly 1.0 for all assets."""
        corr = estimate_correlation(sample_returns, method="ledoit")

        diagonal = np.diag(corr.values)
        np.testing.assert_allclose(diagonal, np.ones(len(diagonal)), atol=1e-12)

    def test_values_bounded_minus_one_to_one(
        self, sample_returns: pd.DataFrame
    ) -> None:
        """All correlation values should be in [-1, 1]."""
        corr = estimate_correlation(sample_returns, method="ledoit")

        assert corr.values.min() >= -1.0 - 1e-10
        assert corr.values.max() <= 1.0 + 1e-10

    def test_symmetric(self, sample_returns: pd.DataFrame) -> None:
        """Correlation matrix should be symmetric."""
        corr = estimate_correlation(sample_returns, method="ledoit")
        assert np.allclose(corr.values, corr.values.T, atol=1e-10)

    def test_shape_matches_assets(self, sample_returns: pd.DataFrame) -> None:
        """Correlation matrix should be NxN with correct asset labels."""
        corr = estimate_correlation(sample_returns, method="sample")

        n = sample_returns.shape[1]
        assert corr.shape == (n, n)
        assert list(corr.index) == list(sample_returns.columns)
        assert list(corr.columns) == list(sample_returns.columns)
