"""Tests for core/models/returns.py — expected return estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.returns import estimate_returns


class TestEstimateReturns:
    """Tests for estimate_returns()."""

    def test_historical_returns_series(self, sample_returns: pd.DataFrame) -> None:
        """Historical method should return a Series indexed by asset name."""
        mu = estimate_returns(sample_returns, method="historical")

        assert isinstance(mu, pd.Series)
        assert len(mu) == sample_returns.shape[1]
        assert list(mu.index) == list(sample_returns.columns)

    def test_ewma_returns_series(self, sample_returns: pd.DataFrame) -> None:
        """EWMA method should return a valid Series."""
        mu = estimate_returns(sample_returns, method="ewma")

        assert isinstance(mu, pd.Series)
        assert len(mu) == sample_returns.shape[1]

    def test_james_stein_returns_series(self, sample_returns: pd.DataFrame) -> None:
        """James-Stein method should return a valid Series."""
        mu = estimate_returns(sample_returns, method="james_stein")

        assert isinstance(mu, pd.Series)
        assert len(mu) == sample_returns.shape[1]

    def test_all_values_finite(self, sample_returns: pd.DataFrame) -> None:
        """All estimated returns should be finite."""
        mu = estimate_returns(sample_returns, method="historical")
        assert np.all(np.isfinite(mu.values))

    def test_invalid_method_raises(self, sample_returns: pd.DataFrame) -> None:
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown return estimation method"):
            estimate_returns(sample_returns, method="bogus")
