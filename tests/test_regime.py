"""Tests for core/models/regime.py — HMM regime detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.regime import detect_regimes


REQUIRED_KEYS = {
    "regimes",
    "regime_names",
    "transition_matrix",
    "regime_means",
    "regime_vols",
    "current_regime",
    "current_regime_name",
}


@pytest.fixture
def synthetic_regime_returns() -> pd.Series:
    """BTC-like synthetic returns with two distinct regimes.

    First half: bear market (negative mean, high vol).
    Second half: bull market (positive mean, moderate vol).
    """
    rng = np.random.default_rng(42)
    n_bear = 200
    n_bull = 200

    bear_returns = rng.normal(-0.005, 0.04, n_bear)  # negative drift, high vol
    bull_returns = rng.normal(0.003, 0.02, n_bull)    # positive drift, lower vol

    all_returns = np.concatenate([bear_returns, bull_returns])
    dates = pd.bdate_range(start="2023-01-02", periods=len(all_returns))

    return pd.Series(all_returns, index=dates, name="BTC")


class TestDetectRegimes:
    """Tests for detect_regimes()."""

    def test_returns_dict_with_all_required_keys(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """detect_regimes should return a dict with all required keys."""
        result = detect_regimes(synthetic_regime_returns, n_regimes=2)

        assert isinstance(result, dict)
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_regime_names_has_n_entries(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """regime_names should have exactly n_regimes entries."""
        for n in [2, 3]:
            result = detect_regimes(synthetic_regime_returns, n_regimes=n)
            assert len(result["regime_names"]) == n

    def test_two_regimes_labels(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """With n_regimes=2, labels should be 'Bull' and 'Bear'."""
        result = detect_regimes(synthetic_regime_returns, n_regimes=2)

        labels = set(result["regime_names"].values())
        assert labels == {"Bull", "Bear"}

    def test_three_regimes_labels(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """With n_regimes=3, labels should be 'Bull', 'Bear', and 'Sideways'."""
        result = detect_regimes(synthetic_regime_returns, n_regimes=3)

        labels = set(result["regime_names"].values())
        assert labels == {"Bull", "Bear", "Sideways"}

    def test_bull_mean_greater_than_bear_mean(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """The 'Bull' regime should have a higher mean return than 'Bear'."""
        result = detect_regimes(synthetic_regime_returns, n_regimes=2)

        name_map = result["regime_names"]
        means = result["regime_means"]

        bull_state = [k for k, v in name_map.items() if v == "Bull"][0]
        bear_state = [k for k, v in name_map.items() if v == "Bear"][0]

        assert means[bull_state] > means[bear_state]

    def test_current_regime_name_is_valid(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """current_regime_name should be one of the name_map values."""
        result = detect_regimes(synthetic_regime_returns, n_regimes=2)

        valid_names = set(result["regime_names"].values())
        assert result["current_regime_name"] in valid_names

    def test_regimes_series_length_matches_lookback(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """The regimes Series should have length = min(lookback_days, len(returns))."""
        lookback = 300
        result = detect_regimes(
            synthetic_regime_returns, n_regimes=2, lookback_days=lookback
        )

        expected_len = min(lookback, len(synthetic_regime_returns))
        assert len(result["regimes"]) == expected_len

    def test_transition_matrix_shape_and_rows_sum_to_one(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """Transition matrix should be NxN with rows summing to 1."""
        n = 2
        result = detect_regimes(synthetic_regime_returns, n_regimes=n)

        tm = result["transition_matrix"]
        assert tm.shape == (n, n)
        np.testing.assert_allclose(tm.sum(axis=1), np.ones(n), atol=1e-10)

    def test_regime_vols_positive(
        self, synthetic_regime_returns: pd.Series
    ) -> None:
        """All regime volatilities should be strictly positive."""
        result = detect_regimes(synthetic_regime_returns, n_regimes=2)
        assert np.all(result["regime_vols"] > 0)
