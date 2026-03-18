"""Tests for core/risk/metrics.py — compute_risk_metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.risk.metrics import compute_drawdown_series, compute_risk_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_returns() -> pd.Series:
    """All-zero returns (100 days)."""
    return pd.Series(np.zeros(100), name="flat")


@pytest.fixture
def positive_returns() -> pd.Series:
    """All-positive returns (100 days of +1%)."""
    return pd.Series(np.full(100, 0.01), name="positive")


@pytest.fixture
def known_returns() -> pd.Series:
    """Small known series for manual verification."""
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.02], name="known")


# ---------------------------------------------------------------------------
# Test: flat series (all zeros)
# ---------------------------------------------------------------------------

class TestFlatReturns:
    def test_sharpe_is_zero(self, flat_returns: pd.Series) -> None:
        m = compute_risk_metrics(flat_returns)
        assert m["sharpe_ratio"] == 0.0

    def test_max_drawdown_is_zero(self, flat_returns: pd.Series) -> None:
        m = compute_risk_metrics(flat_returns)
        assert m["max_drawdown"] == 0.0

    def test_omega_is_inf(self, flat_returns: pd.Series) -> None:
        """No positive or negative returns → omega should handle gracefully."""
        m = compute_risk_metrics(flat_returns)
        # No positive returns and no negative returns → 0/0 edge case
        # Our implementation: neg_sum == 0 → np.inf
        assert m["omega_ratio"] == np.inf

    def test_annualized_return_is_zero(self, flat_returns: pd.Series) -> None:
        m = compute_risk_metrics(flat_returns)
        assert m["annualized_return"] == 0.0

    def test_annualized_volatility_is_zero(self, flat_returns: pd.Series) -> None:
        m = compute_risk_metrics(flat_returns)
        assert m["annualized_volatility"] == 0.0

    def test_max_drawdown_duration_is_zero(self, flat_returns: pd.Series) -> None:
        m = compute_risk_metrics(flat_returns)
        assert m["max_drawdown_duration"] == 0


# ---------------------------------------------------------------------------
# Test: all-positive returns
# ---------------------------------------------------------------------------

class TestPositiveReturns:
    def test_no_drawdown(self, positive_returns: pd.Series) -> None:
        m = compute_risk_metrics(positive_returns)
        assert m["max_drawdown"] == 0.0

    def test_sortino_is_inf(self, positive_returns: pd.Series) -> None:
        """No negative returns → sortino should be inf (or very high)."""
        m = compute_risk_metrics(positive_returns)
        assert m["sortino_ratio"] == np.inf

    def test_positive_days_100_pct(self, positive_returns: pd.Series) -> None:
        m = compute_risk_metrics(positive_returns)
        assert m["positive_days_pct"] == 100.0

    def test_equity_curve_monotonically_increasing(self, positive_returns: pd.Series) -> None:
        m = compute_risk_metrics(positive_returns)
        eq = m["equity_curve"]
        assert all(eq.diff().dropna() > 0)

    def test_annualized_return_positive(self, positive_returns: pd.Series) -> None:
        m = compute_risk_metrics(positive_returns)
        assert m["annualized_return"] > 0


# ---------------------------------------------------------------------------
# Test: known series [0.01, -0.02, 0.03, -0.01, 0.02]
# ---------------------------------------------------------------------------

class TestKnownReturns:
    def test_annualized_return(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        # mean = (0.01 - 0.02 + 0.03 - 0.01 + 0.02) / 5 = 0.006
        # annualized = 0.006 * 365 = 2.19
        expected = round(0.006 * 365, 6)
        assert m["annualized_return"] == expected

    def test_equity_curve_start(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        eq = m["equity_curve"]
        assert eq.iloc[0] == pytest.approx(1.01, abs=1e-10)

    def test_equity_curve_end(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        eq = m["equity_curve"]
        # (1+0.01)*(1-0.02)*(1+0.03)*(1-0.01)*(1+0.02)
        expected = 1.01 * 0.98 * 1.03 * 0.99 * 1.02
        assert eq.iloc[-1] == pytest.approx(expected, rel=1e-10)

    def test_max_drawdown(self, known_returns: pd.Series) -> None:
        """After day 1 (equity=1.01), day 2 drops to 1.01*0.98=0.9898.
        DD = 0.9898/1.01 - 1 = -0.02 exactly.
        """
        m = compute_risk_metrics(known_returns)
        # Peak after day 1 = 1.01, trough after day 2 = 1.01 * 0.98 = 0.9898
        # drawdown = 0.9898 / 1.01 - 1 = -0.02 (exact)
        assert m["max_drawdown"] == pytest.approx(-0.02, abs=1e-6)

    def test_best_and_worst_day(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        assert m["best_day"] == 0.03
        assert m["worst_day"] == -0.02

    def test_var_95(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        expected = round(float(np.percentile(known_returns, 5)), 6)
        assert m["var_95"] == expected

    def test_cvar_95(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        var_95 = np.percentile(known_returns, 5)
        tail = known_returns[known_returns <= var_95]
        if len(tail) > 0:
            expected = round(float(tail.mean()), 6)
        else:
            expected = round(float(var_95), 6)
        assert m["cvar_95"] == expected

    def test_positive_days_pct(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        # 3 positive out of 5 = 60%
        assert m["positive_days_pct"] == 60.0

    def test_skewness_and_kurtosis_are_floats(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        assert isinstance(m["skewness"], float)
        assert isinstance(m["kurtosis"], float)


# ---------------------------------------------------------------------------
# Test: compute_drawdown_series standalone
# ---------------------------------------------------------------------------

class TestComputeDrawdownSeries:
    def test_flat_returns(self, flat_returns: pd.Series) -> None:
        dd = compute_drawdown_series(flat_returns)
        assert (dd == 0).all()

    def test_positive_returns(self, positive_returns: pd.Series) -> None:
        dd = compute_drawdown_series(positive_returns)
        assert (dd == 0).all()

    def test_known_drawdown(self, known_returns: pd.Series) -> None:
        dd = compute_drawdown_series(known_returns)
        # After day 2 (-2% return), drawdown should be negative
        assert dd.iloc[1] < 0
        # After recovery to new high, drawdown should be 0
        # Day 3: 0.9898 * 1.03 = 1.01949 > 1.01 → new high, dd=0
        assert dd.iloc[2] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_series(self) -> None:
        m = compute_risk_metrics(pd.Series(dtype=float))
        assert m["annualized_return"] == 0.0
        assert m["max_drawdown"] == 0.0

    def test_single_return(self) -> None:
        m = compute_risk_metrics(pd.Series([0.05]))
        assert m["annualized_return"] == round(0.05 * 365, 6)
        assert m["equity_curve"].iloc[0] == pytest.approx(1.05)

    def test_custom_ann_factor(self, known_returns: pd.Series) -> None:
        m252 = compute_risk_metrics(known_returns, ann_factor=252)
        m365 = compute_risk_metrics(known_returns, ann_factor=365)
        assert m252["annualized_return"] < m365["annualized_return"]
        assert m252["annualized_volatility"] < m365["annualized_volatility"]

    def test_risk_free_rate_affects_sharpe(self, known_returns: pd.Series) -> None:
        m0 = compute_risk_metrics(known_returns, risk_free_rate=0.0)
        m4 = compute_risk_metrics(known_returns, risk_free_rate=0.04)
        assert m4["sharpe_ratio"] < m0["sharpe_ratio"]

    def test_max_drawdown_duration_type(self, known_returns: pd.Series) -> None:
        m = compute_risk_metrics(known_returns)
        assert isinstance(m["max_drawdown_duration"], int)

    def test_all_negative_returns(self) -> None:
        rets = pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01])
        m = compute_risk_metrics(rets)
        assert m["annualized_return"] < 0
        assert m["max_drawdown"] < 0
        assert m["positive_days_pct"] == 0.0
        assert m["omega_ratio"] == 0.0  # no positive returns, pos_sum=0
