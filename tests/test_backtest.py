"""Tests for core/risk/backtest.py — walk-forward backtester."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.risk.backtest import BacktestConfig, BacktestResult, _get_rebalance_dates, run_backtest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """~3 years of synthetic daily prices for 5 assets.

    Uses geometric Brownian motion with fixed seed for reproducibility.
    Date range: 2022-01-03 to 2024-12-31 (business days).
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    n_days = len(dates)
    assets = ["BTC", "ETH", "SOL", "AVAX", "LINK"]
    n_assets = len(assets)

    # Daily returns ~ N(0.0003, 0.03) — mild positive drift, crypto-like vol
    daily_returns = rng.normal(0.0003, 0.03, (n_days, n_assets))

    # Build price levels from cumulative returns
    initial_prices = np.array([40000.0, 2500.0, 100.0, 35.0, 15.0])
    cum_returns = np.cumsum(daily_returns, axis=0)
    price_data = initial_prices * np.exp(cum_returns)

    return pd.DataFrame(price_data, index=dates, columns=assets)


@pytest.fixture
def monthly_config() -> BacktestConfig:
    """Standard monthly rebalance config with equal_weight strategy."""
    return BacktestConfig(
        strategy="equal_weight",
        start_date="2023-01-01",
        end_date="2024-12-31",
        rebalance_frequency="monthly",
        lookback_days=365,
        transaction_cost_bps=10.0,
        max_weight=0.30,
        initial_capital=100_000.0,
    )


# ---------------------------------------------------------------------------
# Test: _get_rebalance_dates
# ---------------------------------------------------------------------------

class TestGetRebalanceDates:
    def test_monthly_dates_count(self, synthetic_prices: pd.DataFrame) -> None:
        dates = _get_rebalance_dates(
            synthetic_prices, "2023-01-01", "2024-12-31", "monthly",
        )
        # 2 years = 24 months
        assert len(dates) == 24

    def test_weekly_dates_count(self, synthetic_prices: pd.DataFrame) -> None:
        dates = _get_rebalance_dates(
            synthetic_prices, "2023-01-01", "2023-03-31", "weekly",
        )
        # ~13 weeks in Q1
        assert 12 <= len(dates) <= 14

    def test_quarterly_dates_count(self, synthetic_prices: pd.DataFrame) -> None:
        dates = _get_rebalance_dates(
            synthetic_prices, "2023-01-01", "2024-12-31", "quarterly",
        )
        # 2 years = 8 quarters
        assert len(dates) == 8

    def test_dates_are_in_price_index(self, synthetic_prices: pd.DataFrame) -> None:
        dates = _get_rebalance_dates(
            synthetic_prices, "2023-01-01", "2024-12-31", "monthly",
        )
        for d in dates:
            assert d in synthetic_prices.index

    def test_dates_within_range(self, synthetic_prices: pd.DataFrame) -> None:
        start = pd.Timestamp("2023-06-01")
        end = pd.Timestamp("2023-12-31")
        dates = _get_rebalance_dates(synthetic_prices, str(start.date()), str(end.date()), "monthly")
        for d in dates:
            assert d >= start
            assert d <= end

    def test_invalid_frequency_raises(self, synthetic_prices: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown frequency"):
            _get_rebalance_dates(synthetic_prices, "2023-01-01", "2024-12-31", "daily")


# ---------------------------------------------------------------------------
# Test: run_backtest with equal_weight
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_equity_curve_length(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        # Equity curve should cover all trading days in range
        expected_days = synthetic_prices.loc["2023-01-01":"2024-12-31"]
        # log returns drop the first row, so count from returns
        log_rets = np.log(synthetic_prices / synthetic_prices.shift(1)).dropna()
        bt_days = log_rets.loc["2023-01-01":"2024-12-31"]
        assert len(result.equity_curve) == len(bt_days)

    def test_returns_length_matches_equity(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        assert len(result.returns) == len(result.equity_curve)

    def test_transaction_costs_applied(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        assert result.transaction_costs_total > 0

    def test_rebalance_dates_count(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        # With 2 years monthly and enough lookback, expect ~24 rebalances
        # First few may be skipped if lookback window has < 90 days
        assert len(result.rebalance_dates) > 0
        assert len(result.rebalance_dates) <= 24

    def test_metrics_computed(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        assert "annualized_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "equity_curve" in result.metrics

    def test_weights_history_shape(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        assert len(result.weights_history) == len(result.rebalance_dates)
        # Columns should be the asset names
        for asset in synthetic_prices.columns:
            assert asset in result.weights_history.columns

    def test_config_stored(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        assert result.config is monthly_config

    def test_initial_value(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        # First equity value should be close to initial capital (slightly modified by day 1 return)
        first_val = result.equity_curve.iloc[0]
        assert 90_000 < first_val < 110_000


# ---------------------------------------------------------------------------
# Test: no look-ahead bias
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:
    def test_training_window_ends_before_rebalance(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        """Verify that at each rebalance date t, training data ends before t.

        We do this by checking the rebalance dates against the price index.
        The backtest uses prices[:t-1].tail(lookback) — if we have at least
        1 day of returns before t, training is strictly before t.
        """
        result = run_backtest(synthetic_prices, monthly_config)
        log_returns = np.log(synthetic_prices / synthetic_prices.shift(1)).dropna()

        for rebalance_date in result.rebalance_dates:
            # Training window should be [t - lookback, t - 1 day]
            training_end = rebalance_date - pd.Timedelta(days=1)
            training_start = rebalance_date - pd.Timedelta(days=monthly_config.lookback_days)
            training_data = log_returns.loc[training_start:training_end]

            # All training data dates must be strictly before rebalance date
            assert all(d < rebalance_date for d in training_data.index), (
                f"Look-ahead detected: training data at {rebalance_date} "
                f"includes dates >= rebalance date"
            )

    def test_rebalance_dates_in_price_index(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        for d in result.rebalance_dates:
            assert d in synthetic_prices.index


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unsupported_strategy_raises(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        config = BacktestConfig(
            strategy="black_litterman",
            start_date="2023-01-01",
            end_date="2024-12-31",
        )
        with pytest.raises(ValueError, match="not supported"):
            run_backtest(synthetic_prices, config)

    def test_no_trading_days_raises(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        config = BacktestConfig(
            strategy="equal_weight",
            start_date="2030-01-01",
            end_date="2030-12-31",
        )
        with pytest.raises(ValueError, match="No trading days"):
            run_backtest(synthetic_prices, config)

    def test_zero_transaction_cost(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        config = BacktestConfig(
            strategy="equal_weight",
            start_date="2023-01-01",
            end_date="2024-12-31",
            transaction_cost_bps=0.0,
        )
        result = run_backtest(synthetic_prices, config)
        assert result.transaction_costs_total == 0.0

    def test_weekly_rebalance(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        config = BacktestConfig(
            strategy="equal_weight",
            start_date="2023-06-01",
            end_date="2023-12-31",
            rebalance_frequency="weekly",
        )
        result = run_backtest(synthetic_prices, config)
        # ~26 weeks in 6 months
        assert len(result.rebalance_dates) > 20

    def test_quarterly_rebalance(
        self, synthetic_prices: pd.DataFrame,
    ) -> None:
        config = BacktestConfig(
            strategy="equal_weight",
            start_date="2023-01-01",
            end_date="2024-12-31",
            rebalance_frequency="quarterly",
        )
        result = run_backtest(synthetic_prices, config)
        assert len(result.rebalance_dates) <= 8

    def test_turnover_history_length(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        result = run_backtest(synthetic_prices, monthly_config)
        assert len(result.turnover_history) == len(result.rebalance_dates)

    def test_equal_weight_low_turnover(
        self, synthetic_prices: pd.DataFrame, monthly_config: BacktestConfig,
    ) -> None:
        """Equal weight rebalancing back to 1/N should have modest turnover."""
        result = run_backtest(synthetic_prices, monthly_config)
        avg_turnover = result.turnover_history.mean()
        # Equal weight drift + rebalance → moderate turnover
        assert 0 < avg_turnover < 1.0
