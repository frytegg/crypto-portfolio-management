"""Walk-forward backtester with transaction costs.

No look-ahead bias: at rebalance date t, training window = prices[:t-1].tail(lookback).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
import structlog

from core.optimization._base import PortfolioResult
from core.risk.metrics import compute_risk_metrics

log = structlog.get_logger(__name__)

MIN_TRAINING_DAYS = 90


@dataclass
class BacktestConfig:
    """Configuration for a walk-forward backtest run."""

    strategy: Literal[
        "equal_weight", "markowitz", "garch_gmv", "hrp",
        "risk_parity", "cvar", "black_litterman", "regime_aware"
    ]
    start_date: str
    end_date: str
    rebalance_frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    lookback_days: int = 365
    transaction_cost_bps: float = 10.0
    max_weight: float = 0.15
    initial_capital: float = 100_000.0


@dataclass
class BacktestResult:
    """Full result of a completed backtest."""

    equity_curve: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    turnover_history: pd.Series
    transaction_costs_total: float
    rebalance_dates: list[pd.Timestamp]
    metrics: dict
    config: BacktestConfig


def _get_strategy_fn(config: BacktestConfig) -> Callable[[pd.DataFrame], PortfolioResult]:
    """Return the callable for the requested strategy. Only imports what's needed."""
    strategy = config.strategy
    max_w = config.max_weight

    if strategy == "equal_weight":
        from core.optimization.equal_weight import optimize_equal_weight
        return lambda r: optimize_equal_weight(r)

    if strategy == "markowitz":
        from core.optimization.markowitz import optimize_markowitz
        return lambda r: optimize_markowitz(r, objective="Sharpe", max_weight=max_w)

    if strategy == "garch_gmv":
        from core.optimization.markowitz import optimize_garch_gmv
        return lambda r: optimize_garch_gmv(r, max_weight=max_w)

    if strategy == "hrp":
        from core.optimization.hrp import optimize_hrp
        return lambda r: optimize_hrp(r, max_weight=max_w)

    if strategy == "risk_parity":
        from core.optimization.risk_parity import optimize_risk_parity
        return lambda r: optimize_risk_parity(r)

    if strategy == "cvar":
        from core.optimization.cvar import optimize_cvar
        return lambda r: optimize_cvar(r, max_weight=max_w)

    raise ValueError(
        f"Strategy '{strategy}' not supported in backtest. "
        f"Supported: equal_weight, markowitz, garch_gmv, hrp, risk_parity, cvar"
    )


# Strategies supported in walk-forward backtest
_BACKTEST_STRATEGIES = {
    "equal_weight", "markowitz", "garch_gmv", "hrp", "risk_parity", "cvar",
}


def run_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Walk-forward backtest. No look-ahead bias enforced by construction.

    At each rebalance date t, only prices strictly before t are used for
    training. Between rebalances, the portfolio drifts with market returns.
    """
    strategy_fn = _get_strategy_fn(config)

    # Compute log returns from prices
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Filter to backtest date range
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    bt_returns = log_returns.loc[start:end]

    if len(bt_returns) == 0:
        raise ValueError(
            f"No trading days in [{config.start_date}, {config.end_date}]"
        )

    rebalance_dates = _get_rebalance_dates(
        prices, config.start_date, config.end_date, config.rebalance_frequency,
    )
    log.info(
        "backtest_start",
        strategy=config.strategy,
        start=config.start_date,
        end=config.end_date,
        rebalance_count=len(rebalance_dates),
        trading_days=len(bt_returns),
    )

    assets = prices.columns.tolist()
    n_assets = len(assets)

    # Initialize with equal weights
    weights_current = pd.Series(1.0 / n_assets, index=assets)

    # Tracking structures
    equity_values: list[float] = []
    daily_returns_list: list[float] = []
    weights_records: list[dict] = []
    turnover_records: list[dict] = []
    actual_rebalance_dates: list[pd.Timestamp] = []
    total_costs = 0.0
    portfolio_value = config.initial_capital

    rebalance_set = set(rebalance_dates)

    for date in bt_returns.index:
        # --- Rebalance if this is a rebalance date ---
        if date in rebalance_set:
            # Training window: all returns strictly before this date
            training_end = date - pd.Timedelta(days=1)
            training_start = date - pd.Timedelta(days=config.lookback_days)
            training_returns = log_returns.loc[training_start:training_end]

            if len(training_returns) < MIN_TRAINING_DAYS:
                log.debug(
                    "skipping_rebalance_insufficient_data",
                    date=str(date.date()),
                    available=len(training_returns),
                    required=MIN_TRAINING_DAYS,
                )
            else:
                try:
                    result = strategy_fn(training_returns)
                    new_weights = result.weights.reindex(assets, fill_value=0.0)

                    # Normalize to sum to 1 (safety)
                    weight_sum = new_weights.sum()
                    if weight_sum > 0:
                        new_weights = new_weights / weight_sum

                    # Turnover and transaction cost
                    turnover = float(np.abs(new_weights - weights_current).sum())
                    cost = turnover * config.transaction_cost_bps / 10_000
                    portfolio_value *= (1 - cost)
                    total_costs += cost * portfolio_value

                    weights_current = new_weights
                    actual_rebalance_dates.append(date)

                    weights_records.append(
                        {"date": date, **weights_current.to_dict()}
                    )
                    turnover_records.append({"date": date, "turnover": turnover})

                    log.debug(
                        "rebalance",
                        date=str(date.date()),
                        turnover=round(turnover, 4),
                        cost_pct=round(cost * 100, 4),
                    )
                except Exception:
                    log.exception(
                        "strategy_failed_at_rebalance", date=str(date.date()),
                    )

        # --- Daily portfolio return (drift with market) ---
        day_returns = bt_returns.loc[date]
        portfolio_return = float((weights_current * day_returns).sum())
        portfolio_value *= (1 + portfolio_return)

        # Update weights for drift (weights shift with returns)
        drifted = weights_current * (1 + day_returns)
        drifted_sum = drifted.sum()
        if drifted_sum > 0:
            weights_current = drifted / drifted_sum

        equity_values.append(portfolio_value)
        daily_returns_list.append(portfolio_return)

    # --- Build result series ---
    equity_curve = pd.Series(
        equity_values, index=bt_returns.index, name="equity",
    )
    daily_returns = pd.Series(
        daily_returns_list, index=bt_returns.index, name="returns",
    )

    # Weights history DataFrame
    if weights_records:
        weights_history = pd.DataFrame(weights_records).set_index("date")
    else:
        weights_history = pd.DataFrame(columns=["date"] + assets).set_index("date")

    # Turnover history
    if turnover_records:
        turnover_df = pd.DataFrame(turnover_records).set_index("date")
        turnover_history = turnover_df["turnover"]
    else:
        turnover_history = pd.Series(dtype=float, name="turnover")

    # Risk metrics on the backtest returns
    metrics = compute_risk_metrics(daily_returns)

    log.info(
        "backtest_complete",
        final_value=round(portfolio_value, 2),
        total_return_pct=round((portfolio_value / config.initial_capital - 1) * 100, 2),
        rebalances=len(actual_rebalance_dates),
        total_cost_usd=round(total_costs, 2),
    )

    return BacktestResult(
        equity_curve=equity_curve,
        returns=daily_returns,
        weights_history=weights_history,
        turnover_history=turnover_history,
        transaction_costs_total=total_costs,
        rebalance_dates=actual_rebalance_dates,
        metrics=metrics,
        config=config,
    )


def _get_rebalance_dates(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    frequency: str,
) -> list[pd.Timestamp]:
    """Generate rebalance dates aligned to price index.

    - "monthly": first business day of each month
    - "weekly": every Monday
    - "quarterly": first business day of Jan, Apr, Jul, Oct
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    price_dates = prices.index

    if frequency == "monthly":
        # Generate month starts, then find first trading day >= each
        candidates = pd.date_range(start, end, freq="MS")
    elif frequency == "weekly":
        # Every Monday
        candidates = pd.date_range(start, end, freq="W-MON")
    elif frequency == "quarterly":
        # First business day of Jan, Apr, Jul, Oct
        candidates = pd.date_range(start, end, freq="QS")
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    # Map each candidate to the first trading day on or after it
    rebalance_dates = []
    for candidate in candidates:
        valid = price_dates[price_dates >= candidate]
        if len(valid) > 0 and valid[0] <= end:
            rebalance_dates.append(valid[0])

    # Deduplicate (two candidates could map to the same trading day)
    seen: set[pd.Timestamp] = set()
    unique: list[pd.Timestamp] = []
    for d in rebalance_dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    return unique
