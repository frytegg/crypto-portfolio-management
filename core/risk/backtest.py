"""Walk-forward backtester with transaction costs.

No look-ahead bias: at rebalance date t, training window = prices[:t-1].tail(lookback).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import structlog

from core.data.onchain import OnchainSignals

log = structlog.get_logger(__name__)


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


def run_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig,
    onchain_signals: OnchainSignals | None = None,
) -> BacktestResult:
    """Walk-forward backtest. No look-ahead bias enforced by construction."""
    raise NotImplementedError


def _get_rebalance_dates(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    frequency: str,
) -> list[pd.Timestamp]:
    """Generate rebalance dates. Monthly = last trading day of each month."""
    raise NotImplementedError
