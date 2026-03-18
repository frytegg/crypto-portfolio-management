"""Portfolio risk and performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def compute_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    ann_factor: int = 365,
) -> dict:
    """Compute comprehensive risk metrics for a portfolio return series.

    Parameters
    ----------
    returns : pd.Series
        Daily log or simple returns.
    risk_free_rate : float
        Annualized risk-free rate (e.g. 0.04 for 4%).
    ann_factor : int
        Annualization factor (365 for crypto).

    Returns
    -------
    dict with keys:
        annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
        calmar_ratio, omega_ratio, max_drawdown, max_drawdown_duration,
        var_95, cvar_95, var_99, cvar_99, skewness, kurtosis,
        positive_days_pct, best_day, worst_day, equity_curve, drawdown_series
    """
    returns = returns.dropna()

    if len(returns) == 0:
        log.warning("empty_returns_series")
        return _empty_metrics()

    # --- Core stats ---
    ann_ret = round(float(returns.mean() * ann_factor), 6)
    ann_vol = round(float(returns.std() * np.sqrt(ann_factor)), 6)

    # --- Sharpe ratio ---
    sharpe = round((ann_ret - risk_free_rate) / ann_vol, 6) if ann_vol > 0 else 0.0

    # --- Sortino ratio (downside deviation) ---
    downside = returns[returns < 0]
    if len(downside) > 0:
        downside_std = float(downside.std())
        sortino = (
            round(ann_ret / (downside_std * np.sqrt(ann_factor)), 6)
            if downside_std > 0
            else 0.0
        )
    else:
        sortino = round(float(np.inf), 6) if ann_ret > 0 else 0.0

    # --- Equity curve and drawdown ---
    equity_curve = (1 + returns).cumprod()
    drawdown_series = equity_curve / equity_curve.cummax() - 1
    max_dd = round(float(drawdown_series.min()), 6)

    # --- Calmar ratio ---
    calmar = round(ann_ret / abs(max_dd), 6) if max_dd != 0 else 0.0

    # --- Omega ratio ---
    pos_sum = float(returns[returns > 0].sum())
    neg_sum = float(returns[returns < 0].sum())
    omega = round(pos_sum / abs(neg_sum), 6) if neg_sum != 0 else np.inf

    # --- VaR and CVaR ---
    var_95 = round(float(np.percentile(returns, 5)), 6)
    cvar_95_vals = returns[returns <= var_95]
    cvar_95 = round(float(cvar_95_vals.mean()), 6) if len(cvar_95_vals) > 0 else var_95

    var_99 = round(float(np.percentile(returns, 1)), 6)
    cvar_99_vals = returns[returns <= var_99]
    cvar_99 = round(float(cvar_99_vals.mean()), 6) if len(cvar_99_vals) > 0 else var_99

    # --- Distribution stats ---
    skewness = round(float(returns.skew()), 6)
    kurtosis = round(float(returns.kurtosis()), 6)

    # --- Day-level stats ---
    positive_days_pct = round(float((returns > 0).mean() * 100), 6)
    best_day = round(float(returns.max()), 6)
    worst_day = round(float(returns.min()), 6)

    # --- Max drawdown duration ---
    mdd_duration = _max_drawdown_duration(drawdown_series)

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "omega_ratio": omega,
        "max_drawdown": max_dd,
        "max_drawdown_duration": mdd_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "positive_days_pct": positive_days_pct,
        "best_day": best_day,
        "worst_day": worst_day,
        "equity_curve": equity_curve,
        "drawdown_series": drawdown_series,
    }


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown time series from returns."""
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def _max_drawdown_duration(drawdown_series: pd.Series) -> int:
    """Find the duration (in days) of the worst drawdown period.

    Duration = number of days from the peak before the worst drawdown
    to the recovery point (or end of series if no recovery).
    """
    if len(drawdown_series) == 0 or drawdown_series.min() == 0:
        return 0

    # Find the index of the maximum drawdown (trough)
    trough_idx = drawdown_series.idxmin()
    trough_pos = drawdown_series.index.get_loc(trough_idx)

    # Find the peak before the trough: last time drawdown was 0 before trough
    dd_before_trough = drawdown_series.iloc[:trough_pos + 1]
    zero_before = dd_before_trough[dd_before_trough == 0]
    if len(zero_before) > 0:
        peak_pos = drawdown_series.index.get_loc(zero_before.index[-1])
    else:
        peak_pos = 0

    # Find recovery after trough: first time drawdown returns to 0 after trough
    dd_after_trough = drawdown_series.iloc[trough_pos:]
    zero_after = dd_after_trough[dd_after_trough >= 0]
    if len(zero_after) > 0:
        recovery_pos = drawdown_series.index.get_loc(zero_after.index[0])
    else:
        # No recovery — duration extends to end of series
        recovery_pos = len(drawdown_series) - 1

    return int(recovery_pos - peak_pos)


def _empty_metrics() -> dict:
    """Return zeroed-out metrics for empty return series."""
    return {
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "omega_ratio": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration": 0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "var_99": 0.0,
        "cvar_99": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "positive_days_pct": 0.0,
        "best_day": 0.0,
        "worst_day": 0.0,
        "equity_curve": pd.Series(dtype=float),
        "drawdown_series": pd.Series(dtype=float),
    }
