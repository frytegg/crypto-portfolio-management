"""Black-Litterman with on-chain signal views.

CRITICAL (WARNING 4): blacklitterman_stats() MUST be called BEFORE
optimization(model='BL'). Calling optimization(model='BL') without it
gives garbage weights — no error raised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp
import structlog

from core.optimization._base import PortfolioResult

log = structlog.get_logger(__name__)


def generate_onchain_views(
    onchain_signals: dict,
    universe_assets: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Translate on-chain signals into Black-Litterman P, Q, confidence matrices.

    Rules:
        1. TVL momentum > 5% → ETH outperforms BTC (relative view).
        2. Stablecoin supply increase > 3% → market bullish (absolute view).
        3. DEX volume surge > 30% → DeFi tokens outperform (absolute view).

    Args:
        onchain_signals: Dict with keys like "tvl_momentum_30d",
            "stablecoin_supply_change_30d", "dex_volume_trend_7d".
        universe_assets: List of asset ticker strings in portfolio order.

    Returns:
        Tuple of (P, Q, confidences):
            P: DataFrame (n_views x n_assets) — picking matrix.
            Q: Series (n_views,) — view magnitudes.
            confidences: Series (n_views,) — confidence levels [0, 1].
        If no views triggered, P is empty DataFrame.
    """
    views: list[dict] = []

    # Rule 1: TVL momentum — ETH outperforms BTC
    tvl_mom = onchain_signals.get("tvl_momentum_30d", 0)
    if tvl_mom > 0.05:
        p_row = pd.Series(0.0, index=universe_assets)
        if "ETH" in universe_assets:
            p_row["ETH"] = 1.0
        if "BTC" in universe_assets:
            p_row["BTC"] = -1.0
        q = 0.02 / 12  # 2% annualized → monthly
        conf = min(tvl_mom / 0.2, 1.0)
        views.append({"p": p_row, "q": q, "conf": conf})

    # Rule 2: Stablecoin inflow — market bullish
    stable_change = onchain_signals.get("stablecoin_supply_change_30d", 0)
    if stable_change > 0.03:
        p_row = pd.Series(1.0 / len(universe_assets), index=universe_assets)
        q = 0.01 / 12
        conf = min(stable_change / 0.1, 1.0)
        views.append({"p": p_row, "q": q, "conf": conf})

    # Rule 3: DEX volume surge — DeFi tokens outperform
    defi_tokens = [t for t in ["UNI", "AAVE", "MKR", "LINK"] if t in universe_assets]
    dex_trend = onchain_signals.get("dex_volume_trend_7d", 1.0)
    if dex_trend > 1.3 and defi_tokens:
        p_row = pd.Series(0.0, index=universe_assets)
        for t in defi_tokens:
            p_row[t] = 1.0 / len(defi_tokens)
        q = 0.03 / 12
        conf = min((dex_trend - 1.0) / 0.5, 1.0)
        views.append({"p": p_row, "q": q, "conf": conf})

    if not views:
        return (
            pd.DataFrame(columns=universe_assets),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    P = pd.DataFrame([v["p"] for v in views])
    Q = pd.Series([v["q"] for v in views])
    confidences = pd.Series([v["conf"] for v in views])

    log.info(
        "onchain_views_generated",
        n_views=len(views),
        confidences=confidences.round(3).tolist(),
    )

    return P, Q, confidences


def optimize_black_litterman(
    returns: pd.DataFrame,
    onchain_signals: dict,
    risk_free_rate: float = 0.0,
    method_cov: str = "ledoit",
    max_weight: float = 0.15,
) -> PortfolioResult:
    """Black-Litterman optimization with on-chain signal views.

    If no on-chain views are triggered, falls back to max-Sharpe Markowitz.

    CRITICAL (WARNING 4): blacklitterman_stats() is called BEFORE
    optimization(model='BL') — reversing this order gives silent garbage.

    Args:
        returns: T x N DataFrame of daily log returns.
        onchain_signals: Dict of on-chain signal values.
        risk_free_rate: Annual risk-free rate.
        method_cov: Covariance estimator for riskfolio-lib.
        max_weight: Upper bound per asset weight.

    Returns:
        PortfolioResult with BL-optimized weights.

    Raises:
        ValueError: If the solver fails.
    """
    P, Q, confidences = generate_onchain_views(
        onchain_signals, list(returns.columns)
    )

    if P.empty:
        from core.optimization.markowitz import optimize_markowitz

        log.info("bl_no_views_fallback_to_markowitz")
        result = optimize_markowitz(
            returns, objective="Sharpe", max_weight=max_weight
        )
        result.name = "Black-Litterman (no views — fallback to Max Sharpe)"
        result.metadata["fallback"] = True
        return result

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov=method_cov)

    # CRITICAL: must call blacklitterman_stats BEFORE optimization(model='BL')
    port.blacklitterman_stats(
        P=P.values,
        Q=Q.values.reshape(-1, 1),
        rf=risk_free_rate,
        w=None,
        delta=2.5,
        eq=True,
    )

    port.upperlng = max_weight
    w = port.optimization(model="BL", rm="MV", obj="Sharpe", rf=risk_free_rate)

    if w is None or w.isnull().all().all():
        raise ValueError("Black-Litterman optimization failed — solver returned None.")

    weights = w["weights"]
    weights.name = "weights"

    mu = port.mu_bl.values.flatten() if hasattr(port, "mu_bl") and port.mu_bl is not None else port.mu.values.flatten()
    cov = port.cov_bl.values if hasattr(port, "cov_bl") and port.cov_bl is not None else port.cov.values
    w_arr = weights.values

    ann_ret = float(w_arr @ mu) * 365
    ann_vol = float(np.sqrt(w_arr @ cov @ w_arr)) * np.sqrt(365)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    log.info(
        "bl_optimized",
        n_views=len(Q),
        ann_return=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 4),
    )

    return PortfolioResult(
        name="Black-Litterman",
        weights=weights,
        expected_return=ann_ret,
        expected_volatility=ann_vol,
        sharpe_ratio=sharpe,
        metadata={
            "n_views": len(Q),
            "view_confidences": confidences.tolist(),
            "onchain_signals": onchain_signals,
            "fallback": False,
        },
    )
