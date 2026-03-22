"""On-chain data from DeFiLlama.

CRITICAL: DeFiLlama subdomains -- mixing them returns 404:
  - TVL endpoints:         https://api.llama.fi/...
  - Stablecoin endpoints:  https://stablecoins.llama.fi/...
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd
import requests
import structlog

from core.config import settings
from core.data.cache import cache, cache_get_or_fetch, invalidate

log = structlog.get_logger(__name__)

DEFILLAMA_TVL_BASE = "https://api.llama.fi"
DEFILLAMA_STABLES_BASE = "https://stablecoins.llama.fi"

# Timeout for all HTTP requests (seconds)
_REQUEST_TIMEOUT = 30


@dataclass
class OnchainSignals:
    """Derived on-chain signals used for Black-Litterman views and regime detection."""

    tvl_momentum_30d: float
    stablecoin_dominance: float
    stablecoin_supply_change_30d: float
    dex_volume_trend_7d: float
    chain_tvl_shares: dict[str, float]
    as_of: str  # ISO string for serialization

    # Signal interpretations
    tvl_momentum_interpretation: str = "Neutral"
    stablecoin_dominance_interpretation: str = "Neutral"
    stablecoin_supply_interpretation: str = "Neutral"
    dex_volume_interpretation: str = "Neutral"


def _fetch_total_tvl() -> pd.Series:
    """GET https://api.llama.fi/v2/historicalChainTvl -> total TVL time series."""
    url = f"{DEFILLAMA_TVL_BASE}/v2/historicalChainTvl"
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    dates = [pd.Timestamp.utcfromtimestamp(d["date"]) for d in data]
    values = [d["tvl"] for d in data]
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name="total_tvl")
    series.index.name = "date"
    return series


def _fetch_chain_tvl(chain: str) -> pd.Series:
    """GET https://api.llama.fi/v2/historicalChainTvl/{chain} -> chain-specific TVL."""
    url = f"{DEFILLAMA_TVL_BASE}/v2/historicalChainTvl/{chain}"
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    dates = [pd.Timestamp.utcfromtimestamp(d["date"]) for d in data]
    values = [d["tvl"] for d in data]
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=f"{chain.lower()}_tvl")
    series.index.name = "date"
    return series


def _fetch_stablecoin_mcap() -> pd.Series:
    """GET https://stablecoins.llama.fi/stablecoincharts/all -> total stablecoin market cap.

    CRITICAL: Uses stablecoins.llama.fi, NOT api.llama.fi.
    Response is a list of {date, totalCirculating: {peggedUSD, peggedEUR, ...}}.
    We sum all pegs for total stablecoin market cap.
    """
    url = f"{DEFILLAMA_STABLES_BASE}/stablecoincharts/all"
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    dates = []
    values = []
    for entry in data:
        raw_date = entry["date"]
        ts = int(raw_date) if isinstance(raw_date, str) else raw_date
        dates.append(pd.Timestamp.utcfromtimestamp(ts))
        circulating = entry.get("totalCirculating", {})
        total = sum(circulating.values())
        values.append(total)

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name="stablecoin_mcap")
    series.index.name = "date"
    return series


def _fetch_dex_volume() -> pd.Series:
    """GET https://api.llama.fi/overview/dexs?excludeTotalDataChart=false -> daily DEX volume.

    Response has totalDataChart: [[timestamp, volume], ...].
    """
    url = f"{DEFILLAMA_TVL_BASE}/overview/dexs?excludeTotalDataChart=false"
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    chart_data = data.get("totalDataChart", [])
    if not chart_data:
        log.warning("dex_volume_empty_chart")
        return pd.Series(dtype=float, name="dex_volume")

    dates = [pd.Timestamp.utcfromtimestamp(row[0]) for row in chart_data]
    values = [row[1] for row in chart_data]
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name="dex_volume")
    series.index.name = "date"
    return series


def _do_fetch_all() -> dict[str, pd.Series]:
    """Fetch all DeFiLlama endpoints and return structured dict."""
    log.info("onchain_fetching_all")

    total_tvl = _fetch_total_tvl()
    log.info("onchain_fetched", endpoint="total_tvl", points=len(total_tvl))

    eth_tvl = _fetch_chain_tvl("Ethereum")
    log.info("onchain_fetched", endpoint="eth_tvl", points=len(eth_tvl))

    sol_tvl = _fetch_chain_tvl("Solana")
    log.info("onchain_fetched", endpoint="sol_tvl", points=len(sol_tvl))

    stablecoin_mcap = _fetch_stablecoin_mcap()
    log.info("onchain_fetched", endpoint="stablecoin_mcap", points=len(stablecoin_mcap))

    dex_volume = _fetch_dex_volume()
    log.info("onchain_fetched", endpoint="dex_volume", points=len(dex_volume))

    return {
        "total_tvl": total_tvl,
        "eth_tvl": eth_tvl,
        "sol_tvl": sol_tvl,
        "stablecoin_mcap": stablecoin_mcap,
        "dex_volume": dex_volume,
    }


def fetch_onchain_data(force_refresh: bool = False) -> dict[str, pd.Series]:
    """Fetch all DeFiLlama endpoints. Returns dict with keys:
    'total_tvl', 'eth_tvl', 'sol_tvl', 'stablecoin_mcap', 'dex_volume'.
    Cached for 6h.
    """
    if force_refresh:
        invalidate("onchain_data")

    return cache_get_or_fetch(
        key="onchain_data",
        fetch_fn=_do_fetch_all,
        ttl=settings.CACHE_TTL_ONCHAIN,
    )


def _interpret_tvl_momentum(value: float) -> str:
    """Interpret TVL momentum signal."""
    if value > 0.05:
        return "Bullish"
    if value < -0.05:
        return "Bearish"
    return "Neutral"


def _interpret_stablecoin_dominance(value: float) -> str:
    """High stablecoin dominance = risk-off (bearish); low = risk-on (bullish)."""
    if value > 0.15:
        return "Bearish"
    if value < 0.08:
        return "Bullish"
    return "Neutral"


def _interpret_stablecoin_supply(value: float) -> str:
    """Stablecoin supply growth = inflow (bullish)."""
    if value > 0.03:
        return "Bullish"
    if value < -0.03:
        return "Bearish"
    return "Neutral"


def _interpret_dex_volume(value: float) -> str:
    """DEX volume trend above 1 = increasing activity."""
    if value > 1.3:
        return "Bullish"
    if value < 0.7:
        return "Bearish"
    return "Neutral"


def compute_onchain_signals(onchain_data: dict[str, pd.Series]) -> OnchainSignals:
    """Derive the on-chain signals from fetched data. Pure function, no I/O.

    Args:
        onchain_data: Dict from fetch_onchain_data() with keys:
            total_tvl, eth_tvl, sol_tvl, stablecoin_mcap, dex_volume.

    Returns:
        OnchainSignals dataclass with derived values and interpretations.
    """
    total_tvl = onchain_data["total_tvl"]
    eth_tvl = onchain_data["eth_tvl"]
    sol_tvl = onchain_data["sol_tvl"]
    stablecoin_mcap = onchain_data["stablecoin_mcap"]
    dex_volume = onchain_data["dex_volume"]

    # TVL momentum: 30d change
    if len(total_tvl) >= 31 and total_tvl.iloc[-31] > 0:
        tvl_momentum_30d = (total_tvl.iloc[-1] / total_tvl.iloc[-31]) - 1
    else:
        tvl_momentum_30d = 0.0

    # Stablecoin dominance: stablecoin mcap / total TVL
    if len(stablecoin_mcap) > 0 and len(total_tvl) > 0 and total_tvl.iloc[-1] > 0:
        stablecoin_dominance = stablecoin_mcap.iloc[-1] / total_tvl.iloc[-1]
    else:
        stablecoin_dominance = 0.0

    # Stablecoin supply change: 30d
    if len(stablecoin_mcap) >= 31 and stablecoin_mcap.iloc[-31] > 0:
        stablecoin_supply_change_30d = (
            stablecoin_mcap.iloc[-1] / stablecoin_mcap.iloc[-31]
        ) - 1
    else:
        stablecoin_supply_change_30d = 0.0

    # DEX volume trend: 7d MA / 30d MA
    if len(dex_volume) >= 30:
        ma_7 = dex_volume.iloc[-7:].mean()
        ma_30 = dex_volume.iloc[-30:].mean()
        dex_volume_trend_7d = ma_7 / ma_30 if ma_30 > 0 else 1.0
    else:
        dex_volume_trend_7d = 1.0

    # Chain TVL shares
    chain_tvl_shares = {}
    if len(total_tvl) > 0 and total_tvl.iloc[-1] > 0:
        current_total = total_tvl.iloc[-1]
        if len(eth_tvl) > 0:
            chain_tvl_shares["Ethereum"] = eth_tvl.iloc[-1] / current_total
        if len(sol_tvl) > 0:
            chain_tvl_shares["Solana"] = sol_tvl.iloc[-1] / current_total

    as_of = str(total_tvl.index[-1]) if len(total_tvl) > 0 else ""

    signals = OnchainSignals(
        tvl_momentum_30d=tvl_momentum_30d,
        stablecoin_dominance=stablecoin_dominance,
        stablecoin_supply_change_30d=stablecoin_supply_change_30d,
        dex_volume_trend_7d=dex_volume_trend_7d,
        chain_tvl_shares=chain_tvl_shares,
        as_of=as_of,
        tvl_momentum_interpretation=_interpret_tvl_momentum(tvl_momentum_30d),
        stablecoin_dominance_interpretation=_interpret_stablecoin_dominance(stablecoin_dominance),
        stablecoin_supply_interpretation=_interpret_stablecoin_supply(stablecoin_supply_change_30d),
        dex_volume_interpretation=_interpret_dex_volume(dex_volume_trend_7d),
    )

    log.info(
        "onchain_signals_computed",
        tvl_momentum=round(tvl_momentum_30d, 4),
        stable_dom=round(stablecoin_dominance, 4),
        stable_change=round(stablecoin_supply_change_30d, 4),
        dex_trend=round(dex_volume_trend_7d, 4),
    )

    # Cache signals for other modules (e.g., BL optimization)
    cache.set("onchain_signals", asdict(signals), expire=settings.CACHE_TTL_ONCHAIN)

    return signals
