"""Top-50 CoinGecko universe with stablecoin filtering.

Fetches /coins/markets endpoint, filters out stablecoins, returns UniverseAsset list.
Caches result with CACHE_TTL_UNIVERSE TTL.
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

from core.config import settings
from core.data.cache import cache_get_or_fetch

log = structlog.get_logger(__name__)

STABLECOINS: frozenset[str] = frozenset({
    "tether", "usd-coin", "dai", "binance-usd", "trueusd",
    "first-digital-usd", "usdd", "frax", "paypal-usd",
})


@dataclass(frozen=True)
class UniverseAsset:
    """A single asset in the investable universe."""

    coingecko_id: str
    symbol: str
    name: str
    market_cap: float
    market_cap_rank: int
    current_price: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    volume_24h: float
    binance_symbol: str | None
    yfinance_ticker: str


def fetch_universe(force_refresh: bool = False) -> list[UniverseAsset]:
    """Fetch top-50 cryptos from CoinGecko, filter stablecoins. Cached for 4h."""
    raise NotImplementedError


def get_universe_from_cache() -> list[UniverseAsset] | None:
    """Read universe from cache without fetching. Returns None if stale."""
    raise NotImplementedError
