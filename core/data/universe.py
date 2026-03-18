"""Top-50 CoinGecko universe with stablecoin filtering.

Fetches /coins/markets endpoint, filters out stablecoins, returns UniverseAsset list.
Caches result with CACHE_TTL_UNIVERSE TTL.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests
import structlog

from core.config import settings
from core.data.cache import cache, cache_get_or_fetch, invalidate
from core.data.symbol_map import get_binance_symbol, get_yfinance_ticker

log = structlog.get_logger(__name__)

_COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"

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
    """Fetch top-50 cryptos from CoinGecko, filter stablecoins. Cached for 4h.

    Raises:
        RuntimeError: If CoinGecko returns a non-200 status code.
    """
    if force_refresh:
        invalidate("universe")

    return cache_get_or_fetch(
        key="universe",
        fetch_fn=_fetch_from_coingecko,
        ttl=settings.CACHE_TTL_UNIVERSE,
    )


def get_universe_from_cache() -> list[UniverseAsset] | None:
    """Read universe from cache without fetching. Returns None if stale."""
    return cache.get("universe")


def _fetch_from_coingecko() -> list[UniverseAsset]:
    """Call CoinGecko /coins/markets and build UniverseAsset list."""
    headers: dict[str, str] = {}
    if settings.COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = settings.COINGECKO_API_KEY

    log.info("coingecko_fetch_start", url=_COINGECKO_MARKETS_URL)

    resp = requests.get(
        _COINGECKO_MARKETS_URL,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d,30d",
        },
        headers=headers,
        timeout=30,
    )

    if resp.status_code != 200:
        msg = (
            f"CoinGecko API returned status {resp.status_code}: "
            f"{resp.text[:200]}"
        )
        log.error("coingecko_fetch_failed", status=resp.status_code, body=resp.text[:200])
        raise RuntimeError(msg)

    raw_coins: list[dict] = resp.json()
    total_fetched = len(raw_coins)

    # Filter out stablecoins
    filtered = [coin for coin in raw_coins if coin["id"] not in STABLECOINS]
    stablecoins_removed = total_fetched - len(filtered)

    # Build UniverseAsset list
    universe: list[UniverseAsset] = []
    skipped_no_ticker = 0

    for coin in filtered:
        cg_id = coin["id"]
        yf_ticker = get_yfinance_ticker(cg_id)

        if yf_ticker is None:
            # Assets without a yfinance ticker can't be fetched — skip them
            # but still build a fallback ticker for resilience
            symbol = coin.get("symbol", "").upper()
            yf_ticker = f"{symbol}-USD"
            log.debug("unmapped_yfinance_ticker", coingecko_id=cg_id, fallback=yf_ticker)
            skipped_no_ticker += 1

        asset = UniverseAsset(
            coingecko_id=cg_id,
            symbol=coin.get("symbol", "").upper(),
            name=coin.get("name", ""),
            market_cap=float(coin.get("market_cap", 0) or 0),
            market_cap_rank=int(coin.get("market_cap_rank", 0) or 0),
            current_price=float(coin.get("current_price", 0) or 0),
            price_change_24h=float(coin.get("price_change_percentage_24h", 0) or 0),
            price_change_7d=float(coin.get("price_change_percentage_7d_in_currency", 0) or 0),
            price_change_30d=float(coin.get("price_change_percentage_30d_in_currency", 0) or 0),
            volume_24h=float(coin.get("total_volume", 0) or 0),
            binance_symbol=get_binance_symbol(cg_id),
            yfinance_ticker=yf_ticker,
        )
        universe.append(asset)

    log.info(
        "coingecko_fetch_complete",
        total_fetched=total_fetched,
        stablecoins_removed=stablecoins_removed,
        unmapped_tickers=skipped_no_ticker,
        universe_size=len(universe),
    )

    return universe
