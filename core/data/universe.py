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

# Assets excluded from the investable universe:
# - Stablecoins (pegged to USD, no alpha)
# - Wrapped/tokenized assets (WBTC, gold tokens — track underlying, not independent)
# - Yield-bearing stablecoins / RWA tokens (price ~$1, not suitable for MVO)
# - Obscure/illiquid tokens that lack reliable price feeds
EXCLUDED_ASSETS: frozenset[str] = frozenset({
    # Stablecoins
    "tether", "usd-coin", "dai", "binance-usd", "trueusd",
    "first-digital-usd", "usdd", "frax", "paypal-usd", "usds",
    "global-dollar", "circle-usyc",
    # Wrapped / tokenized
    "wrapped-bitcoin", "tether-gold", "pax-gold",
    # RWA / illiquid / unsuitable for portfolio optimization
    "figr-heloc", "rain", "world-liberty-financial",
    "pi-network", "aster", "memecore", "canton-network",
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

    # Filter out excluded assets by ID
    filtered = [coin for coin in raw_coins if coin["id"] not in EXCLUDED_ASSETS]
    stablecoins_removed = total_fetched - len(filtered)

    # Secondary filter: catch unlisted stablecoins by price ~$1.00 AND "USD"/"usd" in name
    pre_count = len(filtered)
    keep: list[dict] = []
    for coin in filtered:
        price = float(coin.get("current_price", 0) or 0)
        name = coin.get("name", "")
        if 0.99 <= price <= 1.01 and ("USD" in name or "usd" in name):
            log.info(
                "stablecoin_heuristic_filtered",
                coingecko_id=coin["id"],
                name=name,
                price=price,
            )
        else:
            keep.append(coin)
    filtered = keep
    stablecoins_removed += pre_count - len(filtered)

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

    # Drop assets with no data source (no yfinance ticker AND no Binance symbol)
    pre_source_count = len(universe)
    final_universe: list[UniverseAsset] = []
    for asset in universe:
        yf_mapped = get_yfinance_ticker(asset.coingecko_id) is not None
        if not yf_mapped and asset.binance_symbol is None:
            log.info(
                "no_data_source_filtered",
                coingecko_id=asset.coingecko_id,
                name=asset.name,
                symbol=asset.symbol,
            )
        else:
            final_universe.append(asset)
    no_source_dropped = pre_source_count - len(final_universe)

    log.info(
        "coingecko_fetch_complete",
        total_fetched=total_fetched,
        excluded_removed=stablecoins_removed,
        no_source_dropped=no_source_dropped,
        unmapped_tickers=skipped_no_ticker,
        universe_size=len(final_universe),
    )

    return final_universe
