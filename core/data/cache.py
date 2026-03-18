"""Disk-backed cache singleton using diskcache.

Thread-safe (SQLite WAL mode), process-safe, persistent across restarts.
All modules import `cache` from here — never create a second Cache instance.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import diskcache
import structlog

from core.config import settings

log = structlog.get_logger(__name__)

T = TypeVar("T")

# Module-level singleton — thread-safe, process-safe (SQLite ACID)
cache: diskcache.Cache = diskcache.Cache(
    settings.CACHE_DIR,
    size_limit=settings.CACHE_SIZE_LIMIT,
    disk_min_file_size=0,
    timeout=1,
)


def cache_get_or_fetch(key: str, fetch_fn: Callable[[], T], ttl: int) -> T:
    """Read from cache if fresh. Otherwise call fetch_fn(), store with TTL, return.

    Exceptions from fetch_fn propagate — never silently swallowed.
    """
    value = cache.get(key)
    if value is not None:
        log.debug("cache_hit", key=key)
        return value

    log.info("cache_miss", key=key, ttl=ttl)
    value = fetch_fn()
    cache.set(key, value, expire=ttl)
    return value


def invalidate(key: str) -> None:
    """Explicitly evict a key from cache."""
    cache.delete(key)
    log.info("cache_invalidated", key=key)


def get_live_price(symbol: str) -> float | None:
    """Read price:{symbol} from cache. Returns None if stale or missing."""
    return cache.get(f"price:{symbol}")


def set_live_price(symbol: str, price: float, ttl: int | None = None) -> None:
    """Write a live price to cache with TTL."""
    if ttl is None:
        ttl = settings.CACHE_TTL_LIVE_PRICE
    cache.set(f"price:{symbol}", price, expire=ttl)
