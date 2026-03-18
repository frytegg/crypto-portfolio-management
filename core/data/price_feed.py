"""Binance WebSocket live price feed.

Streams miniTicker data via combined stream. Writes to diskcache with 60s TTL.
CRITICAL: 24h hard disconnect is mandatory -- reconnect loop is not optional.
"""
from __future__ import annotations

import threading

import diskcache
import structlog

log = structlog.get_logger(__name__)


class BinancePriceFeed:
    """Daemon thread streaming live prices via Binance WebSocket miniTicker."""

    def __init__(self, cache: diskcache.Cache, symbols: list[str]) -> None:
        self._cache = cache
        self._symbols = [s for s in symbols if s is not None]
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Spawn daemon thread. Call once at app startup."""
        raise NotImplementedError

    def stop(self) -> None:
        """Signal thread to stop and close WebSocket cleanly."""
        raise NotImplementedError

    def get_price(self, symbol: str) -> float | None:
        """Read latest price from cache. None if stale (>60s) or missing."""
        raise NotImplementedError

    def _run_forever_with_reconnect(self) -> None:
        """Internal reconnect loop with exponential backoff + 23.5h proactive reconnect."""
        raise NotImplementedError

    def _on_message(self, ws: object, message: str) -> None:
        """Parse combined stream envelope, write to cache."""
        raise NotImplementedError

    def _on_error(self, ws: object, error: Exception) -> None:
        """Log at WARN level, let run_forever exit to trigger reconnect."""
        raise NotImplementedError

    def _on_open(self, ws: object) -> None:
        """Log connection established, reset backoff counter."""
        raise NotImplementedError
