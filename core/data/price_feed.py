"""Binance WebSocket live price feed.

Streams miniTicker data via combined stream. Writes to diskcache with 60s TTL.
CRITICAL: 24h hard disconnect is mandatory -- reconnect loop is not optional.
"""
from __future__ import annotations

import json
import threading

import diskcache
import structlog
import websocket

log = structlog.get_logger(__name__)


class BinancePriceFeed:
    """Daemon thread streaming live prices via Binance WebSocket miniTicker.

    Usage:
        feed = BinancePriceFeed(cache, ["BTCUSDT", "ETHUSDT"])
        feed.start()   # starts daemon thread
        feed.stop()    # signals thread to stop
        feed.get_price("BTCUSDT")  # -> float | None
    """

    def __init__(self, cache: diskcache.Cache, symbols: list[str]) -> None:
        self._cache = cache
        self._symbols = [s for s in symbols if s is not None]
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._connected = False
        self._ws: websocket.WebSocketApp | None = None

    def start(self) -> None:
        """Spawn daemon thread. Call once at app startup."""
        if not self._symbols:
            log.warning("price_feed_no_symbols", msg="No symbols to stream")
            return

        self._thread = threading.Thread(
            target=self._run_forever_with_reconnect,
            name="binance-price-feed",
            daemon=True,
        )
        self._thread.start()
        log.info(
            "price_feed_started",
            n_symbols=len(self._symbols),
            symbols=self._symbols[:5],
        )

    def stop(self) -> None:
        """Signal thread to stop and close WebSocket cleanly."""
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        log.info("price_feed_stopped")

    def get_price(self, symbol: str) -> float | None:
        """Read latest price from cache. None if stale (>60s) or missing."""
        return self._cache.get(f"price:{symbol}")

    def _run_forever_with_reconnect(self) -> None:
        """Internal reconnect loop with exponential backoff.

        Binance hard-disconnects WebSocket connections after exactly 24 hours.
        This loop ensures automatic reconnection with exponential backoff.
        """
        attempt = 0
        while not self._stop_event.is_set():
            streams = "/".join(
                f"{sym.lower()}@miniTicker" for sym in self._symbols
            )
            url = f"wss://stream.binance.com:9443/stream?streams={streams}"

            self._ws = websocket.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open,
            )

            try:
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                log.warning("ws_run_forever_error", error=str(e))

            if self._stop_event.is_set():
                break

            wait = min(2 ** attempt, 60)
            log.warning(
                "ws_reconnecting",
                wait_seconds=wait,
                attempt=attempt + 1,
            )
            # Use stop_event.wait() instead of time.sleep() so stop() unblocks immediately
            if self._stop_event.wait(timeout=wait):
                break
            attempt += 1

    def _on_message(self, ws: object, message: str) -> None:
        """Parse combined stream envelope, write to cache."""
        try:
            envelope = json.loads(message)
            data = envelope["data"]
            symbol = data["s"]       # e.g. "BTCUSDT"
            price = float(data["c"]) # close price
            self._cache.set(f"price:{symbol}", price, expire=60)
        except (KeyError, ValueError, TypeError) as e:
            log.debug("ws_message_parse_error", error=str(e))

    def _on_error(self, ws: object, error: Exception) -> None:
        """Log at ERROR level, let run_forever exit to trigger reconnect.

        If Binance returns HTTP 451 (geo-restricted), stop permanently instead
        of retrying — the hosting region is blocked and reconnection will never
        succeed.
        """
        error_str = str(error)
        if "451" in error_str or "restricted location" in error_str.lower():
            log.warning(
                "ws_geo_restricted",
                msg="Binance WebSocket unavailable from this hosting region (HTTP 451). Live prices disabled.",
            )
            self._stop_event.set()
            return
        log.error("ws_error", error=error_str)

    def _on_close(self, ws: object, close_status_code: int | None = None, close_msg: str | None = None) -> None:
        """Log disconnect, set connected flag to False."""
        self._connected = False
        self._cache.set("ws_connected", False)

        close_msg_str = str(close_msg or "")
        if close_status_code == 451 or "451" in close_msg_str or "restricted" in close_msg_str.lower():
            log.warning(
                "ws_geo_restricted",
                msg="Binance WebSocket unavailable from this hosting region (HTTP 451). Live prices disabled.",
            )
            self._stop_event.set()
            return

        log.info(
            "ws_disconnected",
            status_code=close_status_code,
            message=close_msg,
        )

    def _on_open(self, ws: object) -> None:
        """Log connection established, reset backoff counter."""
        self._connected = True
        self._cache.set("ws_connected", True)
        log.info("ws_connected", n_streams=len(self._symbols))
