"""Tests for BinancePriceFeed — no real network connections.

All WebSocket interactions are mocked. Tests verify:
1. Constructor filters None symbols
2. start() spawns a daemon thread
3. stop() signals the thread and closes the WebSocket
4. get_price() reads from cache
5. _on_message() parses combined stream envelopes correctly
6. _on_message() handles malformed messages without crashing
7. _on_error() logs without crashing
8. _on_open() sets connected flag
9. _on_close() clears connected flag
10. start() with no symbols does not spawn a thread
"""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import diskcache
import pytest

from core.data.price_feed import BinancePriceFeed


@pytest.fixture
def tmp_cache(tmp_path):
    """Create a temporary diskcache instance for testing."""
    return diskcache.Cache(str(tmp_path / "test_cache"), size_limit=10_000_000, timeout=1)


@pytest.fixture
def symbols():
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@pytest.fixture
def feed(tmp_cache, symbols):
    return BinancePriceFeed(tmp_cache, symbols)


class TestConstructor:
    """BinancePriceFeed.__init__"""

    def test_filters_none_symbols(self, tmp_cache):
        feed = BinancePriceFeed(tmp_cache, ["BTCUSDT", None, "ETHUSDT", None])
        assert feed._symbols == ["BTCUSDT", "ETHUSDT"]

    def test_stores_cache_reference(self, tmp_cache, symbols):
        feed = BinancePriceFeed(tmp_cache, symbols)
        assert feed._cache is tmp_cache

    def test_initial_state(self, feed):
        assert feed._thread is None
        assert not feed._stop_event.is_set()
        assert feed._connected is False
        assert feed._ws is None


class TestStartStop:
    """start() and stop() lifecycle."""

    def test_start_spawns_daemon_thread(self, feed):
        with patch.object(feed, "_run_forever_with_reconnect"):
            feed.start()
            assert feed._thread is not None
            assert feed._thread.daemon is True
            assert feed._thread.name == "binance-price-feed"
            feed.stop()

    def test_start_with_empty_symbols_does_not_spawn_thread(self, tmp_cache):
        feed = BinancePriceFeed(tmp_cache, [])
        feed.start()
        assert feed._thread is None

    def test_stop_sets_event(self, feed):
        with patch.object(feed, "_run_forever_with_reconnect"):
            feed.start()
            feed.stop()
            assert feed._stop_event.is_set()

    def test_stop_closes_ws(self, feed):
        mock_ws = MagicMock()
        feed._ws = mock_ws
        feed._stop_event.set()
        feed.stop()
        mock_ws.close.assert_called_once()

    def test_stop_handles_ws_close_error(self, feed):
        """stop() should not crash if ws.close() raises."""
        mock_ws = MagicMock()
        mock_ws.close.side_effect = Exception("Connection already closed")
        feed._ws = mock_ws
        feed._stop_event.set()
        feed.stop()  # Should not raise


class TestGetPrice:
    """get_price() reads from cache."""

    def test_returns_cached_price(self, feed, tmp_cache):
        tmp_cache.set("price:BTCUSDT", 42000.50, expire=60)
        assert feed.get_price("BTCUSDT") == 42000.50

    def test_returns_none_for_missing(self, feed):
        assert feed.get_price("NONEXISTENT") is None

    def test_returns_none_after_expiry(self, feed, tmp_cache):
        tmp_cache.set("price:BTCUSDT", 42000.0, expire=0.01)
        import time
        time.sleep(0.05)
        assert feed.get_price("BTCUSDT") is None


class TestOnMessage:
    """_on_message() parsing combined stream miniTicker envelopes."""

    def test_parses_valid_message(self, feed, tmp_cache):
        msg = json.dumps({
            "stream": "btcusdt@miniTicker",
            "data": {
                "e": "24hrMiniTicker",
                "E": 1234567890,
                "s": "BTCUSDT",
                "c": "42150.75",
                "o": "41000.00",
                "h": "42500.00",
                "l": "40800.00",
                "v": "12345.678",
                "q": "500000000.00",
            },
        })
        feed._on_message(None, msg)
        assert tmp_cache.get("price:BTCUSDT") == 42150.75

    def test_parses_multiple_symbols(self, feed, tmp_cache):
        for symbol, price in [("BTCUSDT", "42000.0"), ("ETHUSDT", "2500.0")]:
            msg = json.dumps({
                "stream": f"{symbol.lower()}@miniTicker",
                "data": {"s": symbol, "c": price},
            })
            feed._on_message(None, msg)

        assert tmp_cache.get("price:BTCUSDT") == 42000.0
        assert tmp_cache.get("price:ETHUSDT") == 2500.0

    def test_handles_malformed_json(self, feed, tmp_cache):
        """Should not crash on invalid JSON."""
        feed._on_message(None, "not json at all")
        # No cache entries should be written
        assert tmp_cache.get("price:BTCUSDT") is None

    def test_handles_missing_data_key(self, feed, tmp_cache):
        """Should not crash when 'data' key is missing."""
        msg = json.dumps({"stream": "btcusdt@miniTicker"})
        feed._on_message(None, msg)

    def test_handles_missing_symbol_key(self, feed, tmp_cache):
        """Should not crash when 's' key is missing from data."""
        msg = json.dumps({"data": {"c": "42000.0"}})
        feed._on_message(None, msg)

    def test_handles_invalid_price(self, feed, tmp_cache):
        """Should not crash when price is not a valid float."""
        msg = json.dumps({"data": {"s": "BTCUSDT", "c": "not_a_number"}})
        feed._on_message(None, msg)
        assert tmp_cache.get("price:BTCUSDT") is None


class TestCallbacks:
    """on_error, on_close, on_open callbacks."""

    def test_on_error_does_not_crash(self, feed):
        feed._on_error(None, Exception("test error"))

    def test_on_open_sets_connected(self, feed):
        assert feed._connected is False
        feed._on_open(None)
        assert feed._connected is True

    def test_on_close_clears_connected(self, feed):
        feed._connected = True
        feed._on_close(None, 1000, "normal")
        assert feed._connected is False

    def test_on_close_handles_none_params(self, feed):
        feed._connected = True
        feed._on_close(None)
        assert feed._connected is False


class TestReconnectLoop:
    """_run_forever_with_reconnect builds correct URL and reconnects."""

    def test_builds_correct_stream_url(self, feed):
        """Verify the combined stream URL format."""
        expected_streams = "btcusdt@miniTicker/ethusdt@miniTicker/solusdt@miniTicker"
        expected_url = f"wss://stream.binance.com:9443/stream?streams={expected_streams}"

        with patch("core.data.price_feed.websocket.WebSocketApp") as mock_ws_cls:
            mock_ws_instance = MagicMock()
            mock_ws_instance.run_forever.side_effect = lambda **kwargs: feed._stop_event.set()
            mock_ws_cls.return_value = mock_ws_instance

            feed._run_forever_with_reconnect()

            mock_ws_cls.assert_called_once()
            call_args = mock_ws_cls.call_args
            assert call_args[0][0] == expected_url

    def test_reconnects_on_disconnect(self, feed):
        """Verify reconnect loop retries after run_forever exits."""
        call_count = 0

        def fake_run_forever(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                feed._stop_event.set()

        with patch("core.data.price_feed.websocket.WebSocketApp") as mock_ws_cls:
            mock_ws_instance = MagicMock()
            mock_ws_instance.run_forever.side_effect = fake_run_forever
            mock_ws_cls.return_value = mock_ws_instance

            feed._run_forever_with_reconnect()

            assert call_count == 2

    def test_stops_when_event_set(self, feed):
        """Verify loop exits when stop event is set."""
        feed._stop_event.set()

        with patch("core.data.price_feed.websocket.WebSocketApp") as mock_ws_cls:
            feed._run_forever_with_reconnect()
            mock_ws_cls.assert_not_called()

    def test_handles_run_forever_exception(self, feed):
        """Verify loop continues after run_forever raises."""
        call_count = 0

        def exploding_run_forever(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network down")
            feed._stop_event.set()

        with patch("core.data.price_feed.websocket.WebSocketApp") as mock_ws_cls:
            mock_ws_instance = MagicMock()
            mock_ws_instance.run_forever.side_effect = exploding_run_forever
            mock_ws_cls.return_value = mock_ws_instance

            feed._run_forever_with_reconnect()

            assert call_count == 2
