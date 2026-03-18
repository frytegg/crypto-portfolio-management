"""Tests for core/data/universe.py — CoinGecko universe builder.

All tests mock requests.get. No network calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.data.universe import STABLECOINS, UniverseAsset, fetch_universe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coingecko_coin(
    cg_id: str,
    symbol: str,
    name: str,
    rank: int,
    price: float = 100.0,
    market_cap: float = 1e10,
) -> dict:
    """Build a fake CoinGecko /coins/markets response item."""
    return {
        "id": cg_id,
        "symbol": symbol,
        "name": name,
        "current_price": price,
        "market_cap": market_cap,
        "market_cap_rank": rank,
        "price_change_percentage_24h": 1.5,
        "price_change_percentage_7d_in_currency": 3.2,
        "price_change_percentage_30d_in_currency": -5.1,
        "total_volume": 5e9,
        "high_24h": price * 1.02,
        "low_24h": price * 0.98,
    }


def _make_fake_response(n_real: int = 8, n_stablecoins: int = 2) -> list[dict]:
    """Build a fake CoinGecko response with n_real assets + n_stablecoins.

    Real assets use IDs from the symbol_map so binance_symbol and yfinance_ticker
    will be resolved correctly.
    """
    real_assets = [
        ("bitcoin", "btc", "Bitcoin", 1, 42000.0, 8e11),
        ("ethereum", "eth", "Ethereum", 2, 2500.0, 3e11),
        ("binancecoin", "bnb", "BNB", 3, 300.0, 4.5e10),
        ("solana", "sol", "Solana", 4, 100.0, 4e10),
        ("ripple", "xrp", "XRP", 5, 0.5, 2.5e10),
        ("cardano", "ada", "Cardano", 6, 0.35, 1.2e10),
        ("polkadot", "dot", "Polkadot", 7, 7.0, 8e9),
        ("chainlink", "link", "Chainlink", 8, 15.0, 7e9),
        ("avalanche-2", "avax", "Avalanche", 9, 35.0, 6e9),
        ("uniswap", "uni", "Uniswap", 10, 8.0, 5e9),
    ]

    stablecoin_assets = [
        ("tether", "usdt", "Tether", 3, 1.0, 9e10),
        ("usd-coin", "usdc", "USD Coin", 6, 1.0, 3e10),
        ("dai", "dai", "Dai", 20, 1.0, 5e9),
    ]

    coins = []
    for i in range(min(n_real, len(real_assets))):
        cg_id, sym, name, rank, price, mcap = real_assets[i]
        coins.append(_make_coingecko_coin(cg_id, sym, name, rank, price, mcap))

    for i in range(min(n_stablecoins, len(stablecoin_assets))):
        cg_id, sym, name, rank, price, mcap = stablecoin_assets[i]
        coins.append(_make_coingecko_coin(cg_id, sym, name, rank, price, mcap))

    return coins


# ---------------------------------------------------------------------------
# Tests for fetch_universe
# ---------------------------------------------------------------------------

class TestFetchUniverse:
    """Tests for the main fetch_universe function."""

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_filters_stablecoins(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """Stablecoins are removed from the universe. 10 coins - 2 stables = 8."""
        # Make cache_get_or_fetch call the fetch_fn directly
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        fake_response = _make_fake_response(n_real=8, n_stablecoins=2)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_get.return_value = mock_resp

        result = fetch_universe()

        assert len(result) == 8
        for asset in result:
            assert asset.coingecko_id not in STABLECOINS

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_all_fields_populated(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """Every UniverseAsset field is populated with a non-default value."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        fake_response = _make_fake_response(n_real=3, n_stablecoins=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_get.return_value = mock_resp

        result = fetch_universe()

        assert len(result) == 3
        for asset in result:
            assert isinstance(asset, UniverseAsset)
            assert asset.coingecko_id != ""
            assert asset.symbol != ""
            assert asset.name != ""
            assert asset.market_cap > 0
            assert asset.market_cap_rank > 0
            assert asset.current_price > 0
            assert asset.volume_24h > 0
            assert asset.yfinance_ticker != ""
            # price changes can be negative, just check they're set
            assert isinstance(asset.price_change_24h, float)
            assert isinstance(asset.price_change_7d, float)
            assert isinstance(asset.price_change_30d, float)

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_symbol_is_uppercase(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """Asset symbols are uppercase."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        fake_response = _make_fake_response(n_real=5, n_stablecoins=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_get.return_value = mock_resp

        result = fetch_universe()

        for asset in result:
            assert asset.symbol == asset.symbol.upper()

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_binance_symbol_resolved(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """Known assets have binance_symbol resolved from symbol_map."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        fake_response = _make_fake_response(n_real=3, n_stablecoins=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_get.return_value = mock_resp

        result = fetch_universe()

        btc = next(a for a in result if a.coingecko_id == "bitcoin")
        assert btc.binance_symbol == "BTCUSDT"

        eth = next(a for a in result if a.coingecko_id == "ethereum")
        assert eth.binance_symbol == "ETHUSDT"

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_yfinance_ticker_resolved(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """Known assets have yfinance_ticker resolved from symbol_map."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        fake_response = _make_fake_response(n_real=5, n_stablecoins=0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_get.return_value = mock_resp

        result = fetch_universe()

        btc = next(a for a in result if a.coingecko_id == "bitcoin")
        assert btc.yfinance_ticker == "BTC-USD"

        sol = next(a for a in result if a.coingecko_id == "solana")
        assert sol.yfinance_ticker == "SOL-USD"

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_raises_on_non_200(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """RuntimeError raised when CoinGecko returns non-200."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"
        mock_get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="CoinGecko API returned status 429"):
            fetch_universe()

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_raises_on_server_error(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """RuntimeError raised on 500 server error."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        with pytest.raises(RuntimeError, match="CoinGecko API returned status 500"):
            fetch_universe()

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_handles_null_fields_gracefully(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """Null/missing fields in CoinGecko response don't crash — default to 0."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        # CoinGecko sometimes returns null for price_change fields
        coin = _make_coingecko_coin("bitcoin", "btc", "Bitcoin", 1)
        coin["price_change_percentage_24h"] = None
        coin["price_change_percentage_7d_in_currency"] = None
        coin["price_change_percentage_30d_in_currency"] = None
        coin["total_volume"] = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [coin]
        mock_get.return_value = mock_resp

        result = fetch_universe()

        assert len(result) == 1
        assert result[0].price_change_24h == 0.0
        assert result[0].price_change_7d == 0.0
        assert result[0].price_change_30d == 0.0
        assert result[0].volume_24h == 0.0

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_correct_api_params(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """CoinGecko API is called with the exact params from the spec."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        fetch_universe()

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs["params"]
        assert params["vs_currency"] == "usd"
        assert params["order"] == "market_cap_desc"
        assert params["per_page"] == 50
        assert params["page"] == 1
        assert params["sparkline"] == "false"
        assert params["price_change_percentage"] == "24h,7d,30d"

    @patch("core.data.universe.cache_get_or_fetch")
    @patch("core.data.universe.requests.get")
    def test_unmapped_asset_gets_fallback_ticker(
        self, mock_get: MagicMock, mock_cache: MagicMock
    ) -> None:
        """Assets not in symbol_map get a fallback {SYMBOL}-USD yfinance ticker."""
        mock_cache.side_effect = lambda key, fetch_fn, ttl: fetch_fn()

        # Use a CoinGecko ID that's NOT in the symbol_map
        coin = _make_coingecko_coin("unknown-coin-xyz", "xyz", "Unknown Coin", 50)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [coin]
        mock_get.return_value = mock_resp

        result = fetch_universe()

        assert len(result) == 1
        assert result[0].yfinance_ticker == "XYZ-USD"
        assert result[0].binance_symbol is None


# ---------------------------------------------------------------------------
# Tests for get_universe_from_cache
# ---------------------------------------------------------------------------

class TestGetUniverseFromCache:
    """Tests for the cache-only reader."""

    @patch("core.data.universe.cache")
    def test_returns_none_on_miss(self, mock_cache: MagicMock) -> None:
        """Returns None when universe is not in cache."""
        from core.data.universe import get_universe_from_cache

        mock_cache.get.return_value = None
        result = get_universe_from_cache()
        assert result is None

    @patch("core.data.universe.cache")
    def test_returns_cached_value(self, mock_cache: MagicMock) -> None:
        """Returns cached universe list when present."""
        from core.data.universe import get_universe_from_cache

        fake_universe = [
            UniverseAsset(
                coingecko_id="bitcoin", symbol="BTC", name="Bitcoin",
                market_cap=8e11, market_cap_rank=1, current_price=42000.0,
                price_change_24h=1.5, price_change_7d=3.2, price_change_30d=-5.1,
                volume_24h=3e10, binance_symbol="BTCUSDT", yfinance_ticker="BTC-USD",
            )
        ]
        mock_cache.get.return_value = fake_universe

        result = get_universe_from_cache()
        assert result == fake_universe


# ---------------------------------------------------------------------------
# Tests for force_refresh
# ---------------------------------------------------------------------------

class TestForceRefresh:
    """Tests for force_refresh behavior."""

    @patch("core.data.universe.invalidate")
    @patch("core.data.universe.cache_get_or_fetch")
    def test_invalidates_cache_on_force_refresh(
        self, mock_cache: MagicMock, mock_invalidate: MagicMock
    ) -> None:
        """force_refresh=True calls invalidate('universe') before fetching."""
        mock_cache.return_value = []

        fetch_universe(force_refresh=True)

        mock_invalidate.assert_called_once_with("universe")

    @patch("core.data.universe.invalidate")
    @patch("core.data.universe.cache_get_or_fetch")
    def test_no_invalidate_without_force(
        self, mock_cache: MagicMock, mock_invalidate: MagicMock
    ) -> None:
        """Normal call does not invalidate cache."""
        mock_cache.return_value = []

        fetch_universe(force_refresh=False)

        mock_invalidate.assert_not_called()
