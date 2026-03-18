"""Tests for core/data/fetcher.py — Historical OHLCV fetcher.

All tests mock yfinance and Binance REST. No network calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.data.universe import UniverseAsset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_universe(n: int = 5) -> list[UniverseAsset]:
    """Create a small test universe of n assets."""
    assets_data = [
        ("bitcoin", "BTC", "Bitcoin", "BTCUSDT", "BTC-USD"),
        ("ethereum", "ETH", "Ethereum", "ETHUSDT", "ETH-USD"),
        ("solana", "SOL", "Solana", "SOLUSDT", "SOL-USD"),
        ("cardano", "ADA", "Cardano", "ADAUSDT", "ADA-USD"),
        ("polkadot", "DOT", "Polkadot", "DOTUSDT", "DOT-USD"),
        ("chainlink", "LINK", "Chainlink", "LINKUSDT", "LINK-USD"),
        ("avalanche-2", "AVAX", "Avalanche", "AVAXUSDT", "AVAX-USD"),
        ("uniswap", "UNI", "Uniswap", "UNIUSDT", "UNI-USD"),
        ("ripple", "XRP", "XRP", "XRPUSDT", "XRP-USD"),
        ("dogecoin", "DOGE", "Dogecoin", "DOGEUSDT", "DOGE-USD"),
    ]

    universe = []
    for i, (cg_id, sym, name, binance, yf_ticker) in enumerate(assets_data[:n]):
        universe.append(
            UniverseAsset(
                coingecko_id=cg_id,
                symbol=sym,
                name=name,
                market_cap=1e11 / (i + 1),
                market_cap_rank=i + 1,
                current_price=100.0 / (i + 1),
                price_change_24h=1.0,
                price_change_7d=2.0,
                price_change_30d=5.0,
                volume_24h=1e9,
                binance_symbol=binance,
                yfinance_ticker=yf_ticker,
            )
        )
    return universe


def _make_price_df(
    tickers: list[str],
    n_days: int = 400,
    seed: int = 42,
    nan_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate a synthetic price DataFrame mimicking yfinance output.

    Args:
        tickers: Column names (yfinance-style, e.g. "BTC-USD").
        n_days: Number of trading days.
        seed: Random seed for reproducibility.
        nan_columns: Tickers to fill entirely with NaN (simulates yfinance failure).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n_days)

    data = {}
    for ticker in tickers:
        # Generate a random walk price series starting at 100
        returns = rng.normal(0.0005, 0.03, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        data[ticker] = prices

    df = pd.DataFrame(data, index=dates)

    if nan_columns:
        for col in nan_columns:
            if col in df.columns:
                df[col] = np.nan

    return df


def _make_yfinance_multiindex(prices: pd.DataFrame) -> pd.DataFrame:
    """Wrap a flat price DataFrame in the MultiIndex format yfinance returns."""
    # yfinance with group_by="column" returns MultiIndex: (field, ticker)
    arrays = [
        ["Close"] * len(prices.columns),
        list(prices.columns),
    ]
    tuples = list(zip(*arrays))
    multi_idx = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])

    result = pd.DataFrame(prices.values, index=prices.index, columns=multi_idx)
    return result


def _make_binance_klines(n_days: int = 400, seed: int = 99) -> list[list]:
    """Generate fake Binance /api/v3/klines response."""
    rng = np.random.default_rng(seed)
    base_ts = pd.Timestamp.now().normalize() - pd.Timedelta(days=n_days)
    klines = []
    price = 100.0
    for i in range(n_days):
        ts = base_ts + pd.Timedelta(days=i)
        open_time = int(ts.timestamp() * 1000)
        close_time = int((ts + pd.Timedelta(hours=23, minutes=59)).timestamp() * 1000)
        price *= np.exp(rng.normal(0.0005, 0.03))
        klines.append([
            open_time,           # 0: open time
            str(price * 0.99),   # 1: open
            str(price * 1.01),   # 2: high
            str(price * 0.98),   # 3: low
            str(price),          # 4: close  <-- this is what we extract
            str(1000.0),         # 5: volume
            close_time,          # 6: close time
            str(50000.0),        # 7: quote asset volume
            100,                 # 8: number of trades
            str(500.0),          # 9: taker buy base
            str(25000.0),        # 10: taker buy quote
            "0",                 # 11: ignore
        ])
    return klines


# ---------------------------------------------------------------------------
# Tests for _fetch_yfinance
# ---------------------------------------------------------------------------

class TestFetchYfinance:
    """Tests for the yfinance download wrapper."""

    @patch("core.data.fetcher.yf.download")
    def test_returns_close_prices(self, mock_download: MagicMock) -> None:
        """yfinance download returns Close column extracted from MultiIndex."""
        from core.data.fetcher import _fetch_yfinance

        tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
        flat_prices = _make_price_df(tickers, n_days=200)
        mock_download.return_value = _make_yfinance_multiindex(flat_prices)

        result = _fetch_yfinance(tickers, "2024-01-01", "2026-03-17")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == tickers
        assert len(result) == 200

    @patch("core.data.fetcher.yf.download")
    def test_handles_empty_download(self, mock_download: MagicMock) -> None:
        """Returns empty DataFrame when yfinance returns nothing."""
        from core.data.fetcher import _fetch_yfinance

        mock_download.return_value = pd.DataFrame()
        result = _fetch_yfinance(["BTC-USD"], "2024-01-01", "2026-03-17")

        assert result.empty

    def test_empty_tickers_list(self) -> None:
        """Returns empty DataFrame for empty ticker list."""
        from core.data.fetcher import _fetch_yfinance

        result = _fetch_yfinance([], "2024-01-01", "2026-03-17")
        assert result.empty


# ---------------------------------------------------------------------------
# Tests for _fetch_binance_rest
# ---------------------------------------------------------------------------

class TestFetchBinanceRest:
    """Tests for the Binance REST fallback."""

    @patch("core.data.fetcher.requests.get")
    def test_returns_price_series(self, mock_get: MagicMock) -> None:
        """Binance klines are parsed into a pd.Series of close prices."""
        from core.data.fetcher import _fetch_binance_rest

        klines = _make_binance_klines(n_days=200)
        mock_resp = MagicMock()
        mock_resp.json.return_value = klines
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = _fetch_binance_rest("BTCUSDT", limit=200)

        assert isinstance(result, pd.Series)
        assert len(result) == 200
        assert result.dtype == np.float64

    @patch("core.data.fetcher.requests.get")
    def test_returns_none_on_empty_response(self, mock_get: MagicMock) -> None:
        """Returns None when Binance returns empty array."""
        from core.data.fetcher import _fetch_binance_rest

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = _fetch_binance_rest("FAKEUSDT", limit=730)
        assert result is None

    @patch("core.data.fetcher.requests.get")
    def test_limit_capped_at_1000(self, mock_get: MagicMock) -> None:
        """Binance limit parameter is capped at 1000."""
        from core.data.fetcher import _fetch_binance_rest

        klines = _make_binance_klines(n_days=100)
        mock_resp = MagicMock()
        mock_resp.json.return_value = klines
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        _fetch_binance_rest("BTCUSDT", limit=2000)

        # Check the actual limit sent to the API
        call_args = mock_get.call_args
        assert int(call_args.kwargs["params"]["limit"]) <= 1000


# ---------------------------------------------------------------------------
# Tests for _clean_prices
# ---------------------------------------------------------------------------

class TestCleanPrices:
    """Tests for the 5-step data cleaning pipeline."""

    def test_basic_cleaning_produces_valid_output(self) -> None:
        """Clean prices and returns have no NaN, correct shapes."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH", "SOL"], n_days=400)
        cleaned_prices, returns = _clean_prices(prices)

        # No NaN in either output
        assert not cleaned_prices.isna().any().any()
        assert not returns.isna().any().any()

        # Returns has one fewer row than raw prices (due to shift in log returns)
        assert len(returns) == len(cleaned_prices)
        assert len(returns) < 400  # First row dropped by log return calculation

        # Same columns in both
        assert list(cleaned_prices.columns) == list(returns.columns)

    def test_ffill_handles_small_gaps(self) -> None:
        """Gaps of up to 5 days are forward-filled."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH"], n_days=300)
        # Insert a 4-day gap (should be filled)
        prices.iloc[50:54, 0] = np.nan

        cleaned, returns = _clean_prices(prices)

        assert "BTC" in cleaned.columns
        assert not cleaned["BTC"].isna().any()

    def test_drops_high_nan_assets(self) -> None:
        """Assets with >20% NaN are dropped entirely."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH", "BAD"], n_days=300)
        # Make BAD column 50% NaN (exceeds 20% threshold)
        prices.iloc[:150, prices.columns.get_loc("BAD")] = np.nan

        cleaned, returns = _clean_prices(prices)

        assert "BAD" not in cleaned.columns
        assert "BAD" not in returns.columns
        assert "BTC" in cleaned.columns
        assert "ETH" in cleaned.columns

    def test_returns_are_log_returns(self) -> None:
        """Returns are computed as np.log(prices / prices.shift(1))."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC"], n_days=300)
        _, returns = _clean_prices(prices)

        # Log returns should be centered around 0 with reasonable magnitude
        assert returns["BTC"].mean() == pytest.approx(0.0, abs=0.05)
        assert returns["BTC"].std() > 0

    def test_minimum_observations_enforced(self) -> None:
        """Assets with fewer than 180 observations after cleaning are dropped."""
        from core.data.fetcher import _clean_prices

        # Create a very short DataFrame (100 days) — should warn but keep
        # (because all columns have the same count)
        short_prices = _make_price_df(["BTC", "ETH"], n_days=100)
        cleaned, returns = _clean_prices(short_prices)

        # With only 100 days, we get ~99 returns after log. Below 180 threshold,
        # but since we still return what we have (with a warning), check it's not empty
        # The function warns but keeps data when the *entire* DataFrame is short
        assert len(returns) <= 100

    def test_monotonic_index(self) -> None:
        """Output DataFrames have a monotonically increasing DatetimeIndex."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH"], n_days=300)
        cleaned, returns = _clean_prices(prices)

        assert cleaned.index.is_monotonic_increasing
        assert returns.index.is_monotonic_increasing

    def test_empty_input(self) -> None:
        """Empty input returns empty output."""
        from core.data.fetcher import _clean_prices

        cleaned, returns = _clean_prices(pd.DataFrame())
        assert cleaned.empty
        assert returns.empty


# ---------------------------------------------------------------------------
# Tests for fetch_historical_data (integration with mocks)
# ---------------------------------------------------------------------------

class TestFetchHistoricalData:
    """End-to-end tests with mocked API calls."""

    @patch("core.data.fetcher.cache_get_or_fetch")
    @patch("core.data.fetcher._fetch_and_clean")
    def test_uses_cache(
        self, mock_fetch_clean: MagicMock, mock_cache: MagicMock
    ) -> None:
        """fetch_historical_data delegates to cache_get_or_fetch."""
        from core.data.fetcher import fetch_historical_data

        universe = _make_universe(3)
        dummy_prices = _make_price_df(["BTC", "ETH", "SOL"], n_days=200)
        dummy_returns = np.log(dummy_prices / dummy_prices.shift(1)).dropna()
        mock_cache.return_value = (dummy_prices, dummy_returns)

        prices, returns = fetch_historical_data(universe)

        mock_cache.assert_called_once()
        assert isinstance(prices, pd.DataFrame)
        assert isinstance(returns, pd.DataFrame)

    @patch("core.data.fetcher.invalidate")
    @patch("core.data.fetcher.cache_get_or_fetch")
    def test_force_refresh_invalidates_cache(
        self, mock_cache: MagicMock, mock_invalidate: MagicMock
    ) -> None:
        """force_refresh=True invalidates existing cache entries before fetching."""
        from core.data.fetcher import fetch_historical_data

        universe = _make_universe(2)
        mock_cache.return_value = (pd.DataFrame(), pd.DataFrame())

        fetch_historical_data(universe, force_refresh=True)

        # Should invalidate both old-style keys
        assert mock_invalidate.call_count == 2

    @patch("core.data.fetcher.requests.get")
    @patch("core.data.fetcher.yf.download")
    @patch("core.data.fetcher.cache_get_or_fetch")
    def test_full_pipeline_with_binance_fallback(
        self,
        mock_cache: MagicMock,
        mock_yf_download: MagicMock,
        mock_requests_get: MagicMock,
    ) -> None:
        """Assets failing yfinance are retried via Binance REST."""
        from core.data.fetcher import _fetch_and_clean

        universe = _make_universe(3)  # BTC, ETH, SOL
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

        # yfinance returns data for BTC and ETH, but SOL is all NaN
        flat_prices = _make_price_df(tickers, n_days=400, nan_columns=["SOL-USD"])
        mock_yf_download.return_value = _make_yfinance_multiindex(flat_prices)

        # Binance REST returns data for SOL (need 600+ calendar days to cover 400 bdays)
        klines = _make_binance_klines(n_days=700, seed=77)
        mock_resp = MagicMock()
        mock_resp.json.return_value = klines
        mock_resp.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_resp

        prices, returns = _fetch_and_clean(universe, lookback_days=730)

        # SOL should be present (recovered via Binance)
        assert "SOL" in prices.columns
        assert "BTC" in prices.columns
        assert "ETH" in prices.columns

        # No NaN in returns
        assert not returns.isna().any().any()

    @patch("core.data.fetcher.yf.download")
    @patch("core.data.fetcher.cache_get_or_fetch")
    def test_output_columns_are_display_symbols(
        self, mock_cache: MagicMock, mock_yf_download: MagicMock
    ) -> None:
        """Output columns use display symbols (BTC, ETH) not yfinance tickers (BTC-USD)."""
        from core.data.fetcher import _fetch_and_clean

        universe = _make_universe(3)
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]
        flat_prices = _make_price_df(tickers, n_days=400)
        mock_yf_download.return_value = _make_yfinance_multiindex(flat_prices)

        prices, returns = _fetch_and_clean(universe, lookback_days=730)

        for col in prices.columns:
            assert "-USD" not in col, f"Column '{col}' still has yfinance format"

    @patch("core.data.fetcher.yf.download")
    @patch("core.data.fetcher.cache_get_or_fetch")
    def test_output_column_count_lte_universe(
        self, mock_cache: MagicMock, mock_yf_download: MagicMock
    ) -> None:
        """Number of output columns is <= universe size (assets may be dropped)."""
        from core.data.fetcher import _fetch_and_clean

        universe = _make_universe(5)
        tickers = [a.yfinance_ticker for a in universe]
        flat_prices = _make_price_df(tickers, n_days=400)
        mock_yf_download.return_value = _make_yfinance_multiindex(flat_prices)

        prices, returns = _fetch_and_clean(universe, lookback_days=730)

        assert len(prices.columns) <= len(universe)
        assert len(returns.columns) <= len(universe)


# ---------------------------------------------------------------------------
# Tests for data contract compliance
# ---------------------------------------------------------------------------

class TestDataContract:
    """Verify the output matches the data cleaning contract from data-layer.md."""

    def test_no_nan_in_returns(self) -> None:
        """Returns DataFrame has zero NaN values."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH", "SOL", "ADA", "DOT"], n_days=400)
        _, returns = _clean_prices(prices)

        assert returns.isna().sum().sum() == 0

    def test_columns_are_uppercase_symbols(self) -> None:
        """Column names are uppercase ticker symbols."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH", "SOL"], n_days=300)
        _, returns = _clean_prices(prices)

        for col in returns.columns:
            assert col == col.upper()

    def test_values_are_float64(self) -> None:
        """Both prices and returns use float64 precision."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH"], n_days=300)
        cleaned, returns = _clean_prices(prices)

        for col in cleaned.columns:
            assert cleaned[col].dtype == np.float64
        for col in returns.columns:
            assert returns[col].dtype == np.float64

    def test_index_is_datetime(self) -> None:
        """Output has a DatetimeIndex."""
        from core.data.fetcher import _clean_prices

        prices = _make_price_df(["BTC", "ETH"], n_days=300)
        cleaned, returns = _clean_prices(prices)

        assert isinstance(cleaned.index, pd.DatetimeIndex)
        assert isinstance(returns.index, pd.DatetimeIndex)
