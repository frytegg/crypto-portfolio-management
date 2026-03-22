"""Tests for core/data/onchain.py — all HTTP requests are mocked.

Tests verify:
1. _fetch_total_tvl parses DeFiLlama TVL response correctly
2. _fetch_chain_tvl uses correct subdomain and parses chain data
3. _fetch_stablecoin_mcap uses stablecoins.llama.fi (NOT api.llama.fi)
4. _fetch_dex_volume parses totalDataChart from overview/dexs endpoint
5. compute_onchain_signals produces correct signal values
6. Signal interpretations are correct for bullish/neutral/bearish ranges
7. fetch_onchain_data caches with 6h TTL
8. Edge cases: empty data, short series, zero values
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import diskcache
import numpy as np
import pandas as pd
import pytest

from core.data.onchain import (
    OnchainSignals,
    _fetch_chain_tvl,
    _fetch_dex_volume,
    _fetch_stablecoin_mcap,
    _fetch_total_tvl,
    _interpret_dex_volume,
    _interpret_stablecoin_dominance,
    _interpret_stablecoin_supply,
    _interpret_tvl_momentum,
    compute_onchain_signals,
    fetch_onchain_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tvl_response():
    """Simulate DeFiLlama /v2/historicalChainTvl response."""
    base_ts = 1704067200  # 2024-01-01
    return [
        {"date": base_ts + i * 86400, "tvl": 50_000_000_000 + i * 100_000_000}
        for i in range(60)
    ]


@pytest.fixture
def mock_chain_tvl_response():
    """Simulate DeFiLlama /v2/historicalChainTvl/Ethereum response."""
    base_ts = 1704067200
    return [
        {"date": base_ts + i * 86400, "tvl": 25_000_000_000 + i * 50_000_000}
        for i in range(60)
    ]


@pytest.fixture
def mock_stablecoin_response():
    """Simulate stablecoins.llama.fi/stablecoincharts/all response."""
    base_ts = 1704067200
    return [
        {
            "date": base_ts + i * 86400,
            "totalCirculating": {
                "peggedUSD": 130_000_000_000 + i * 200_000_000,
                "peggedEUR": 500_000_000,
            },
        }
        for i in range(60)
    ]


@pytest.fixture
def mock_dex_response():
    """Simulate api.llama.fi/overview/dexs response."""
    base_ts = 1704067200
    return {
        "totalDataChart": [
            [base_ts + i * 86400, 2_000_000_000 + i * 10_000_000]
            for i in range(60)
        ],
    }


@pytest.fixture
def sample_onchain_data():
    """Pre-built onchain_data dict with pd.Series for signal computation tests."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    # TVL grows from 50B to 56B over 60 days (~12% growth over 30 days)
    total_tvl = pd.Series(
        np.linspace(50e9, 56e9, 60),
        index=dates,
        name="total_tvl",
    )
    eth_tvl = pd.Series(
        np.linspace(25e9, 28e9, 60),
        index=dates,
        name="eth_tvl",
    )
    sol_tvl = pd.Series(
        np.linspace(5e9, 6e9, 60),
        index=dates,
        name="sol_tvl",
    )
    # Stablecoin grows from 130B to 140B (~7.7% over 30d)
    stablecoin_mcap = pd.Series(
        np.linspace(130e9, 140e9, 60),
        index=dates,
        name="stablecoin_mcap",
    )
    # DEX volume: last 7d average should be higher than 30d for "surge"
    dex_base = np.full(60, 2e9)
    dex_base[-7:] = 3e9  # Last 7 days spike
    dex_volume = pd.Series(dex_base, index=dates, name="dex_volume")

    return {
        "total_tvl": total_tvl,
        "eth_tvl": eth_tvl,
        "sol_tvl": sol_tvl,
        "stablecoin_mcap": stablecoin_mcap,
        "dex_volume": dex_volume,
        "total_crypto_mcap": 2.5e12,  # ~$2.5T total crypto market cap
    }


# ---------------------------------------------------------------------------
# Tests: individual fetch functions
# ---------------------------------------------------------------------------

class TestFetchTotalTvl:
    def test_parses_response(self, mock_tvl_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_tvl_response
        mock_resp.raise_for_status = MagicMock()

        with patch("core.data.onchain.requests.get", return_value=mock_resp) as mock_get:
            result = _fetch_total_tvl()

            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert "api.llama.fi" in url
            assert "historicalChainTvl" in url

        assert isinstance(result, pd.Series)
        assert len(result) == 60
        assert result.name == "total_tvl"
        assert result.iloc[0] == 50_000_000_000

    def test_uses_correct_subdomain(self, mock_tvl_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_tvl_response
        mock_resp.raise_for_status = MagicMock()

        with patch("core.data.onchain.requests.get", return_value=mock_resp) as mock_get:
            _fetch_total_tvl()
            url = mock_get.call_args[0][0]
            assert url.startswith("https://api.llama.fi")
            assert "stablecoins" not in url


class TestFetchChainTvl:
    def test_parses_ethereum(self, mock_chain_tvl_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_chain_tvl_response
        mock_resp.raise_for_status = MagicMock()

        with patch("core.data.onchain.requests.get", return_value=mock_resp) as mock_get:
            result = _fetch_chain_tvl("Ethereum")
            url = mock_get.call_args[0][0]
            assert "Ethereum" in url

        assert isinstance(result, pd.Series)
        assert result.name == "ethereum_tvl"
        assert len(result) == 60


class TestFetchStablecoinMcap:
    def test_uses_stablecoins_subdomain(self, mock_stablecoin_response):
        """CRITICAL: must use stablecoins.llama.fi, NOT api.llama.fi."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_stablecoin_response
        mock_resp.raise_for_status = MagicMock()

        with patch("core.data.onchain.requests.get", return_value=mock_resp) as mock_get:
            result = _fetch_stablecoin_mcap()
            url = mock_get.call_args[0][0]
            assert "stablecoins.llama.fi" in url
            assert "api.llama.fi" not in url

        assert isinstance(result, pd.Series)
        assert result.name == "stablecoin_mcap"

    def test_sums_all_pegs(self, mock_stablecoin_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_stablecoin_response
        mock_resp.raise_for_status = MagicMock()

        with patch("core.data.onchain.requests.get", return_value=mock_resp):
            result = _fetch_stablecoin_mcap()

        # First entry: peggedUSD=130B + peggedEUR=500M
        assert result.iloc[0] == 130_000_000_000 + 500_000_000


class TestFetchDexVolume:
    def test_parses_total_data_chart(self, mock_dex_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_dex_response
        mock_resp.raise_for_status = MagicMock()

        with patch("core.data.onchain.requests.get", return_value=mock_resp) as mock_get:
            result = _fetch_dex_volume()
            url = mock_get.call_args[0][0]
            assert "overview/dexs" in url

        assert isinstance(result, pd.Series)
        assert result.name == "dex_volume"
        assert len(result) == 60

    def test_handles_empty_chart(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"totalDataChart": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("core.data.onchain.requests.get", return_value=mock_resp):
            result = _fetch_dex_volume()

        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: compute_onchain_signals
# ---------------------------------------------------------------------------

class TestComputeOnchainSignals:
    def test_computes_tvl_momentum(self, sample_onchain_data):
        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(sample_onchain_data)

        tvl = sample_onchain_data["total_tvl"]
        expected = (tvl.iloc[-1] / tvl.iloc[-31]) - 1
        assert abs(signals.tvl_momentum_30d - expected) < 1e-6

    def test_computes_stablecoin_dominance(self, sample_onchain_data):
        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(sample_onchain_data)

        # Denominator is total crypto market cap, not DeFi TVL
        expected = (
            sample_onchain_data["stablecoin_mcap"].iloc[-1]
            / sample_onchain_data["total_crypto_mcap"]
        )
        assert abs(signals.stablecoin_dominance - expected) < 1e-6
        # Sanity: with $140B stablecoins / $2.5T total = ~5.6%
        assert 0.01 < signals.stablecoin_dominance < 0.20

    def test_computes_stablecoin_supply_change(self, sample_onchain_data):
        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(sample_onchain_data)

        mcap = sample_onchain_data["stablecoin_mcap"]
        expected = (mcap.iloc[-1] / mcap.iloc[-31]) - 1
        assert abs(signals.stablecoin_supply_change_30d - expected) < 1e-6

    def test_computes_dex_volume_trend(self, sample_onchain_data):
        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(sample_onchain_data)

        dex = sample_onchain_data["dex_volume"]
        expected = dex.iloc[-7:].mean() / dex.iloc[-30:].mean()
        assert abs(signals.dex_volume_trend_7d - expected) < 1e-6

    def test_computes_chain_tvl_shares(self, sample_onchain_data):
        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(sample_onchain_data)

        total = sample_onchain_data["total_tvl"].iloc[-1]
        eth = sample_onchain_data["eth_tvl"].iloc[-1]
        sol = sample_onchain_data["sol_tvl"].iloc[-1]

        assert abs(signals.chain_tvl_shares["Ethereum"] - eth / total) < 1e-6
        assert abs(signals.chain_tvl_shares["Solana"] - sol / total) < 1e-6

    def test_caches_signals(self, sample_onchain_data):
        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(sample_onchain_data)

            mock_cache.set.assert_called_once()
            call_args = mock_cache.set.call_args
            assert call_args[0][0] == "onchain_signals"
            assert "tvl_momentum_30d" in call_args[0][1]


class TestComputeOnchainSignalsEdgeCases:
    def test_short_series_defaults(self):
        """If series have fewer than 31 points, should default to 0."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        short_data = {
            "total_tvl": pd.Series(np.full(10, 50e9), index=dates),
            "eth_tvl": pd.Series(np.full(10, 25e9), index=dates),
            "sol_tvl": pd.Series(np.full(10, 5e9), index=dates),
            "stablecoin_mcap": pd.Series(np.full(10, 130e9), index=dates),
            "dex_volume": pd.Series(np.full(10, 2e9), index=dates),
        }

        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(short_data)

        assert signals.tvl_momentum_30d == 0.0
        assert signals.stablecoin_supply_change_30d == 0.0
        assert signals.dex_volume_trend_7d == 1.0

    def test_zero_total_tvl(self):
        """If total TVL is zero, dominance and shares should default safely."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        zero_data = {
            "total_tvl": pd.Series(np.zeros(60), index=dates),
            "eth_tvl": pd.Series(np.zeros(60), index=dates),
            "sol_tvl": pd.Series(np.zeros(60), index=dates),
            "stablecoin_mcap": pd.Series(np.full(60, 130e9), index=dates),
            "dex_volume": pd.Series(np.full(60, 2e9), index=dates),
        }

        with patch("core.data.onchain.cache") as mock_cache:
            mock_cache.set = MagicMock()
            signals = compute_onchain_signals(zero_data)

        assert signals.stablecoin_dominance == 0.0
        assert signals.chain_tvl_shares == {}


# ---------------------------------------------------------------------------
# Tests: signal interpretations
# ---------------------------------------------------------------------------

class TestSignalInterpretations:
    def test_tvl_momentum_bullish(self):
        assert _interpret_tvl_momentum(0.10) == "Bullish"

    def test_tvl_momentum_neutral(self):
        assert _interpret_tvl_momentum(0.02) == "Neutral"

    def test_tvl_momentum_bearish(self):
        assert _interpret_tvl_momentum(-0.10) == "Bearish"

    def test_stablecoin_dominance_bearish(self):
        assert _interpret_stablecoin_dominance(0.20) == "Bearish"

    def test_stablecoin_dominance_neutral(self):
        assert _interpret_stablecoin_dominance(0.10) == "Neutral"

    def test_stablecoin_dominance_bullish(self):
        assert _interpret_stablecoin_dominance(0.05) == "Bullish"

    def test_stablecoin_supply_bullish(self):
        assert _interpret_stablecoin_supply(0.05) == "Bullish"

    def test_stablecoin_supply_neutral(self):
        assert _interpret_stablecoin_supply(0.01) == "Neutral"

    def test_stablecoin_supply_bearish(self):
        assert _interpret_stablecoin_supply(-0.05) == "Bearish"

    def test_dex_volume_bullish(self):
        assert _interpret_dex_volume(1.5) == "Bullish"

    def test_dex_volume_neutral(self):
        assert _interpret_dex_volume(1.0) == "Neutral"

    def test_dex_volume_bearish(self):
        assert _interpret_dex_volume(0.5) == "Bearish"

    def test_boundary_values(self):
        """Exact boundary values should be Neutral."""
        assert _interpret_tvl_momentum(0.05) == "Neutral"
        assert _interpret_tvl_momentum(-0.05) == "Neutral"
        assert _interpret_stablecoin_supply(0.03) == "Neutral"
        assert _interpret_stablecoin_supply(-0.03) == "Neutral"


# ---------------------------------------------------------------------------
# Tests: fetch_onchain_data with caching
# ---------------------------------------------------------------------------

class TestFetchOnchainDataCaching:
    def test_uses_cache(self):
        """fetch_onchain_data should use cache_get_or_fetch."""
        fake_data = {"total_tvl": pd.Series([1, 2, 3])}

        with patch("core.data.onchain.cache_get_or_fetch", return_value=fake_data) as mock:
            result = fetch_onchain_data()
            mock.assert_called_once()
            assert mock.call_args[1]["key"] == "onchain_data"
            assert result is fake_data

    def test_force_refresh_invalidates(self):
        """force_refresh=True should call invalidate before cache_get_or_fetch."""
        fake_data = {"total_tvl": pd.Series([1, 2, 3])}

        with (
            patch("core.data.onchain.cache_get_or_fetch", return_value=fake_data),
            patch("core.data.onchain.invalidate") as mock_invalidate,
        ):
            fetch_onchain_data(force_refresh=True)
            mock_invalidate.assert_called_once_with("onchain_data")
