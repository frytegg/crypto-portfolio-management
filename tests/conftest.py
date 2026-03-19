"""Shared test fixtures for all test modules.

All fixtures are deterministic (fixed random seed). No network calls anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


ASSET_NAMES = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOT", "LINK", "AVAX", "UNI"]


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """252 trading days x 10 assets of synthetic log returns.

    Generated with fixed seed for reproducibility.
    Realistic crypto-like properties: high vol, mild correlation.
    Mean ~N(0.001, 0.02) daily.
    """
    rng = np.random.default_rng(42)
    n_days = 252
    n_assets = len(ASSET_NAMES)

    # Base correlation structure (crypto assets are moderately correlated)
    base_corr = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(base_corr, 1.0)

    # Asset-specific daily vols (annualized 40-80% -> daily ~2.5-5%)
    daily_vols = rng.uniform(0.025, 0.05, n_assets)
    cov = np.outer(daily_vols, daily_vols) * base_corr

    # Ensure PSD
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Generate returns with small positive drift
    mean_returns = rng.uniform(-0.0005, 0.002, n_assets)
    returns_data = rng.multivariate_normal(mean_returns, cov, size=n_days)

    dates = pd.bdate_range(start="2024-01-02", periods=n_days)
    return pd.DataFrame(returns_data, index=dates, columns=ASSET_NAMES)


@pytest.fixture
def large_returns(sample_returns: pd.DataFrame) -> pd.DataFrame:
    """730 trading days x 45 assets. Simulates full universe for optimization tests.

    Uses block bootstrap from sample_returns to generate realistic autocorrelation.
    """
    rng = np.random.default_rng(123)
    n_days = 730
    n_extra_assets = 35
    block_size = 10

    # Generate extra assets with similar properties
    extra_names = [f"ASSET_{i}" for i in range(n_extra_assets)]
    daily_vols = rng.uniform(0.02, 0.06, n_extra_assets)
    extra_returns = rng.normal(0, 1, (n_days, n_extra_assets)) * daily_vols

    # Block bootstrap the original 10 assets to 730 days
    base = sample_returns.values
    n_blocks = n_days // block_size + 1
    block_starts = rng.integers(0, len(base) - block_size, size=n_blocks)
    bootstrapped = np.vstack([base[s : s + block_size] for s in block_starts])[:n_days]

    # Combine
    all_returns = np.hstack([bootstrapped, extra_returns])
    all_names = list(sample_returns.columns) + extra_names
    dates = pd.bdate_range(start="2022-06-01", periods=n_days)

    return pd.DataFrame(all_returns, index=dates, columns=all_names)


@pytest.fixture
def btc_returns() -> pd.Series:
    """730 rows of BTC-like returns for regime tests.

    Two distinct regime structure: bear (first 365 days) and bull (last 365 days).
    """
    rng = np.random.default_rng(99)
    n_bear = 365
    n_bull = 365

    bear = rng.normal(-0.002, 0.04, n_bear)
    bull = rng.normal(0.003, 0.025, n_bull)

    all_returns = np.concatenate([bear, bull])
    dates = pd.bdate_range(start="2023-01-02", periods=len(all_returns))

    return pd.Series(all_returns, index=dates, name="BTC")


@pytest.fixture
def sample_prices(sample_returns: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct price levels from sample_returns, starting at 100."""
    cum_returns = sample_returns.cumsum()
    prices = 100.0 * np.exp(cum_returns)
    return prices


@pytest.fixture
def sample_universe() -> list[dict]:
    """10 hardcoded universe asset dicts. No network calls.

    Returns list of dicts (not UniverseAsset dataclass) to avoid import coupling.
    Convert to UniverseAsset in tests that need the dataclass.
    """
    return [
        {
            "coingecko_id": "bitcoin",
            "symbol": "BTC",
            "name": "Bitcoin",
            "market_cap": 800_000_000_000,
            "market_cap_rank": 1,
            "current_price": 40000.0,
            "price_change_24h": 2.5,
            "price_change_7d": 5.0,
            "price_change_30d": 10.0,
            "volume_24h": 30_000_000_000,
            "binance_symbol": "BTCUSDT",
            "yfinance_ticker": "BTC-USD",
        },
        {
            "coingecko_id": "ethereum",
            "symbol": "ETH",
            "name": "Ethereum",
            "market_cap": 300_000_000_000,
            "market_cap_rank": 2,
            "current_price": 2500.0,
            "price_change_24h": 1.8,
            "price_change_7d": 3.5,
            "price_change_30d": 8.0,
            "volume_24h": 15_000_000_000,
            "binance_symbol": "ETHUSDT",
            "yfinance_ticker": "ETH-USD",
        },
        {
            "coingecko_id": "binancecoin",
            "symbol": "BNB",
            "name": "BNB",
            "market_cap": 45_000_000_000,
            "market_cap_rank": 3,
            "current_price": 300.0,
            "price_change_24h": 0.5,
            "price_change_7d": 2.0,
            "price_change_30d": 5.0,
            "volume_24h": 1_000_000_000,
            "binance_symbol": "BNBUSDT",
            "yfinance_ticker": "BNB-USD",
        },
        {
            "coingecko_id": "solana",
            "symbol": "SOL",
            "name": "Solana",
            "market_cap": 40_000_000_000,
            "market_cap_rank": 4,
            "current_price": 100.0,
            "price_change_24h": 3.0,
            "price_change_7d": 7.0,
            "price_change_30d": 15.0,
            "volume_24h": 2_000_000_000,
            "binance_symbol": "SOLUSDT",
            "yfinance_ticker": "SOL-USD",
        },
        {
            "coingecko_id": "ripple",
            "symbol": "XRP",
            "name": "XRP",
            "market_cap": 25_000_000_000,
            "market_cap_rank": 5,
            "current_price": 0.5,
            "price_change_24h": -1.0,
            "price_change_7d": -2.0,
            "price_change_30d": 0.5,
            "volume_24h": 1_500_000_000,
            "binance_symbol": "XRPUSDT",
            "yfinance_ticker": "XRP-USD",
        },
        {
            "coingecko_id": "cardano",
            "symbol": "ADA",
            "name": "Cardano",
            "market_cap": 12_000_000_000,
            "market_cap_rank": 6,
            "current_price": 0.35,
            "price_change_24h": -0.5,
            "price_change_7d": 1.0,
            "price_change_30d": 3.0,
            "volume_24h": 500_000_000,
            "binance_symbol": "ADAUSDT",
            "yfinance_ticker": "ADA-USD",
        },
        {
            "coingecko_id": "polkadot",
            "symbol": "DOT",
            "name": "Polkadot",
            "market_cap": 8_000_000_000,
            "market_cap_rank": 7,
            "current_price": 7.0,
            "price_change_24h": 1.2,
            "price_change_7d": 4.0,
            "price_change_30d": 6.0,
            "volume_24h": 300_000_000,
            "binance_symbol": "DOTUSDT",
            "yfinance_ticker": "DOT-USD",
        },
        {
            "coingecko_id": "chainlink",
            "symbol": "LINK",
            "name": "Chainlink",
            "market_cap": 7_000_000_000,
            "market_cap_rank": 8,
            "current_price": 15.0,
            "price_change_24h": 2.0,
            "price_change_7d": 5.5,
            "price_change_30d": 12.0,
            "volume_24h": 400_000_000,
            "binance_symbol": "LINKUSDT",
            "yfinance_ticker": "LINK-USD",
        },
        {
            "coingecko_id": "avalanche-2",
            "symbol": "AVAX",
            "name": "Avalanche",
            "market_cap": 6_000_000_000,
            "market_cap_rank": 9,
            "current_price": 35.0,
            "price_change_24h": 1.5,
            "price_change_7d": 3.0,
            "price_change_30d": 7.0,
            "volume_24h": 250_000_000,
            "binance_symbol": "AVAXUSDT",
            "yfinance_ticker": "AVAX-USD",
        },
        {
            "coingecko_id": "uniswap",
            "symbol": "UNI",
            "name": "Uniswap",
            "market_cap": 5_000_000_000,
            "market_cap_rank": 10,
            "current_price": 8.0,
            "price_change_24h": 0.8,
            "price_change_7d": 2.5,
            "price_change_30d": 4.0,
            "volume_24h": 200_000_000,
            "binance_symbol": "UNIUSDT",
            "yfinance_ticker": "UNI-USD",
        },
    ]
