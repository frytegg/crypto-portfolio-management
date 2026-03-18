"""Historical OHLCV data fetcher -- yfinance primary, Binance REST fallback.

Data cleaning pipeline:
1. Forward-fill gaps up to 5 consecutive days
2. Drop assets with >20% NaN values
3. Compute log returns: np.log(prices / prices.shift(1))
4. Drop rows with any remaining NaN
5. Enforce minimum 180 observations per asset
"""
from __future__ import annotations

import pandas as pd
import structlog

from core.config import settings
from core.data.cache import cache_get_or_fetch
from core.data.universe import UniverseAsset

log = structlog.get_logger(__name__)


def fetch_historical_data(
    universe: list[UniverseAsset],
    lookback_days: int | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download 2yr daily OHLCV for all universe assets.

    Returns:
        prices: DatetimeIndex DataFrame, columns=ticker symbols, values=close prices
        returns: DatetimeIndex DataFrame, columns=ticker symbols, values=daily log returns
    """
    raise NotImplementedError


def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch close prices from yfinance.download()."""
    raise NotImplementedError


def _fetch_binance_rest(symbol: str, limit: int = 730) -> pd.Series:
    """Fetch daily close from Binance /api/v3/klines for one symbol."""
    raise NotImplementedError


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply the 5-step cleaning pipeline."""
    raise NotImplementedError
