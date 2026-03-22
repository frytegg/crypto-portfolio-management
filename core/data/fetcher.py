"""Historical OHLCV data fetcher -- yfinance primary, Binance REST fallback, CoinGecko tertiary.

Fallback chain: yfinance → Binance REST → CoinGecko market_chart.
CoinGecko calls are rate-limited (3s between calls) and run sequentially.

Data cleaning pipeline:
1. Forward-fill gaps up to 5 consecutive days
2. Drop assets with >20% NaN values
3. Compute log returns: np.log(prices / prices.shift(1))
4. Drop rows with any remaining NaN
5. Enforce minimum 180 observations per asset
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import structlog
import yfinance as yf

from core.config import settings
from core.data.cache import cache, cache_get_or_fetch, invalidate
from core.data.symbol_map import get_display_symbol
from core.data.universe import UniverseAsset

log = structlog.get_logger(__name__)

_fetch_lock = threading.Lock()

# Use a browser-like User-Agent to avoid 403 blocks in cloud environments
_yf_session = requests.Session()
_yf_session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"})

_MIN_OBSERVATIONS = 180
_MAX_NAN_RATIO = 0.20
_FFILL_LIMIT = 5
_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
_COINGECKO_RATE_LIMIT_DELAY = 3.0  # seconds between calls (30 calls/min on demo key)


def fetch_historical_data(
    universe: list[UniverseAsset],
    lookback_days: int | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download 2yr daily OHLCV for all universe assets.

    Primary source is yfinance; assets that fail yfinance fall back to Binance REST,
    then to CoinGecko historical as a tertiary fallback. Results are cached with a 4-hour TTL.

    Thread-safe: a threading lock prevents concurrent fetches. If a second call arrives
    while the first is in progress, it waits for completion and returns the cached result.

    Returns:
        prices: DatetimeIndex DataFrame, columns=ticker symbols (e.g. "BTC"), values=close prices
        returns: DatetimeIndex DataFrame, columns=ticker symbols, values=daily log returns
    """
    if not _fetch_lock.acquire(blocking=False):
        log.info("fetch_already_in_progress", msg="skipping duplicate call, waiting for first fetch")
        # Wait for the in-progress fetch to complete, then return cached result
        _fetch_lock.acquire(blocking=True)
        _fetch_lock.release()
        cached = cache.get("historical_prices_and_returns")
        if cached is not None:
            return cached
        return pd.DataFrame(), pd.DataFrame()

    try:
        if lookback_days is None:
            lookback_days = settings.DEFAULT_LOOKBACK_DAYS

        if force_refresh:
            invalidate("historical_prices")
            invalidate("historical_returns")

        def _fetch() -> tuple[pd.DataFrame, pd.DataFrame]:
            return _fetch_and_clean(universe, lookback_days)

        return cache_get_or_fetch(
            key="historical_prices_and_returns",
            fetch_fn=_fetch,
            ttl=settings.CACHE_TTL_PRICES,
        )
    finally:
        _fetch_lock.release()


def _fetch_and_clean(
    universe: list[UniverseAsset], lookback_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrate fetching from yfinance + Binance + CoinGecko fallback, then clean."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Build ticker -> display symbol mapping
    yf_tickers = [a.yfinance_ticker for a in universe]
    ticker_to_display = {a.yfinance_ticker: get_display_symbol(a.coingecko_id) for a in universe}

    # Primary: yfinance batch download
    prices = _fetch_yfinance(yf_tickers, start_str, end_str)

    # Rename columns from yfinance tickers ("BTC-USD") to display symbols ("BTC")
    yf_total_failure = prices.empty
    if not yf_total_failure:
        prices = prices.rename(columns=ticker_to_display)

    # Identify failed assets (all NaN or missing columns)
    # If yfinance returned nothing, ALL assets are considered failed
    failed_tickers: list[str] = []
    for asset in universe:
        display = get_display_symbol(asset.coingecko_id)
        if yf_total_failure or display not in prices.columns or prices[display].isna().all():
            failed_tickers.append(display)

    if yf_total_failure:
        log.warning(
            "yfinance_total_failure",
            msg=f"Attempting Binance REST fallback for all {len(failed_tickers)} assets",
        )

    # Fallback: Binance REST for failed assets
    binance_fallback_count = 0
    binance_series_list: list[pd.Series] = []
    for asset in universe:
        display = get_display_symbol(asset.coingecko_id)
        if display not in failed_tickers:
            continue
        if asset.binance_symbol is None:
            log.warn(
                "no_binance_fallback",
                asset=display,
                reason="no Binance symbol mapped",
            )
            continue

        try:
            series = _fetch_binance_rest(asset.binance_symbol, limit=lookback_days)
            if series is not None and len(series) > 0:
                series.name = display
                if prices.empty:
                    # No yfinance index to align to — collect and merge later
                    binance_series_list.append(series)
                else:
                    # Reindex Binance data to match existing DataFrame dates.
                    # Binance returns calendar days; yfinance uses business days.
                    # Reindex + ffill bridges the weekend gaps.
                    aligned = series.reindex(prices.index, method="ffill")
                    if display in prices.columns:
                        prices = prices.drop(columns=[display])
                    prices[display] = aligned
                binance_fallback_count += 1
                log.info("binance_fallback_success", asset=display, rows=len(series))
        except Exception as exc:
            log.error("binance_fallback_failed", asset=display, error=str(exc))

    # If yfinance failed entirely, build prices DataFrame from Binance series
    if prices.empty and binance_series_list:
        prices = pd.concat(binance_series_list, axis=1).sort_index()
        prices = prices.ffill(limit=_FFILL_LIMIT)
        log.info("binance_only_prices_built", assets=len(prices.columns), rows=len(prices))

    # Tertiary fallback: CoinGecko historical for assets still missing
    still_failed: list[UniverseAsset] = []
    for asset in universe:
        display = get_display_symbol(asset.coingecko_id)
        has_data = (
            not prices.empty
            and display in prices.columns
            and not prices[display].isna().all()
        )
        if not has_data:
            still_failed.append(asset)

    coingecko_fallback_count = 0
    coingecko_series_list: list[pd.Series] = []
    if still_failed:
        log.info(
            "coingecko_fallback_start",
            assets=len(still_failed),
            msg=f"Attempting CoinGecko fallback for {len(still_failed)} assets",
        )
        for i, asset in enumerate(still_failed):
            display = get_display_symbol(asset.coingecko_id)
            # Rate-limit: 3s between calls (30 calls/min on demo key)
            if i > 0:
                time.sleep(_COINGECKO_RATE_LIMIT_DELAY)

            series = _fetch_coingecko_historical(asset.coingecko_id, days=lookback_days)
            if series is not None and len(series) > 0:
                series.name = display
                if prices.empty:
                    coingecko_series_list.append(series)
                else:
                    aligned = series.reindex(prices.index, method="ffill")
                    if display in prices.columns:
                        prices = prices.drop(columns=[display])
                    prices[display] = aligned
                coingecko_fallback_count += 1
                log.info("coingecko_fallback_success", asset=display, days=len(series))

        # If both yfinance and Binance failed, build from CoinGecko series
        if prices.empty and coingecko_series_list:
            prices = pd.concat(coingecko_series_list, axis=1).sort_index()
            prices = prices.ffill(limit=_FFILL_LIMIT)
            log.info("coingecko_only_prices_built", assets=len(prices.columns), rows=len(prices))

    initial_asset_count = len(prices.columns)

    # Clean prices and compute returns
    prices, returns = _clean_prices(prices)

    final_asset_count = len(prices.columns)
    dropped_count = initial_asset_count - final_asset_count

    log.info(
        "fetch_historical_complete",
        universe_size=len(universe),
        yfinance_fetched=initial_asset_count - binance_fallback_count - coingecko_fallback_count,
        binance_fallback=binance_fallback_count,
        coingecko_fallback=coingecko_fallback_count,
        assets_dropped=dropped_count,
        final_assets=final_asset_count,
        observations=len(returns),
    )

    return prices, returns


def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch close prices from yfinance.download().

    Returns a DataFrame with DatetimeIndex and columns = yfinance ticker names.
    Missing/failed tickers will have NaN columns.
    Returns an empty DataFrame on total failure (caller should fall back to Binance).
    """
    if not tickers:
        return pd.DataFrame()

    log.info("yfinance_download_start", tickers=len(tickers), start=start, end=end)

    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
            session=_yf_session,
        )
    except Exception as exc:
        log.error(
            "yfinance_download_failed",
            error=str(exc),
            tickers=len(tickers),
            msg="Falling back to Binance REST for all assets",
        )
        return pd.DataFrame()

    if raw.empty:
        log.error(
            "yfinance_download_empty",
            tickers=len(tickers),
            msg="yfinance returned empty DataFrame — falling back to Binance REST for all assets",
        )
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when group_by="column" and multiple tickers
    # Extract "Close" level. For a single ticker, it may return flat columns.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    elif "Close" in raw.columns:
        # Single ticker case: columns are flat ("Open", "High", "Low", "Close", ...)
        prices = raw[["Close"]]
        prices.columns = tickers[:1]
    else:
        log.error("yfinance_unexpected_format", columns=list(raw.columns[:10]))
        return pd.DataFrame()

    # Ensure the index is a clean DatetimeIndex (no timezone)
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    log.info("yfinance_download_complete", assets=len(prices.columns), rows=len(prices))
    return prices


def _fetch_binance_rest(symbol: str, limit: int = 730) -> pd.Series | None:
    """Fetch daily close from Binance /api/v3/klines for one symbol.

    Args:
        symbol: Binance pair, e.g. "BTCUSDT"
        limit: Number of daily candles (max 1000)

    Returns:
        pd.Series with DatetimeIndex and close prices, or None on failure.
    """
    log.info("binance_rest_fetch", symbol=symbol, limit=limit)

    # Binance klines endpoint has a max limit of 1000 per request
    limit = min(limit, 1000)

    resp = requests.get(
        _BINANCE_KLINES_URL,
        params={"symbol": symbol, "interval": "1d", "limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if not data:
        log.warn("binance_rest_empty", symbol=symbol)
        return None

    # Each kline: [open_time, open, high, low, close, volume, close_time, ...]
    # Index 0 = open_time (ms), Index 4 = close price
    timestamps = [pd.Timestamp(candle[0], unit="ms").normalize() for candle in data]
    closes = [float(candle[4]) for candle in data]

    series = pd.Series(closes, index=pd.DatetimeIndex(timestamps), name=symbol)
    series = series[~series.index.duplicated(keep="last")]

    return series


def _fetch_coingecko_historical(coingecko_id: str, days: int = 365) -> pd.Series | None:
    """Fetch daily close prices from CoinGecko market_chart endpoint.

    Args:
        coingecko_id: CoinGecko asset ID, e.g. "bitcoin", "ethereum"
        days: Number of days of history (max 365 — demo key limit)

    Returns:
        pd.Series with DatetimeIndex and daily close prices, or None on failure.
    """
    # CoinGecko free demo key only supports up to 365 days; 730 returns HTTP 401
    days = min(days, 365)
    log.info(
        "coingecko_historical_call",
        coingecko_id=coingecko_id,
        key_present=bool(settings.COINGECKO_API_KEY),
    )

    url = _COINGECKO_MARKET_CHART_URL.format(coingecko_id=coingecko_id)
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    headers = {"x-cg-demo-api-key": settings.COINGECKO_API_KEY}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            log.warn(
                "coingecko_historical_http_error",
                coingecko_id=coingecko_id,
                status=resp.status_code,
            )
            return None

        data = resp.json().get("prices")
        if not data:
            log.warn("coingecko_historical_empty", coingecko_id=coingecko_id)
            return None

        series = pd.Series(
            {pd.Timestamp(ts, unit="ms").normalize(): price for ts, price in data},
            name=coingecko_id,
        )
        # Drop duplicate dates (CoinGecko can return an extra partial-day entry)
        series = series[~series.index.duplicated(keep="last")]
        return series

    except Exception as exc:
        log.error("coingecko_historical_failed", coingecko_id=coingecko_id, error=str(exc))
        return None


def _clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the 5-step cleaning pipeline.

    Steps:
        1. Forward-fill gaps up to 5 consecutive days
        2. Drop assets with >20% NaN values
        3. Compute log returns: np.log(prices / prices.shift(1))
        4. Drop rows where any remaining NaN exists
        5. Enforce minimum 180 observations per asset

    Returns:
        (cleaned_prices, returns) — both aligned with the same index/columns.
    """
    # Step 1: Forward-fill gaps up to 5 consecutive days
    prices = prices.ffill(limit=_FFILL_LIMIT)

    # Step 2: Drop assets with >20% NaN values
    nan_ratio = prices.isna().mean()
    valid_assets = nan_ratio[nan_ratio <= _MAX_NAN_RATIO].index.tolist()
    dropped_nan = set(prices.columns) - set(valid_assets)
    if dropped_nan:
        log.info("dropped_high_nan_assets", assets=sorted(dropped_nan), threshold=_MAX_NAN_RATIO)
    prices = prices[valid_assets]

    if prices.empty:
        log.error("no_assets_survived_nan_filter")
        return pd.DataFrame(), pd.DataFrame()

    # Step 3: Compute log returns
    returns = np.log(prices / prices.shift(1))

    # Step 4: Drop rows where any NaN exists (includes the first row from shift)
    valid_rows = returns.dropna().index
    returns = returns.loc[valid_rows]
    prices = prices.loc[valid_rows]

    # Step 5: Enforce minimum 180 observations per asset
    if len(returns) < _MIN_OBSERVATIONS:
        # If the entire DataFrame is too short, keep what we have but warn
        log.warn(
            "insufficient_total_observations",
            observations=len(returns),
            minimum=_MIN_OBSERVATIONS,
        )
    else:
        # Check per-asset observation counts (after row drops, all columns have same count)
        # This matters if we later support per-asset start dates
        asset_counts = returns.count()
        valid_assets = asset_counts[asset_counts >= _MIN_OBSERVATIONS].index.tolist()
        dropped_short = set(returns.columns) - set(valid_assets)
        if dropped_short:
            log.info(
                "dropped_short_history_assets",
                assets=sorted(dropped_short),
                threshold=_MIN_OBSERVATIONS,
            )
        returns = returns[valid_assets]
        prices = prices[valid_assets]

    # Ensure monotonically increasing DatetimeIndex
    prices = prices.sort_index()
    returns = returns.sort_index()

    return prices, returns
