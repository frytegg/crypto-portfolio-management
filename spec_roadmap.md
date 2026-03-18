# Crypto Portfolio Management — Full Technical Specification & Roadmap

> **Purpose**: This document is the single source of truth for refactoring the crypto portfolio management project. It is designed to be consumed by an LLM to generate precise, self-contained Claude Code implementation prompts. Every section contains exact file paths, library APIs, function signatures, and acceptance criteria.

> **Current state**: A single-file Streamlit app (`app.py`, 548 lines) with 4 basic strategies (Equal Weight, GMV-Shrink via scipy SLSQP, Inverse Volatility, Momentum Top-K), data from yfinance + CoinGecko Top 10, and basic risk metrics. No GARCH, no Markowitz efficient frontier, no on-chain data, no regime detection, no backtesting engine.

> **Target state**: A modular, professional-grade Plotly Dash application with 7 allocation strategies (Markowitz MVO, GARCH-GMV, HRP, Risk Parity, Mean-CVaR, Regime-Aware, Equal Weight benchmark), GARCH volatility forecasting, on-chain signal integration fed into Black-Litterman views, walk-forward backtesting, 50 dynamically-fetched crypto assets, real-time Binance WebSocket prices, quantstats tearsheet export, and Railway deployment.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack](#2-tech-stack)
3. [Project Structure](#3-project-structure)
4. [Data Layer](#4-data-layer)
5. [Models Layer](#5-models-layer)
6. [Optimization Layer](#6-optimization-layer)
7. [Risk & Analytics Layer](#7-risk--analytics-layer)
8. [Dashboard Layer](#8-dashboard-layer)
9. [Configuration & Environment](#9-configuration--environment)
10. [Deployment](#10-deployment)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Appendix A: Current app.py Reference](#appendix-a-current-apppy-reference)
13. [Appendix B: Symbol Mapping Table](#appendix-b-symbol-mapping-table)
14. [Appendix C: API Rate Limits](#appendix-c-api-rate-limits)

---

## 1. Architecture Overview

### Design Principles

- **Separation of concerns**: `core/` contains pure Python logic (zero UI imports). All functions accept and return pandas DataFrames, numpy arrays, or plain Python types. The `dashboard/` layer is Dash-specific and calls into `core/`.
- **Single process, multi-threaded**: One gunicorn worker with 4 threads. One daemon background thread for Binance WebSocket. Dash callbacks run in Flask request threads. Shared state via thread-safe `diskcache`.
- **Cache-first data access**: All API data (CoinGecko, yfinance, DeFiLlama) is fetched once and written to persistent disk cache with TTL. Dash callbacks always read from cache, never directly from APIs.
- **Compute-on-demand**: Heavy computations (optimization, GARCH fitting, backtesting) run only when the user clicks "Optimize" or "Run Backtest" — never on page load or interval tick.

### Data Flow Diagram

```
[CoinGecko API] ─── top 50 universe ───→ [diskcache]
[yfinance / Binance REST] ── OHLCV ───→ [diskcache]
[DeFiLlama API] ──── on-chain data ───→ [diskcache]
[Binance WebSocket] ── live prices ───→ [diskcache] (background thread, 1s updates)
                                              │
                                              ▼
                                    [Dash callbacks read cache]
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                [core/models/]        [core/optimization/]   [core/risk/]
                GARCH, Regime,        Markowitz, HRP,        VaR, CVaR,
                Covariance            Risk Parity, CVaR,     Backtest,
                                      Black-Litterman        Tearsheet
                         │                    │                    │
                         └────────────────────┼────────────────────┘
                                              ▼
                                    [dashboard/components/]
                                    Charts, tables, cards
                                              │
                                              ▼
                                        [User browser]
```

---

## 2. Tech Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `dash` | >=2.14.0 | Web framework (replaces Streamlit) |
| `dash-bootstrap-components` | >=1.5.0 | UI components + DARKLY theme |
| `dash-bootstrap-templates` | >=1.1.0 | Plotly figure templates matching Bootstrap themes |
| `plotly` | >=5.18.0 | Interactive charts |
| `gunicorn` | >=21.0.0 | Production WSGI server |
| `pandas` | >=2.0.0 | DataFrames |
| `numpy` | >=1.24.0 | Numerical computation |
| `scipy` | >=1.10.0 | Statistical functions (used by riskfolio-lib internally) |
| `riskfolio-lib` | >=6.0.0 | Portfolio optimization (MVO, HRP, HERC, Risk Parity, CVaR, Black-Litterman) |
| `cvxpy` | >=1.5.2 | Convex optimization backend for riskfolio-lib |
| `clarabel` | >=0.6.0 | Solver for cvxpy (pre-built binary, no Rust needed) |
| `arch` | >=7.0.0 | GARCH/GJR-GARCH/EGARCH volatility models |
| `hmmlearn` | >=0.3.0 | Hidden Markov Model for regime detection |
| `scikit-learn` | >=1.3.0 | Used by hmmlearn, also Ledoit-Wolf covariance |
| `quantstats-lumi` | >=0.3.0 | Tearsheet generation and risk metrics (pandas 2.x compatible fork of quantstats) |
| `yfinance` | >=0.2.40 | Historical OHLCV data |
| `aiohttp` | >=3.9.0 | Async HTTP for parallel data fetching |
| `websocket-client` | >=1.6.0 | Binance WebSocket (synchronous, thread-based) |
| `diskcache` | >=5.6.0 | Persistent SQLite-backed cache |
| `requests` | >=2.31.0 | Synchronous HTTP for CoinGecko/DeFiLlama |
| `python-dotenv` | >=1.0.0 | .env file loading |

### Why These Choices

- **Dash over Streamlit**: Callback model avoids full-page reruns. `dcc.Interval` updates only the live price section, not the entire app. Professional dark themes via DBC. Standard Flask/WSGI deployment.
- **riskfolio-lib over scipy.optimize.minimize**: Supports 35+ risk measures, HRP/HERC/NCO, Black-Litterman, efficient frontier plotting, Ledoit-Wolf/Gerber/denoised covariance estimation. Replaces 30 lines of hand-rolled SLSQP with 5 lines of battle-tested code.
- **arch over manual volatility**: The canonical Python GARCH library. GJR-GARCH with Student-t is 3 lines of code. **CRITICAL**: GJR-GARCH is specified via `vol='GARCH', o=1` — there is NO `vol='GJR-GARCH'` string in arch.
- **websocket-client over websockets (asyncio)**: Dash is WSGI/threaded. `websocket-client` runs natively in a `threading.Thread` with `run_forever()` — no asyncio event loop conflicts.
- **diskcache over st.cache_data**: Persists across restarts, thread-safe for WebSocket writer + Dash callback readers, SQLite-backed ACID transactions.
- **quantstats-lumi over quantstats**: The original quantstats has known pandas 2.0+ compatibility issues. `quantstats-lumi` is a maintained fork that fixes these. Import as `import quantstats_lumi as qs`.

---

## 3. Project Structure

```
crypto-portfolio-management/
│
├── app.py                          # Dash app entry point
├── requirements.txt                # All dependencies
├── Dockerfile                      # Production container
├── Procfile                        # Railway start command
├── railway.toml                    # Railway config-as-code
├── .env.example                    # Template (committed)
├── .env                            # Local values (gitignored)
├── .gitignore
├── .dockerignore
├── spec_roadmap.md                 # This file
│
├── core/                           # Pure Python — NO UI imports
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── universe.py             # Dynamic top-50 universe builder
│   │   ├── fetcher.py              # Historical OHLCV fetching (yfinance + Binance REST fallback)
│   │   ├── onchain.py              # On-chain indicators (DeFiLlama + CoinGecko)
│   │   ├── price_feed.py           # Binance WebSocket live price feed (background thread)
│   │   ├── cache.py                # diskcache singleton + helpers
│   │   └── symbol_map.py           # CoinGecko ID ↔ Binance symbol ↔ yfinance ticker mapping
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── covariance.py           # Covariance estimation (Ledoit-Wolf, Gerber, denoised)
│   │   ├── garch.py                # GARCH/GJR-GARCH/EGARCH volatility forecasting
│   │   ├── regime.py               # HMM regime detection (bull/bear/sideways)
│   │   └── returns.py              # Return estimation (historical mean, CAPM, shrinkage)
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── markowitz.py            # Mean-Variance optimization + efficient frontier
│   │   ├── hrp.py                  # Hierarchical Risk Parity
│   │   ├── risk_parity.py          # Equal Risk Contribution (true risk parity)
│   │   ├── cvar.py                 # Mean-CVaR optimization
│   │   ├── black_litterman.py      # Black-Litterman with on-chain views
│   │   ├── regime_alloc.py         # Regime-aware allocation (switches strategy per regime)
│   │   └── equal_weight.py         # 1/N benchmark
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── metrics.py              # VaR, CVaR, Sortino, Calmar, Omega, MDD, Sharpe
│   │   └── backtest.py             # Walk-forward backtester with transaction costs
│   │
│   └── analytics/
│       ├── __init__.py
│       └── tearsheet.py            # quantstats HTML tearsheet generation
│
├── dashboard/                      # Dash-specific code
│   ├── __init__.py
│   ├── layout.py                   # Main app layout (tabs, sidebar, structure)
│   ├── theme.py                    # DARKLY theme + Plotly figure template config
│   │
│   ├── callbacks/
│   │   ├── __init__.py
│   │   ├── data_cb.py              # Universe loading, historical data fetch
│   │   ├── optimization_cb.py      # Strategy computation triggers
│   │   ├── garch_cb.py             # GARCH model fitting and display
│   │   ├── regime_cb.py            # Regime detection display
│   │   ├── onchain_cb.py           # On-chain data display
│   │   ├── backtest_cb.py          # Backtester trigger and display
│   │   ├── live_cb.py              # Real-time price update (dcc.Interval)
│   │   └── report_cb.py            # Tearsheet download
│   │
│   └── components/
│       ├── __init__.py
│       ├── metric_card.py          # Single KPI card (dbc.Card)
│       ├── efficient_frontier.py   # Interactive efficient frontier chart
│       ├── correlation_heatmap.py  # NxN correlation matrix heatmap
│       ├── equity_chart.py         # Equity curve line chart
│       ├── drawdown_chart.py       # Drawdown area chart
│       ├── weights_chart.py        # Portfolio weights bar/pie chart
│       ├── garch_chart.py          # Conditional volatility + forecast chart
│       ├── regime_chart.py         # Price chart with regime colored bands
│       ├── dendrogram.py           # HRP cluster dendrogram
│       ├── onchain_charts.py       # TVL, stablecoin flows, exchange volumes
│       └── comparison_table.py     # Strategy comparison table
│
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Shared fixtures (sample returns DataFrame, etc.)
    ├── test_covariance.py
    ├── test_garch.py
    ├── test_optimization.py
    ├── test_risk_metrics.py
    ├── test_backtest.py
    └── test_universe.py
```

### File Responsibilities (detailed)

#### `app.py` — Entry Point
- Creates the Dash app instance with DARKLY theme
- Imports and registers all callbacks from `dashboard/callbacks/`
- Starts the Binance WebSocket background thread
- Exposes `server = app.server` for gunicorn
- Reads `PORT` from environment, defaults to 8050
- Contains a `/health` endpoint for Railway healthcheck
- **~50 lines max** — no business logic here

#### `core/data/universe.py` — Universe Builder
- Fetches top 50 cryptos by market cap from CoinGecko `/coins/markets`
- Filters out stablecoins (USDT, USDC, DAI, BUSD, TUSD, FDUSD) — they have zero/near-zero volatility and break optimization
- Returns a list of `UniverseAsset` dataclass objects with: `coingecko_id`, `symbol`, `name`, `market_cap`, `market_cap_rank`, `binance_symbol` (nullable), `yfinance_ticker`
- Caches result in diskcache with 4h TTL
- After filtering stablecoins, the effective universe will be ~44-47 assets

#### `core/data/fetcher.py` — Historical Data
- Primary source: `yfinance.download()` with `threads=True`
- Fallback: Binance REST `GET /api/v3/klines` for assets not on yfinance
- Fetches daily OHLCV for all universe assets
- Default lookback: 2 years (730 days)
- Returns a clean `pd.DataFrame` with DatetimeIndex, columns = ticker symbols, values = adjusted close prices
- Handles missing data: forward-fill gaps up to 5 days, drop assets with >20% missing data
- Caches result in diskcache with 4h TTL
- Computes returns: `prices.pct_change().dropna()`

#### `core/data/onchain.py` — On-Chain Data
- **DeFiLlama API** (fully free, no key):
  - `GET https://api.llama.fi/v2/historicalChainTvl` — total crypto TVL over time
  - `GET https://api.llama.fi/v2/historicalChainTvl/{chain}` — per-chain TVL (Ethereum, Solana, BSC, Arbitrum, etc.)
  - `GET https://stablecoins.llama.fi/stablecoincharts/all` — stablecoin total market cap over time
  - `GET https://stablecoins.llama.fi/stablecoins` — current stablecoin list with market caps
  - `GET https://api.llama.fi/overview/dexs` — DEX volume overview
- **CoinGecko** (free demo API):
  - `GET /coins/{id}` — developer_score, community_score, public_interest_score
  - `GET /coins/{id}/market_chart?vs_currency=usd&days=365&interval=daily` — historical prices for assets not on yfinance
- Returns structured DataFrames indexed by date
- Caches with 6h TTL (on-chain data updates slowly)

#### `core/data/price_feed.py` — Live Prices
- Binance WebSocket combined stream: `wss://stream.binance.com:9443/stream?streams={sym1}@miniTicker/{sym2}@miniTicker/...`
- Runs in a daemon `threading.Thread`
- Uses `websocket-client` library's `WebSocketApp.run_forever(ping_interval=20, ping_timeout=10)`
- Writes each price update to `diskcache` with key `price:{SYMBOL}` and 60s TTL
- Auto-reconnects with exponential backoff (1s, 2s, 4s, 8s, ... max 60s) on disconnect
- Handles Binance's 24h hard disconnect gracefully
- Message format: combined stream envelope → `data.s` (symbol), `data.c` (close price)
- Only streams assets that have a valid Binance symbol mapping (skips stablecoins, XMR, LEO, OKB)

#### `core/data/cache.py` — Cache Singleton
```python
import diskcache
import os

CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")

cache = diskcache.Cache(
    CACHE_DIR,
    size_limit=int(200e6),  # 200 MB max
    disk_min_file_size=0,
    timeout=1,
)
```
- Thread-safe and process-safe (SQLite ACID)
- Shared by WebSocket writer thread and Dash callback reader threads
- Helper functions: `cache_get_or_fetch(key, fetch_fn, ttl)` pattern

#### `core/data/symbol_map.py` — Symbol Mapping
- Static dict: `COINGECKO_TO_BINANCE: dict[str, str | None]` — maps CoinGecko IDs to Binance USDT pairs
- Static dict: `COINGECKO_TO_YFINANCE: dict[str, str]` — maps CoinGecko IDs to yfinance tickers (e.g., `"bitcoin" → "BTC-USD"`)
- `None` values indicate assets not available on that exchange (stablecoins, XMR, LEO, OKB)
- Functions: `get_binance_symbols(cg_ids)`, `get_yfinance_tickers(cg_ids)`, `binance_to_display_name(symbol)`
- See [Appendix B](#appendix-b-symbol-mapping-table) for the full mapping table

---

## 4. Data Layer

### 4.1 Universe Construction (`core/data/universe.py`)

**CoinGecko API call:**
```python
GET https://api.coingecko.com/api/v3/coins/markets
Params:
  vs_currency: "usd"
  order: "market_cap_desc"
  per_page: 50
  page: 1
  sparkline: false
  price_change_percentage: "24h,7d,30d"
Headers:
  x-cg-demo-api-key: {COINGECKO_API_KEY}  # Free demo key
```

**Response fields used:**
- `id` — CoinGecko ID (e.g., "bitcoin")
- `symbol` — ticker symbol (e.g., "btc")
- `name` — full name (e.g., "Bitcoin")
- `current_price` — current USD price
- `market_cap` — market cap in USD
- `market_cap_rank` — rank by market cap
- `price_change_percentage_24h` — 24h price change %
- `price_change_percentage_7d_in_currency` — 7d price change %
- `price_change_percentage_30d_in_currency` — 30d price change %
- `total_volume` — 24h trading volume
- `high_24h`, `low_24h` — 24h price range

**Stablecoin filter:**
```python
STABLECOINS = {"tether", "usd-coin", "dai", "binance-usd", "trueusd", "first-digital-usd", "usdd", "frax", "paypal-usd"}
universe = [coin for coin in raw_top_50 if coin["id"] not in STABLECOINS]
```

**Output dataclass:**
```python
@dataclass
class UniverseAsset:
    coingecko_id: str       # "bitcoin"
    symbol: str             # "BTC"
    name: str               # "Bitcoin"
    market_cap: float       # 1_200_000_000_000.0
    market_cap_rank: int    # 1
    current_price: float    # 42150.50
    price_change_24h: float # -2.3
    price_change_7d: float  # 5.1
    price_change_30d: float # 12.8
    volume_24h: float       # 28_000_000_000.0
    binance_symbol: str | None  # "BTCUSDT" or None
    yfinance_ticker: str    # "BTC-USD"
```

### 4.2 Historical OHLCV (`core/data/fetcher.py`)

**Primary: yfinance**
```python
import yfinance as yf

raw = yf.download(
    tickers=yf_tickers,       # list of "BTC-USD", "ETH-USD", ...
    start="2024-03-17",       # 2 years back from today
    end="2026-03-17",         # today
    interval="1d",
    auto_adjust=False,
    progress=False,
    group_by="column",
    threads=True,
)
prices = raw["Close"]  # pd.DataFrame, columns=ticker names
```

**Fallback: Binance REST** (for assets failing yfinance)
```
GET https://api.binance.com/api/v3/klines
Params:
  symbol: "BTCUSDT"
  interval: "1d"
  limit: 730
```
Returns array of arrays: `[open_time, open, high, low, close, volume, ...]`

**Data cleaning pipeline:**
1. Forward-fill gaps up to 5 consecutive days
2. Drop assets with >20% NaN values
3. Compute log returns: `np.log(prices / prices.shift(1))`
4. Drop rows where any remaining NaN exists
5. Minimum 180 observations required per asset

**Output:**
- `prices: pd.DataFrame` — DatetimeIndex, columns = asset symbols, values = close prices
- `returns: pd.DataFrame` — DatetimeIndex, columns = asset symbols, values = daily log returns

### 4.3 On-Chain Indicators (`core/data/onchain.py`)

**DeFiLlama endpoints (all free, no API key):**

| Endpoint | Returns | Cache TTL |
|----------|---------|-----------|
| `GET https://api.llama.fi/v2/historicalChainTvl` | Daily total crypto TVL time series | 6h |
| `GET https://api.llama.fi/v2/historicalChainTvl/{chain}` | Daily TVL for a specific chain (Ethereum, Solana, etc.) | 6h |
| `GET https://stablecoins.llama.fi/stablecoincharts/all` | Daily total stablecoin market cap | 6h |
| `GET https://stablecoins.llama.fi/stablecoins?includePrices=false` | Current stablecoin list with mcaps | 6h |
| `GET https://api.llama.fi/overview/dexs?excludeTotalDataChart=false` | DEX volume overview with daily chart | 6h |

**Signal derivation (computed in `core/data/onchain.py`):**

| Signal Name | Computation | Interpretation |
|-------------|-------------|----------------|
| `tvl_momentum_30d` | (TVL_today / TVL_30d_ago) - 1 | >0 = capital inflow = bullish |
| `stablecoin_dominance` | stablecoin_mcap / total_crypto_mcap | Rising = risk-off, falling = risk-on |
| `stablecoin_supply_change_30d` | (supply_today / supply_30d_ago) - 1 | >0 = new capital entering crypto |
| `dex_volume_trend_7d` | 7d SMA of daily DEX volume / 30d SMA | >1 = rising activity = bullish |
| `chain_tvl_share_{chain}` | chain_tvl / total_tvl | Relative chain dominance |

**These signals feed into Black-Litterman views** (see section 6.5).

### 4.4 Live Price Feed (`core/data/price_feed.py`)

**Implementation class: `BinancePriceFeed`**

```python
class BinancePriceFeed:
    def __init__(self, cache: diskcache.Cache, symbols: list[str]):
        """
        Args:
            cache: diskcache.Cache instance (thread-safe)
            symbols: list of Binance symbols, e.g., ["BTCUSDT", "ETHUSDT", ...]
        """

    def start(self) -> None:
        """Start the daemon background thread. Call once at app startup."""

    def stop(self) -> None:
        """Signal the thread to stop and close the WebSocket."""

    def get_price(self, symbol: str) -> float | None:
        """Read latest price from cache. Returns None if stale (>60s) or missing."""
```

**WebSocket URL construction:**
```python
streams = "/".join(f"{sym.lower()}@miniTicker" for sym in self.symbols)
url = f"wss://stream.binance.com:9443/stream?streams={streams}"
```

**Message parsing (combined stream envelope):**
```python
envelope = json.loads(message)
data = envelope["data"]
symbol = data["s"]      # "BTCUSDT"
price = float(data["c"])  # close/last price
self.cache.set(f"price:{symbol}", price, expire=60)
```

**Reconnection logic:**
- On disconnect: wait `min(2^attempt, 60)` seconds, then reconnect
- Reset backoff counter on successful connection
- Log every reconnection attempt at WARN level

---

## 5. Models Layer

### 5.1 Covariance Estimation (`core/models/covariance.py`)

**Methods to implement (all via riskfolio-lib):**

```python
import riskfolio as rp
import numpy as np
import pandas as pd

def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",  # "ledoit_wolf" | "gerber" | "denoised" | "sample"
) -> pd.DataFrame:
    """
    Estimate the covariance matrix from returns.

    Args:
        returns: T x N DataFrame of asset returns
        method: estimation method

    Returns:
        N x N covariance matrix as DataFrame
    """
```

**riskfolio-lib covariance methods:**
- `method="hist"` — sample covariance (default, unreliable for crypto)
- `method="ledoit"` — Ledoit-Wolf shrinkage (recommended default)
- `method="oas"` — Oracle Approximating Shrinkage
- `method="gerber1"` / `"gerber2"` — Gerber statistic (robust to outliers, good for crypto)
- `method="denoise"` — Random Matrix Theory denoised covariance (Marchenko-Pastur)

The method is set on the `rp.Portfolio` object:
```python
port = rp.Portfolio(returns=returns)
port.assets_stats(method_cov="ledoit")  # computes mu and cov internally
```

### 5.2 GARCH Volatility Forecasting (`core/models/garch.py`)

**Core implementation using `arch` library:**

```python
from arch import arch_model
import pandas as pd
import numpy as np

def fit_garch(
    returns: pd.Series,
    p: int = 1,
    o: int = 1,        # o=1 → GJR-GARCH (asymmetric volatility)
    q: int = 1,
    dist: str = "StudentsT",  # "Normal" | "StudentsT" | "SkewStudent"
    rescale: bool = True,
) -> dict:
    """
    Fit a GJR-GARCH(p,o,q) model to a single asset's return series.

    Args:
        returns: pd.Series of daily log returns
        p: GARCH lag order
        o: Asymmetric (GJR) lag order. 0 = standard GARCH, 1 = GJR-GARCH
        q: ARCH lag order
        dist: Error distribution. "StudentsT" is strongly recommended for crypto
        rescale: If True, multiply returns by 100 before fitting (arch convention)

    Returns:
        dict with keys:
            "conditional_volatility": pd.Series — fitted conditional vol (same index as returns)
            "forecast_variance": float — 1-step-ahead variance forecast
            "forecast_vol": float — sqrt of forecast_variance
            "params": dict — fitted model parameters
            "aic": float — Akaike Information Criterion
            "bic": float — Bayesian Information Criterion
            "model_result": arch result object (for advanced use)
    """
    scaled = returns * 100 if rescale else returns

    am = arch_model(
        scaled,
        vol="GARCH",
        p=p, o=o, q=q,
        dist=dist,
        mean="Constant",
        rescale=False,
    )
    result = am.fit(disp="off", show_warning=False)

    cond_vol = result.conditional_volatility
    if rescale:
        cond_vol = cond_vol / 100  # scale back

    forecast = result.forecast(horizon=1, reindex=False)
    fcast_var = forecast.variance.iloc[-1, 0]
    if rescale:
        fcast_var = fcast_var / (100 ** 2)

    return {
        "conditional_volatility": cond_vol,
        "forecast_variance": fcast_var,
        "forecast_vol": np.sqrt(fcast_var),
        "params": dict(result.params),
        "aic": result.aic,
        "bic": result.bic,
        "model_result": result,
    }


def build_garch_covariance(
    returns: pd.DataFrame,
    garch_params: dict | None = None,
) -> pd.DataFrame:
    """
    Build a GARCH-implied covariance matrix using the DCC-like approach:
    1. Fit GJR-GARCH(1,1,1) to each asset → get forecasted vol σ_i
    2. Compute constant correlation matrix R from standardized residuals
    3. Covariance = D @ R @ D, where D = diag(σ_1, ..., σ_N)

    Args:
        returns: T x N DataFrame of daily log returns
        garch_params: optional dict of per-asset GARCH parameters

    Returns:
        N x N GARCH-implied covariance matrix as DataFrame
    """
```

**Per-asset GARCH fitting loop:**
```python
def fit_all_garch(returns: pd.DataFrame) -> dict[str, dict]:
    """Fit GARCH to every asset in the universe. Returns dict keyed by column name."""
    results = {}
    for col in returns.columns:
        try:
            results[col] = fit_garch(returns[col])
        except Exception as e:
            # Fallback: use sample volatility if GARCH fails to converge
            vol = returns[col].std()
            results[col] = {
                "conditional_volatility": pd.Series(vol, index=returns.index),
                "forecast_vol": vol,
                "forecast_variance": vol ** 2,
                "params": {},
                "aic": np.nan,
                "bic": np.nan,
                "model_result": None,
                "error": str(e),
            }
    return results
```

### 5.3 Regime Detection (`core/models/regime.py`)

**Implementation using `hmmlearn.GaussianHMM`:**

```python
from hmmlearn import hmm
import numpy as np
import pandas as pd

def detect_regimes(
    returns: pd.Series,
    n_regimes: int = 2,      # 2 = bull/bear, 3 = bull/bear/sideways
    lookback_days: int = 730, # train on last 2 years
    random_state: int = 42,
) -> dict:
    """
    Detect market regimes using a Gaussian Hidden Markov Model.

    Args:
        returns: pd.Series of daily returns (typically BTC or a market index)
        n_regimes: number of hidden states
        lookback_days: training window
        random_state: for reproducibility

    Returns:
        dict with keys:
            "regimes": pd.Series — integer regime labels (same index as returns)
            "regime_names": dict[int, str] — e.g., {0: "Bear", 1: "Bull"}
            "transition_matrix": np.ndarray — NxN transition probability matrix
            "regime_means": np.ndarray — mean return per regime
            "regime_vols": np.ndarray — volatility per regime
            "current_regime": int — regime at the last observation
            "current_regime_name": str — "Bull", "Bear", or "Sideways"
    """
    X = returns.values[-lookback_days:].reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(X)

    hidden_states = model.predict(X)

    # Label regimes by mean return: highest mean = Bull, lowest = Bear
    means = model.means_.flatten()
    sorted_indices = np.argsort(means)
    # Map: sorted_indices[0] = Bear, sorted_indices[-1] = Bull
    name_map = {}
    if n_regimes == 2:
        name_map[sorted_indices[0]] = "Bear"
        name_map[sorted_indices[1]] = "Bull"
    elif n_regimes == 3:
        name_map[sorted_indices[0]] = "Bear"
        name_map[sorted_indices[1]] = "Sideways"
        name_map[sorted_indices[2]] = "Bull"

    regime_series = pd.Series(
        hidden_states,
        index=returns.index[-lookback_days:],
        name="regime",
    )

    return {
        "regimes": regime_series,
        "regime_names": name_map,
        "transition_matrix": model.transmat_,
        "regime_means": model.means_.flatten(),
        "regime_vols": np.sqrt(model.covars_.flatten()),
        "current_regime": int(hidden_states[-1]),
        "current_regime_name": name_map[hidden_states[-1]],
    }
```

### 5.4 Return Estimation (`core/models/returns.py`)

Simple expected return estimators:

```python
def estimate_returns(
    returns: pd.DataFrame,
    method: str = "historical",  # "historical" | "ewma" | "capm" | "james_stein"
) -> pd.Series:
    """Estimate expected returns for each asset."""
```

For riskfolio-lib, the method is set on the Portfolio object:
```python
port.assets_stats(method_mu="hist")  # or "ewma1", "ewma2", "JS" (James-Stein)
```

---

## 6. Optimization Layer

All optimization functions return a `PortfolioResult` dataclass:

```python
@dataclass
class PortfolioResult:
    name: str                    # Strategy name
    weights: pd.Series           # Asset weights (index = asset names, values = floats summing to 1)
    expected_return: float       # Annualized expected return
    expected_volatility: float   # Annualized expected volatility
    sharpe_ratio: float          # (return - rf) / vol
    metadata: dict               # Strategy-specific info (e.g., GARCH params, regime state)
```

### 6.1 Markowitz Mean-Variance Optimization (`core/optimization/markowitz.py`)

**Uses `riskfolio-lib`:**

```python
import riskfolio as rp
import pandas as pd

def optimize_markowitz(
    returns: pd.DataFrame,
    objective: str = "Sharpe",  # "Sharpe" | "MinRisk" | "MaxRet" | "Utility"
    risk_measure: str = "MV",   # "MV" (variance) | "CVaR" | "CDaR" | "EVaR"
    risk_free_rate: float = 0.0,
    method_cov: str = "ledoit",
    method_mu: str = "hist",
    max_weight: float = 0.15,   # 15% max per asset for 50-asset universe
    min_weight: float = 0.0,
) -> PortfolioResult:
    """
    Run Markowitz mean-variance optimization.

    riskfolio-lib API:
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu=method_mu, method_cov=method_cov)
        port.upperlng = max_weight  # upper bound per asset (long only)
        port.lowerlng = min_weight  # lower bound per asset
        w = port.optimization(model="Classic", rm=risk_measure, obj=objective, rf=risk_free_rate)
    """


def compute_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    risk_measure: str = "MV",
    method_cov: str = "ledoit",
    method_mu: str = "hist",
    max_weight: float = 0.15,
) -> dict:
    """
    Compute the efficient frontier.

    riskfolio-lib API:
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu=method_mu, method_cov=method_cov)
        frontier = port.efficient_frontier(model="Classic", rm=risk_measure, points=n_points, rf=0)

    Returns:
        dict with keys:
            "frontier_returns": list[float]
            "frontier_risks": list[float]
            "frontier_weights": pd.DataFrame — (n_points x N_assets)
            "max_sharpe_weights": pd.Series
            "min_vol_weights": pd.Series
            "asset_returns": pd.Series — individual asset expected returns
            "asset_risks": pd.Series — individual asset volatilities
    """


def plot_efficient_frontier(
    returns: pd.DataFrame,
    method_cov: str = "ledoit",
    max_weight: float = 0.15,
) -> plotly.graph_objects.Figure:
    """
    Generate the efficient frontier plot using riskfolio-lib's built-in plotting,
    then convert to Plotly for Dash compatibility.

    riskfolio-lib has matplotlib-based plots:
        ax = rp.PlotFunctions.plot_frontier(
            w_frontier=frontier,
            mu=port.mu,
            cov=port.cov,
            returns=returns,
            rm=risk_measure,
            rf=0,
        )

    Since Dash uses Plotly, we extract the frontier data and build a Plotly figure.
    Plot elements:
        - Efficient frontier curve (line)
        - Individual assets (scatter points with labels)
        - Max Sharpe portfolio (star marker)
        - Min Volatility portfolio (diamond marker)
        - Capital Market Line (dashed line from rf through max Sharpe)
    """
```

### 6.2 Hierarchical Risk Parity (`core/optimization/hrp.py`)

```python
import riskfolio as rp

def optimize_hrp(
    returns: pd.DataFrame,
    codependence: str = "pearson",  # "pearson" | "spearman" | "gerber1" | "gerber2"
    covariance: str = "ledoit",     # covariance estimation for risk budgeting
    linkage: str = "ward",          # "ward" | "single" | "complete" | "average"
    risk_measure: str = "MV",       # risk measure for leaf allocation
    max_weight: float = 0.15,
) -> PortfolioResult:
    """
    Hierarchical Risk Parity (Lopez de Prado).

    riskfolio-lib API:
        port = rp.HCPortfolio(returns=returns)
        w = port.optimization(
            model="HRP",
            codependence=codependence,
            covariance=covariance,
            rm=risk_measure,
            linkage=linkage,
        )

    Also generate dendrogram data for visualization:
        ax = rp.PlotFunctions.plot_dendrogram(
            returns=returns,
            codependence=codependence,
            linkage=linkage,
        )
    """
```

### 6.3 True Risk Parity — Equal Risk Contribution (`core/optimization/risk_parity.py`)

```python
import riskfolio as rp

def optimize_risk_parity(
    returns: pd.DataFrame,
    risk_measure: str = "MV",     # risk contribution measured by
    method_cov: str = "ledoit",
    risk_budget: list[float] | None = None,  # None = equal risk contribution
) -> PortfolioResult:
    """
    True Risk Parity (Equal Risk Contribution).
    Each asset contributes equally to total portfolio risk.

    riskfolio-lib API:
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_cov=method_cov)
        # Risk budgeting: equal contribution
        port.risk_contrib = None  # None = equal risk contribution
        w = port.rp_optimization(model="Classic", rm=risk_measure, rf=0)

    NOT the same as inverse volatility (which ignores correlations).
    """
```

### 6.4 Mean-CVaR Optimization (`core/optimization/cvar.py`)

```python
import riskfolio as rp

def optimize_cvar(
    returns: pd.DataFrame,
    alpha: float = 0.05,       # CVaR confidence level (5% tail)
    objective: str = "MinRisk", # "MinRisk" | "Sharpe" | "MaxRet"
    method_cov: str = "ledoit",
    method_mu: str = "hist",
    max_weight: float = 0.15,
) -> PortfolioResult:
    """
    Mean-CVaR optimization. Uses Conditional Value-at-Risk instead of variance.
    Better suited for crypto's fat-tailed return distributions.

    riskfolio-lib API:
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu=method_mu, method_cov=method_cov)
        port.alpha = alpha
        port.upperlng = max_weight
        w = port.optimization(model="Classic", rm="CVaR", obj=objective, rf=0)
    """
```

### 6.5 Black-Litterman with On-Chain Views (`core/optimization/black_litterman.py`)

```python
import riskfolio as rp
import numpy as np
import pandas as pd

def optimize_black_litterman(
    returns: pd.DataFrame,
    market_caps: pd.Series,         # market cap per asset (for equilibrium returns)
    views: pd.DataFrame,            # P matrix (K x N) — view portfolios
    view_returns: pd.Series,        # Q vector (K x 1) — expected returns per view
    view_confidences: pd.Series,    # confidence per view (0 to 1)
    risk_free_rate: float = 0.0,
    method_cov: str = "ledoit",
    max_weight: float = 0.15,
) -> PortfolioResult:
    """
    Black-Litterman model combining market equilibrium with subjective views.

    On-chain signals translate to views:
    - tvl_momentum_30d > 5% → "Ethereum outperforms market by 2% next month"
    - stablecoin_supply_change > 3% → "Crypto market returns 1% above equilibrium"
    - dex_volume_trend > 1.5 → "DeFi tokens (UNI, AAVE, MKR) outperform by 3%"

    riskfolio-lib API:
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov=method_cov)
        port.blacklitterman_stats(
            P=views,          # K x N matrix
            Q=view_returns,   # K x 1 vector
            delta=2.5,        # risk aversion coefficient
            rf=risk_free_rate,
            eq=True,          # use equilibrium returns
        )
        w = port.optimization(model="BL", rm="MV", obj="Sharpe")
    """


def generate_onchain_views(
    onchain_signals: dict,
    universe_assets: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Translate on-chain signals into Black-Litterman views.

    Returns (P, Q, confidences):
        P: K x N DataFrame — each row is a view portfolio
        Q: K-length Series — expected excess return per view
        confidences: K-length Series — confidence level per view (0-1)

    View generation rules:
    1. If tvl_momentum_30d > 0.05:
       → View: ETH outperforms BTC by 2% (P row: +1 on ETH, -1 on BTC)
       → Confidence: min(tvl_momentum_30d / 0.2, 1.0)

    2. If stablecoin_supply_change_30d > 0.03:
       → View: Market returns 1% above equilibrium (P row: equal weight on all assets)
       → Confidence: min(stablecoin_supply_change_30d / 0.1, 1.0)

    3. If dex_volume_trend_7d > 1.3:
       → View: DeFi tokens (UNI, AAVE, MKR, LINK, SNX) outperform by 3%
       → Confidence: min((dex_volume_trend_7d - 1.0) / 0.5, 1.0)

    4. For each chain with tvl_change > 0.1:
       → View: That chain's native token outperforms by 1.5%
       → Confidence: 0.5
    """
```

### 6.6 Regime-Aware Allocation (`core/optimization/regime_alloc.py`)

```python
def optimize_regime_aware(
    returns: pd.DataFrame,
    regime_info: dict,           # output from detect_regimes()
    bull_strategy: str = "max_sharpe",  # strategy to use in bull regime
    bear_strategy: str = "min_vol",     # strategy to use in bear regime
    method_cov: str = "ledoit",
    max_weight: float = 0.15,
) -> PortfolioResult:
    """
    Regime-aware allocation: switches strategy based on current detected regime.

    Logic:
        if current_regime == "Bull":
            Run max Sharpe Markowitz (risk-on)
        elif current_regime == "Bear":
            Run min volatility or HRP (defensive)
        elif current_regime == "Sideways":
            Run Risk Parity (balanced)

    The metadata dict includes regime detection details for display.
    """
```

### 6.7 GARCH-Enhanced GMV (`core/optimization/markowitz.py` — additional function)

```python
def optimize_garch_gmv(
    returns: pd.DataFrame,
    max_weight: float = 0.15,
) -> PortfolioResult:
    """
    Global Minimum Variance using GARCH-forecasted covariance matrix.

    Steps:
    1. Call fit_all_garch(returns) → get per-asset forecast volatilities
    2. Call build_garch_covariance(returns) → get GARCH-implied covariance matrix
    3. Pass to riskfolio-lib with custom covariance:

        port = rp.Portfolio(returns=returns)
        port.mu = expected_returns  # from returns.py
        port.cov = garch_cov_matrix  # from garch.py
        port.upperlng = max_weight
        w = port.optimization(model="Classic", rm="MV", obj="MinRisk", rf=0)

    This is the star strategy — it demonstrates GARCH + portfolio optimization integration.
    """
```

### 6.8 Equal Weight Benchmark (`core/optimization/equal_weight.py`)

```python
def equal_weight(n_assets: int, asset_names: list[str]) -> PortfolioResult:
    """1/N equal weight allocation. Benchmark for all other strategies."""
```

---

## 7. Risk & Analytics Layer

### 7.1 Risk Metrics (`core/risk/metrics.py`)

```python
def compute_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    ann_factor: int = 365,       # crypto trades 365 days/year
) -> dict:
    """
    Compute comprehensive risk metrics for a portfolio return series.

    Returns dict with keys:
        "annualized_return": float
        "annualized_volatility": float
        "sharpe_ratio": float
        "sortino_ratio": float      # downside deviation only
        "calmar_ratio": float       # return / max drawdown
        "omega_ratio": float        # probability-weighted gain/loss ratio
        "max_drawdown": float       # worst peak-to-trough decline
        "max_drawdown_duration": int # days in worst drawdown
        "var_95": float             # Value-at-Risk at 95% confidence
        "cvar_95": float            # Conditional VaR (Expected Shortfall) at 95%
        "var_99": float             # VaR at 99%
        "cvar_99": float            # CVaR at 99%
        "skewness": float           # return distribution skewness
        "kurtosis": float           # return distribution excess kurtosis
        "positive_days_pct": float  # % of days with positive returns
        "best_day": float           # best single-day return
        "worst_day": float          # worst single-day return
        "equity_curve": pd.Series   # cumulative wealth (starting at 1.0)
        "drawdown_series": pd.Series # drawdown at each point in time
    """
```

### 7.2 Walk-Forward Backtester (`core/risk/backtest.py`)

```python
@dataclass
class BacktestConfig:
    start_date: str                    # "2024-03-17"
    end_date: str                      # "2026-03-17"
    rebalance_frequency: str           # "monthly" | "weekly" | "quarterly"
    strategy: str                      # "markowitz" | "hrp" | "risk_parity" | "cvar" | "black_litterman" | "regime" | "garch_gmv" | "equal_weight"
    lookback_days: int = 365           # training window for optimization
    transaction_cost_bps: float = 10   # 10 bps = 0.1% per trade (Binance taker fee)
    max_weight: float = 0.15
    initial_capital: float = 100_000   # USD

@dataclass
class BacktestResult:
    equity_curve: pd.Series            # daily portfolio value
    returns: pd.Series                 # daily returns
    weights_history: pd.DataFrame      # weights at each rebalance date
    turnover_history: pd.Series        # turnover at each rebalance
    transaction_costs_total: float     # cumulative transaction costs
    rebalance_dates: list[pd.Timestamp]
    metrics: dict                      # output of compute_risk_metrics()


def run_backtest(
    prices: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """
    Walk-forward backtest.

    Algorithm:
    1. Set rebalance dates based on frequency
    2. At each rebalance date t:
       a. Take returns from [t - lookback_days, t] as training window
       b. Run the specified strategy to get target weights
       c. Compute turnover = sum(|w_new - w_old|)
       d. Apply transaction cost = turnover * transaction_cost_bps / 10000
       e. Rebalance the portfolio
    3. Between rebalances, portfolio drifts with market prices
    4. Track equity curve, returns, weights history, turnover
    5. Compute risk metrics on the full backtest return series

    IMPORTANT: No look-ahead bias. At time t, only data from [0, t] is used.
    """
```

### 7.3 Tearsheet Generation (`core/analytics/tearsheet.py`)

```python
import quantstats as qs

def generate_tearsheet(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,  # e.g., BTC returns
    title: str = "Portfolio Performance Report",
    output_path: str = "/tmp/tearsheet.html",
) -> str:
    """
    Generate a quantstats HTML tearsheet.

    quantstats API:
        qs.reports.html(
            returns,
            benchmark=benchmark_returns,
            title=title,
            output=output_path,
        )

    The HTML file includes:
    - Cumulative returns chart
    - Rolling Sharpe ratio
    - Drawdown analysis
    - Monthly returns heatmap
    - Annual returns bar chart
    - Distribution of returns
    - Worst drawdowns table
    - Risk metrics summary

    Returns: path to the generated HTML file
    """
```

---

## 8. Dashboard Layer

### 8.1 Theme Configuration (`dashboard/theme.py`)

```python
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

# DARKLY theme — professional dark trading terminal look
THEME = dbc.themes.DARKLY
THEME_NAME = "darkly"

def init_theme():
    """Call at app startup to load matching Plotly figure templates."""
    load_figure_template(THEME_NAME)

# Plotly layout defaults for all figures
FIGURE_LAYOUT = dict(
    template=THEME_NAME,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#AAAAAA"),
    margin=dict(l=40, r=20, t=40, b=40),
)
```

### 8.2 App Layout (`dashboard/layout.py`)

**Main layout structure:**

```
┌──────────────────────────────────────────────────────────────────┐
│  HEADER: "Quantitative Crypto Portfolio Management"              │
│  Subtitle: Live status indicator + last update time              │
├──────────────────────────────────────────────────────────────────┤
│  TABS:                                                           │
│  [Market] [Frontier] [Strategies] [GARCH] [Risk] [On-Chain]     │
│  [Backtest] [Report]                                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TAB CONTENT AREA (changes based on selected tab)                │
│                                                                  │
│                                                                  │
│                                                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
│  HIDDEN: dcc.Interval (5s for live prices)                       │
│  HIDDEN: dcc.Store (selected assets, date range, strategy params)│
└──────────────────────────────────────────────────────────────────┘
```

**Tab specifications:**

#### Tab 1: Market Overview
- **Top bar**: Live prices for top 10 assets (updated every 5s via `dcc.Interval`)
- **Universe table**: All ~45 assets with: rank, name, symbol, price, 24h%, 7d%, 30d%, market cap, volume
- **Market heatmap**: Treemap colored by 24h% change, sized by market cap
- **Controls**: Universe size selector (top 20 / 30 / 50), date range picker

#### Tab 2: Efficient Frontier
- **Interactive efficient frontier chart** (Plotly scatter + line):
  - Frontier curve (blue line)
  - Individual assets (labeled scatter points)
  - Max Sharpe portfolio (gold star)
  - Min Volatility portfolio (green diamond)
  - Capital Market Line (dashed)
  - Current portfolio (if user has selected a strategy — red dot)
- **Controls**: Covariance method dropdown, return estimation method dropdown, max weight slider
- **Asset expected returns & volatilities table** below the chart

#### Tab 3: Strategy Lab
- **Strategy selector**: Checkboxes to enable/disable strategies for comparison
- **For each enabled strategy**:
  - Weights bar chart
  - Key metrics row (Ann. Return, Volatility, Sharpe, Sortino, Calmar, MDD, CVaR)
- **Comparison section**:
  - Combined equity curves chart (all strategies overlaid)
  - Comparison table (strategies as columns, metrics as rows)
  - Weights comparison heatmap (strategies x assets)
- **Controls**: Max weight slider, risk-free rate input, optimize button
- **"Optimize" button** triggers computation (strategies do NOT auto-compute on page load)

#### Tab 4: GARCH Volatility
- **Per-asset GARCH results** (selectable via dropdown):
  - Conditional volatility time series chart
  - Forecasted vs realized volatility comparison
  - Model parameters table (alpha, beta, gamma, nu)
  - AIC/BIC model comparison
- **Volatility heatmap**: Assets x time, colored by GARCH conditional vol
- **GARCH-implied correlation matrix** heatmap
- **Controls**: GARCH variant selector (GARCH / GJR-GARCH / EGARCH), distribution selector (Normal / Student-t / Skew-t)

#### Tab 5: Risk Dashboard
- **Correlation heatmap**: NxN matrix of asset correlations (clustered)
- **Drawdown analysis**: For currently selected strategy
  - Drawdown chart (area plot)
  - Top 5 worst drawdowns table (start date, end date, depth, recovery time)
- **Return distribution**: Histogram with normal overlay, QQ plot
- **Rolling metrics**: 30d rolling Sharpe, 30d rolling volatility
- **Regime detection panel**:
  - Price chart with regime bands (Bull = green, Bear = red, Sideways = yellow)
  - Current regime indicator
  - Transition probability matrix display
  - Regime statistics table (mean return, volatility, duration per regime)

#### Tab 6: On-Chain Signals
- **Total crypto TVL chart** (line, from DeFiLlama)
- **TVL by top 5 chains** (stacked area chart)
- **Stablecoin market cap chart** (line)
- **DEX volume trend chart** (bar chart, 7d vs 30d MA)
- **Signal summary table**: All derived signals with current values and interpretation (bullish/neutral/bearish)
- **Black-Litterman views table**: How signals translate to portfolio views (auto-generated from `generate_onchain_views()`)

#### Tab 7: Backtest Engine
- **Configuration panel**:
  - Strategy selector dropdown
  - Date range picker (start, end)
  - Rebalance frequency dropdown (weekly / monthly / quarterly)
  - Transaction cost input (bps)
  - Lookback window slider (days)
  - Initial capital input
  - **"Run Backtest" button**
- **Results** (appear after button click):
  - Equity curve chart
  - Drawdown chart
  - Monthly returns heatmap (using Plotly heatmap, not quantstats)
  - Turnover chart (bar chart at each rebalance date)
  - Weights evolution chart (stacked area over time)
  - Full metrics table
  - Total transaction costs display

#### Tab 8: Report Export
- **Strategy selector**: Choose which strategy's results to export
- **Benchmark selector**: Choose benchmark (BTC, ETH, Equal Weight, or None)
- **"Generate Report" button**: Creates quantstats HTML tearsheet
- **"Download Report" button**: `dcc.Download` component serves the HTML file
- Preview of key metrics while waiting

### 8.3 Callbacks Architecture

**Key Dash patterns used:**

```python
# State management via dcc.Store (client-side JSON)
dcc.Store(id="universe-store", storage_type="session")      # universe data
dcc.Store(id="prices-store", storage_type="memory")         # historical prices (large)
dcc.Store(id="returns-store", storage_type="memory")        # returns matrix
dcc.Store(id="strategy-results-store", storage_type="memory")  # optimization results

# Live price updates (every 5 seconds)
dcc.Interval(id="live-interval", interval=5000, n_intervals=0)

# Callback pattern: button-triggered computation
@callback(
    Output("strategy-results-store", "data"),
    Input("optimize-button", "n_clicks"),
    State("universe-store", "data"),
    State("max-weight-slider", "value"),
    prevent_initial_call=True,  # IMPORTANT: don't run on page load
)
def run_optimization(n_clicks, universe_data, max_weight):
    # Heavy computation here — only runs when button is clicked
    ...
```

**Callback dependency chain:**
```
[Page load] → load_universe() → [universe-store]
                                       ↓
[universe-store] → load_historical_data() → [prices-store] + [returns-store]
                                                     ↓
[Optimize button click] + [prices-store] → run_optimization() → [strategy-results-store]
                                                                        ↓
[strategy-results-store] → update_equity_chart(), update_weights_chart(), update_metrics(), ...

[live-interval] → update_live_prices() → [live-price-table] (independent of optimization chain)
```

### 8.4 Key Component Specifications

#### `dashboard/components/metric_card.py`
```python
def create_metric_card(title: str, value: str, subtitle: str = "", color: str = "primary") -> dbc.Card:
    """
    Create a single KPI metric card.

    Uses dbc.Card with:
    - dbc.CardBody containing:
      - html.H6(title, className="card-subtitle text-muted")
      - html.H3(value, className="card-title")
      - html.P(subtitle, className="card-text text-muted small")
    - className="mb-3 shadow-sm"
    """
```

#### `dashboard/components/efficient_frontier.py`
```python
def create_efficient_frontier_figure(
    frontier_data: dict,   # output of compute_efficient_frontier()
    current_portfolio: tuple[float, float] | None = None,
) -> go.Figure:
    """
    Build the interactive Plotly efficient frontier chart.

    Traces:
    1. go.Scatter — frontier curve (mode="lines", line_color="#00d4ff")
    2. go.Scatter — individual assets (mode="markers+text", text=asset_names)
    3. go.Scatter — max Sharpe point (mode="markers", marker_symbol="star", marker_color="gold")
    4. go.Scatter — min vol point (mode="markers", marker_symbol="diamond", marker_color="lime")
    5. go.Scatter — CML line (mode="lines", line_dash="dash")
    6. go.Scatter — current portfolio (mode="markers", marker_symbol="circle", marker_color="red") [optional]

    Axes: x="Annualized Volatility", y="Annualized Expected Return"
    """
```

#### `dashboard/components/regime_chart.py`
```python
def create_regime_chart(
    prices: pd.Series,         # BTC or market index prices
    regime_data: dict,         # output of detect_regimes()
) -> go.Figure:
    """
    Price chart with regime-colored background bands.

    Uses go.Scatter for price line + go.layout.Shape for each regime period.
    Colors: Bull = rgba(0, 200, 83, 0.15), Bear = rgba(255, 82, 82, 0.15), Sideways = rgba(255, 193, 7, 0.15)
    """
```

---

## 9. Configuration & Environment

### `.env.example` (committed to git)
```
APP_ENV=development
APP_PORT=8050
APP_DEBUG=true
APP_LOG_LEVEL=info

# CoinGecko (free demo key — register at coingecko.com/en/api)
COINGECKO_API_KEY=

# Cache
CACHE_DIR=.cache
CACHE_TTL_PRICES=14400
CACHE_TTL_ONCHAIN=21600

# Binance WebSocket (no key needed for public streams)
BINANCE_WS_ENABLED=true
```

### `.gitignore`
```
.env
.env.*
!.env.example
__pycache__/
*.pyc
.cache/
*.egg-info/
dist/
build/
.pytest_cache/
.venv/
venv/
node_modules/
*.mp4
```

### `.dockerignore`
```
.env
.env.*
!.env.example
__pycache__/
.git/
.venv/
venv/
*.mp4
*.pdf
.cache/
.pytest_cache/
```

---

## 10. Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8050
EXPOSE $PORT

CMD ["sh", "-c", "gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120"]
```

### Procfile (Railway)
```
web: gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

### railway.toml
```toml
[build]
builder = "RAILPACK"

[deploy]
startCommand = "gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 5
```

### Railway-specific notes
- Railway injects `PORT` environment variable — the app MUST bind to it
- Set all other env vars via Railway dashboard or CLI: `railway variables set KEY=VALUE`
- Free trial: 30 days, $5 credit. After that, Hobby plan at $5/month (covers low-traffic dashboard)
- Railway supports outbound WebSocket connections natively (Binance WS works)
- Services do NOT sleep on Railway (unlike Render free tier)
- `gunicorn --workers 1` is critical: multiple workers = multiple WebSocket threads = redundant connections

### Local development
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -r requirements.txt
python app.py
# Opens at http://localhost:8050
```

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Days 1–3)

**Goal**: Project skeleton, data pipeline, cache, basic Dash app running.

| # | Task | Files | Acceptance Criteria |
|---|------|-------|---------------------|
| 1.1 | Initialize project structure | All dirs + `__init__.py` files, `.env.example`, `.gitignore`, `.dockerignore`, `requirements.txt` | `pip install -r requirements.txt` succeeds. All dirs exist. |
| 1.2 | Implement diskcache singleton | `core/data/cache.py` | Cache reads/writes work, TTL expiry works, thread-safe test passes |
| 1.3 | Implement symbol mapping | `core/data/symbol_map.py` | `get_binance_symbols()` and `get_yfinance_tickers()` return correct values for top 50 |
| 1.4 | Implement universe builder | `core/data/universe.py` | Fetches top 50 from CoinGecko, filters stablecoins, returns `list[UniverseAsset]`, caches result |
| 1.5 | Implement historical data fetcher | `core/data/fetcher.py` | Downloads 2yr daily OHLCV for ~45 assets, cleans data, computes returns, caches to disk |
| 1.6 | Set up Dash app skeleton | `app.py`, `dashboard/theme.py`, `dashboard/layout.py` | Dash app runs at localhost:8050 with DARKLY theme, 8 empty tabs, header visible |
| 1.7 | Market Overview tab | `dashboard/callbacks/data_cb.py`, `dashboard/components/` | Universe table displays, market treemap renders, data loads from cache |

### Phase 2: Core Models (Days 4–7)

**Goal**: Covariance estimation, GARCH volatility, regime detection all working.

| # | Task | Files | Acceptance Criteria |
|---|------|-------|---------------------|
| 2.1 | Implement covariance estimation | `core/models/covariance.py` | Ledoit-Wolf, Gerber, and denoised methods produce valid NxN positive semi-definite matrices |
| 2.2 | Implement GARCH module | `core/models/garch.py` | `fit_garch()` fits GJR-GARCH to BTC returns, returns conditional vol + 1-step forecast. `fit_all_garch()` processes all assets. `build_garch_covariance()` produces valid cov matrix. |
| 2.3 | Implement regime detection | `core/models/regime.py` | `detect_regimes()` fits HMM on BTC returns, returns regime labels, transition matrix, correctly labels Bull/Bear |
| 2.4 | Implement return estimation | `core/models/returns.py` | Historical mean and James-Stein estimators work correctly |
| 2.5 | GARCH tab UI | `dashboard/callbacks/garch_cb.py`, `dashboard/components/garch_chart.py` | Dropdown selects asset, conditional vol chart renders, forecast values display, model params table shows |
| 2.6 | Risk Dashboard tab (regime section) | `dashboard/callbacks/regime_cb.py`, `dashboard/components/regime_chart.py` | Regime bands overlay on price chart, transition matrix displays, current regime indicator works |

### Phase 3: Optimization Strategies (Days 8–12)

**Goal**: All 7 strategies implemented, efficient frontier working, strategy comparison functional.

| # | Task | Files | Acceptance Criteria |
|---|------|-------|---------------------|
| 3.1 | Equal Weight benchmark | `core/optimization/equal_weight.py` | Returns 1/N weights, `PortfolioResult` dataclass works |
| 3.2 | Markowitz MVO + efficient frontier | `core/optimization/markowitz.py` | Max Sharpe, Min Vol, and custom objective work via riskfolio-lib. Efficient frontier computes 50 points. |
| 3.3 | GARCH-Enhanced GMV | Extends `core/optimization/markowitz.py` | Uses GARCH covariance matrix, produces valid weights |
| 3.4 | Hierarchical Risk Parity | `core/optimization/hrp.py` | HRP produces valid weights, dendrogram data extractable |
| 3.5 | True Risk Parity (ERC) | `core/optimization/risk_parity.py` | Equal risk contribution verified: each asset's risk contribution within 1% of 1/N |
| 3.6 | Mean-CVaR optimization | `core/optimization/cvar.py` | CVaR optimization produces valid weights, different from MV optimization |
| 3.7 | Black-Litterman with on-chain views | `core/optimization/black_litterman.py` | Views generated from on-chain data, BL optimization produces weights that reflect views |
| 3.8 | Regime-Aware allocation | `core/optimization/regime_alloc.py` | Switches between max Sharpe (bull) and min vol (bear) based on HMM regime |
| 3.9 | Efficient Frontier tab UI | `dashboard/callbacks/optimization_cb.py`, `dashboard/components/efficient_frontier.py` | Interactive frontier chart with individual assets, max Sharpe point, min vol point, CML |
| 3.10 | Strategy Lab tab UI | `dashboard/callbacks/optimization_cb.py`, components | All strategies displayed with weights, equity curves, metrics comparison table. Optimize button triggers computation. |
| 3.11 | Risk Dashboard tab (correlations + drawdown) | `dashboard/components/correlation_heatmap.py`, `dashboard/components/drawdown_chart.py` | Clustered correlation heatmap renders for ~45 assets. Drawdown chart and top-5 table work. |

### Phase 4: Backtesting & Analytics (Days 13–15)

**Goal**: Walk-forward backtester, risk metrics, tearsheet export.

| # | Task | Files | Acceptance Criteria |
|---|------|-------|---------------------|
| 4.1 | Risk metrics module | `core/risk/metrics.py` | All metrics compute correctly: Sharpe, Sortino, Calmar, Omega, VaR, CVaR, MDD, skew, kurtosis |
| 4.2 | Walk-forward backtester | `core/risk/backtest.py` | Backtest runs with monthly rebalancing, transaction costs applied, no look-ahead bias, equity curve produced |
| 4.3 | Backtest tab UI | `dashboard/callbacks/backtest_cb.py` | Config panel works, "Run Backtest" button triggers computation, results display: equity curve, drawdown, monthly heatmap, turnover, weights evolution |
| 4.4 | Tearsheet generation | `core/analytics/tearsheet.py` | quantstats HTML file generated successfully |
| 4.5 | Report Export tab UI | `dashboard/callbacks/report_cb.py` | Strategy/benchmark selectors work, Generate button creates file, Download button serves it |

### Phase 5: Live Data & On-Chain (Days 16–18)

**Goal**: Binance WebSocket live prices, on-chain data integration, Black-Litterman views.

| # | Task | Files | Acceptance Criteria |
|---|------|-------|---------------------|
| 5.1 | Binance WebSocket price feed | `core/data/price_feed.py` | Background thread connects, receives prices, writes to cache. Auto-reconnects on disconnect. |
| 5.2 | Live price callback | `dashboard/callbacks/live_cb.py` | Market Overview tab updates prices every 5s without triggering other callbacks |
| 5.3 | On-chain data fetcher | `core/data/onchain.py` | DeFiLlama TVL, stablecoin, DEX volume data fetched and cached |
| 5.4 | On-Chain signal derivation | `core/data/onchain.py` | All 5 signals computed correctly with interpretations |
| 5.5 | On-Chain tab UI | `dashboard/callbacks/onchain_cb.py`, `dashboard/components/onchain_charts.py` | TVL chart, stablecoin chart, DEX volume chart, signal table, BL views table all render |

### Phase 6: Polish & Deploy (Days 19–21)

**Goal**: Testing, error handling, Docker, Railway deployment, demo-ready.

| # | Task | Files | Acceptance Criteria |
|---|------|-------|---------------------|
| 6.1 | Unit tests | `tests/` | Tests pass for: covariance estimation, GARCH fitting, risk metrics, optimization (at least one test per strategy), backtest |
| 6.2 | Error handling & loading states | All callback files | Loading spinners on all heavy computations. Graceful error messages on API failures. Fallback to cached data when APIs are down. |
| 6.3 | Health endpoint | `app.py` | `GET /health` returns 200 OK |
| 6.4 | Dockerfile | `Dockerfile` | `docker build` succeeds, `docker run` serves app on port 8050 |
| 6.5 | Railway deployment | `Procfile`, `railway.toml` | App deployed and accessible at `*.up.railway.app`, live prices working |
| 6.6 | Pre-demo data seeding | Script or startup logic | On first deploy, pre-fetch and cache all universe + historical data so the app loads instantly for the presentation |
| 6.7 | Final UI polish | `dashboard/layout.py`, CSS | Consistent spacing, proper chart titles, responsive layout, no visual glitches |

---

## Appendix A: Current app.py Reference

The current `app.py` (548 lines) contains the following reusable logic that should be extracted into `core/`:

| Current function | Target location | Notes |
|------------------|-----------------|-------|
| `get_ann_factor()` | `core/risk/metrics.py` | Keep as utility |
| `calculate_risk_metrics()` | `core/risk/metrics.py` | Expand with more metrics |
| `perf_metrics()` | `core/risk/metrics.py` | Merge into `compute_risk_metrics()` |
| `normalize_base100()` | `core/risk/metrics.py` or utility | Keep as utility |
| `fetch_top_cryptos()` | `core/data/universe.py` | Expand to 50, add stablecoin filter |
| `load_data_yfinance()` | `core/data/fetcher.py` | Add Binance fallback, disk caching |
| `weights_equal_weight()` | `core/optimization/equal_weight.py` | Keep, wrap in PortfolioResult |
| `weights_gmv_shrink()` | DELETE — replaced by riskfolio-lib | Hand-rolled SLSQP replaced |
| `weights_inverse_vol()` | DELETE — replaced by true risk parity | Inverse vol is not true risk parity |
| `weights_momentum_topk()` | Consider keeping as an additional strategy or removing | Not in the core 7 strategies |
| `render_strategy_block()` | `dashboard/components/` | Decompose into individual components |

---

## Appendix B: Symbol Mapping Table

```python
COINGECKO_TO_BINANCE = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "tether": None,                  # IS USDT
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "usd-coin": None,                # Stablecoin
    "xrp": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
    "tron": "TRXUSDT",
    "toncoin": "TONUSDT",
    "cardano": "ADAUSDT",
    "avalanche-2": "AVAXUSDT",
    "shiba-inu": "SHIBUSDT",
    "polkadot": "DOTUSDT",
    "chainlink": "LINKUSDT",
    "bitcoin-cash": "BCHUSDT",
    "near": "NEARUSDT",
    "litecoin": "LTCUSDT",
    "uniswap": "UNIUSDT",
    "dai": None,                     # Stablecoin
    "leo-token": None,               # Not on Binance (Bitfinex)
    "internet-computer": "ICPUSDT",
    "ethereum-classic": "ETCUSDT",
    "stellar": "XLMUSDT",
    "monero": None,                  # Delisted Feb 2024
    "cosmos": "ATOMUSDT",
    "okb": None,                     # Not on Binance (OKX)
    "aptos": "APTUSDT",
    "filecoin": "FILUSDT",
    "hedera-hashgraph": "HBARUSDT",
    "cronos": None,                  # Limited availability
    "mantle": "MNTLUSDT",
    "vechain": "VETUSDT",
    "injective-protocol": "INJUSDT",
    "the-graph": "GRTUSDT",
    "arbitrum": "ARBUSDT",
    "aave": "AAVEUSDT",
    "stacks": "STXUSDT",
    "optimism": "OPUSDT",
    "kaspa": None,                   # Verify current listing
    "maker": "MKRUSDT",
    "immutable-x": "IMXUSDT",
    "theta-token": "THETAUSDT",
    "algorand": "ALGOUSDT",
    "fantom": "FTMUSDT",
    "eos": "EOSUSDT",
    "quant-network": "QNTUSDT",
    "flow": "FLOWUSDT",
    "render-token": "RENDERUSDT",
    "sui": "SUIUSDT",
    "pepe": "PEPEUSDT",
}

COINGECKO_TO_YFINANCE = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "xrp": "XRP-USD",
    "dogecoin": "DOGE-USD",
    "tron": "TRX-USD",
    "toncoin": "TON-USD",
    "cardano": "ADA-USD",
    "avalanche-2": "AVAX-USD",
    "shiba-inu": "SHIB-USD",
    "polkadot": "DOT-USD",
    "chainlink": "LINK-USD",
    "bitcoin-cash": "BCH-USD",
    "near": "NEAR-USD",
    "litecoin": "LTC-USD",
    "uniswap": "UNI-USD",
    "internet-computer": "ICP-USD",
    "ethereum-classic": "ETC-USD",
    "stellar": "XLM-USD",
    "cosmos": "ATOM-USD",
    "aptos": "APT-USD",
    "filecoin": "FIL-USD",
    "hedera-hashgraph": "HBAR-USD",
    "vechain": "VET-USD",
    "injective-protocol": "INJ-USD",
    "the-graph": "GRT-USD",
    "arbitrum": "ARB11841-USD",
    "aave": "AAVE-USD",
    "stacks": "STX-USD",
    "optimism": "OP-USD",
    "maker": "MKR-USD",
    "immutable-x": "IMX-USD",
    "theta-token": "THETA-USD",
    "algorand": "ALGO-USD",
    "fantom": "FTM-USD",
    "eos": "EOS-USD",
    "quant-network": "QNT-USD",
    "flow": "FLOW-USD",
    "render-token": "RNDR-USD",
    "sui": "SUI-USD",
    "pepe": "PEPE-USD",
    "mantle": "MNT-USD",
}
```

> **Note**: This mapping must be verified against live APIs before implementation. The top 50 changes frequently. Build a one-time bootstrap script (see `core/data/symbol_map.py` spec) that cross-references CoinGecko `/coins/markets` with Binance `/api/v3/exchangeInfo` to regenerate this dict.

---

## Appendix C: API Rate Limits

| API | Rate Limit | Key Required | Notes |
|-----|-----------|--------------|-------|
| CoinGecko Demo | 30 calls/min, 10k calls/month | Yes (free demo key) | Use `/coins/markets?per_page=50` to fetch all in 1 call |
| CoinGecko (no key) | 5-15 calls/min | No | Unreliable, may get 429 errors |
| yfinance | Undocumented, ~2000/hour | No | Cache aggressively, intermittent 403s reported |
| Binance REST | 1200 weight/min per IP | No | `GET /klines` = 1-5 weight per call |
| Binance WebSocket | 1024 streams/connection, 300 connections/5min | No | 24h hard disconnect |
| DeFiLlama | No documented limits | No | Free, no key needed |

**Rate limit strategy**: Fetch all data at startup, cache with 4h TTL for prices and 6h TTL for on-chain. During normal operation, the app makes near-zero API calls — everything reads from cache. Only the Binance WebSocket runs continuously.

---

## Appendix D: Critical Implementation Warnings

These are non-obvious gotchas discovered during research that MUST be respected during implementation:

1. **arch library**: GJR-GARCH is `arch_model(y, vol='GARCH', p=1, o=1, q=1)`. The `o=1` parameter makes it GJR-GARCH. There is NO `vol='GJR-GARCH'` string. Using `vol='EGARCH'` is a separate model.

2. **riskfolio-lib call order**: `port.blacklitterman_stats()` MUST be called before `port.optimization(model='BL', ...)`. If you call optimization first, you get garbage results.

3. **riskfolio-lib risk parity convergence**: `port.rp_optimization()` uses a log-barrier formulation that can fail on highly correlated crypto assets. Use `port.rrp_optimization()` (relaxed risk parity) as a fallback.

4. **riskfolio-lib efficient frontier output**: `port.efficient_frontier()` returns a DataFrame of weights per point (columns = frontier points, rows = assets). It does NOT return risk/return coordinates directly — those are computed inside `PlotFunctions.plot_frontier()`. To get coordinates for a custom Plotly chart, you must compute `sqrt(w.T @ cov @ w)` and `w.T @ mu` for each weight column.

5. **riskfolio-lib solvers**: Default is CLARABEL (free). MOSEK is needed for RLVaR/RLDaR risk measures. Always set `solvers=['CLARABEL', 'SCS']` as fallback chain on HCPortfolio.

6. **CoinGecko Demo API**: Historical data is capped at 365 days (policy change February 2024). For 2-year backtests, use yfinance as the primary historical source. CoinGecko is for universe metadata + fallback only.

7. **hmmlearn state ordering**: State labels are arbitrary and change between runs. ALWAYS identify Bull/Bear by comparing `model.means_` post-fit: `bull_state = np.argmax(model.means_.flatten())`.

8. **quantstats pandas 2.0**: Use `quantstats-lumi` (not `quantstats`) for pandas 2.x compatibility. Import: `import quantstats_lumi as qs`.

9. **DeFiLlama subdomains**: TVL endpoints use `api.llama.fi`, stablecoin endpoints use `stablecoins.llama.fi`. Using the wrong subdomain returns 404.

10. **arch conditional_volatility units**: The result is in the same units as input `y`. If you pass `returns * 100` (percentage returns), conditional_volatility is in percentage points. Divide by 100 when converting back.

11. **Binance WebSocket 24h limit**: Connections are hard-disconnected after exactly 24 hours regardless of ping/pong health. The reconnect loop is not optional — it is mandatory.

12. **gunicorn workers**: Use `--workers 1 --threads 4`. Multiple workers create multiple WebSocket background threads, each opening redundant Binance connections.

---

## Appendix E: Key riskfolio-lib Risk Measures

These are the valid `rm` parameter values for `port.optimization()`:

| Code | Name | Best For |
|------|------|----------|
| `"MV"` | Mean-Variance (standard deviation) | Traditional Markowitz |
| `"CVaR"` | Conditional Value-at-Risk | Tail risk (crypto) |
| `"EVaR"` | Entropic Value-at-Risk | Conservative tail risk |
| `"CDaR"` | Conditional Drawdown-at-Risk | Drawdown-focused |
| `"MDD"` | Maximum Drawdown | Max loss from peak |
| `"UCI"` | Ulcer Index | Pain index |
| `"WR"` | Worst Realization | Worst-case return |

**Recommended for this project**: `"MV"` for Markowitz, `"CVaR"` for Mean-CVaR strategy, `"MV"` for HRP/Risk Parity.

---

## Appendix F: Exact riskfolio-lib API Signatures

These are the confirmed exact API signatures from riskfolio-lib v7.2 documentation. Use these verbatim — do not guess parameter names.

### Portfolio Constructor
```python
port = rp.Portfolio(
    returns=df_returns,      # pd.DataFrame (T x N)
    sht=False,               # allow short positions
    uppersht=0.2,            # max absolute short weight
    upperlng=1,              # max long weight per asset
    budget=1,                # sum of weights constraint
    alpha=0.05,              # significance level for CVaR/EVaR
)
```

### assets_stats — method_cov valid values
`'hist'`, `'semi'`, `'ewma1'`, `'ewma2'`, `'ledoit'`, `'oas'`, `'shrunk'`, `'gl'`, `'jlogo'`, `'fixed'`, `'spectral'`, `'shrink'`, `'gerber1'`, `'gerber2'`

### assets_stats — method_mu valid values
`'hist'`, `'ewma1'`, `'ewma2'`, `'JS'` (James-Stein), `'BS'` (Bayes-Stein), `'BOP'`

### optimization — obj valid values
`'MinRisk'`, `'Utility'`, `'Sharpe'`, `'MaxRet'`

### optimization — model valid values
`'Classic'`, `'BL'` (Black-Litterman), `'FM'` (Factor Model), `'BLFM'`

### Black-Litterman stats
```python
port.blacklitterman_stats(
    P=P,              # np.array or pd.DataFrame (K x N) — views matrix
    Q=Q,              # pd.DataFrame (K x 1) — expected excess returns per view
    rf=0,             # risk-free rate
    w=None,           # equilibrium weights (None = market cap weighted)
    delta=None,       # risk aversion (None = auto-compute)
    eq=True,          # use equilibrium returns
)
# MUST be called BEFORE port.optimization(model='BL', ...)
```

### HCPortfolio optimization — model valid values
`'HRP'`, `'HERC'`, `'HERC2'`, `'NCO'`

### HCPortfolio optimization — codependence valid values
`'pearson'`, `'spearman'`, `'kendall'`, `'gerber1'`, `'gerber2'`, `'distance'`, `'mutual_info'`, `'tail'`, `'custom_cov'`

### HCPortfolio optimization — linkage valid values
`'single'`, `'complete'`, `'average'`, `'weighted'`, `'centroid'`, `'median'`, `'ward'`, `'DBHT'`
