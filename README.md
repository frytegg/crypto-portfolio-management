# Crypto Portfolio Management Dashboard

A modular, research-grade portfolio optimization dashboard for cryptocurrency assets. Implements 7 allocation strategies, GARCH volatility forecasting, Hidden Markov Model regime detection, on-chain signal integration, and walk-forward backtesting — all wrapped in a real-time interactive Plotly Dash interface.

**Live demo:** [crypto-portfolio-management.onrender.com](https://crypto-portfolio-management.onrender.com/)

---

## Features

### Portfolio Optimization (7 Strategies)

| Strategy | Description |
|---|---|
| **Equal Weight (1/N)** | Naive diversification baseline — allocates equally across all assets |
| **Markowitz MVO** | Classic mean-variance optimization on the efficient frontier |
| **GARCH-Enhanced GMV** | Global Minimum Variance using GJR-GARCH covariance forecasts |
| **Hierarchical Risk Parity (HRP)** | Tree-based clustering for robust, estimation-error-resistant allocation |
| **Equal Risk Contribution** | Risk parity — each asset contributes equally to total portfolio risk |
| **Mean-CVaR** | Minimizes Conditional Value-at-Risk (expected shortfall) at the 95th percentile |
| **Black-Litterman** | Combines market equilibrium with on-chain signal views via Bayesian updating |
| **Regime-Aware** | Dynamically shifts allocation based on HMM-detected market regimes |

### Volatility Modeling

- **GJR-GARCH(1,1,1)** with Student's t-distribution for heavy-tailed crypto returns
- Multi-step ahead volatility forecasting
- Per-asset conditional volatility visualization

### Regime Detection

- **Gaussian Hidden Markov Model** (2 or 3 states) fitted on market returns
- Automatic bull/bear state labeling by mean return comparison
- Regime probability timeline and transition matrix visualization

### On-Chain Signals

- **Total Value Locked (TVL)** trends from DeFiLlama
- **Stablecoin market cap** flows (risk-on/risk-off indicator)
- **DEX volume** trends
- **Per-chain TVL** breakdown
- Signals feed into Black-Litterman views for conviction-weighted allocation

### Walk-Forward Backtesting

- Configurable rebalance frequency (weekly, monthly, quarterly)
- Transaction cost modeling (configurable basis points)
- No look-ahead bias — training window strictly uses past data only
- Equity curve, drawdown chart, and full risk metric comparison across strategies

### Risk Analytics

- Annualized return, volatility, Sharpe, Sortino, Calmar, Omega ratios
- Value-at-Risk (95%, 99%) and Conditional VaR
- Maximum drawdown and drawdown duration
- Skewness, kurtosis, and tail risk statistics
- QuantStats-powered tearsheet report generation

### Live Market Data

- **Binance WebSocket** real-time price feed with automatic reconnection
- Live price ticker bar updating every 5 seconds
- Cache-backed architecture — dashboard callbacks never hit external APIs directly

---

## Architecture

```
crypto-portfolio-management/
├── app.py                     # Application entry point (Dash + gunicorn)
├── core/                      # Pure Python — no Dash/Plotly imports
│   ├── config.py              # pydantic-settings configuration
│   ├── logger.py              # structlog setup
│   ├── analytics/
│   │   └── tearsheet.py       # QuantStats report generation
│   ├── data/
│   │   ├── cache.py           # diskcache (SQLite-backed, 200MB limit)
│   │   ├── fetcher.py         # Historical OHLCV from yfinance/Binance
│   │   ├── onchain.py         # DeFiLlama TVL, stablecoins, DEX volume
│   │   ├── price_feed.py      # Binance WebSocket live prices
│   │   ├── symbol_map.py      # CoinGecko ↔ Binance ↔ yfinance mapping
│   │   └── universe.py        # Top-N asset selection via CoinGecko
│   ├── models/
│   │   ├── covariance.py      # Covariance matrix estimation
│   │   ├── garch.py           # GJR-GARCH volatility forecasting
│   │   ├── regime.py          # HMM regime detection
│   │   └── returns.py         # Return computation and cleaning
│   ├── optimization/
│   │   ├── _base.py           # PortfolioResult dataclass
│   │   ├── markowitz.py       # Mean-variance optimization
│   │   ├── hrp.py             # Hierarchical Risk Parity
│   │   ├── risk_parity.py     # Equal Risk Contribution
│   │   ├── cvar.py            # Mean-CVaR optimization
│   │   ├── black_litterman.py # Black-Litterman with on-chain views
│   │   ├── regime_alloc.py    # Regime-aware allocation
│   │   └── equal_weight.py    # 1/N baseline
│   └── risk/
│       ├── backtest.py        # Walk-forward backtester
│       └── metrics.py         # Portfolio risk metrics
├── dashboard/                 # Dash-only — imports from core/
│   ├── layout.py              # Tab-based layout assembly
│   ├── theme.py               # DARKLY theme, colors, figure template
│   ├── callbacks/             # One file per domain
│   │   ├── data_cb.py         # Data loading and refresh
│   │   ├── optimization_cb.py # Strategy execution
│   │   ├── garch_cb.py        # GARCH tab logic
│   │   ├── regime_cb.py       # Regime tab logic
│   │   ├── onchain_cb.py      # On-chain signals tab
│   │   ├── backtest_cb.py     # Backtest execution
│   │   ├── live_cb.py         # Live price updates
│   │   └── report_cb.py       # Tearsheet generation
│   └── components/            # Reusable chart builders
│       ├── metric_card.py     # KPI display cards
│       ├── efficient_frontier.py
│       ├── correlation_heatmap.py
│       ├── equity_chart.py
│       ├── drawdown_chart.py
│       ├── weights_chart.py
│       ├── garch_chart.py
│       ├── regime_chart.py
│       ├── dendrogram.py
│       ├── onchain_charts.py
│       └── comparison_table.py
└── tests/                     # pytest suite (70%+ coverage target)
```

### Design Principles

- **Strict separation**: `core/` is pure computation (zero Dash imports), `dashboard/` handles presentation
- **Cache-first**: Callbacks read from `diskcache` — never call external APIs during user interaction
- **Heavy computation on demand**: All optimization and backtests run only on button click (`prevent_initial_call=True`)
- **Single worker**: `gunicorn --workers 1 --threads 4` — required for WebSocket correctness and cache consistency

### Data Flow

```
Startup (background thread):
  CoinGecko → universe → cache
  yfinance/Binance → prices/returns → cache
  DeFiLlama → TVL, stablecoins, DEX volume → cache
  Binance WebSocket → live prices → cache (continuous)

Runtime:
  User action → callback reads cache → compute on demand → update charts
  Live ticker → 5s interval → read cache["price:{SYMBOL}"] → update display
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| **Web Framework** | Dash 2.18, dash-bootstrap-components (DARKLY theme) |
| **Optimization** | riskfolio-lib, cvxpy, CLARABEL solver |
| **Volatility** | arch (GJR-GARCH) |
| **Regime Detection** | hmmlearn (Gaussian HMM) |
| **Data** | pandas, numpy, scipy, yfinance |
| **Live Prices** | websocket-client (Binance streams) |
| **On-Chain** | DeFiLlama API (requests) |
| **Cache** | diskcache (SQLite-backed) |
| **Analytics** | quantstats-lumi |
| **Config** | pydantic-settings, python-dotenv |
| **Logging** | structlog (structured JSON) |
| **Server** | gunicorn |
| **Testing** | pytest, pytest-cov |

---

## Getting Started

### Prerequisites

- Python 3.11+
- A [CoinGecko Demo API key](https://www.coingecko.com/en/api) (free tier works)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/crypto-portfolio-management.git
cd crypto-portfolio-management

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the example env file
cp .env.example .env
```

Edit `.env` with your settings:

```env
APP_ENV=development
APP_DEBUG=true
PORT=8050
COINGECKO_API_KEY=your_api_key_here
CACHE_DIR=.cache
BINANCE_WS_ENABLED=true
```

### Running Locally

```bash
# Development server (with hot reload)
python app.py

# Production server
gunicorn app:server --bind 0.0.0.0:8050 --workers 1 --threads 4 --timeout 120
```

Open [http://localhost:8050](http://localhost:8050) in your browser.

---

## Deployment

The application is deployed on **Render** using Docker.

### Render

The project includes a `Dockerfile` and `Procfile` ready for Render deployment:

1. Connect your GitHub repository to a new Render Web Service
2. Set the environment to **Docker**
3. Add environment variables in the Render dashboard:
   - `APP_ENV=production`
   - `COINGECKO_API_KEY=your_key`
   - `BINANCE_WS_ENABLED=true`
4. Deploy

> **Important:** The app must run with a single worker (`--workers 1`) to maintain WebSocket connection integrity and cache consistency.

### Docker (Self-hosted)

```bash
docker build -t crypto-portfolio .
docker run -p 8050:8050 --env-file .env crypto-portfolio
```

---

## Testing

```bash
# Run full test suite with coverage
pytest

# Run with verbose output, stop on first failure
pytest -x -v
```

Coverage target: **70%+** on `core/` modules.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| **Market Overview** | Asset universe, correlation heatmap, data staleness indicator |
| **Optimization** | Run all 7 strategies, view weights, efficient frontier, comparison table |
| **GARCH Volatility** | Per-asset GJR-GARCH fit, conditional volatility plots, forecasts |
| **Regime Detection** | HMM regime probabilities, state timeline, transition matrix |
| **Risk Dashboard** | Full risk metrics: VaR, CVaR, drawdowns, Sharpe, Sortino, Calmar |
| **On-Chain Signals** | TVL trends, stablecoin flows, DEX volume, per-chain breakdown |
| **Backtest** | Walk-forward backtest with equity curves, drawdown charts, strategy comparison |
| **Live Prices** | Real-time Binance WebSocket price feed |
| **Report** | QuantStats tearsheet generation |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment: development, test, production |
| `APP_LOG_LEVEL` | `info` | Log level: debug, info, warn, error |
| `APP_DEBUG` | `false` | Enable Dash debug mode |
| `PORT` | `8050` | Server port |
| `COINGECKO_API_KEY` | — | CoinGecko API key (required for universe fetching) |
| `CACHE_DIR` | `.cache` | Path to diskcache directory |
| `CACHE_SIZE_LIMIT` | `200000000` | Cache size limit in bytes (200MB) |
| `BINANCE_WS_ENABLED` | `true` | Enable Binance WebSocket live price feed |
| `DEFAULT_MAX_WEIGHT` | `0.15` | Maximum single-asset weight constraint |
| `DEFAULT_LOOKBACK_DAYS` | `730` | Historical data lookback (2 years) |
| `TRANSACTION_COST_BPS` | `10.0` | Transaction cost in basis points for backtests |

---

## License

This project is for educational and research purposes.
