# Crypto Portfolio Management Dashboard

## What This Project Does
Modular Plotly Dash application for multi-strategy crypto portfolio optimization (7 strategies), GARCH volatility forecasting, HMM regime detection, on-chain signal integration (Black-Litterman views), and walk-forward backtesting. Deployable on Railway.

## Architecture
- **Language**: Python 3.11+
- **Framework**: Dash + dash-bootstrap-components (DARKLY theme)
- **Optimization**: riskfolio-lib + cvxpy + CLARABEL solver
- **Volatility**: arch (GJR-GARCH)
- **Regime**: hmmlearn (GaussianHMM)
- **Cache**: diskcache (SQLite-backed, 200MB limit)
- **Live prices**: websocket-client (Binance miniTicker)
- **Analytics**: quantstats-lumi (pandas 2.x compatible)

### Iron Laws
1. `core/` = pure Python. ZERO Dash/Plotly imports
2. `dashboard/` = Dash-only. May import core/
3. Callbacks NEVER call external APIs — read from diskcache only
4. Heavy computation only on button click (`prevent_initial_call=True`)
5. gunicorn: `--workers 1 --threads 4` strictly (never change worker count)

## Directory Structure
```
core/data/       → universe, fetcher, onchain, price_feed, cache, symbol_map
core/models/     → covariance, garch, regime, returns
core/optimization/ → markowitz, hrp, risk_parity, cvar, black_litterman, regime_alloc, equal_weight
core/risk/       → metrics, backtest
core/analytics/  → tearsheet
dashboard/callbacks/ → data_cb, optimization_cb, garch_cb, regime_cb, onchain_cb, backtest_cb, live_cb, report_cb
dashboard/components/ → metric_card, efficient_frontier, correlation_heatmap, equity_chart, drawdown_chart, weights_chart, garch_chart, regime_chart, dendrogram, onchain_charts, comparison_table
```

## Key Commands
- `python app.py` — run development server (port 8050)
- `pytest` — run tests with coverage
- `pytest -x -v` — run tests, stop on first failure
- `gunicorn app:server -w 1 --threads 4 -b 0.0.0.0:8050` — production

## Critical Implementation Warnings
Read `.claude/rules/critical-warnings.md` BEFORE touching any module. Top 3:
1. **GJR-GARCH**: `arch_model(y, vol='GARCH', p=1, o=1, q=1)` — `vol='GJR-GARCH'` does NOT exist
2. **Black-Litterman**: `blacklitterman_stats()` MUST be called BEFORE `optimization(model='BL')` — no error raised otherwise
3. **quantstats**: `import quantstats_lumi as qs` — NOT `quantstats`

## Environment Variables
See `.env.example` — key ones: `COINGECKO_API_KEY`, `CACHE_DIR`, `PORT`

## Data Flow
```
Startup: CoinGecko → universe → cache | yfinance/Binance → prices → cache | DeFiLlama → onchain → cache
Runtime: callbacks read cache → compute on demand → update charts
Live:    BinancePriceFeed daemon thread → cache["price:{SYMBOL}"] every tick
```

## File Conventions
- Optimization strategies: `core/optimization/{name}.py` with `optimize_{name}()` returning `PortfolioResult`
- Dashboard callbacks: `dashboard/callbacks/{domain}_cb.py`
- Dashboard components: `dashboard/components/{name}.py` with `create_{name}()` returning `go.Figure` or `dbc` component
- Tests: `tests/test_{module_name}.py`
