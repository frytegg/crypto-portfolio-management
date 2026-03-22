# Critical Implementation Warnings — Read Before Touching These Modules

## arch Library (core/models/garch.py)

WARNING 1: GJR-GARCH parameter name
  CORRECT:   arch_model(y, vol='GARCH', p=1, o=1, q=1)
  WRONG:     arch_model(y, vol='GJR-GARCH', ...)   <- does not exist, raises ValueError
  The `o=1` parameter is what makes it GJR-GARCH (asymmetric term).

WARNING 2: Input scaling
  Always pass returns * 100 to arch_model for numerical stability.
  Divide conditional_volatility by 100 when converting back.
  Divide forecast_variance by 10000 when converting back.

WARNING 3: conditional_volatility units
  result.conditional_volatility is in the SAME units as input y.
  If y was returns*100, the output is in percentage points (not decimals).

## riskfolio-lib (core/optimization/)

WARNING 4: Black-Litterman call order — THIS CAUSES SILENT WRONG RESULTS
  CORRECT order:
    port.assets_stats(...)
    port.blacklitterman_stats(P=..., Q=...)   <- MUST come before optimization
    w = port.optimization(model='BL', ...)
  WRONG order (gives garbage weights, no error raised):
    port.assets_stats(...)
    w = port.optimization(model='BL', ...)    <- blacklitterman_stats not called

WARNING 5: efficient_frontier() output shape
  port.efficient_frontier() returns a weights DataFrame:
    rows = assets, columns = frontier points (point_0, point_1, ...)
  It does NOT return (risk, return) coordinates.
  To get coordinates for Plotly: compute manually for each column w:
    vol = np.sqrt((w.T @ cov @ w).iloc[0, 0]) * np.sqrt(365)
    ret = (w.T @ mu).iloc[0, 0] * 365

WARNING 6: Risk parity convergence on crypto
  port.rp_optimization() uses log-barrier formulation — can fail on highly
  correlated assets (common in crypto bear markets).
  Always wrap in try/except and fall back to port.rrp_optimization().

WARNING 7: Solver specification for HCPortfolio
  Always set: port = rp.HCPortfolio(returns=returns)
  Then: w = port.optimization(..., solvers=['CLARABEL', 'SCS'])
  CLARABEL is the default free solver. SCS is the fallback.

## hmmlearn (core/models/regime.py)

WARNING 8: State label ordering is arbitrary
  model.fit() assigns state 0, 1, 2 randomly between runs.
  ALWAYS identify Bull/Bear by comparing means AFTER fitting:
    bull_state = int(np.argmax(model.means_.flatten()))
    bear_state = int(np.argmin(model.means_.flatten()))
  Never hardcode state 0 = Bear or state 1 = Bull.

## quantstats (core/analytics/tearsheet.py)

WARNING 9: Wrong package name
  CORRECT:   import quantstats_lumi as qs
  WRONG:     import quantstats as qs    <- pandas 2.x incompatible, breaks silently

## DeFiLlama (core/data/onchain.py)

WARNING 10: Subdomain split — wrong subdomain returns 404 with no useful error
  TVL endpoints:         https://api.llama.fi/v2/historicalChainTvl
  Stablecoin endpoints:  https://stablecoins.llama.fi/stablecoincharts/all
  Using api.llama.fi for stablecoin endpoints = 404
  Using stablecoins.llama.fi for TVL endpoints = 404

## Binance WebSocket (core/data/price_feed.py)

WARNING 11: 24-hour hard disconnect is mandatory to handle
  Binance forcibly closes all WebSocket connections after exactly 24 hours.
  The reconnect loop MUST be implemented. This is not optional.
  Recommended: proactive reconnect at 23.5h to avoid the hard disconnect.

## Deployment (Procfile, Dockerfile, render.yaml)

WARNING 12: Worker count is critical for WebSocket correctness
  CORRECT:   gunicorn app:server --workers 1 --threads 4
  WRONG:     gunicorn app:server --workers 4    <- 4 WebSocket threads = 4x connections
  Multiple workers also break the shared diskcache singleton contract.
