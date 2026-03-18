# Monetary Value Safety for Portfolio Math

## Portfolio Optimization Context
This project does NOT execute trades — it computes optimal allocations.
However, precision matters for:
- Covariance matrix estimation (small numerical errors compound)
- Weight constraints (weights must sum to exactly 1.0)
- Risk metrics (VaR, CVaR sensitive to tail precision)
- Backtest P&L tracking

## Rules
- Use numpy/pandas float64 for all portfolio math — sufficient for optimization
- NEVER use float32 for returns, covariance, or weight computation
- When comparing weights sum to 1.0, use np.isclose(weights.sum(), 1.0, atol=1e-6)
- Transaction costs in backtest: use Decimal if tracking cumulative P&L over many periods
- GARCH volatility: always rescale (returns*100 input, /100 output) — see critical-warnings.md

## Display Precision
- Weights: 2 decimal places (e.g., 15.23%)
- Returns: 2 decimal places (e.g., 12.45%)
- Volatility: 2 decimal places (e.g., 23.67%)
- Sharpe/Sortino: 2 decimal places (e.g., 1.45)
- Prices: 2 decimal places for USD display
