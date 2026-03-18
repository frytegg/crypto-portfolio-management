# Data Layer Rules — Cache-First Architecture

## Cache TTL Reference
| Cache Key Pattern   | TTL      | Source           | Notes                          |
|---------------------|----------|------------------|--------------------------------|
| "universe"          | 4 hours  | CoinGecko        | Top-50 list, market cap ranks  |
| "prices"            | 4 hours  | yfinance/Binance | Full 2yr OHLCV DataFrame       |
| "returns"           | 4 hours  | computed          | log returns from prices        |
| "onchain:tvl"       | 6 hours  | DeFiLlama        | Daily total TVL series         |
| "onchain:chains"    | 6 hours  | DeFiLlama        | Per-chain TVL                  |
| "onchain:stables"   | 6 hours  | DeFiLlama        | Stablecoin market cap          |
| "onchain:dex"       | 6 hours  | DeFiLlama        | DEX volume chart               |
| "price:{SYMBOL}"    | 60 sec   | Binance WebSocket | One key per asset              |

## Rules for Callbacks
1. Callbacks read from diskcache — never from APIs
2. data_cb.py is the ONLY module that triggers data loading/refresh
3. Any callback that reads from cache must handle None (cache miss) gracefully —
   show a loading spinner or "Data loading..." placeholder, not an error page
4. live_cb.py fires every 5 seconds for price updates only — must complete in <500ms

## API Rate Limits
| API            | Limit              | Key      | Strategy                          |
|----------------|--------------------|----------|-----------------------------------|
| CoinGecko Demo | 30/min, 10k/month  | Required | 1 call at startup, cache 4h       |
| yfinance       | ~2000/hour (est.)  | None     | Batch download once, cache 4h     |
| Binance REST   | 1200 weight/min    | None     | Fallback only, 1-5 weight/call    |
| Binance WS     | 1024 streams/conn  | None     | One persistent connection         |
| DeFiLlama      | None documented    | None     | Fetch once, cache 6h              |

## Data Cleaning Contract (fetcher.py)
Every consumer of prices/returns DataFrames can assume:
- No NaN values in the returns DataFrame
- All assets have at least 180 observations
- DatetimeIndex is monotonically increasing
- Column names are uppercase ticker symbols (e.g., "BTC", "ETH")
- Values are daily log returns (not percentage, not cumulative)
