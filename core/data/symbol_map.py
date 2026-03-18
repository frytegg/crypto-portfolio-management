"""Static mapping dictionaries: CoinGecko ID <-> Binance symbol <-> yfinance ticker.

These maps cover the top-50 crypto assets. They are hardcoded because:
1. CoinGecko IDs are stable (slug-based, e.g. "bitcoin")
2. Binance symbols follow {BASE}USDT convention
3. yfinance tickers follow {BASE}-USD convention

For assets not in these maps, fetcher.py will attempt dynamic resolution.
"""

from __future__ import annotations

# CoinGecko ID -> yfinance ticker
COINGECKO_TO_YFINANCE: dict[str, str] = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "ripple": "XRP-USD",
    "cardano": "ADA-USD",
    "dogecoin": "DOGE-USD",
    "polkadot": "DOT-USD",
    "chainlink": "LINK-USD",
    "avalanche-2": "AVAX-USD",
    "uniswap": "UNI-USD",
    "tron": "TRX-USD",
    "polygon-ecosystem-token": "POL-USD",
    "litecoin": "LTC-USD",
    "bitcoin-cash": "BCH-USD",
    "near": "NEAR-USD",
    "stellar": "XLM-USD",
    "internet-computer": "ICP-USD",
    "aptos": "APT-USD",
    "filecoin": "FIL-USD",
    "cosmos": "ATOM-USD",
    "hedera-hashgraph": "HBAR-USD",
    "arbitrum": "ARB-USD",
    "optimism": "OP-USD",
    "render-token": "RENDER-USD",
    "injective-protocol": "INJ-USD",
    "the-graph": "GRT-USD",
    "aave": "AAVE-USD",
    "algorand": "ALGO-USD",
    "fantom": "FTM-USD",
    "theta-token": "THETA-USD",
    "eos": "EOS-USD",
    "flow": "FLOW-USD",
    "axie-infinity": "AXS-USD",
    "decentraland": "MANA-USD",
    "the-sandbox": "SAND-USD",
    "maker": "MKR-USD",
    "quant-network": "QNT-USD",
    "sui": "SUI-USD",
    "sei-network": "SEI-USD",
    "celestia": "TIA-USD",
    "jupiter-exchange-solana": "JUP-USD",
    "starknet": "STRK-USD",
    "fetch-ai": "FET-USD",
    "ondo-finance": "ONDO-USD",
}

# CoinGecko ID -> Binance symbol (USDT pair)
COINGECKO_TO_BINANCE: dict[str, str] = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "ripple": "XRPUSDT",
    "cardano": "ADAUSDT",
    "dogecoin": "DOGEUSDT",
    "polkadot": "DOTUSDT",
    "chainlink": "LINKUSDT",
    "avalanche-2": "AVAXUSDT",
    "uniswap": "UNIUSDT",
    "tron": "TRXUSDT",
    "polygon-ecosystem-token": "POLUSDT",
    "litecoin": "LTCUSDT",
    "bitcoin-cash": "BCHUSDT",
    "near": "NEARUSDT",
    "stellar": "XLMUSDT",
    "internet-computer": "ICPUSDT",
    "aptos": "APTUSDT",
    "filecoin": "FILUSDT",
    "cosmos": "ATOMUSDT",
    "hedera-hashgraph": "HBARUSDT",
    "arbitrum": "ARBUSDT",
    "optimism": "OPUSDT",
    "render-token": "RENDERUSDT",
    "injective-protocol": "INJUSDT",
    "the-graph": "GRTUSDT",
    "aave": "AAVEUSDT",
    "algorand": "ALGOUSDT",
    "fantom": "FTMUSDT",
    "theta-token": "THETAUSDT",
    "eos": "EOSUSDT",
    "flow": "FLOWUSDT",
    "axie-infinity": "AXSUSDT",
    "decentraland": "MANAUSDT",
    "the-sandbox": "SANDUSDT",
    "maker": "MKRUSDT",
    "quant-network": "QNTUSDT",
    "sui": "SUIUSDT",
    "sei-network": "SEIUSDT",
    "celestia": "TIAUSDT",
    "jupiter-exchange-solana": "JUPUSDT",
    "starknet": "STRKUSDT",
    "fetch-ai": "FETUSDT",
    "ondo-finance": "ONDOUSDT",
}

# Reverse lookups
YFINANCE_TO_COINGECKO: dict[str, str] = {v: k for k, v in COINGECKO_TO_YFINANCE.items()}
BINANCE_TO_COINGECKO: dict[str, str] = {v: k for k, v in COINGECKO_TO_BINANCE.items()}


def get_yfinance_ticker(coingecko_id: str) -> str | None:
    """Resolve CoinGecko ID to yfinance ticker. Returns None if unmapped."""
    return COINGECKO_TO_YFINANCE.get(coingecko_id)


def get_binance_symbol(coingecko_id: str) -> str | None:
    """Resolve CoinGecko ID to Binance USDT pair. Returns None if unmapped."""
    return COINGECKO_TO_BINANCE.get(coingecko_id)


def get_display_symbol(coingecko_id: str) -> str:
    """Get short display symbol (e.g. 'BTC') from CoinGecko ID."""
    ticker = COINGECKO_TO_YFINANCE.get(coingecko_id, "")
    return ticker.replace("-USD", "") if ticker else coingecko_id.upper()
