"""On-chain data from DeFiLlama and CoinGecko.

CRITICAL: DeFiLlama subdomains -- mixing them returns 404:
  - TVL endpoints:         https://api.llama.fi/...
  - Stablecoin endpoints:  https://stablecoins.llama.fi/...
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import structlog

from core.config import settings
from core.data.cache import cache_get_or_fetch

log = structlog.get_logger(__name__)

DEFILLAMA_TVL_BASE = "https://api.llama.fi"
DEFILLAMA_STABLES_BASE = "https://stablecoins.llama.fi"


@dataclass
class OnchainSignals:
    """Derived on-chain signals used for Black-Litterman views and regime detection."""

    tvl_momentum_30d: float
    stablecoin_dominance: float
    stablecoin_supply_change_30d: float
    dex_volume_trend_7d: float
    chain_tvl_shares: dict[str, float]
    as_of: pd.Timestamp


def fetch_onchain_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch all DeFiLlama endpoints. Returns dict with keys:
    'total_tvl', 'chain_tvl', 'stablecoin_mcap', 'stablecoins', 'dex_volume'.
    Cached for 6h.
    """
    raise NotImplementedError


def compute_onchain_signals(onchain_data: dict[str, pd.DataFrame]) -> OnchainSignals:
    """Derive the 5 signals from fetched data. Pure function, no I/O."""
    raise NotImplementedError
