"""Application configuration via pydantic-settings.

Reads from .env file and environment variables. Validates at import time —
if a required variable is missing or invalid, the app crashes immediately
with a clear error message.
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — import `settings` singleton, never use os.environ."""

    # Application
    APP_ENV: Literal["development", "test", "production"] = "development"
    APP_LOG_LEVEL: Literal["debug", "info", "warn", "error"] = "info"
    APP_DEBUG: bool = False
    PORT: int = 8050

    # API keys
    COINGECKO_API_KEY: str = ""

    # Cache
    CACHE_DIR: str = ".cache"
    CACHE_SIZE_LIMIT: int = 200_000_000  # 200 MB

    # Cache TTLs (seconds)
    CACHE_TTL_UNIVERSE: int = 14_400   # 4 hours
    CACHE_TTL_PRICES: int = 14_400     # 4 hours
    CACHE_TTL_ONCHAIN: int = 21_600    # 6 hours
    CACHE_TTL_LIVE_PRICE: int = 60     # 60 seconds per key

    # Binance WebSocket
    BINANCE_WS_ENABLED: bool = True

    # Portfolio defaults
    DEFAULT_MAX_WEIGHT: float = 0.15
    DEFAULT_LOOKBACK_DAYS: int = 730   # 2 years
    TRANSACTION_COST_BPS: float = 10.0

    @field_validator("COINGECKO_API_KEY")
    @classmethod
    def warn_missing_api_key(cls, v: str) -> str:
        if not v:
            warnings.warn(
                "COINGECKO_API_KEY not set — CoinGecko rate limits will be severe",
                stacklevel=2,
            )
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Module-level singleton — import this everywhere
settings = Settings()
