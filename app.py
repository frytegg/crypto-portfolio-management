"""Application entry point.

Exposes `server` for gunicorn: `gunicorn app:server -w 1 --threads 4`
Run locally: `python app.py`
"""

from __future__ import annotations

from datetime import datetime, timezone

import dash
import structlog
from flask import jsonify

from core.config import settings
from core.data.cache import cache  # noqa: F401 — ensure cache dir is created
from core.logger import setup_logging
from dashboard.callbacks import register_all_callbacks
from dashboard.layout import create_layout
from dashboard.theme import EXTERNAL_STYLESHEETS, load_figure_template

# 1. Logging
setup_logging()
log = structlog.get_logger(__name__)

# 2. Plotly template
load_figure_template()

# 3. Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,
    title="Crypto Portfolio Management",
    update_title="Loading...",
)
server = app.server  # Flask server for gunicorn

# Health endpoint — used by Render / load balancers / uptime monitors
@server.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}), 200

# 4. Layout
app.layout = create_layout()

# 5. Callbacks
register_all_callbacks(app)

# 6. Startup tasks (data seeding + WebSocket)
log.info(
    "app_startup",
    env=settings.APP_ENV,
    port=settings.PORT,
    cache_dir=settings.CACHE_DIR,
    ws_enabled=settings.BINANCE_WS_ENABLED,
)

import threading

from core.data.universe import fetch_universe
from core.data.fetcher import fetch_historical_data
from core.data.onchain import fetch_onchain_data
from core.data.price_feed import BinancePriceFeed

_STALE_THRESHOLD_SECONDS = 4 * 3600  # 4 hours


def _is_cache_stale() -> bool:
    """Check if cached data is older than 4 hours or missing."""
    ts = cache.get("meta:data_updated_at")
    if ts is None:
        return True
    try:
        updated_at = datetime.fromisoformat(ts)
        age = (datetime.now(timezone.utc) - updated_at).total_seconds()
        return age > _STALE_THRESHOLD_SECONDS
    except (ValueError, TypeError):
        return True


def _startup_data_seeding() -> None:
    """Seed all data caches on startup (runs in background thread).

    1. Fetch universe (if empty or stale)
    2. Fetch historical prices/returns
    3. Fetch on-chain data
    4. Pre-cache equal_weight optimization
    """
    try:
        # 1. Universe
        universe = fetch_universe()
        log.info("startup_universe_fetched", n_assets=len(universe))

        # 2. Historical prices/returns
        fetch_historical_data(universe)
        cache.set("meta:data_updated_at", datetime.now(timezone.utc).isoformat())
        log.info("startup_historical_data_fetched")

        # 3. On-chain data
        try:
            fetch_onchain_data()
            log.info("startup_onchain_data_fetched")
        except Exception as exc:
            log.warning("startup_onchain_fetch_failed", error=str(exc))

        # 4. Pre-cache equal_weight
        try:
            returns_df = cache.get("returns")
            if returns_df is not None and not returns_df.empty:
                from core.optimization.equal_weight import optimize_equal_weight
                result = optimize_equal_weight(returns_df)
                cache.set("precached_equal_weight", {
                    "name": result.name,
                    "weights": {k: float(v) for k, v in result.weights.items()},
                    "expected_return": float(result.expected_return),
                    "expected_volatility": float(result.expected_volatility),
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "metadata": result.metadata,
                })
                log.info("startup_equal_weight_precached")
        except Exception as exc:
            log.warning("startup_equal_weight_precache_failed", error=str(exc))

        # 5. Start WebSocket price feed
        if settings.BINANCE_WS_ENABLED and universe:
            binance_symbols = [a.binance_symbol for a in universe if a.binance_symbol]
            price_feed = BinancePriceFeed(cache, binance_symbols)
            price_feed.start()
            log.info("startup_price_feed_started", n_symbols=len(binance_symbols))
        else:
            log.info("startup_price_feed_skipped", ws_enabled=settings.BINANCE_WS_ENABLED)

        log.info("startup_data_seeding_complete")

    except Exception as exc:
        log.error("startup_data_seeding_failed", error=str(exc), exc_info=True)


# Run data seeding in background thread so gunicorn can start serving immediately
threading.Thread(target=_startup_data_seeding, name="data-seeding", daemon=True).start()

if __name__ == "__main__":
    app.run(debug=settings.APP_DEBUG, port=settings.PORT)
