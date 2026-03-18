"""Application entry point.

Exposes `server` for gunicorn: `gunicorn app:server -w 1 --threads 4`
Run locally: `python app.py`
"""

from __future__ import annotations

import dash
import structlog

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

# 4. Layout
app.layout = create_layout()

# 5. Callbacks
register_all_callbacks(app)

# 6. Startup tasks (data prefetch + WebSocket)
log.info(
    "app_startup",
    env=settings.APP_ENV,
    port=settings.PORT,
    cache_dir=settings.CACHE_DIR,
    ws_enabled=settings.BINANCE_WS_ENABLED,
)

# TODO: Uncomment after implementing data layer
# from core.data.universe import fetch_universe
# from core.data.fetcher import fetch_historical_data
# from core.data.onchain import fetch_onchain_data
# from core.data.price_feed import BinancePriceFeed
#
# universe = fetch_universe()
# fetch_historical_data(universe)
# fetch_onchain_data()
#
# if settings.BINANCE_WS_ENABLED:
#     symbols = [a.binance_symbol for a in universe if a.binance_symbol]
#     price_feed = BinancePriceFeed(cache, symbols)
#     price_feed.start()

if __name__ == "__main__":
    app.run(debug=settings.APP_DEBUG, port=settings.PORT)
