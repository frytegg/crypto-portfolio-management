"""Live price feed callbacks — fires every 5 seconds via dcc.Interval.

Reads price:{SYMBOL} from diskcache for top-10 assets and updates the live
price row in the Market Overview tab header. Does NOT trigger any other callback.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html, no_update
import structlog

from core.data.cache import get_live_price
from core.data.universe import UniverseAsset
from dashboard.theme import COLORS

log = structlog.get_logger(__name__)


def _fmt_price(price: float) -> str:
    """Format price with $ prefix and appropriate decimals."""
    if price >= 1.0:
        return f"${price:,.2f}"
    return f"${price:.4f}"


def _fmt_pct(value: float) -> str:
    """Format percentage with sign."""
    return f"{value:+.2f}%"


def _pct_color(value: float) -> str:
    """Return green/red/muted color based on sign."""
    if value > 0:
        return COLORS["success"]
    if value < 0:
        return COLORS["danger"]
    return COLORS["text_muted"]


@callback(
    Output("live-price-row", "children"),
    Input("live-interval", "n_intervals"),
    State("universe-store", "data"),
    prevent_initial_call=True,
)
def update_live_prices(
    n_intervals: int,
    universe_data: list[dict] | None,
) -> list:
    """Read live prices from cache and update top-10 price badges.

    Fires every 5 seconds. Only updates the live price row — does not
    trigger any other callback or computation.
    """
    if not universe_data:
        return no_update

    # Reconstruct top-10 assets
    top_assets = []
    for a_dict in universe_data[:10]:
        try:
            top_assets.append(UniverseAsset(**a_dict))
        except (TypeError, KeyError):
            continue

    if not top_assets:
        return no_update

    badges = []
    any_live = False

    for asset in top_assets:
        # Try live price from WebSocket cache, fall back to CoinGecko price
        live = None
        if asset.binance_symbol:
            live = get_live_price(asset.binance_symbol)

        if live is not None:
            price = live
            any_live = True
            # Compute change vs last known close from CoinGecko
            if asset.current_price and asset.current_price > 0:
                pct_change = ((price - asset.current_price) / asset.current_price) * 100
            else:
                pct_change = asset.price_change_24h
        else:
            price = asset.current_price
            pct_change = asset.price_change_24h

        color = _pct_color(pct_change)

        badge = dbc.Col(
            html.Div(
                [
                    html.Span(
                        asset.symbol,
                        style={"fontWeight": "bold", "marginRight": "4px"},
                    ),
                    html.Span(
                        _fmt_price(price),
                        style={"marginRight": "4px"},
                    ),
                    html.Span(
                        _fmt_pct(pct_change),
                        style={"color": color, "fontSize": "0.85em"},
                    ),
                    # Green dot indicator when receiving live data
                    html.Span(
                        " \u25CF" if live is not None else "",
                        style={
                            "color": COLORS["success"],
                            "fontSize": "0.6em",
                            "verticalAlign": "super",
                        },
                    ),
                ],
                className="text-center p-2",
                style={
                    "backgroundColor": COLORS["card_bg"],
                    "borderRadius": "6px",
                },
            ),
            className="mb-2",
        )
        badges.append(badge)

    return badges
