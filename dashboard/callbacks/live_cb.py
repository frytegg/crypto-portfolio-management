"""Live price feed callbacks — fires every 5 seconds via dcc.Interval.

Reads price:{SYMBOL} from diskcache for top-10 assets and updates the live
price row in the Market Overview tab header. Does NOT trigger any other callback.

Also updates the data staleness indicator in the layout header.
"""
from __future__ import annotations

from datetime import datetime, timezone

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html, no_update
import structlog

from core.data.cache import cache, get_live_price
from core.data.universe import UniverseAsset
from dashboard.theme import COLORS

log = structlog.get_logger(__name__)

# Staleness threshold — data older than 6 hours triggers a warning badge
_STALE_THRESHOLD_SECONDS = 6 * 3600


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
    Output("data-staleness-indicator", "children"),
    Input("live-interval", "n_intervals"),
    State("universe-store", "data"),
    prevent_initial_call=True,
)
def update_live_prices(
    n_intervals: int,
    universe_data: list[dict] | None,
) -> tuple:
    """Read live prices from cache and update top-10 price badges.

    Fires every 5 seconds. Only updates the live price row — does not
    trigger any other callback or computation. Also updates the data
    staleness indicator.
    """
    try:
        # --- Staleness indicator ---
        staleness_badge = _build_staleness_badge()

        if not universe_data:
            return no_update, staleness_badge

        # Reconstruct top-10 assets
        top_assets = []
        for a_dict in universe_data[:10]:
            try:
                top_assets.append(UniverseAsset(**a_dict))
            except (TypeError, KeyError):
                continue

        if not top_assets:
            return no_update, staleness_badge

        badges = []

        for asset in top_assets:
            # Try live price from WebSocket cache, fall back to CoinGecko price
            live = None
            if asset.binance_symbol:
                live = get_live_price(asset.binance_symbol)

            if live is not None:
                price = live
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

        return badges, staleness_badge

    except Exception as exc:
        log.error("update_live_prices_failed", error=str(exc), exc_info=True)
        return no_update, no_update


def _build_staleness_badge() -> html.Div:
    """Build a status indicator showing WebSocket state and last update time.

    - Green dot + "Live" when WebSocket is connected (cache key "ws_connected")
    - Yellow dot + "Cached" when using cached prices
    - "Prices as of: HH:MM UTC" timestamp
    - Red warning if data is stale (>6h)
    """
    ws_connected = cache.get("ws_connected") is True
    updated_at_str = cache.get("meta:data_updated_at")

    if not updated_at_str:
        return html.Div(
            dbc.Badge("No data loaded", color="secondary", className="p-2"),
        )

    try:
        updated_at = datetime.fromisoformat(updated_at_str)
        now = datetime.now(timezone.utc)
        age_seconds = (now - updated_at).total_seconds()

        # Format timestamp as HH:MM UTC
        time_str = updated_at.strftime("%H:%M UTC")

        # Connection status dot + label
        if ws_connected:
            dot_color = COLORS["success"]
            status_label = "Live"
        else:
            dot_color = COLORS["warning"]
            status_label = "Cached"

        status_dot = html.Span(
            "\u25CF ",
            style={"color": dot_color, "fontSize": "0.9em"},
        )

        children = [
            status_dot,
            html.Span(
                status_label,
                style={"color": dot_color, "fontWeight": "bold", "fontSize": "0.85em"},
            ),
            html.Span(
                f"  \u2022  Prices as of: {time_str}",
                style={"color": COLORS["text_muted"], "fontSize": "0.85em"},
            ),
        ]

        # Add stale warning
        if age_seconds > _STALE_THRESHOLD_SECONDS:
            age_hours = int(age_seconds // 3600)
            children.append(
                dbc.Badge(
                    f"Stale ({age_hours}h)",
                    color="warning",
                    className="ms-2 p-1",
                    style={"fontSize": "0.75em"},
                ),
            )

        return html.Div(children, style={"display": "inline-flex", "alignItems": "center"})

    except (ValueError, TypeError):
        return html.Div()
