"""Regime detection callbacks.

Tab layout with:
- Regime chart (price line with colored background rectangles)
- Transition matrix table
- Current regime badge
- Regime statistics cards
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
import structlog

from core.data.cache import cache
from core.models.regime import detect_regimes
from dashboard.components.metric_card import create_metric_card
from dashboard.components.regime_chart import create_regime_chart
from dashboard.theme import COLORS

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tab layout builder (called from data_cb.py tab routing)
# ---------------------------------------------------------------------------

def build_regime_tab(returns_summary: dict | None) -> html.Div:
    """Build the Regime Detection tab layout."""
    if not returns_summary or not returns_summary.get("columns"):
        return html.Div(
            dbc.Spinner(
                html.H5("Waiting for market data...", className="text-muted"),
                color="primary",
            ),
            className="text-center mt-5",
        )

    return html.Div([
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Number of Regimes", className="fw-bold mb-1"),
                dbc.RadioItems(
                    id="regime-n-regimes",
                    options=[
                        {"label": "2 (Bull / Bear)", "value": 2},
                        {"label": "3 (Bull / Sideways / Bear)", "value": 3},
                    ],
                    value=2,
                    inline=True,
                    className="mb-3",
                ),
            ], md=4),
            dbc.Col([
                dbc.Button(
                    "Detect Regimes",
                    id="regime-detect-button",
                    color="primary",
                    className="mt-4",
                ),
            ], md=3),
        ], className="mb-3"),

        # Spinner wrapping results
        dbc.Spinner(
            html.Div(id="regime-results-container"),
            color="primary",
            type="border",
        ),
    ])


# ---------------------------------------------------------------------------
# Callback: Detect regimes on button click
# ---------------------------------------------------------------------------

@callback(
    Output("regime-results-store", "data"),
    Output("regime-results-container", "children"),
    Input("regime-detect-button", "n_clicks"),
    State("returns-store", "data"),
    State("regime-n-regimes", "value"),
    prevent_initial_call=True,
)
def run_regime_detection(
    n_clicks: int | None,
    returns_summary: dict | None,
    n_regimes: int,
) -> tuple:
    """Run HMM regime detection on BTC returns and render results."""
    if not n_clicks or not returns_summary:
        return no_update, no_update

    returns_df: pd.DataFrame | None = cache.get("returns")
    if returns_df is None or returns_df.empty:
        return no_update, html.P("No returns data available.", className="text-muted")

    # Use BTC as the market indicator for regime detection
    # Fall back to first column if BTC not in universe
    columns = returns_summary.get("columns", [])
    btc_col = "BTC" if "BTC" in columns else (columns[0] if columns else None)
    if btc_col is None:
        return no_update, html.P("No asset data for regime detection.", className="text-muted")

    log.info("detecting_regimes", asset=btc_col, n_regimes=n_regimes)

    try:
        regime_data = detect_regimes(
            returns_df[btc_col],
            n_regimes=n_regimes,
        )

        # Serialize for dcc.Store
        serialized = {
            "regimes_values": regime_data["regimes"].values.tolist(),
            "regimes_index": [str(d) for d in regime_data["regimes"].index],
            "regime_names": {str(k): v for k, v in regime_data["regime_names"].items()},
            "transition_matrix": regime_data["transition_matrix"].tolist(),
            "regime_means": regime_data["regime_means"].tolist(),
            "regime_vols": regime_data["regime_vols"].tolist(),
            "current_regime": regime_data["current_regime"],
            "current_regime_name": regime_data["current_regime_name"],
            "btc_col": btc_col,
        }

        # Build the visual output
        content = _build_regime_content(regime_data, btc_col)

        return serialized, content
    except Exception as exc:
        log.error("run_regime_detection_failed", error=str(exc), exc_info=True)
        return no_update, dbc.Alert(
            f"Regime detection failed: {exc}", color="danger", dismissable=True,
        )


# ---------------------------------------------------------------------------
# Layout builders
# ---------------------------------------------------------------------------

def _build_regime_content(regime_data: dict, btc_col: str) -> html.Div:
    """Build the regime detection results layout."""
    # Get prices for the chart
    prices_df: pd.DataFrame | None = cache.get("prices")
    prices = pd.Series(dtype=float)
    if prices_df is not None and btc_col in prices_df.columns:
        prices = prices_df[btc_col]

    # 1. Current regime badge
    current_name = regime_data["current_regime_name"]
    badge_color = {
        "Bull": "success",
        "Bear": "danger",
        "Sideways": "warning",
    }.get(current_name, "secondary")

    badge = dbc.Badge(
        f"Current Regime: {current_name}",
        color=badge_color,
        className="fs-5 p-2",
    )

    # 2. Regime chart
    regime_chart = create_regime_chart(prices, regime_data)

    # 3. Regime statistics cards
    name_map = regime_data["regime_names"]
    means = regime_data["regime_means"]
    vols = regime_data["regime_vols"]

    stat_cards = []
    for state, name in sorted(name_map.items(), key=lambda x: x[1]):
        card_color = {"Bull": "success", "Bear": "danger", "Sideways": "warning"}.get(
            name, "secondary"
        )
        stat_cards.append(
            dbc.Col(
                create_metric_card(
                    f"{name} Regime",
                    f"Mean: {means[state] * 100:.3f}%",
                    subtitle=f"Vol: {vols[state] * 100:.3f}%",
                    color=card_color,
                ),
                md=4 if len(name_map) <= 3 else 3,
            )
        )

    # 4. Transition matrix table
    transition_table = _build_transition_table(
        regime_data["transition_matrix"], name_map
    )

    return html.Div([
        dbc.Row([dbc.Col(badge, className="mb-3")]),
        dbc.Row(stat_cards, className="mb-4"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=regime_chart, config={"displayModeBar": False}),
                md=8,
            ),
            dbc.Col(transition_table, md=4),
        ]),
    ])


def _build_transition_table(
    transition_matrix: np.ndarray,
    name_map: dict[int, str],
) -> dbc.Card:
    """Build a card showing the regime transition probability matrix."""
    states = sorted(name_map.keys())
    names = [name_map[s] for s in states]

    header = html.Thead(
        html.Tr(
            [html.Th("From \\ To")] + [html.Th(n) for n in names]
        )
    )

    rows = []
    for i, from_name in enumerate(names):
        cells = [html.Td(from_name, style={"fontWeight": "bold"})]
        for j in range(len(names)):
            prob = transition_matrix[states[i]][states[j]]
            # Highlight high probabilities
            style = {}
            if prob > 0.5:
                style = {"color": COLORS["success"], "fontWeight": "bold"}
            cells.append(html.Td(f"{prob:.3f}", style=style))
        rows.append(html.Tr(cells))

    return dbc.Card([
        dbc.CardHeader("Transition Probabilities"),
        dbc.CardBody(
            dbc.Table(
                [header, html.Tbody(rows)],
                bordered=True,
                hover=True,
                color="dark",
                size="sm",
                className="mb-0",
            )
        ),
    ])
