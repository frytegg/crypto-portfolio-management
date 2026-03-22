"""On-chain data and signal callbacks.

Tab 6: On-Chain Signals — TVL, stablecoin, DEX volume charts + signal table.
"""
from __future__ import annotations

from dataclasses import asdict

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
import structlog

from core.data.cache import cache
from core.data.onchain import (
    OnchainSignals,
    compute_onchain_signals,
    fetch_onchain_data,
)
from core.optimization.black_litterman import generate_onchain_views
from dashboard.components.onchain_charts import (
    create_dex_volume_chart,
    create_stablecoin_chart,
    create_tvl_chart,
)
from dashboard.theme import COLORS

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Signal interpretation colors
# ---------------------------------------------------------------------------

_INTERP_COLORS = {
    "Bullish": COLORS["success"],
    "Neutral": COLORS["warning"],
    "Bearish": COLORS["danger"],
}


def _fmt_pct(value: float) -> str:
    """Format as percentage with sign."""
    return f"{value:+.2%}"


def _fmt_ratio(value: float) -> str:
    """Format ratio to 2 decimal places."""
    return f"{value:.2f}"


def _fmt_share(value: float) -> str:
    """Format TVL share as percentage."""
    return f"{value:.1%}"


# ---------------------------------------------------------------------------
# Tab layout builder (called from data_cb tab routing)
# ---------------------------------------------------------------------------

def build_onchain_tab() -> html.Div:
    """Build the On-Chain Signals tab layout."""
    return html.Div([
        html.H4("On-Chain Signals", className="mb-2"),
        html.P(
            "DeFi metrics from DeFiLlama — TVL, stablecoin flows, DEX volume. "
            "Signals are used as Black-Litterman views for portfolio optimization.",
            className="text-muted mb-3",
            style={"fontSize": "0.9em"},
        ),

        dbc.Button(
            "Refresh On-Chain Data",
            id="onchain-refresh-btn",
            color="primary",
            className="mb-4",
        ),

        # Loading spinner wrapping all content
        dbc.Spinner(
            html.Div(id="onchain-content"),
            color="primary",
            type="border",
        ),
    ])


# ---------------------------------------------------------------------------
# Main callback: load/refresh on-chain data and render all charts + tables
# ---------------------------------------------------------------------------

@callback(
    Output("onchain-content", "children"),
    Input("onchain-refresh-btn", "n_clicks"),
    prevent_initial_call=False,
)
def update_onchain_tab(n_clicks: int | None) -> html.Div:
    """Fetch on-chain data (from cache or API) and render charts + signal table.

    Fires on page load (prevent_initial_call=False) and on refresh button click.
    """
    force = n_clicks is not None and n_clicks > 0

    try:
        onchain_data = fetch_onchain_data(force_refresh=force)
    except Exception as exc:
        log.error("onchain_fetch_failed", error=str(exc), exc_info=True)
        return html.Div(
            dbc.Alert(
                f"Failed to fetch on-chain data: {exc}",
                color="danger",
                dismissable=True,
            ),
            className="mt-3",
        )

    total_tvl = onchain_data.get("total_tvl")
    eth_tvl = onchain_data.get("eth_tvl")
    sol_tvl = onchain_data.get("sol_tvl")
    stablecoin_mcap = onchain_data.get("stablecoin_mcap")
    dex_volume = onchain_data.get("dex_volume")

    if total_tvl is None or len(total_tvl) == 0:
        return html.Div(
            dbc.Alert("No TVL data available.", color="warning"),
            className="mt-3",
        )

    # Compute signals
    try:
        signals = compute_onchain_signals(onchain_data)
    except Exception as exc:
        log.error("onchain_signals_failed", error=str(exc), exc_info=True)
        signals = None

    # --- Charts ---
    charts_row_1 = dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=create_tvl_chart(total_tvl, eth_tvl, sol_tvl),
                config={"displayModeBar": False, "displaylogo": False},
            ),
            md=12,
        ),
    ], className="mb-3")

    charts_row_2 = dbc.Row([
        dbc.Col(
            dcc.Graph(
                figure=create_stablecoin_chart(stablecoin_mcap),
                config={"displayModeBar": False, "displaylogo": False},
            ),
            md=6,
        ),
        dbc.Col(
            dcc.Graph(
                figure=create_dex_volume_chart(dex_volume),
                config={"displayModeBar": False, "displaylogo": False},
            ),
            md=6,
        ),
    ], className="mb-3")

    # --- Signal table ---
    signal_table = _build_signal_table(signals) if signals else html.Div()

    # --- BL views table ---
    bl_table = _build_bl_views_table(signals) if signals else html.Div()

    return html.Div([
        charts_row_1,
        charts_row_2,
        html.Hr(),
        dbc.Row([
            dbc.Col(signal_table, md=6),
            dbc.Col(bl_table, md=6),
        ]),
    ])


# ---------------------------------------------------------------------------
# Signal interpretation table
# ---------------------------------------------------------------------------

def _build_signal_table(signals: OnchainSignals) -> html.Div:
    """Build a styled table showing each signal with value and interpretation."""
    rows_data = [
        (
            "TVL Momentum (30d)",
            _fmt_pct(signals.tvl_momentum_30d),
            signals.tvl_momentum_interpretation,
            "Positive = DeFi growth; >5% is bullish",
        ),
        (
            "Stablecoin Dominance",
            _fmt_share(signals.stablecoin_dominance),
            signals.stablecoin_dominance_interpretation,
            "High = risk-off; low = risk-on",
        ),
        (
            "Stablecoin Supply (30d)",
            _fmt_pct(signals.stablecoin_supply_change_30d),
            signals.stablecoin_supply_interpretation,
            "Inflow >3% = capital entering crypto",
        ),
        (
            "DEX Volume Trend",
            _fmt_ratio(signals.dex_volume_trend_7d),
            signals.dex_volume_interpretation,
            "7d/30d ratio; >1.3 = volume surge",
        ),
    ]

    # Chain TVL shares
    for chain, share in signals.chain_tvl_shares.items():
        rows_data.append((
            f"{chain} TVL Share",
            _fmt_share(share),
            "—",
            f"Share of total DeFi TVL on {chain}",
        ))

    rows = []
    for label, value, interp, tooltip in rows_data:
        color = _INTERP_COLORS.get(interp, COLORS["text_muted"])
        rows.append(html.Tr([
            html.Td(label, title=tooltip),
            html.Td(value, style={"fontWeight": "bold", "textAlign": "right"}),
            html.Td(
                interp,
                style={
                    "color": color,
                    "fontWeight": "bold",
                    "textAlign": "center",
                },
            ),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Signal"),
                html.Th("Value", style={"textAlign": "right"}),
                html.Th("Interpretation", style={"textAlign": "center"}),
            ])),
            html.Tbody(rows),
        ],
        bordered=True,
        className="table-dark",
        hover=True,
        responsive=True,
        size="sm",
    )

    as_of = signals.as_of if signals.as_of else "Unknown"

    return html.Div([
        html.H5("On-Chain Signal Summary", className="mb-2"),
        html.P(
            f"As of: {as_of}",
            className="text-muted",
            style={"fontSize": "0.85em"},
        ),
        table,
    ])


# ---------------------------------------------------------------------------
# Black-Litterman views table
# ---------------------------------------------------------------------------

def _build_bl_views_table(signals: OnchainSignals) -> html.Div:
    """Build a table showing which BL views would be triggered by current signals.

    Calls generate_onchain_views with a representative asset list to show
    the user what views the optimizer would use.
    """
    signals_dict = asdict(signals)

    # Use a representative universe for display purposes
    display_assets = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOT", "LINK",
        "AVAX", "UNI", "AAVE", "MKR",
    ]

    try:
        P, Q, confidences = generate_onchain_views(signals_dict, display_assets)
    except Exception as exc:
        log.warning("bl_views_display_failed", error=str(exc))
        return html.Div(
            dbc.Alert("Could not generate BL views preview.", color="secondary"),
        )

    if P.empty:
        return html.Div([
            html.H5("Black-Litterman Views", className="mb-2"),
            dbc.Alert(
                "No views triggered by current on-chain signals. "
                "The optimizer would fall back to Max Sharpe (Markowitz).",
                color="info",
            ),
        ])

    rows = []
    view_descriptions = {
        0: "TVL Momentum: ETH outperforms BTC",
        1: "Stablecoin Inflow: Market broadly bullish",
        2: "DEX Volume Surge: DeFi tokens outperform",
    }

    for i in range(len(Q)):
        p_row = P.iloc[i]
        nonzero = p_row[p_row.abs() > 0.001]
        assets_str = ", ".join(
            f"{name} ({val:+.2f})" for name, val in nonzero.items()
        )
        desc = view_descriptions.get(i, f"View {i + 1}")

        rows.append(html.Tr([
            html.Td(desc),
            html.Td(assets_str, style={"fontSize": "0.85em"}),
            html.Td(f"{float(Q.iloc[i]):.4f}", style={"textAlign": "right"}),
            html.Td(
                f"{float(confidences.iloc[i]):.1%}",
                style={"textAlign": "right", "fontWeight": "bold"},
            ),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("View"),
                html.Th("Assets (P matrix)"),
                html.Th("Expected Return (Q)", style={"textAlign": "right"}),
                html.Th("Confidence", style={"textAlign": "right"}),
            ])),
            html.Tbody(rows),
        ],
        bordered=True,
        className="table-dark",
        hover=True,
        responsive=True,
        size="sm",
    )

    return html.Div([
        html.H5("Black-Litterman Views", className="mb-2"),
        html.P(
            "Views generated from on-chain signals for the BL optimizer.",
            className="text-muted",
            style={"fontSize": "0.85em"},
        ),
        table,
    ])
