"""Data loading and Market Overview tab callbacks.

Reads universe, prices, returns from diskcache. Populates initial views.
Tab routing dispatches active tab to the appropriate layout builder.
"""
from __future__ import annotations

from dataclasses import asdict

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html, no_update
import structlog

from core.data.cache import cache, get_live_price
from core.data.fetcher import fetch_historical_data
from core.data.universe import UniverseAsset, fetch_universe
from dashboard.callbacks.backtest_cb import build_backtest_tab
from dashboard.callbacks.garch_cb import build_garch_tab
from dashboard.callbacks.optimization_cb import build_optimization_tab
from dashboard.callbacks.regime_cb import build_regime_tab
from dashboard.callbacks.report_cb import build_report_tab
from dashboard.components.metric_card import create_metric_card
from dashboard.theme import COLORS, FIGURE_LAYOUT

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_large_number(value: float) -> str:
    """Format large numbers with B/M/K suffix."""
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    if value >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:.2f}"


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


# ---------------------------------------------------------------------------
# Callback 1: Startup data load
# ---------------------------------------------------------------------------

@callback(
    Output("universe-store", "data"),
    Output("returns-store", "data"),
    Input("main-tabs", "active_tab"),  # fires on page load
    prevent_initial_call=False,
)
def load_startup_data(_active_tab: str) -> tuple:
    """On page load, fetch universe and historical data, store metadata.

    This runs once on startup. Universe metadata is serialized to the store
    for use by the table callback. Returns store gets a lightweight summary
    (column names only) since actual DataFrames are read from diskcache.
    """
    try:
        universe = fetch_universe()
        prices, returns = fetch_historical_data(universe)

        universe_data = [asdict(a) for a in universe]
        returns_summary = {
            "columns": list(returns.columns),
            "n_observations": len(returns),
            "start_date": str(returns.index[0].date()) if len(returns) > 0 else "",
            "end_date": str(returns.index[-1].date()) if len(returns) > 0 else "",
        }

        log.info(
            "startup_data_loaded",
            universe_size=len(universe),
            assets=len(returns.columns),
            observations=len(returns),
        )
        return universe_data, returns_summary

    except Exception as exc:
        log.error("startup_data_load_failed", error=str(exc))
        return [], {}


# ---------------------------------------------------------------------------
# Callback 2: Tab routing
# ---------------------------------------------------------------------------

@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("universe-store", "data"),
    Input("returns-store", "data"),
)
def render_tab_content(
    active_tab: str,
    universe_data: list[dict] | None,
    returns_summary: dict | None,
) -> html.Div:
    """Route tab selection to appropriate tab layout."""
    if active_tab == "tab-overview":
        return _build_overview_tab(universe_data, returns_summary)
    if active_tab == "tab-optimization":
        return build_optimization_tab(returns_summary)
    if active_tab == "tab-garch":
        return build_garch_tab(returns_summary)
    if active_tab == "tab-regime":
        return build_regime_tab(returns_summary)
    if active_tab == "tab-risk":
        return _build_risk_tab(returns_summary)
    if active_tab == "tab-backtest":
        return build_backtest_tab(returns_summary)
    if active_tab == "tab-report":
        return build_report_tab(returns_summary)

    # Placeholder for remaining tabs
    tab_labels = {
        "tab-onchain": "On-Chain Signals",
        "tab-live": "Live Prices",
    }
    label = tab_labels.get(active_tab, active_tab)
    return html.Div(
        html.H4(f"{label} — Coming Soon", className="text-muted text-center mt-5"),
    )


# ---------------------------------------------------------------------------
# Market Overview tab layout builder
# ---------------------------------------------------------------------------

def _build_overview_tab(
    universe_data: list[dict] | None,
    returns_summary: dict | None,
) -> html.Div:
    """Build the complete Market Overview tab content."""
    if not universe_data:
        return html.Div(
            dbc.Spinner(
                html.H5("Loading market data...", className="text-muted"),
                color="primary",
                size="lg",
            ),
            className="text-center mt-5",
        )

    universe = [UniverseAsset(**a) for a in universe_data]

    # Top-row KPI cards
    total_mcap = sum(a.market_cap for a in universe)
    total_volume = sum(a.volume_24h for a in universe)
    n_assets = len(universe)
    avg_change_24h = sum(a.price_change_24h for a in universe) / n_assets if n_assets else 0

    obs_text = ""
    if returns_summary:
        obs_text = (
            f"{returns_summary.get('n_observations', '?')} days "
            f"({returns_summary.get('start_date', '?')} to {returns_summary.get('end_date', '?')})"
        )

    kpi_row = dbc.Row(
        [
            dbc.Col(
                create_metric_card("Total Market Cap", _fmt_large_number(total_mcap)),
                md=3,
            ),
            dbc.Col(
                create_metric_card("24h Volume", _fmt_large_number(total_volume)),
                md=3,
            ),
            dbc.Col(
                create_metric_card(
                    "Universe Size",
                    str(n_assets),
                    subtitle=obs_text,
                    color="info",
                ),
                md=3,
            ),
            dbc.Col(
                create_metric_card(
                    "Avg 24h Change",
                    _fmt_pct(avg_change_24h),
                    color="success" if avg_change_24h >= 0 else "danger",
                ),
                md=3,
            ),
        ],
        className="mb-4",
    )

    # Top-10 live price ticker row
    live_row = _build_live_price_row(universe[:10])

    # Market treemap
    treemap = _build_market_treemap(universe)

    # Universe table
    table = _build_universe_table(universe)

    return html.Div([kpi_row, live_row, treemap, html.Hr(), table])


# ---------------------------------------------------------------------------
# Live price row (top-10)
# ---------------------------------------------------------------------------

def _build_live_price_row(top_assets: list[UniverseAsset]) -> dbc.Row:
    """Build a horizontal row showing top-10 asset prices."""
    badges = []
    for asset in top_assets:
        # Try live price from WebSocket cache, fall back to CoinGecko price
        live = None
        if asset.binance_symbol:
            live = get_live_price(asset.binance_symbol)
        price = live if live is not None else asset.current_price

        color = _pct_color(asset.price_change_24h)
        badges.append(
            dbc.Col(
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
                            _fmt_pct(asset.price_change_24h),
                            style={"color": color, "fontSize": "0.85em"},
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
        )

    return dbc.Row(badges, className="mb-4 g-2")


# ---------------------------------------------------------------------------
# Market Treemap
# ---------------------------------------------------------------------------

def _build_market_treemap(universe: list[UniverseAsset]) -> dcc.Graph:
    """Build a market cap treemap colored by 24h price change."""
    labels = [f"{a.symbol}<br>{_fmt_pct(a.price_change_24h)}" for a in universe]
    values = [a.market_cap for a in universe]
    changes = [a.price_change_24h for a in universe]
    hover_text = [
        f"{a.name} ({a.symbol})<br>"
        f"Price: {_fmt_price(a.current_price)}<br>"
        f"Market Cap: {_fmt_large_number(a.market_cap)}<br>"
        f"24h: {_fmt_pct(a.price_change_24h)}"
        for a in universe
    ]

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            values=values,
            parents=[""] * len(universe),
            marker=dict(
                colors=changes,
                colorscale=[[0, "red"], [0.5, "gray"], [1, "green"]],
                cmid=0,
                colorbar=dict(title="24h %", thickness=15),
                line=dict(width=1, color=COLORS["bg"]),
            ),
            text=hover_text,
            hoverinfo="text",
            textinfo="label",
            textfont=dict(size=12),
        )
    )
    fig.update_layout(
        **FIGURE_LAYOUT,
        title="Market Cap Treemap (colored by 24h change)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=450,
    )

    return dcc.Graph(figure=fig, id="market-treemap", config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Universe Table
# ---------------------------------------------------------------------------

def _build_universe_table(universe: list[UniverseAsset]) -> html.Div:
    """Build the interactive universe DataTable."""
    table_data = []
    for a in universe:
        table_data.append({
            "Rank": a.market_cap_rank,
            "Name": a.name,
            "Symbol": a.symbol,
            "Price (USD)": _fmt_price(a.current_price),
            "24h%": a.price_change_24h,
            "7d%": a.price_change_7d,
            "30d%": a.price_change_30d,
            "Market Cap": _fmt_large_number(a.market_cap),
            "Volume 24h": _fmt_large_number(a.volume_24h),
        })

    columns = [
        {"name": "Rank", "id": "Rank", "type": "numeric"},
        {"name": "Name", "id": "Name"},
        {"name": "Symbol", "id": "Symbol"},
        {"name": "Price (USD)", "id": "Price (USD)"},
        {"name": "24h%", "id": "24h%", "type": "numeric", "format": {"specifier": "+.2f"}},
        {"name": "7d%", "id": "7d%", "type": "numeric", "format": {"specifier": "+.2f"}},
        {"name": "30d%", "id": "30d%", "type": "numeric", "format": {"specifier": "+.2f"}},
        {"name": "Market Cap", "id": "Market Cap"},
        {"name": "Volume 24h", "id": "Volume 24h"},
    ]

    pct_columns = ["24h%", "7d%", "30d%"]

    # Conditional styles for percentage columns (green > 0, red < 0)
    style_data_conditional = []
    for col in pct_columns:
        style_data_conditional.extend([
            {
                "if": {
                    "filter_query": f"{{{col}}} > 0",
                    "column_id": col,
                },
                "color": COLORS["success"],
                "fontWeight": "bold",
            },
            {
                "if": {
                    "filter_query": f"{{{col}}} < 0",
                    "column_id": col,
                },
                "color": COLORS["danger"],
                "fontWeight": "bold",
            },
        ])

    table = dash_table.DataTable(
        id="universe-table",
        data=table_data,
        columns=columns,
        sort_action="native",
        sort_mode="single",
        sort_by=[{"column_id": "Rank", "direction": "asc"}],
        page_size=50,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["card_bg"],
            "color": COLORS["text"],
            "fontWeight": "bold",
            "borderBottom": f"2px solid {COLORS['primary']}",
        },
        style_cell={
            "backgroundColor": COLORS["bg"],
            "color": COLORS["text"],
            "border": f"1px solid {COLORS['grid']}",
            "padding": "8px 12px",
            "textAlign": "right",
            "fontFamily": "Inter, -apple-system, sans-serif",
            "fontSize": "0.9em",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Name"}, "textAlign": "left", "minWidth": "120px"},
            {"if": {"column_id": "Symbol"}, "textAlign": "center", "fontWeight": "bold"},
            {"if": {"column_id": "Rank"}, "textAlign": "center", "width": "60px"},
        ],
        style_data_conditional=style_data_conditional,
    )

    return html.Div(
        [
            html.H5("Universe Assets", className="mb-3"),
            table,
        ]
    )


# ---------------------------------------------------------------------------
# Risk Dashboard tab (Tab 5)
# ---------------------------------------------------------------------------

def _build_risk_tab(returns_summary: dict | None) -> html.Div:
    """Build the Risk Dashboard tab with correlation heatmap, drawdown,
    rolling Sharpe, and rolling volatility charts."""
    if not returns_summary or not returns_summary.get("columns"):
        return html.Div(
            dbc.Spinner(
                html.H5("Waiting for market data...", className="text-muted"),
                color="primary",
            ),
            className="text-center mt-5",
        )

    returns = cache.get("returns")
    if returns is None:
        return html.Div(
            dbc.Alert("Return data not yet loaded.", color="warning"),
            className="mt-3",
        )

    import numpy as np
    import plotly.graph_objects as go_
    from dashboard.components.correlation_heatmap import create_correlation_heatmap
    from dashboard.components.drawdown_chart import create_drawdown_chart

    # --- Correlation heatmap ---
    corr_fig = create_correlation_heatmap(returns)

    # --- Equal-weight portfolio for drawdown/rolling stats ---
    n_assets = len(returns.columns)
    port_returns = (returns * (1.0 / n_assets)).sum(axis=1)
    equity = (1 + port_returns).cumprod()
    dd_series = equity / equity.cummax() - 1

    dd_fig = create_drawdown_chart({"Equal Weight Portfolio": dd_series})

    # --- Rolling 30d Sharpe ---
    window = 30
    roll_mean = port_returns.rolling(window).mean()
    roll_std = port_returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(365)
    roll_sharpe = roll_sharpe.dropna()

    sharpe_fig = go_.Figure()
    sharpe_fig.add_trace(go_.Scatter(
        x=roll_sharpe.index,
        y=roll_sharpe.values,
        mode="lines",
        line=dict(color=COLORS["info"], width=1.5),
        hovertemplate="%{y:.2f}<extra>Rolling Sharpe</extra>",
    ))
    sharpe_fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"])
    sharpe_fig.update_layout(**FIGURE_LAYOUT)
    sharpe_fig.update_layout(
        title="Rolling 30-Day Sharpe Ratio (Equal Weight)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        height=350,
        showlegend=False,
    )

    # --- Rolling 30d volatility ---
    roll_vol = roll_std * np.sqrt(365) * 100  # Annualized, in %
    roll_vol = roll_vol.dropna()

    vol_fig = go_.Figure()
    vol_fig.add_trace(go_.Scatter(
        x=roll_vol.index,
        y=roll_vol.values,
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["warning"], width=1.5),
        fillcolor="rgba(243,156,18,0.2)",
        hovertemplate="%{y:.1f}%<extra>Rolling Vol</extra>",
    ))
    vol_fig.update_layout(**FIGURE_LAYOUT)
    vol_fig.update_layout(
        title="Rolling 30-Day Annualized Volatility (Equal Weight)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=350,
        showlegend=False,
    )

    return html.Div([
        html.H4("Risk Dashboard", className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=corr_fig), md=6),
            dbc.Col(dcc.Graph(figure=dd_fig), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=sharpe_fig), md=6),
            dbc.Col(dcc.Graph(figure=vol_fig), md=6),
        ], className="mb-3"),
    ])
