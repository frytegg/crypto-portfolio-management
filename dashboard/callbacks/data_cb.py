"""Data loading and Market Overview tab callbacks.

Reads universe, prices, returns from diskcache. Populates initial views.
Tab routing dispatches active tab to the appropriate layout builder.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
import structlog

from core.config import settings
from core.data.cache import cache, get_live_price
from core.data.fetcher import fetch_historical_data
from core.data.universe import UniverseAsset, fetch_universe
from dashboard.callbacks.backtest_cb import build_backtest_tab
from dashboard.callbacks.garch_cb import build_garch_tab
from dashboard.callbacks.onchain_cb import build_onchain_tab
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

        # Store prices and returns individually so all callbacks can read them
        # via cache.get("prices") / cache.get("returns"). The fetcher caches the
        # tuple under "historical_prices_and_returns", but callbacks expect these
        # shorthand keys.
        cache.set("prices", prices, expire=settings.CACHE_TTL_PRICES)
        cache.set("returns", returns, expire=settings.CACHE_TTL_PRICES)
        log.info("returns_store_populated", shape=returns.shape)

        universe_data = [asdict(a) for a in universe]
        returns_summary = {
            "columns": list(returns.columns),
            "n_observations": len(returns),
            "start_date": str(returns.index[0].date()) if len(returns) > 0 else "",
            "end_date": str(returns.index[-1].date()) if len(returns) > 0 else "",
        }

        # Track data freshness for staleness indicator
        cache.set("meta:data_updated_at", datetime.now(timezone.utc).isoformat())

        log.info(
            "startup_data_loaded",
            universe_size=len(universe),
            assets=len(returns.columns),
            observations=len(returns),
        )
        return universe_data, returns_summary

    except Exception as exc:
        log.error("startup_data_load_failed", error=str(exc), exc_info=True)
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
    try:
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

        if active_tab == "tab-onchain":
            return build_onchain_tab()
        if active_tab == "tab-live":
            return _build_live_tab(universe_data)

        return html.Div(
            html.H4(f"{active_tab} — Coming Soon", className="text-muted text-center mt-5"),
        )
    except Exception as exc:
        log.error("render_tab_content_failed", tab=active_tab, error=str(exc), exc_info=True)
        return html.Div(
            dbc.Alert(
                f"Error loading tab: {exc}",
                color="danger",
                dismissable=True,
            ),
            className="mt-3",
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

    # Market treemap
    treemap = _build_market_treemap(universe)

    # Universe table
    table = _build_universe_table(universe)

    return html.Div([kpi_row, treemap, html.Hr(), table])



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
                colorscale=[[0, COLORS["danger"]], [0.5, COLORS["text_muted"]], [1, COLORS["success"]]],
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
        **{**FIGURE_LAYOUT, "margin": dict(l=10, r=10, t=50, b=10)},
        title="Market Cap Treemap (colored by 24h change)",
        height=450,
    )

    return dcc.Graph(figure=fig, id="market-treemap", config={"displayModeBar": False, "displaylogo": False})


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

_STRATEGY_OPTIONS = [
    {"label": "Equal Weight", "value": "equal_weight"},
    {"label": "Markowitz MVO", "value": "markowitz"},
    {"label": "GARCH-GMV", "value": "garch_gmv"},
    {"label": "Hierarchical Risk Parity", "value": "hrp"},
    {"label": "Equal Risk Contribution", "value": "risk_parity"},
    {"label": "Mean-CVaR", "value": "cvar"},
    {"label": "Black-Litterman", "value": "black_litterman"},
    {"label": "Regime-Aware", "value": "regime_aware"},
]

_STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    o["value"]: o["label"] for o in _STRATEGY_OPTIONS
}


def _build_risk_tab(returns_summary: dict | None) -> html.Div:
    """Build the Risk Dashboard tab with strategy selector, correlation heatmap,
    drawdown, rolling Sharpe, and rolling volatility charts."""
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

    # --- Correlation heatmap (asset-level, not strategy-dependent) ---
    from dashboard.components.correlation_heatmap import create_correlation_heatmap
    corr_fig = create_correlation_heatmap(returns)

    _no_bar = {"displayModeBar": False, "displaylogo": False}

    return html.Div([
        html.H4("Risk Dashboard", className="mb-3"),

        # Strategy selector
        dbc.Row([
            dbc.Col([
                html.Label("Strategy for Risk Metrics", className="fw-bold mb-1"),
                dcc.Dropdown(
                    id="risk-strategy-selector",
                    options=_STRATEGY_OPTIONS,
                    placeholder="Select a strategy",
                    clearable=False,
                    persistence=True,
                    persistence_type="session",
                    style={"color": "#AAAAAA"},
                ),
            ], md=3),
            dbc.Col([
                html.P(
                    "Select a strategy to view its drawdown and rolling metrics. "
                    "Run optimization first for non-equal-weight strategies.",
                    className="text-muted small mt-3",
                ),
            ], md=9),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(
                dcc.Graph(figure=corr_fig, config=_no_bar),
                md=6,
            ),
            dbc.Col(
                dcc.Graph(id="risk-drawdown-chart", figure=go.Figure(), config=_no_bar),
                md=6,
            ),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(
                dcc.Graph(id="risk-sharpe-chart", figure=go.Figure(), config=_no_bar),
                md=6,
            ),
            dbc.Col(
                dcc.Graph(id="risk-vol-chart", figure=go.Figure(), config=_no_bar),
                md=6,
            ),
        ], className="mb-3"),
    ])


# ---------------------------------------------------------------------------
# Callback: Update risk charts on strategy selection
# ---------------------------------------------------------------------------

@callback(
    Output("risk-drawdown-chart", "figure"),
    Output("risk-sharpe-chart", "figure"),
    Output("risk-vol-chart", "figure"),
    Input("risk-strategy-selector", "value"),
    State("strategy-results-store", "data"),
    prevent_initial_call=True,
)
def update_risk_charts(
    strategy: str | None,
    strategy_store: dict | None,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Recompute drawdown, rolling Sharpe, and rolling volatility for the
    selected strategy."""
    from dashboard.components.drawdown_chart import create_drawdown_chart

    returns_df: pd.DataFrame | None = cache.get("returns")
    if returns_df is None or returns_df.empty or not strategy:
        empty = go.Figure()
        empty.update_layout(**FIGURE_LAYOUT, title="Select a strategy")
        return empty, empty, empty

    display_name = _STRATEGY_DISPLAY_NAMES.get(strategy, strategy)

    # Build portfolio return series from weights
    if strategy == "equal_weight":
        n = len(returns_df.columns)
        port_returns = (returns_df * (1.0 / n)).sum(axis=1)
    else:
        weights = _get_strategy_weights(strategy, strategy_store, returns_df)
        if weights is None:
            empty = go.Figure()
            empty.update_layout(
                **FIGURE_LAYOUT,
                title=f"No weights for {display_name} — run optimization first",
            )
            return empty, empty, empty
        # Align weights to returns columns, filling missing with 0
        w = pd.Series(weights).reindex(returns_df.columns, fill_value=0.0)
        port_returns = (returns_df * w).sum(axis=1)

    # --- Drawdown ---
    equity = (1 + port_returns).cumprod()
    dd_series = equity / equity.cummax() - 1
    dd_fig = create_drawdown_chart({f"{display_name} Portfolio": dd_series})

    # --- Rolling 30d Sharpe ---
    window = 30
    roll_mean = port_returns.rolling(window).mean()
    roll_std = port_returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(365)
    roll_sharpe = roll_sharpe.dropna()

    sharpe_fig = go.Figure()
    sharpe_fig.add_trace(go.Scatter(
        x=roll_sharpe.index,
        y=roll_sharpe.values,
        mode="lines",
        line=dict(color=COLORS["info"], width=1.5),
        hovertemplate="%{y:.2f}<extra>Rolling Sharpe</extra>",
    ))
    sharpe_fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_muted"])
    sharpe_fig.update_layout(**FIGURE_LAYOUT)
    sharpe_fig.update_layout(
        title=f"Rolling 30-Day Sharpe Ratio ({display_name})",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        height=350,
        showlegend=False,
    )

    # --- Rolling 30d volatility ---
    roll_vol = roll_std * np.sqrt(365) * 100
    roll_vol = roll_vol.dropna()

    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(
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
        title=f"Rolling 30-Day Annualized Volatility ({display_name})",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=350,
        showlegend=False,
    )

    return dd_fig, sharpe_fig, vol_fig


def _get_strategy_weights(
    strategy: str,
    strategy_store: dict | None,
    returns_df: pd.DataFrame,
) -> dict[str, float] | None:
    """Retrieve strategy weights from the client store or cache."""
    # Try client-side store first (from optimization tab)
    if strategy_store and strategy in strategy_store:
        return strategy_store[strategy].get("weights")

    # Try precached equal_weight
    if strategy == "equal_weight":
        precached = cache.get("precached_equal_weight")
        if precached:
            return precached.get("weights")

    return None


# ---------------------------------------------------------------------------
# Live Prices tab (Tab 8)
# ---------------------------------------------------------------------------

def _build_live_tab(universe_data: list[dict] | None) -> html.Div:
    """Build the dedicated Live Prices tab with all tracked assets."""
    if not universe_data:
        return html.Div(
            dbc.Spinner(
                html.H5("Waiting for universe data...", className="text-muted"),
                color="primary",
            ),
            className="text-center mt-5",
        )

    assets = []
    for a_dict in universe_data:
        try:
            assets.append(UniverseAsset(**a_dict))
        except (TypeError, KeyError):
            continue

    rows = []
    for asset in assets:
        live = None
        if asset.binance_symbol:
            live = get_live_price(asset.binance_symbol)

        if live is not None:
            price = live
            source = "Binance WS"
            if asset.current_price and asset.current_price > 0:
                pct_change = ((price - asset.current_price) / asset.current_price) * 100
            else:
                pct_change = asset.price_change_24h
        else:
            price = asset.current_price
            source = "CoinGecko"
            pct_change = asset.price_change_24h

        color = _pct_color(pct_change)

        rows.append(
            html.Tr([
                html.Td(
                    html.Span(asset.symbol, style={"fontWeight": "bold"}),
                ),
                html.Td(asset.name),
                html.Td(_fmt_price(price)),
                html.Td(
                    _fmt_pct(pct_change),
                    style={"color": color, "fontWeight": "bold"},
                ),
                html.Td(source, style={
                    "color": COLORS["success"] if source == "Binance WS" else COLORS["text_muted"],
                    "fontSize": "0.85em",
                }),
            ])
        )

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th("Symbol"),
                html.Th("Name"),
                html.Th("Price (USD)"),
                html.Th("Change"),
                html.Th("Source"),
            ])),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        className="mt-3",
    )

    return html.Div([
        html.H4("Live Prices", className="mb-2"),
        html.P(
            "Prices update every 5 seconds via Binance WebSocket. "
            "Green dot indicates live data; fallback to CoinGecko snapshot.",
            className="text-muted mb-3",
            style={"fontSize": "0.9em"},
        ),
        table,
    ])
