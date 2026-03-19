"""Main dashboard layout — assembles all tabs and shared components.

Decision: callbacks read diskcache directly (no dcc.Store for large DataFrames).
dcc.Store is used only for computed results (optimization, backtest, GARCH, regime).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard.theme import STRATEGY_COLORS

# DARKLY-themed accent color for loading spinners — matches Markowitz MVO
_LOADING_COLOR = STRATEGY_COLORS["markowitz"]


def create_layout() -> dbc.Container:
    """Build the top-level Dash layout. Called once in app.py."""
    return dbc.Container(
        fluid=True,
        className="p-3",
        children=[
            # Header with staleness indicator
            dbc.Row(
                [
                    dbc.Col(
                        html.H2(
                            "Crypto Portfolio Management",
                            className="text-center my-3",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        html.Div(
                            id="data-staleness-indicator",
                            className="text-end mt-3",
                        ),
                        md=4,
                    ),
                ],
            ),
            # Live price update interval (every 5 seconds)
            dcc.Interval(id="live-interval", interval=5_000, n_intervals=0),
            # Client-side stores for metadata and computed results
            dcc.Store(id="universe-store"),
            dcc.Store(id="returns-store"),
            dcc.Store(id="strategy-results-store"),
            dcc.Store(id="backtest-results-store"),
            dcc.Store(id="garch-results-store"),
            dcc.Store(id="regime-results-store"),
            dcc.Store(id="onchain-signals-store"),
            # Main tabs
            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-overview",
                children=[
                    dbc.Tab(label="Market Overview", tab_id="tab-overview"),
                    dbc.Tab(label="Optimization", tab_id="tab-optimization"),
                    dbc.Tab(label="GARCH Volatility", tab_id="tab-garch"),
                    dbc.Tab(label="Regime Detection", tab_id="tab-regime"),
                    dbc.Tab(label="Risk Dashboard", tab_id="tab-risk"),
                    dbc.Tab(label="On-Chain Signals", tab_id="tab-onchain"),
                    dbc.Tab(label="Backtest", tab_id="tab-backtest"),
                    dbc.Tab(label="Live Prices", tab_id="tab-live"),
                    dbc.Tab(label="Report", tab_id="tab-report"),
                ],
            ),
            # Tab content container wrapped in loading spinner
            dcc.Loading(
                id="loading-tab-content",
                type="circle",
                color=_LOADING_COLOR,
                children=html.Div(id="tab-content", className="mt-3"),
            ),
        ],
    )
