"""Tearsheet and report export callbacks — Tab 8.

Generate HTML tearsheet via quantstats-lumi and enable download.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
import structlog

from core.data.cache import cache
from dashboard.theme import COLORS

log = structlog.get_logger(__name__)

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


# ---------------------------------------------------------------------------
# Tab layout builder
# ---------------------------------------------------------------------------

def build_report_tab(returns_summary: dict | None) -> html.Div:
    """Build the Report Export tab layout."""
    if not returns_summary or not returns_summary.get("columns"):
        return html.Div(
            dbc.Spinner(
                html.H5("Waiting for market data...", className="text-muted"),
                color="primary",
            ),
            className="text-center mt-5",
        )

    asset_options = [
        {"label": col, "value": col}
        for col in returns_summary["columns"]
    ]

    return html.Div([
        html.H4("Performance Report Export", className="mb-3"),

        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Strategy", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="report-strategy-selector",
                            options=_STRATEGY_OPTIONS,
                            placeholder="Select a strategy",
                            clearable=False,
                            persistence=True,
                            persistence_type="session",
                            style={"color": "#AAAAAA"},
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Report Title", className="fw-bold mb-1"),
                        dbc.Input(
                            id="report-title-input",
                            type="text",
                            value="Portfolio Performance Report",
                            placeholder="Report title...",
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Benchmark Asset (optional)", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="report-benchmark-dropdown",
                            options=[{"label": "None", "value": ""}] + asset_options,
                            clearable=True,
                            placeholder="Select benchmark...",
                            persistence=True,
                            persistence_type="session",
                            style={"color": "#AAAAAA"},
                        ),
                    ], md=3),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-file-alt me-2"), "Generate Report"],
                            id="report-generate-btn",
                            color="primary",
                            size="lg",
                            className="mt-3 w-100",
                        ),
                        html.Small(
                            "Equal Weight works immediately. Other strategies "
                            "require running optimization first.",
                            className="text-muted d-block mt-1",
                            style={"fontSize": "0.8em"},
                        ),
                    ], md=3),
                ]),
            ]),
        ], className="mb-4"),

        # Status
        html.Div(id="report-status", className="mb-3"),

        # Download component
        dcc.Download(id="report-download"),

        # Preview area
        html.Div(id="report-preview"),
    ])


# ---------------------------------------------------------------------------
# Callback: Generate tearsheet
# ---------------------------------------------------------------------------

@callback(
    Output("report-status", "children"),
    Output("report-download", "data"),
    Input("report-generate-btn", "n_clicks"),
    State("report-strategy-selector", "value"),
    State("report-title-input", "value"),
    State("report-benchmark-dropdown", "value"),
    State("strategy-results-store", "data"),
    prevent_initial_call=True,
)
def generate_report(
    n_clicks: int | None,
    strategy: str | None,
    title: str,
    benchmark_asset: str,
    strategy_store: dict | None,
) -> tuple:
    """Generate HTML tearsheet and trigger download."""
    if not n_clicks:
        return no_update, no_update

    if not strategy:
        return (
            dbc.Alert("Please select a strategy first.", color="warning"),
            no_update,
        )

    returns_df = cache.get("returns")
    if returns_df is None:
        return (
            dbc.Alert("No return data available. Load data first.", color="warning"),
            no_update,
        )

    display_name = _STRATEGY_DISPLAY_NAMES.get(strategy, strategy)

    # Build portfolio returns from strategy weights
    if strategy == "equal_weight":
        n_assets = len(returns_df.columns)
        portfolio_returns = (returns_df * (1.0 / n_assets)).sum(axis=1)
    else:
        weights = _get_strategy_weights(strategy, strategy_store, returns_df)
        if weights is None:
            return (
                dbc.Alert(
                    f"No weights available for {display_name}. "
                    "Run 'Run All Strategies' on the Optimization tab first.",
                    color="warning",
                ),
                no_update,
            )
        w = pd.Series(weights).reindex(returns_df.columns, fill_value=0.0)
        portfolio_returns = (returns_df * w).sum(axis=1)

    portfolio_returns.name = display_name

    # Benchmark
    benchmark = None
    if benchmark_asset and benchmark_asset in returns_df.columns:
        benchmark = returns_df[benchmark_asset]
        benchmark.name = benchmark_asset

    # Build full title
    full_title = f"{title or 'Portfolio Performance Report'} — {display_name}"

    # Output filename
    safe_name = strategy.replace(" ", "_")
    output_path = str(
        Path(tempfile.gettempdir()) / f"tearsheet_{safe_name}.html"
    )

    try:
        from core.analytics.tearsheet import generate_tearsheet

        output_path = generate_tearsheet(
            returns=portfolio_returns,
            benchmark=benchmark,
            title=full_title,
            output_path=output_path,
        )

        status = dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"),
             f"Report generated for {display_name}!"],
            color="success",
        )

        download_filename = f"tearsheet_{safe_name}.html"
        return status, dcc.send_file(output_path, filename=download_filename)

    except Exception as exc:
        log.error("tearsheet_generation_failed", error=str(exc), exc_info=True)
        return (
            dbc.Alert(f"Report generation failed: {exc}", color="danger", dismissable=True),
            no_update,
        )


def _get_strategy_weights(
    strategy: str,
    strategy_store: dict | None,
    returns_df: pd.DataFrame,
) -> dict[str, float] | None:
    """Retrieve strategy weights from the client store or cache."""
    if strategy_store and strategy in strategy_store:
        return strategy_store[strategy].get("weights")

    if strategy == "equal_weight":
        precached = cache.get("precached_equal_weight")
        if precached:
            return precached.get("weights")

    return None
