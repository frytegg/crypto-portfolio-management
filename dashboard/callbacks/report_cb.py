"""Tearsheet and report export callbacks — Tab 8.

Generate HTML tearsheet via quantstats-lumi and enable download.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
import structlog

from core.data.cache import cache
from dashboard.theme import COLORS

log = structlog.get_logger(__name__)


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
                        html.Label("Report Title", className="fw-bold mb-1"),
                        dbc.Input(
                            id="report-title-input",
                            type="text",
                            value="Portfolio Performance Report",
                            placeholder="Report title...",
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label("Benchmark Asset (optional)", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="report-benchmark-dropdown",
                            options=[{"label": "None", "value": ""}] + asset_options,
                            value="",
                            clearable=True,
                            placeholder="Select benchmark...",
                        ),
                    ], md=3),
                    dbc.Col([
                        html.P(
                            "Uses the most recently computed backtest returns, "
                            "or portfolio returns from the optimization tab.",
                            className="text-muted small mt-3",
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
                    ], md=2),
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
    State("report-title-input", "value"),
    State("report-benchmark-dropdown", "value"),
    prevent_initial_call=True,
)
def generate_report(
    n_clicks: int | None,
    title: str,
    benchmark_asset: str,
) -> tuple:
    """Generate HTML tearsheet and trigger download."""
    if not n_clicks:
        return no_update, no_update

    # Try to get returns from backtest first, then from optimization/portfolio
    returns_df = cache.get("returns")
    if returns_df is None:
        return (
            dbc.Alert("No return data available. Run a backtest or load data first.", color="warning"),
            no_update,
        )

    # Use equal-weight portfolio returns as default
    import numpy as np
    import pandas as pd

    n_assets = len(returns_df.columns)
    portfolio_returns = (returns_df * (1.0 / n_assets)).sum(axis=1)
    portfolio_returns.name = "Portfolio"

    # Benchmark
    benchmark = None
    if benchmark_asset and benchmark_asset in returns_df.columns:
        benchmark = returns_df[benchmark_asset]
        benchmark.name = benchmark_asset

    try:
        from core.analytics.tearsheet import generate_tearsheet

        status = dbc.Alert(
            [html.I(className="fas fa-spinner fa-spin me-2"), "Generating tearsheet..."],
            color="info",
        )

        output_path = generate_tearsheet(
            returns=portfolio_returns,
            benchmark=benchmark,
            title=title or "Portfolio Performance Report",
        )

        status = dbc.Alert(
            [html.I(className="fas fa-check-circle me-2"), "Report generated successfully!"],
            color="success",
        )

        return status, dcc.send_file(output_path, filename="portfolio_tearsheet.html")

    except Exception as exc:
        log.exception("tearsheet_generation_failed")
        return (
            dbc.Alert(f"Report generation failed: {exc}", color="danger"),
            no_update,
        )
