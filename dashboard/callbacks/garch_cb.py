"""GARCH volatility forecasting callbacks.

Tab layout with:
- Asset dropdown to select which asset's GARCH to display
- Conditional volatility chart (dual y-axis with price)
- Forecast value card + model params table
- Volatility heatmap across all assets
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
import plotly.graph_objects as go
import structlog

from core.data.cache import cache
from core.models.garch import fit_all_garch
from dashboard.components.garch_chart import create_garch_vol_chart, create_vol_heatmap
from dashboard.components.metric_card import create_metric_card
from dashboard.theme import COLORS, FIGURE_LAYOUT

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tab layout builder (called from data_cb.py tab routing)
# ---------------------------------------------------------------------------

def build_garch_tab(returns_summary: dict | None) -> html.Div:
    """Build the GARCH Volatility tab layout."""
    if not returns_summary or not returns_summary.get("columns"):
        return html.Div(
            dbc.Spinner(
                html.H5("Waiting for market data...", className="text-muted"),
                color="primary",
            ),
            className="text-center mt-5",
        )

    columns = returns_summary["columns"]

    return html.Div([
        # Controls row
        dbc.Row([
            dbc.Col([
                html.Label("Select Asset", className="fw-bold mb-1"),
                dcc.Dropdown(
                    id="garch-asset-dropdown",
                    options=[{"label": c, "value": c} for c in columns],
                    value=columns[0] if columns else None,
                    clearable=False,
                    className="mb-3",
                ),
            ], md=3),
            dbc.Col([
                html.Div(id="garch-status-badge", className="mt-4"),
            ], md=9),
        ], className="mb-3"),

        # Fit all button
        dbc.Row([
            dbc.Col(
                dbc.Button(
                    "Fit GARCH Models",
                    id="garch-fit-button",
                    color="primary",
                    className="mb-3",
                ),
                md=3,
            ),
        ]),

        # Spinner wrapping the results
        dbc.Spinner(
            html.Div(id="garch-results-container"),
            color="primary",
            type="border",
        ),
    ])


# ---------------------------------------------------------------------------
# Callback 1: Fit all GARCH models on button click
# ---------------------------------------------------------------------------

@callback(
    Output("garch-results-store", "data"),
    Output("garch-status-badge", "children"),
    Input("garch-fit-button", "n_clicks"),
    State("returns-store", "data"),
    prevent_initial_call=True,
)
def fit_garch_models(
    n_clicks: int | None,
    returns_summary: dict | None,
) -> tuple:
    """Fit GJR-GARCH(1,1,1) to all assets and store serialized results."""
    if not n_clicks or not returns_summary:
        return no_update, no_update

    returns_df: pd.DataFrame | None = cache.get("returns")
    if returns_df is None or returns_df.empty:
        return no_update, dbc.Badge("No returns data", color="warning")

    log.info("fitting_garch_all_assets", n_assets=len(returns_df.columns))

    try:
        garch_results = fit_all_garch(returns_df)

        # Serialize for dcc.Store — convert Series to JSON-friendly dicts
        serialized: dict = {}
        for asset, result in garch_results.items():
            cond_vol = result["conditional_volatility"]
            serialized[asset] = {
                "conditional_volatility_values": cond_vol.values.tolist(),
                "conditional_volatility_index": [
                    str(d) for d in cond_vol.index
                ],
                "forecast_variance": float(result["forecast_variance"]),
                "forecast_vol": float(result["forecast_vol"]),
                "params": {k: float(v) for k, v in result["params"].items()},
                "aic": float(result["aic"]) if np.isfinite(result["aic"]) else None,
                "bic": float(result["bic"]) if np.isfinite(result["bic"]) else None,
                "error": result.get("error"),
            }

        n_success = sum(1 for r in serialized.values() if r.get("error") is None)
        badge = dbc.Badge(
            f"Fitted {n_success}/{len(serialized)} assets",
            color="success" if n_success == len(serialized) else "warning",
            className="fs-6",
        )

        log.info("garch_fitting_complete", n_success=n_success, n_total=len(serialized))
        return serialized, badge
    except Exception as exc:
        log.error("fit_garch_models_failed", error=str(exc), exc_info=True)
        return no_update, dbc.Alert(f"GARCH fitting failed: {exc}", color="danger", dismissable=True)


# ---------------------------------------------------------------------------
# Callback 2: Render selected asset's GARCH chart + params
# ---------------------------------------------------------------------------

@callback(
    Output("garch-results-container", "children"),
    Input("garch-asset-dropdown", "value"),
    Input("garch-results-store", "data"),
    prevent_initial_call=True,
)
def render_garch_asset(
    selected_asset: str | None,
    garch_data: dict | None,
) -> html.Div:
    """Render GARCH chart, forecast card, params table, and heatmap."""
    if not garch_data or not selected_asset:
        return html.Div(
            html.P(
                "Click 'Fit GARCH Models' to run volatility analysis.",
                className="text-muted text-center mt-4",
            )
        )

    if selected_asset not in garch_data:
        return html.Div(
            html.P(f"No GARCH data for {selected_asset}.", className="text-muted")
        )

    try:
        # Deserialize the selected asset's result
        asset_data = garch_data[selected_asset]
        garch_result = _deserialize_garch_result(asset_data)

        # Get prices from cache for the overlay
        prices_df: pd.DataFrame | None = cache.get("prices")
        prices = pd.Series(dtype=float)
        if prices_df is not None and selected_asset in prices_df.columns:
            prices = prices_df[selected_asset]

        # 1. Conditional volatility chart
        vol_chart = create_garch_vol_chart(selected_asset, garch_result, prices)

        # 2. Forecast card
        forecast_vol = asset_data["forecast_vol"]
        forecast_card = create_metric_card(
            "1-Day Forecast Vol",
            f"{forecast_vol:.4f}",
            subtitle=f"Annualized: {forecast_vol * np.sqrt(365):.2%}",
            color="danger",
        )

        # 3. Params table
        params_table = _build_params_table(asset_data)

        # 4. Volatility heatmap (all assets)
        all_garch_results = {
            asset: _deserialize_garch_result(data)
            for asset, data in garch_data.items()
        }
        heatmap = create_vol_heatmap(all_garch_results)

        return html.Div([
            # Chart + cards row
            dbc.Row([
                dbc.Col(dcc.Graph(figure=vol_chart, config={"displayModeBar": False}), md=8),
                dbc.Col([forecast_card, params_table], md=4),
            ], className="mb-4"),
            # Heatmap
            dbc.Row([
                dbc.Col(
                    dcc.Graph(figure=heatmap, config={"displayModeBar": False}),
                    md=12,
                ),
            ]),
        ])
    except Exception as exc:
        log.error("render_garch_asset_failed", asset=selected_asset, error=str(exc), exc_info=True)
        return html.Div(
            dbc.Alert(f"Error rendering GARCH for {selected_asset}: {exc}", color="danger", dismissable=True),
            className="mt-3",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deserialize_garch_result(data: dict) -> dict:
    """Reconstruct GARCH result dict from JSON-serialized store data."""
    index = pd.to_datetime(data["conditional_volatility_index"])
    cond_vol = pd.Series(
        data["conditional_volatility_values"],
        index=index,
        name="cond_vol",
    )
    return {
        "conditional_volatility": cond_vol,
        "forecast_variance": data["forecast_variance"],
        "forecast_vol": data["forecast_vol"],
        "params": data["params"],
        "aic": data["aic"],
        "bic": data["bic"],
        "model_result": None,
    }


def _build_params_table(asset_data: dict) -> dbc.Card:
    """Build a card showing GARCH model parameters."""
    params = asset_data.get("params", {})
    aic = asset_data.get("aic")
    bic = asset_data.get("bic")
    error = asset_data.get("error")

    rows = []
    for name, value in params.items():
        rows.append(html.Tr([
            html.Td(name, style={"fontWeight": "bold"}),
            html.Td(f"{value:.6f}"),
        ]))

    if aic is not None:
        rows.append(html.Tr([html.Td("AIC", style={"fontWeight": "bold"}), html.Td(f"{aic:.2f}")]))
    if bic is not None:
        rows.append(html.Tr([html.Td("BIC", style={"fontWeight": "bold"}), html.Td(f"{bic:.2f}")]))

    header_text = "Model Parameters"
    if error:
        header_text = "Fallback (sample vol)"

    return dbc.Card([
        dbc.CardHeader(header_text),
        dbc.CardBody(
            dbc.Table(
                [html.Tbody(rows)],
                bordered=True,
                hover=True,
                responsive=True,
                color="dark",
                size="sm",
                className="mb-0",
            ) if rows else html.P("No parameters", className="text-muted mb-0"),
        ),
    ], className="mt-3")
