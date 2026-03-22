"""Optimization callbacks — Efficient Frontier (Tab 2) and Strategy Lab (Tab 3).

Tab layout is a single "Optimization" tab with two sections:
1. Efficient Frontier — compute and visualize the mean-variance frontier
2. Strategy Lab — run all strategies, compare weights/equity/metrics
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
import plotly.graph_objects as go
import structlog

from core.data.cache import cache
from dashboard.components.comparison_table import create_comparison_table
from dashboard.components.efficient_frontier import create_efficient_frontier_figure
from dashboard.components.equity_chart import create_equity_chart
from dashboard.components.weights_chart import (
    create_weights_bar,
    create_weights_bar_chart,
)
from dashboard.theme import COLORS, FIGURE_LAYOUT, STRATEGY_COLORS

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tab layout builder (called from data_cb.py tab routing)
# ---------------------------------------------------------------------------

def build_optimization_tab(returns_summary: dict | None) -> html.Div:
    """Build the combined Efficient Frontier + Strategy Lab tab."""
    if not returns_summary or not returns_summary.get("columns"):
        return html.Div(
            dbc.Spinner(
                html.H5("Waiting for market data...", className="text-muted"),
                color="primary",
            ),
            className="text-center mt-5",
        )

    return html.Div([
        # ===================================================================
        # Section 1: Efficient Frontier
        # ===================================================================
        html.H4("Efficient Frontier", className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("Covariance Method", className="fw-bold mb-1"),
                dcc.Dropdown(
                    id="cov-method-dropdown",
                    options=[
                        {"label": "Ledoit-Wolf", "value": "ledoit"},
                        {"label": "Historical", "value": "hist"},
                        {"label": "Oracle Approx. Shrinkage", "value": "oas"},
                    ],
                    value="ledoit",
                    clearable=False,
                    className="mb-2",
                ),
            ], md=3),
            dbc.Col([
                html.Label("Max Weight per Asset", className="fw-bold mb-1"),
                dcc.Slider(
                    id="max-weight-slider",
                    min=0.05,
                    max=1.0,
                    step=0.05,
                    value=0.15,
                    marks={v: f"{v:.0%}" for v in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], md=5),
            dbc.Col([
                dbc.Button(
                    "Compute Frontier",
                    id="frontier-compute-btn",
                    color="info",
                    className="mt-4",
                ),
            ], md=2),
        ], className="mb-3"),

        dbc.Spinner(
            dcc.Graph(id="efficient-frontier-graph", figure=go.Figure(), config={"displayModeBar": False, "displaylogo": False}),
            color="info",
            type="border",
        ),

        html.Hr(className="my-4"),

        # ===================================================================
        # Section 2: Strategy Lab
        # ===================================================================
        html.H4("Strategy Lab", className="mb-3"),

        dbc.Row([
            dbc.Col(
                dbc.Button(
                    "Run All Strategies",
                    id="optimize-btn",
                    color="primary",
                    size="lg",
                    className="mb-3",
                ),
                md=3,
            ),
            dbc.Col(
                html.Div(id="optimize-status-badge", className="mt-2"),
                md=9,
            ),
        ]),

        dbc.Spinner(
            html.Div(id="strategy-results-container"),
            color="primary",
            type="border",
        ),
    ])


# ---------------------------------------------------------------------------
# Callback 1: Compute efficient frontier
# ---------------------------------------------------------------------------

@callback(
    Output("efficient-frontier-graph", "figure"),
    Input("frontier-compute-btn", "n_clicks"),
    State("returns-store", "data"),
    State("cov-method-dropdown", "value"),
    State("max-weight-slider", "value"),
    prevent_initial_call=True,
)
def compute_frontier(
    n_clicks: int | None,
    returns_summary: dict | None,
    cov_method: str,
    max_weight: float,
) -> go.Figure:
    """Compute and render the efficient frontier."""
    if not n_clicks or not returns_summary:
        return no_update

    returns_df: pd.DataFrame | None = cache.get("returns")
    if returns_df is None or returns_df.empty:
        return no_update

    from core.optimization.markowitz import compute_efficient_frontier

    log.info("computing_efficient_frontier", cov_method=cov_method, max_weight=max_weight)

    try:
        frontier_data = compute_efficient_frontier(
            returns_df,
            n_points=50,
            method_cov=cov_method,
            max_weight=max_weight,
        )
        fig = create_efficient_frontier_figure(frontier_data)
    except Exception as exc:
        log.error("frontier_computation_failed", error=str(exc), exc_info=True)
        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT, title=f"Frontier failed: {exc}")

    return fig


# ---------------------------------------------------------------------------
# Callback 2: Run all strategies → store results
# ---------------------------------------------------------------------------

@callback(
    Output("strategy-results-store", "data"),
    Output("optimize-status-badge", "children"),
    Input("optimize-btn", "n_clicks"),
    State("returns-store", "data"),
    State("cov-method-dropdown", "value"),
    State("max-weight-slider", "value"),
    prevent_initial_call=True,
)
def run_all_strategies(
    n_clicks: int | None,
    returns_summary: dict | None,
    cov_method: str | None,
    max_weight: float | None,
) -> tuple:
    """Run all optimization strategies and serialize results to store."""
    if not n_clicks or not returns_summary:
        return no_update, no_update

    returns_df: pd.DataFrame | None = cache.get("returns")
    if returns_df is None or returns_df.empty:
        return no_update, dbc.Badge("No returns data", color="warning")

    cov_method = cov_method or "ledoit"
    max_weight = max_weight or 0.15

    log.info("running_all_strategies", n_assets=len(returns_df.columns))

    from core.optimization.equal_weight import optimize_equal_weight
    from core.optimization.markowitz import optimize_markowitz, optimize_garch_gmv
    from core.optimization.hrp import optimize_hrp
    from core.optimization.risk_parity import optimize_risk_parity
    from core.optimization.cvar import optimize_cvar
    from core.optimization.black_litterman import optimize_black_litterman
    from core.optimization.regime_alloc import optimize_regime_aware
    from core.models.regime import detect_regimes

    results: dict[str, dict] = {}
    errors: list[str] = []

    # Strategy execution list
    strategies = [
        ("equal_weight", lambda: optimize_equal_weight(returns_df)),
        ("markowitz", lambda: optimize_markowitz(
            returns_df, objective="Sharpe", method_cov=cov_method, max_weight=max_weight,
        )),
        ("garch_gmv", lambda: optimize_garch_gmv(returns_df, max_weight=max_weight)),
        ("hrp", lambda: optimize_hrp(returns_df, covariance=cov_method, max_weight=max_weight)),
        ("risk_parity", lambda: optimize_risk_parity(returns_df, method_cov=cov_method)),
        ("cvar", lambda: optimize_cvar(
            returns_df, method_cov=cov_method, max_weight=max_weight,
        )),
    ]

    # Black-Litterman: use onchain signals from cache if available
    def _run_bl() -> object:
        onchain_signals = cache.get("onchain_signals")
        if onchain_signals is None:
            # Provide neutral signals that trigger no views → fallback to Markowitz
            onchain_signals = {}
        signal_dict = (
            onchain_signals if isinstance(onchain_signals, dict) else {}
        )
        return optimize_black_litterman(
            returns_df, onchain_signals=signal_dict,
            method_cov=cov_method, max_weight=max_weight,
        )
    strategies.append(("black_litterman", _run_bl))

    # Regime-aware: detect regimes on BTC (or first column)
    def _run_regime() -> object:
        btc_col = "BTC" if "BTC" in returns_df.columns else returns_df.columns[0]
        regime_info = detect_regimes(returns_df[btc_col], n_regimes=2)
        return optimize_regime_aware(
            returns_df, regime_info=regime_info,
            method_cov=cov_method, max_weight=max_weight,
        )
    strategies.append(("regime_aware", _run_regime))

    for name, run_fn in strategies:
        try:
            result = run_fn()
            results[name] = _serialize_result(result)
            log.info("strategy_complete", strategy=name)
        except Exception as exc:
            log.warning("strategy_failed", strategy=name, error=str(exc))
            errors.append(name)

    n_ok = len(results)
    n_total = len(strategies)
    badge_color = "success" if not errors else "warning"
    badge_text = f"{n_ok}/{n_total} strategies complete"
    if errors:
        badge_text += f" (failed: {', '.join(errors)})"

    badge = dbc.Badge(badge_text, color=badge_color, className="fs-6")

    log.info("all_strategies_complete", n_ok=n_ok, n_errors=len(errors))
    return results, badge


# ---------------------------------------------------------------------------
# Callback 3: Render strategy comparison from store
# ---------------------------------------------------------------------------

@callback(
    Output("strategy-results-container", "children"),
    Input("strategy-results-store", "data"),
    prevent_initial_call=True,
)
def render_strategy_results(store_data: dict | None) -> html.Div:
    """Render all strategy comparison visualizations from store data."""
    if not store_data:
        # Check for pre-cached equal_weight from startup
        precached = cache.get("precached_equal_weight")
        if precached:
            store_data = {"equal_weight": precached}
        else:
            return html.Div(
                html.P(
                    "Click 'Run All Strategies' to optimize.",
                    className="text-muted text-center mt-4",
                )
            )

    try:
        returns_df: pd.DataFrame | None = cache.get("returns")

        # Deserialize weights
        all_weights: dict[str, pd.Series] = {}
        for name, data in store_data.items():
            w = pd.Series(data["weights"])
            w.name = "weights"
            all_weights[name] = w

        # --- A. Weights comparison bar chart ---
        weights_chart = create_weights_bar(all_weights)

        # --- B. Individual weights bar charts (first 4 strategies) ---
        individual_charts = []
        for name, w in list(all_weights.items())[:4]:
            display_name = data.get("name", name) if (data := store_data.get(name)) else name
            chart = create_weights_bar_chart(w, title=f"{display_name}")
            individual_charts.append(
                dbc.Col(dcc.Graph(figure=chart, config={"displayModeBar": False, "displaylogo": False}), md=6)
            )

        # --- C. Equity curves ---
        equity_fig = go.Figure()
        if returns_df is not None:
            equity_curves: dict[str, pd.Series] = {}
            for name, w in all_weights.items():
                # Align weights with returns columns
                common = returns_df.columns.intersection(w.index)
                if len(common) == 0:
                    continue
                w_aligned = w.reindex(common).fillna(0)
                w_aligned = w_aligned / w_aligned.sum()  # renormalize
                port_ret = (returns_df[common] * w_aligned).sum(axis=1)
                equity = (1 + port_ret).cumprod()
                equity_curves[name] = equity

            if equity_curves:
                equity_fig = create_equity_chart(equity_curves)

        # --- D. Strategy weights heatmap ---
        heatmap_fig = _create_weights_heatmap(all_weights)

        # --- E. Comparison table ---
        metrics_table = _build_metrics_table(store_data)

        return html.Div([
            # Comparison table
            html.H5("Strategy Comparison", className="mt-3 mb-2"),
            metrics_table,

            # Grouped weights chart
            html.H5("Weight Comparison", className="mt-4 mb-2"),
            dcc.Graph(figure=weights_chart, config={"displayModeBar": False, "displaylogo": False}),

            # Equity curves
            html.H5("Equity Curves", className="mt-4 mb-2"),
            dcc.Graph(figure=equity_fig, config={"displayModeBar": False, "displaylogo": False}),

            # Individual weight charts
            html.H5("Individual Allocations", className="mt-4 mb-2"),
            dbc.Row(individual_charts),

            # Weights heatmap
            html.H5("Weight Heatmap", className="mt-4 mb-2"),
            dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False, "displaylogo": False}),
        ])
    except Exception as exc:
        log.error("render_strategy_results_failed", error=str(exc), exc_info=True)
        return html.Div(
            dbc.Alert(f"Error rendering strategy results: {exc}", color="danger", dismissable=True),
            className="mt-3",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_result(result: object) -> dict:
    """Serialize a PortfolioResult to a JSON-compatible dict."""
    return {
        "name": result.name,
        "weights": {k: float(v) for k, v in result.weights.items()},
        "expected_return": float(result.expected_return),
        "expected_volatility": float(result.expected_volatility),
        "sharpe_ratio": float(result.sharpe_ratio),
        "metadata": result.metadata,
    }


def _build_metrics_table(store_data: dict) -> dbc.Table:
    """Build the comparison table from serialized strategy results."""
    metrics_by_strategy: dict[str, dict] = {}

    returns_df: pd.DataFrame | None = cache.get("returns")

    for name, data in store_data.items():
        display_name = data.get("name", name)
        w = pd.Series(data["weights"])

        metrics: dict[str, str] = {
            "Ann. Return": f"{data['expected_return']:.2%}",
            "Volatility": f"{data['expected_volatility']:.2%}",
            "Sharpe Ratio": f"{data['sharpe_ratio']:.2f}",
        }

        # Compute Sortino, Max DD, CVaR from actual returns if available
        if returns_df is not None:
            common = returns_df.columns.intersection(w.index)
            if len(common) > 0:
                w_aligned = w.reindex(common).fillna(0)
                w_aligned = w_aligned / w_aligned.sum()
                port_ret = (returns_df[common] * w_aligned).sum(axis=1)

                # Sortino
                downside = port_ret[port_ret < 0]
                downside_std = float(downside.std() * np.sqrt(365)) if len(downside) > 1 else 0.0
                ann_ret = float(port_ret.mean() * 365)
                sortino = ann_ret / downside_std if downside_std > 0 else 0.0
                metrics["Sortino Ratio"] = f"{sortino:.2f}"

                # Max Drawdown
                equity = (1 + port_ret).cumprod()
                peak = equity.cummax()
                dd = (equity - peak) / peak
                max_dd = float(dd.min())
                metrics["Max Drawdown"] = f"{max_dd:.2%}"

                # CVaR 95%
                cvar_95 = float(port_ret.quantile(0.05))
                metrics["CVaR 95%"] = f"{cvar_95:.4f}"

        metrics_by_strategy[display_name] = metrics

    return create_comparison_table(metrics_by_strategy)


def _create_weights_heatmap(all_weights: dict[str, pd.Series]) -> go.Figure:
    """Create a heatmap of strategy weights (strategies x assets)."""
    if not all_weights:
        fig = go.Figure()
        fig.update_layout(**FIGURE_LAYOUT, title="No data")
        return fig

    # Collect all assets, sorted by average weight across strategies
    all_assets: set[str] = set()
    for w in all_weights.values():
        all_assets.update(w.index)

    avg_w = {}
    for asset in all_assets:
        vals = [float(w.get(asset, 0)) for w in all_weights.values()]
        avg_w[asset] = sum(vals) / len(vals)

    # Top 20 assets for readability
    sorted_assets = sorted(avg_w, key=avg_w.get, reverse=True)[:20]
    strategy_names = list(all_weights.keys())

    z = []
    for asset in sorted_assets:
        row = [float(all_weights[s].get(asset, 0)) for s in strategy_names]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        x=strategy_names,
        y=sorted_assets,
        z=z,
        colorscale="Blues",
        text=[[f"{v:.1%}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="Strategy: %{x}<br>Asset: %{y}<br>Weight: %{z:.2%}<extra></extra>",
    ))

    fig.update_layout(
        **FIGURE_LAYOUT,
        title="Weight Heatmap (Top 20 Assets)",
        xaxis_title="Strategy",
        yaxis_title="Asset",
        height=max(400, len(sorted_assets) * 25 + 100),
    )

    return fig
