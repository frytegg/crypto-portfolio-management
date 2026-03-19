"""Walk-forward backtest callbacks — Tab 7.

Config panel → Run Backtest → equity curve, drawdown, monthly heatmap,
turnover bar chart, weights evolution, and metrics table.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import structlog

from core.data.cache import cache
from core.risk.backtest import BacktestConfig, BacktestResult, run_backtest, _BACKTEST_STRATEGIES
from core.risk.metrics import compute_drawdown_series
from dashboard.components.drawdown_chart import create_drawdown_chart
from dashboard.components.equity_chart import create_equity_chart
from dashboard.components.metric_card import create_metric_card
from dashboard.theme import COLORS, FIGURE_LAYOUT, STRATEGY_COLORS

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tab layout builder (called from data_cb.py tab routing)
# ---------------------------------------------------------------------------

def build_backtest_tab(returns_summary: dict | None) -> html.Div:
    """Build the Backtest Engine tab layout."""
    if not returns_summary or not returns_summary.get("columns"):
        return html.Div(
            dbc.Spinner(
                html.H5("Waiting for market data...", className="text-muted"),
                color="primary",
            ),
            className="text-center mt-5",
        )

    start_date = returns_summary.get("start_date", "2023-01-01")
    end_date = returns_summary.get("end_date", "2025-01-01")

    strategy_options = [
        {"label": "Equal Weight (1/N)", "value": "equal_weight"},
        {"label": "Markowitz MVO", "value": "markowitz"},
        {"label": "GARCH-Enhanced GMV", "value": "garch_gmv"},
        {"label": "Hierarchical Risk Parity", "value": "hrp"},
        {"label": "Equal Risk Contribution", "value": "risk_parity"},
        {"label": "Mean-CVaR", "value": "cvar"},
    ]

    return html.Div([
        html.H4("Walk-Forward Backtest Engine", className="mb-3"),

        # Config panel
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Strategy", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="bt-strategy-dropdown",
                            options=strategy_options,
                            value="equal_weight",
                            clearable=False,
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Rebalance Frequency", className="fw-bold mb-1"),
                        dcc.Dropdown(
                            id="bt-rebalance-dropdown",
                            options=[
                                {"label": "Weekly", "value": "weekly"},
                                {"label": "Monthly", "value": "monthly"},
                                {"label": "Quarterly", "value": "quarterly"},
                            ],
                            value="monthly",
                            clearable=False,
                        ),
                    ], md=2),
                    dbc.Col([
                        html.Label("Date Range", className="fw-bold mb-1"),
                        dcc.DatePickerRange(
                            id="bt-date-range",
                            start_date=start_date,
                            end_date=end_date,
                            display_format="YYYY-MM-DD",
                            className="d-block",
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Label("Transaction Cost (bps)", className="fw-bold mb-1"),
                        dbc.Input(
                            id="bt-tx-cost-input",
                            type="number",
                            value=10,
                            min=0,
                            max=100,
                            step=1,
                        ),
                    ], md=2),
                    dbc.Col([
                        html.Label("Initial Capital ($)", className="fw-bold mb-1"),
                        dbc.Input(
                            id="bt-capital-input",
                            type="number",
                            value=100_000,
                            min=1000,
                            step=1000,
                        ),
                    ], md=2),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Lookback Window (days)", className="fw-bold mb-1"),
                        dcc.Slider(
                            id="bt-lookback-slider",
                            min=90,
                            max=730,
                            step=30,
                            value=365,
                            marks={v: str(v) for v in [90, 180, 365, 540, 730]},
                            tooltip={"placement": "bottom"},
                        ),
                    ], md=6),
                    dbc.Col([
                        html.Label("Max Weight per Asset", className="fw-bold mb-1"),
                        dcc.Slider(
                            id="bt-max-weight-slider",
                            min=0.05,
                            max=1.0,
                            step=0.05,
                            value=0.15,
                            marks={v: f"{v:.0%}" for v in [0.05, 0.10, 0.15, 0.30, 0.50, 1.0]},
                            tooltip={"placement": "bottom"},
                        ),
                    ], md=4),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Run Backtest"],
                            id="bt-run-btn",
                            color="success",
                            size="lg",
                            className="mt-4 w-100",
                        ),
                    ], md=2),
                ]),
            ]),
        ], className="mb-4"),

        # Status badge
        html.Div(id="bt-status-badge", className="mb-3"),

        # Results container (populated by callback)
        dbc.Spinner(
            html.Div(id="bt-results-container"),
            color="success",
            type="border",
        ),
    ])


# ---------------------------------------------------------------------------
# Callback: Run backtest
# ---------------------------------------------------------------------------

@callback(
    Output("bt-results-container", "children"),
    Output("bt-status-badge", "children"),
    Input("bt-run-btn", "n_clicks"),
    State("bt-strategy-dropdown", "value"),
    State("bt-rebalance-dropdown", "value"),
    State("bt-date-range", "start_date"),
    State("bt-date-range", "end_date"),
    State("bt-tx-cost-input", "value"),
    State("bt-lookback-slider", "value"),
    State("bt-max-weight-slider", "value"),
    State("bt-capital-input", "value"),
    prevent_initial_call=True,
)
def run_backtest_callback(
    n_clicks: int | None,
    strategy: str,
    rebalance_freq: str,
    start_date: str,
    end_date: str,
    tx_cost_bps: float,
    lookback_days: int,
    max_weight: float,
    initial_capital: float,
) -> tuple:
    """Execute walk-forward backtest and render all result charts."""
    if not n_clicks:
        return no_update, no_update

    prices = cache.get("prices")
    if prices is None:
        return (
            html.Div(
                dbc.Alert("No price data available. Load data first.", color="warning"),
            ),
            dbc.Badge("No data", color="warning"),
        )

    config = BacktestConfig(
        strategy=strategy,
        start_date=start_date[:10],  # strip time if present
        end_date=end_date[:10],
        rebalance_frequency=rebalance_freq,
        lookback_days=lookback_days,
        transaction_cost_bps=tx_cost_bps or 10.0,
        max_weight=max_weight,
        initial_capital=initial_capital or 100_000.0,
    )

    try:
        result = run_backtest(prices, config)
    except Exception as exc:
        log.error("backtest_failed", strategy=strategy, error=str(exc), exc_info=True)
        return (
            html.Div(dbc.Alert(f"Backtest failed: {exc}", color="danger", dismissable=True)),
            dbc.Badge("Failed", color="danger"),
        )

    content = _build_backtest_results(result)
    badge = dbc.Badge(
        f"Backtest complete — {len(result.rebalance_dates)} rebalances",
        color="success",
        className="fs-6",
    )
    return content, badge


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def _build_backtest_results(result: BacktestResult) -> html.Div:
    """Build the full results display from a BacktestResult."""
    metrics = result.metrics

    # --- KPI cards ---
    total_return = (result.equity_curve.iloc[-1] / result.config.initial_capital - 1) * 100
    kpi_row = dbc.Row([
        dbc.Col(create_metric_card(
            "Total Return",
            f"{total_return:+.2f}%",
            color="success" if total_return >= 0 else "danger",
        ), md=2),
        dbc.Col(create_metric_card(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            color="info",
        ), md=2),
        dbc.Col(create_metric_card(
            "Max Drawdown",
            f"{metrics['max_drawdown'] * 100:.2f}%",
            color="danger",
        ), md=2),
        dbc.Col(create_metric_card(
            "Ann. Volatility",
            f"{metrics['annualized_volatility'] * 100:.2f}%",
            color="warning",
        ), md=2),
        dbc.Col(create_metric_card(
            "Final Value",
            f"${result.equity_curve.iloc[-1]:,.0f}",
        ), md=2),
        dbc.Col(create_metric_card(
            "Total Costs",
            f"${result.transaction_costs_total:,.0f}",
            color="warning",
        ), md=2),
    ], className="mb-4")

    # --- Equity curve ---
    equity_fig = create_equity_chart({result.config.strategy: result.equity_curve})

    # --- Drawdown chart ---
    dd_series = result.equity_curve / result.equity_curve.cummax() - 1
    dd_fig = create_drawdown_chart({result.config.strategy: dd_series})

    # --- Monthly returns heatmap ---
    monthly_fig = _create_monthly_heatmap(result.returns)

    # --- Turnover bar chart ---
    turnover_fig = _create_turnover_chart(result.turnover_history)

    # --- Weights evolution ---
    weights_fig = _create_weights_evolution(result.weights_history)

    # --- Metrics table ---
    metrics_table = _build_metrics_table(metrics)

    return html.Div([
        kpi_row,
        dbc.Row([
            dbc.Col(dcc.Graph(figure=equity_fig), md=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=dd_fig), md=6),
            dbc.Col(dcc.Graph(figure=monthly_fig), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=turnover_fig), md=6),
            dbc.Col(dcc.Graph(figure=weights_fig), md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(metrics_table, md=12),
        ]),
    ])


def _create_monthly_heatmap(returns: pd.Series) -> go.Figure:
    """Reshape daily returns to year x month matrix and render as heatmap."""
    if len(returns) == 0:
        return go.Figure()

    # Aggregate monthly returns (compounded)
    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda x: ((1 + x).prod() - 1) * 100
    )
    monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["year", "month"])

    # Pivot to year x month
    years = sorted(monthly.index.get_level_values("year").unique())
    months = list(range(1, 13))
    month_labels = [calendar.month_abbr[m] for m in months]

    z = []
    for year in years:
        row = []
        for month in months:
            if (year, month) in monthly.index:
                row.append(round(float(monthly.loc[(year, month)]), 2))
            else:
                row.append(None)
        z.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=month_labels,
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            colorbar=dict(title="%", thickness=12),
        )
    )

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="Monthly Returns Heatmap",
        height=250,
        margin=dict(l=60, r=30, t=50, b=30),
    )
    fig.update_yaxes(autorange="reversed")

    return fig


def _create_turnover_chart(turnover_history: pd.Series) -> go.Figure:
    """Bar chart of portfolio turnover at each rebalance date."""
    fig = go.Figure()

    if len(turnover_history) > 0:
        fig.add_trace(go.Bar(
            x=turnover_history.index,
            y=turnover_history.values * 100,
            marker_color=COLORS["info"],
            opacity=0.8,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra>Turnover</extra>",
        ))

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="Portfolio Turnover at Rebalance",
        xaxis_title="Date",
        yaxis_title="Turnover (%)",
        height=350,
        showlegend=False,
    )

    return fig


def _create_weights_evolution(weights_history: pd.DataFrame) -> go.Figure:
    """Stacked area chart showing asset weight evolution over time."""
    fig = go.Figure()

    if len(weights_history) == 0:
        fig.update_layout(**FIGURE_LAYOUT, title="Weights Evolution", height=350)
        return fig

    # Sort assets by average weight (largest on bottom)
    avg_weights = weights_history.mean().sort_values(ascending=False)
    top_assets = avg_weights.head(15).index.tolist()

    for asset in reversed(top_assets):  # Reverse so largest is on bottom
        if asset in weights_history.columns:
            fig.add_trace(go.Scatter(
                x=weights_history.index,
                y=weights_history[asset].values * 100,
                mode="lines",
                name=asset,
                stackgroup="one",
                hovertemplate="%{y:.1f}%<extra>" + asset + "</extra>",
            ))

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="Portfolio Weights Evolution (Top 15 Assets)",
        xaxis_title="Date",
        yaxis_title="Weight (%)",
        height=350,
        legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        hovermode="x unified",
    )

    return fig


def _build_metrics_table(metrics: dict) -> dbc.Card:
    """Build a summary table of backtest risk metrics."""
    rows = [
        ("Annualized Return", f"{metrics['annualized_return'] * 100:.2f}%"),
        ("Annualized Volatility", f"{metrics['annualized_volatility'] * 100:.2f}%"),
        ("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}"),
        ("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}" if np.isfinite(metrics['sortino_ratio']) else "∞"),
        ("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}"),
        ("Omega Ratio", f"{metrics['omega_ratio']:.3f}" if np.isfinite(metrics['omega_ratio']) else "∞"),
        ("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%"),
        ("Max DD Duration", f"{metrics['max_drawdown_duration']} days"),
        ("VaR 95%", f"{metrics['var_95'] * 100:.2f}%"),
        ("CVaR 95%", f"{metrics['cvar_95'] * 100:.2f}%"),
        ("VaR 99%", f"{metrics['var_99'] * 100:.2f}%"),
        ("CVaR 99%", f"{metrics['cvar_99'] * 100:.2f}%"),
        ("Skewness", f"{metrics['skewness']:.3f}"),
        ("Excess Kurtosis", f"{metrics['kurtosis']:.3f}"),
        ("Positive Days", f"{metrics['positive_days_pct']:.1f}%"),
        ("Best Day", f"{metrics['best_day'] * 100:.2f}%"),
        ("Worst Day", f"{metrics['worst_day'] * 100:.2f}%"),
    ]

    table_rows = [
        html.Tr([
            html.Td(name, style={"fontWeight": "bold"}),
            html.Td(value, style={"textAlign": "right"}),
        ])
        for name, value in rows
    ]

    return dbc.Card([
        dbc.CardHeader(html.H5("Risk Metrics Summary", className="mb-0")),
        dbc.CardBody(
            dbc.Table(
                [html.Tbody(table_rows)],
                bordered=True,
                dark=True,
                hover=True,
                size="sm",
                className="mb-0",
            ),
        ),
    ])
