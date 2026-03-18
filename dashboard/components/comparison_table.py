"""Strategy comparison table component."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from dashboard.theme import COLORS


def create_comparison_table(metrics_by_strategy: dict[str, dict]) -> dbc.Table:
    """Create a comparison table: rows = metric names, columns = strategies.

    Best value per row is highlighted in green.

    Args:
        metrics_by_strategy: {strategy_name: {metric_name: value, ...}, ...}
            Values should be pre-formatted strings or floats.
    """
    if not metrics_by_strategy:
        return html.P("No strategy results to compare.", className="text-muted")

    strategies = list(metrics_by_strategy.keys())
    # Collect all metric names from the first strategy (all should have the same keys)
    metric_names = list(next(iter(metrics_by_strategy.values())).keys())

    # Metrics where higher is better
    higher_is_better = {
        "Expected Return", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Omega Ratio", "Ann. Return",
    }

    # Build header row
    header = html.Thead(
        html.Tr(
            [html.Th("Metric")] + [html.Th(s) for s in strategies]
        )
    )

    # Build body rows with best-value highlighting
    rows = []
    for metric in metric_names:
        raw_values = {}
        for strat in strategies:
            val = metrics_by_strategy[strat].get(metric, "")
            raw_values[strat] = val

        # Try to find numeric best value for highlighting
        numeric_vals = {}
        for strat, val in raw_values.items():
            try:
                numeric_vals[strat] = float(str(val).rstrip("%").replace(",", ""))
            except (ValueError, TypeError):
                pass

        best_strat = None
        if numeric_vals:
            if metric in higher_is_better:
                best_strat = max(numeric_vals, key=numeric_vals.get)
            else:
                # Lower is better for risk metrics (volatility, drawdown, VaR, CVaR)
                best_strat = min(numeric_vals, key=numeric_vals.get)

        cells = [html.Td(metric, style={"fontWeight": "bold"})]
        for strat in strategies:
            val = raw_values.get(strat, "")
            style = {}
            if strat == best_strat:
                style = {"color": COLORS["success"], "fontWeight": "bold"}
            cells.append(html.Td(str(val), style=style))

        rows.append(html.Tr(cells))

    body = html.Tbody(rows)

    return dbc.Table(
        [header, body],
        bordered=True,
        hover=True,
        responsive=True,
        color="dark",
        className="mt-3",
        size="sm",
    )
