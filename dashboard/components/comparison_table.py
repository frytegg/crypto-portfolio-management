"""Strategy comparison table component."""
from __future__ import annotations

import dash_bootstrap_components as dbc


def create_comparison_table(strategy_results: list[dict]) -> dbc.Table:
    """Create sortable comparison table of all strategy metrics."""
    raise NotImplementedError
