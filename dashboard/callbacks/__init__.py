"""Callback registration aggregator.

Call register_all_callbacks(app) once from app.py after layout is assigned.
Each callback module registers its own @callback decorators on import.
"""

from __future__ import annotations

import dash


def register_all_callbacks(app: dash.Dash) -> None:
    """Import all callback modules to register them with the Dash app.

    Import order matters for avoiding circular imports:
    data_cb -> live_cb -> optimization_cb -> garch_cb ->
    regime_cb -> onchain_cb -> backtest_cb -> report_cb
    """
    from dashboard.callbacks import (  # noqa: F401
        data_cb,
        live_cb,
        optimization_cb,
        garch_cb,
        regime_cb,
        onchain_cb,
        backtest_cb,
        report_cb,
    )
