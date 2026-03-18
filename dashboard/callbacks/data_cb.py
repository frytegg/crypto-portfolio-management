"""Data loading and refresh callbacks.

Reads universe, prices, returns from diskcache. Populates initial views.
"""
from __future__ import annotations

from dash import Input, Output, callback
import structlog

log = structlog.get_logger(__name__)


@callback(
    Output("tab-content", "children", allow_duplicate=True),
    Input("main-tabs", "active_tab"),
    prevent_initial_call=True,
)
def render_tab_content(active_tab: str):
    """Route tab selection to appropriate tab layout."""
    raise NotImplementedError
