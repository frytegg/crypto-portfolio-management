"""Live price feed callbacks — fires every 5 seconds via dcc.Interval."""
from __future__ import annotations

from dash import Input, Output, callback
import structlog

log = structlog.get_logger(__name__)

# Placeholder — callbacks will be registered when tab layouts are built
