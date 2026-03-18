"""Tearsheet generation via quantstats-lumi.

CRITICAL: import quantstats_lumi as qs -- NOT quantstats (pandas 2.x incompatible).
"""
from __future__ import annotations

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def generate_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Portfolio Tearsheet",
    output_path: str | None = None,
) -> str:
    """Generate HTML tearsheet. Returns HTML string (or writes to output_path)."""
    raise NotImplementedError
