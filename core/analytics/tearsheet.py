"""Tearsheet generation via quantstats-lumi.

CRITICAL: import quantstats_lumi as qs -- NOT quantstats (pandas 2.x incompatible).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def generate_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Portfolio Performance Report",
    output_path: str | None = None,
) -> str:
    """Generate HTML tearsheet using quantstats-lumi.

    Args:
        returns: Daily portfolio returns series.
        benchmark: Optional benchmark returns (e.g., BTC).
        title: Report title.
        output_path: Where to save HTML. If None, uses a temp file.

    Returns:
        Absolute path to the generated HTML file.
    """
    import quantstats_lumi as qs  # noqa: WPS433 — deferred import, heavy

    qs.extend_pandas()

    if output_path is None:
        output_path = str(
            Path(tempfile.gettempdir()) / "portfolio_tearsheet.html"
        )

    log.info(
        "generating_tearsheet",
        title=title,
        output_path=output_path,
        n_returns=len(returns),
        has_benchmark=benchmark is not None,
    )

    if benchmark is not None:
        qs.reports.html(
            returns,
            benchmark=benchmark,
            title=title,
            output=output_path,
            download_filename=output_path,
        )
    else:
        qs.reports.html(
            returns,
            title=title,
            output=output_path,
            download_filename=output_path,
        )

    log.info("tearsheet_generated", path=output_path)
    return output_path
