"""Tearsheet generation via quantstats-lumi.

CRITICAL: import quantstats_lumi as qs -- NOT quantstats (pandas 2.x incompatible).
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import quantstats_lumi as qs
import structlog

log = structlog.get_logger(__name__)

# Force DejaVu Sans (ships with matplotlib) instead of Arial (Windows/Mac only).
# Must be set before quantstats renders any charts.
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "sans-serif"

# Suppress matplotlib font-not-found warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*findfont.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Font family.*")


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

    qs.extend_pandas()

    returns = returns.copy()
    returns.name = "Portfolio"

    if benchmark is not None:
        benchmark = benchmark.copy()
        if not benchmark.name:
            benchmark.name = "Benchmark"

    try:
        qs.reports.html(
            returns,
            benchmark=benchmark,
            title=title,
            output=output_path,
            download_filename=output_path,
        )

        if not Path(output_path).exists():
            raise RuntimeError("quantstats did not generate the output file")

        log.info("tearsheet_generated", path=output_path)
        return output_path

    except Exception as exc:
        log.error("tearsheet_generation_failed", error=str(exc))
        raise
