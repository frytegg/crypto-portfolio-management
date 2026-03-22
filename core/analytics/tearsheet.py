"""Tearsheet generation via quantstats-lumi.

CRITICAL: import quantstats_lumi as qs -- NOT quantstats (pandas 2.x incompatible).

Generation runs in a subprocess to avoid crashing the Dash debug reloader.
quantstats uses matplotlib internally, which can trigger file-change detection
in werkzeug's stat-based reloader (font cache, __pycache__, etc).
"""
from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def _generate_in_subprocess(
    returns_json: str,
    benchmark_json: str | None,
    benchmark_name: str | None,
    title: str,
    output_path: str,
) -> None:
    """Worker function that runs in a separate process."""
    from io import StringIO

    import matplotlib
    import matplotlib.pyplot as plt

    # Use DejaVu Sans (ships with matplotlib) instead of Arial (Windows/Mac only).
    # Prevents font-not-found warnings and broken charts on Linux (Render, Docker).
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    import quantstats_lumi as qs

    qs.extend_pandas()

    returns = pd.read_json(StringIO(returns_json), typ="series")
    returns.name = "Portfolio"

    benchmark = None
    if benchmark_json is not None:
        benchmark = pd.read_json(StringIO(benchmark_json), typ="series")
        benchmark.name = benchmark_name or "Benchmark"

    qs.reports.html(
        returns,
        benchmark=benchmark,
        title=title,
        output=output_path,
        download_filename=output_path,
    )


def generate_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Portfolio Performance Report",
    output_path: str | None = None,
) -> str:
    """Generate HTML tearsheet using quantstats-lumi in a subprocess.

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

    # Serialize data for the subprocess
    returns_json = returns.to_json()
    benchmark_json = benchmark.to_json() if benchmark is not None else None
    benchmark_name = str(benchmark.name) if benchmark is not None and benchmark.name else None

    proc = multiprocessing.Process(
        target=_generate_in_subprocess,
        args=(returns_json, benchmark_json, benchmark_name, title, output_path),
    )
    proc.start()
    proc.join(timeout=60)  # 60s max

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError("Tearsheet generation timed out after 60 seconds")

    if proc.exitcode != 0:
        raise RuntimeError(
            f"Tearsheet subprocess failed with exit code {proc.exitcode}"
        )

    if not Path(output_path).exists():
        raise FileNotFoundError(f"Tearsheet was not created at {output_path}")

    log.info("tearsheet_generated", path=output_path)
    return output_path
