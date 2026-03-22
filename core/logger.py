"""Structured logging setup via structlog.

Call `setup_logging()` once at app startup (in app.py).
Then in any module: `log = structlog.get_logger(__name__)`
"""

from __future__ import annotations

import logging
import sys

import structlog

from core.config import settings


def setup_logging() -> None:
    """Configure structlog. JSON in production, colored console in development."""
    is_production = settings.APP_ENV == "production"
    log_level = getattr(logging, settings.APP_LOG_LEVEL.upper(), logging.INFO)

    # Force UTF-8 on stdout/stderr to prevent UnicodeEncodeError on Windows
    # (cp1252 can't encode structlog's colored output or traceback characters)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Configure stdlib logging (for third-party libs that use it)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_production:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
