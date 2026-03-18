"""Dashboard theme configuration — DARKLY bootstrap theme + Plotly template."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio

# Bootstrap theme URL — DARKLY from Bootswatch
THEME = dbc.themes.DARKLY
EXTERNAL_STYLESHEETS = [THEME, dbc.icons.FONT_AWESOME]

# Color palette for charts
COLORS = {
    "bg": "#222222",
    "card_bg": "#2d2d2d",
    "text": "#e0e0e0",
    "text_muted": "#999999",
    "primary": "#375a7f",
    "success": "#00bc8c",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "info": "#3498db",
    "accent_1": "#8e44ad",
    "accent_2": "#1abc9c",
    "accent_3": "#e67e22",
    "grid": "#333333",
}

# Strategy colors — consistent across all charts
STRATEGY_COLORS: dict[str, str] = {
    "equal_weight": "#95a5a6",
    "markowitz": "#3498db",
    "garch_gmv": "#e74c3c",
    "hrp": "#2ecc71",
    "risk_parity": "#f39c12",
    "cvar": "#9b59b6",
    "black_litterman": "#1abc9c",
    "regime_aware": "#e67e22",
}

# Plotly figure layout defaults — apply to every figure
FIGURE_LAYOUT: dict = {
    "template": "plotly_dark",
    "paper_bgcolor": COLORS["bg"],
    "plot_bgcolor": COLORS["bg"],
    "font": {"color": COLORS["text"], "family": "Inter, -apple-system, sans-serif"},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
    "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
    "colorway": list(STRATEGY_COLORS.values()),
}


def load_figure_template() -> None:
    """Register custom Plotly template. Call once at startup."""
    custom_template = go.layout.Template(layout=go.Layout(**FIGURE_LAYOUT))
    pio.templates["portfolio_dark"] = custom_template
    pio.templates.default = "portfolio_dark"
