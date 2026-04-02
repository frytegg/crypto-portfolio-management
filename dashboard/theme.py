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

# Strategy colors — keyed by internal name (used in backtest, equity_chart)
STRATEGY_COLORS: dict[str, str] = {
    "equal_weight": "#888888",
    "markowitz": "#00d4ff",
    "garch_gmv": "#ff6b35",
    "hrp": "#7bed9f",
    "risk_parity": "#ffa502",
    "cvar": "#ff4757",
    "black_litterman": "#eccc68",
    "regime_aware": "#a29bfe",
}

# Strategy colors — keyed by display name (used in comparison table, weights chart)
STRATEGY_DISPLAY_COLORS: dict[str, str] = {
    "Equal Weight":      "#888888",
    "Markowitz MVO":     "#00d4ff",
    "GARCH-GMV":         "#ff6b35",
    "Hierarchical Risk Parity": "#7bed9f",
    "Equal Risk Contribution":  "#ffa502",
    "Mean-CVaR":         "#ff4757",
    "Black-Litterman":   "#eccc68",
    "Black-Litterman (Fallback: Markowitz)": "#eccc68",
    "Regime-Aware":      "#a29bfe",
}



# Strategy dropdown options and display name mapping (used across callbacks)
STRATEGY_OPTIONS = [
    {"label": "Equal Weight", "value": "equal_weight"},
    {"label": "Markowitz MVO", "value": "markowitz"},
    {"label": "GARCH-GMV", "value": "garch_gmv"},
    {"label": "Hierarchical Risk Parity", "value": "hrp"},
    {"label": "Equal Risk Contribution", "value": "risk_parity"},
    {"label": "Mean-CVaR", "value": "cvar"},
    {"label": "Black-Litterman", "value": "black_litterman"},
    {"label": "Regime-Aware", "value": "regime_aware"},
]

STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    o["value"]: o["label"] for o in STRATEGY_OPTIONS
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba() string for semi-transparent fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Plotly figure layout defaults — apply to every figure
FIGURE_LAYOUT: dict = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#AAAAAA", "family": "Inter, -apple-system, sans-serif"},
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    "colorway": list(STRATEGY_COLORS.values()),
}


def load_figure_template() -> None:
    """Register custom Plotly template. Call once at startup."""
    template_layout = {
        **FIGURE_LAYOUT,
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
    }
    custom_template = go.layout.Template(layout=go.Layout(**template_layout))
    pio.templates["portfolio_dark"] = custom_template
    pio.templates.default = "portfolio_dark"
