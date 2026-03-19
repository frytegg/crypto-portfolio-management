"""Correlation matrix heatmap component."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.covariance import LedoitWolf

from dashboard.theme import COLORS, FIGURE_LAYOUT


def create_correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap with hierarchical clustering order.

    Uses Ledoit-Wolf shrinkage covariance estimator for robustness,
    then converts to correlation matrix. Assets are sorted by Ward
    hierarchical clustering to reveal block structure.

    Args:
        returns: T x N DataFrame of daily returns.

    Returns:
        go.Figure with annotated heatmap.
    """
    # Ledoit-Wolf covariance → correlation
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    # Hierarchical clustering for better visual ordering
    dist = 1 - corr
    np.fill_diagonal(dist, 0.0)
    # Ensure symmetry and non-negativity
    dist = np.clip((dist + dist.T) / 2, 0, 2)
    condensed = dist[np.triu_indices(len(dist), k=1)]
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    # Reorder
    assets = [returns.columns[i] for i in order]
    corr_ordered = corr[np.ix_(order, order)]

    # Annotation text
    text = [[f"{corr_ordered[i, j]:.2f}" for j in range(len(assets))] for i in range(len(assets))]

    fig = go.Figure(
        go.Heatmap(
            z=corr_ordered,
            x=assets,
            y=assets,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(
                title="Correlation",
                thickness=15,
                len=0.8,
            ),
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(**FIGURE_LAYOUT)
    fig.update_layout(
        title="Asset Correlation Matrix (Ledoit-Wolf)",
        height=max(450, len(assets) * 25 + 100),
        margin=dict(l=80, r=30, t=50, b=100),
    )
    fig.update_xaxes(tickangle=-45, side="bottom")
    fig.update_yaxes(autorange="reversed")

    return fig
