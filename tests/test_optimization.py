"""Tests for equal weight, Markowitz MVO, efficient frontier, and GARCH-GMV."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.optimization._base import PortfolioResult
from core.optimization.equal_weight import optimize_equal_weight
from core.optimization.markowitz import (
    compute_efficient_frontier,
    optimize_garch_gmv,
    optimize_markowitz,
)


# ---------------------------------------------------------------------------
# Equal Weight
# ---------------------------------------------------------------------------

class TestEqualWeight:
    def test_weights_sum_to_one(self, large_returns: pd.DataFrame) -> None:
        result = optimize_equal_weight(large_returns)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)

    def test_all_weights_equal(self, large_returns: pd.DataFrame) -> None:
        result = optimize_equal_weight(large_returns)
        n = len(large_returns.columns)
        expected = 1.0 / n
        assert np.allclose(result.weights.values, expected, atol=1e-10)

    def test_returns_portfolio_result(self, large_returns: pd.DataFrame) -> None:
        result = optimize_equal_weight(large_returns)
        assert isinstance(result, PortfolioResult)
        assert result.name == "Equal Weight"
        assert result.metadata["n_assets"] == len(large_returns.columns)

    def test_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_equal_weight(large_returns)
        assert result.expected_volatility > 0


# ---------------------------------------------------------------------------
# Markowitz MVO
# ---------------------------------------------------------------------------

class TestMarkowitz:
    def test_weights_sum_to_one(self, large_returns: pd.DataFrame) -> None:
        result = optimize_markowitz(large_returns)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_weights_within_bounds(self, large_returns: pd.DataFrame) -> None:
        max_w = 0.15
        result = optimize_markowitz(large_returns, max_weight=max_w)
        assert (result.weights <= max_w + 1e-6).all(), (
            f"Max weight violated: {result.weights.max():.4f} > {max_w}"
        )

    def test_min_weight_bound(self, large_returns: pd.DataFrame) -> None:
        result = optimize_markowitz(large_returns, min_weight=0.0)
        assert (result.weights >= -1e-6).all()

    def test_returns_portfolio_result(self, large_returns: pd.DataFrame) -> None:
        result = optimize_markowitz(large_returns)
        assert isinstance(result, PortfolioResult)
        assert result.name == "Markowitz MVO"
        assert "objective" in result.metadata
        assert "risk_measure" in result.metadata

    def test_min_risk_objective(self, large_returns: pd.DataFrame) -> None:
        result = optimize_markowitz(large_returns, objective="MinRisk")
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)
        assert result.expected_volatility > 0

    def test_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_markowitz(large_returns)
        assert result.expected_volatility > 0


# ---------------------------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------------------------

class TestEfficientFrontier:
    def test_correct_number_of_points(self, large_returns: pd.DataFrame) -> None:
        n_pts = 30
        frontier = compute_efficient_frontier(large_returns, n_points=n_pts)
        # riskfolio may return slightly fewer points if some fail
        assert len(frontier["frontier_returns"]) >= n_pts - 5
        assert len(frontier["frontier_risks"]) == len(frontier["frontier_returns"])

    def test_returns_all_keys(self, large_returns: pd.DataFrame) -> None:
        frontier = compute_efficient_frontier(large_returns, n_points=20)
        expected_keys = {
            "frontier_returns",
            "frontier_risks",
            "frontier_weights",
            "max_sharpe_weights",
            "min_vol_weights",
            "asset_returns",
            "asset_risks",
        }
        assert set(frontier.keys()) == expected_keys

    def test_frontier_risks_positive(self, large_returns: pd.DataFrame) -> None:
        frontier = compute_efficient_frontier(large_returns, n_points=20)
        assert all(r > 0 for r in frontier["frontier_risks"])

    def test_asset_risks_match_asset_count(self, large_returns: pd.DataFrame) -> None:
        frontier = compute_efficient_frontier(large_returns, n_points=20)
        assert len(frontier["asset_returns"]) == len(large_returns.columns)
        assert len(frontier["asset_risks"]) == len(large_returns.columns)

    def test_max_sharpe_weights_sum_to_one(self, large_returns: pd.DataFrame) -> None:
        frontier = compute_efficient_frontier(large_returns, n_points=20)
        assert np.isclose(frontier["max_sharpe_weights"].sum(), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# GARCH-GMV
# ---------------------------------------------------------------------------

class TestGarchGMV:
    def test_returns_valid_weights(self, large_returns: pd.DataFrame) -> None:
        result = optimize_garch_gmv(large_returns)
        assert isinstance(result, PortfolioResult)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_weights_within_bounds(self, large_returns: pd.DataFrame) -> None:
        max_w = 0.15
        result = optimize_garch_gmv(large_returns, max_weight=max_w)
        assert (result.weights <= max_w + 1e-6).all()

    def test_metadata(self, large_returns: pd.DataFrame) -> None:
        result = optimize_garch_gmv(large_returns)
        assert result.metadata["strategy"] == "GARCH-GMV"
        assert result.metadata["uses_garch_cov"] is True

    def test_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_garch_gmv(large_returns)
        assert result.expected_volatility > 0
