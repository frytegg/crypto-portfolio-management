"""Tests for all 7 optimization strategies + efficient frontier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.optimization._base import PortfolioResult
from core.optimization.black_litterman import (
    generate_onchain_views,
    optimize_black_litterman,
)
from core.optimization.cvar import optimize_cvar
from core.optimization.equal_weight import optimize_equal_weight
from core.optimization.hrp import get_hrp_dendrogram_data, optimize_hrp
from core.optimization.markowitz import (
    compute_efficient_frontier,
    optimize_garch_gmv,
    optimize_markowitz,
)
from core.optimization.regime_alloc import optimize_regime_aware
from core.optimization.risk_parity import optimize_risk_parity


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


# ---------------------------------------------------------------------------
# HRP
# ---------------------------------------------------------------------------

class TestHRP:
    def test_weights_sum_to_one(self, large_returns: pd.DataFrame) -> None:
        result = optimize_hrp(large_returns)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_returns_portfolio_result(self, large_returns: pd.DataFrame) -> None:
        result = optimize_hrp(large_returns)
        assert isinstance(result, PortfolioResult)
        assert result.name == "Hierarchical Risk Parity"
        assert "codependence" in result.metadata
        assert "linkage" in result.metadata

    def test_all_weights_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_hrp(large_returns)
        assert (result.weights >= -1e-6).all()

    def test_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_hrp(large_returns)
        assert result.expected_volatility > 0

    def test_dendrogram_data(self, large_returns: pd.DataFrame) -> None:
        data = get_hrp_dendrogram_data(large_returns)
        assert "linkage_matrix" in data
        assert "asset_names" in data
        n = len(large_returns.columns)
        assert data["linkage_matrix"].shape == (n - 1, 4)
        assert len(data["asset_names"]) == n


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------

class TestRiskParity:
    def test_weights_sum_to_one(self, large_returns: pd.DataFrame) -> None:
        result = optimize_risk_parity(large_returns)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_risk_contributions_approximately_equal(
        self, large_returns: pd.DataFrame
    ) -> None:
        """Each asset's risk contribution should be close to 1/N."""
        result = optimize_risk_parity(large_returns, risk_measure="MV")
        w = result.weights.values
        cov = large_returns.cov().values
        n = len(w)

        # Marginal risk contribution
        marginal_contrib = cov @ w
        # Risk contribution per asset
        risk_contrib = w * marginal_contrib
        total_risk = w @ cov @ w
        risk_contrib_pct = risk_contrib / total_risk

        # Each should be within 5% of 1/N
        target = 1.0 / n
        assert np.allclose(risk_contrib_pct, target, atol=0.05), (
            f"Risk contributions not equal: "
            f"max deviation = {np.max(np.abs(risk_contrib_pct - target)):.4f}"
        )

    def test_returns_portfolio_result(self, large_returns: pd.DataFrame) -> None:
        result = optimize_risk_parity(large_returns)
        assert isinstance(result, PortfolioResult)
        assert result.name == "Equal Risk Contribution"
        assert "method" in result.metadata

    def test_custom_risk_budget(self, large_returns: pd.DataFrame) -> None:
        n = len(large_returns.columns)
        # Double budget for first asset, equal for rest
        budget = [2.0] + [1.0] * (n - 1)
        result = optimize_risk_parity(large_returns, risk_budget=budget)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_risk_parity(large_returns)
        assert result.expected_volatility > 0


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------

class TestCVaR:
    def test_weights_sum_to_one(self, large_returns: pd.DataFrame) -> None:
        result = optimize_cvar(large_returns)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_weights_within_bounds(self, large_returns: pd.DataFrame) -> None:
        max_w = 0.15
        result = optimize_cvar(large_returns, max_weight=max_w)
        assert (result.weights <= max_w + 1e-6).all(), (
            f"Max weight violated: {result.weights.max():.4f} > {max_w}"
        )

    def test_returns_portfolio_result(self, large_returns: pd.DataFrame) -> None:
        result = optimize_cvar(large_returns)
        assert isinstance(result, PortfolioResult)
        assert result.name == "Mean-CVaR"
        assert result.metadata["alpha"] == 0.05
        assert result.metadata["objective"] == "MinRisk"

    def test_different_from_mv(self, large_returns: pd.DataFrame) -> None:
        """CVaR optimization should produce different weights than MV."""
        cvar_result = optimize_cvar(large_returns, objective="MinRisk")
        mv_result = optimize_markowitz(
            large_returns, objective="MinRisk", risk_measure="MV"
        )
        # Weights should not be identical (different risk measures)
        assert not np.allclose(
            cvar_result.weights.values, mv_result.weights.values, atol=1e-3
        ), "CVaR and MV produced identical weights — expected different allocations"

    def test_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_cvar(large_returns)
        assert result.expected_volatility > 0


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

class TestBlackLitterman:
    """Tests for BL optimization with on-chain signal views."""

    # Signals that trigger all 3 view rules
    BULLISH_SIGNALS: dict = {
        "tvl_momentum_30d": 0.10,
        "stablecoin_supply_change_30d": 0.05,
        "dex_volume_trend_7d": 1.5,
    }

    # Signals that trigger no views
    NEUTRAL_SIGNALS: dict = {
        "tvl_momentum_30d": 0.01,
        "stablecoin_supply_change_30d": 0.01,
        "dex_volume_trend_7d": 1.0,
    }

    def test_generate_views_bullish(self, large_returns: pd.DataFrame) -> None:
        assets = list(large_returns.columns)
        P, Q, conf = generate_onchain_views(self.BULLISH_SIGNALS, assets)
        # All 3 rules should fire (LINK is in large_returns via sample_returns)
        assert len(Q) >= 1  # at least TVL momentum fires
        assert P.shape[1] == len(assets)
        assert len(conf) == len(Q)

    def test_generate_views_no_signals(self, large_returns: pd.DataFrame) -> None:
        assets = list(large_returns.columns)
        P, Q, conf = generate_onchain_views(self.NEUTRAL_SIGNALS, assets)
        assert P.empty
        assert len(Q) == 0

    def test_bl_with_views_weights_sum(self, large_returns: pd.DataFrame) -> None:
        result = optimize_black_litterman(
            large_returns, onchain_signals=self.BULLISH_SIGNALS
        )
        assert isinstance(result, PortfolioResult)
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)
        assert result.metadata["fallback"] is False
        assert result.metadata["n_views"] >= 1

    def test_bl_no_views_fallback(self, large_returns: pd.DataFrame) -> None:
        result = optimize_black_litterman(
            large_returns, onchain_signals=self.NEUTRAL_SIGNALS
        )
        assert result.metadata["fallback"] is True
        assert "fallback" in result.name.lower()
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_bl_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        result = optimize_black_litterman(
            large_returns, onchain_signals=self.BULLISH_SIGNALS
        )
        assert result.expected_volatility > 0


# ---------------------------------------------------------------------------
# Regime-Aware
# ---------------------------------------------------------------------------

class TestRegimeAware:
    """Tests for regime-aware allocation using mocked regime_info dicts."""

    @staticmethod
    def _make_regime_info(regime_name: str) -> dict:
        """Create a minimal regime_info dict for testing."""
        return {
            "current_regime_name": regime_name,
            "regime_means": np.array([-0.001, 0.002]),
            "transition_matrix": np.array([[0.95, 0.05], [0.10, 0.90]]),
        }

    def test_bull_uses_max_sharpe(self, large_returns: pd.DataFrame) -> None:
        info = self._make_regime_info("Bull")
        result = optimize_regime_aware(large_returns, regime_info=info)
        assert "Bull" in result.name
        assert result.metadata["strategy_used"] == "Max Sharpe (Markowitz)"
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_bear_uses_min_vol(self, large_returns: pd.DataFrame) -> None:
        info = self._make_regime_info("Bear")
        result = optimize_regime_aware(large_returns, regime_info=info)
        assert "Bear" in result.name
        assert result.metadata["strategy_used"] == "Min Volatility (Markowitz)"
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_sideways_uses_risk_parity(self, large_returns: pd.DataFrame) -> None:
        info = self._make_regime_info("Sideways")
        result = optimize_regime_aware(large_returns, regime_info=info)
        assert "Sideways" in result.name
        assert result.metadata["strategy_used"] == "Risk Parity"
        assert np.isclose(result.weights.sum(), 1.0, atol=1e-4)

    def test_metadata_contains_regime_info(self, large_returns: pd.DataFrame) -> None:
        info = self._make_regime_info("Bull")
        result = optimize_regime_aware(large_returns, regime_info=info)
        assert "regime_means" in result.metadata
        assert "transition_matrix" in result.metadata
        assert result.metadata["current_regime"] == "Bull"

    def test_volatility_positive(self, large_returns: pd.DataFrame) -> None:
        info = self._make_regime_info("Bear")
        result = optimize_regime_aware(large_returns, regime_info=info)
        assert result.expected_volatility > 0
