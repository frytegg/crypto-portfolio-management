"""HMM regime detection via hmmlearn.

CRITICAL: State labels are arbitrary between runs.
Always identify bull/bear by comparing means after fitting:
    bull_state = int(np.argmax(model.means_.flatten()))
    bear_state = int(np.argmin(model.means_.flatten()))
Never hardcode state 0 = Bear or state 1 = Bull.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from hmmlearn import hmm

log = structlog.get_logger(__name__)


def detect_regimes(
    returns: pd.Series,
    n_regimes: int = 2,
    lookback_days: int = 730,
    random_state: int = 42,
) -> dict:
    """Detect market regimes using a Gaussian Hidden Markov Model.

    Fits an HMM to the most recent `lookback_days` of returns and labels
    each hidden state by its mean return (highest mean = Bull, lowest = Bear).

    Args:
        returns: pd.Series of daily log returns (typically BTC or a market index).
        n_regimes: Number of hidden states. 2 = bull/bear, 3 = bull/bear/sideways.
        lookback_days: Number of most recent observations to use for training.
                       If returns has fewer observations, all data is used.
        random_state: Random seed for reproducibility.

    Returns:
        dict with keys:
            "regimes": pd.Series — integer regime labels with DatetimeIndex
                matching the lookback window of the input returns.
            "regime_names": dict[int, str] — maps state index to label,
                e.g. {0: "Bear", 1: "Bull"}.
            "transition_matrix": np.ndarray — N x N transition probability matrix.
            "regime_means": np.ndarray — mean return per state (from model.means_).
            "regime_vols": np.ndarray — volatility per state (sqrt of model.covars_).
            "current_regime": int — regime at the last observation.
            "current_regime_name": str — "Bull", "Bear", or "Sideways".
    """
    # Use at most lookback_days observations (or all if fewer available)
    n_obs = min(lookback_days, len(returns))
    X = returns.values[-n_obs:].reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(X)

    hidden_states = model.predict(X)

    # Label regimes by mean return — state indices are arbitrary
    means = model.means_.flatten()
    sorted_indices = np.argsort(means)

    name_map: dict[int, str] = {}
    if n_regimes == 2:
        name_map[sorted_indices[0]] = "Bear"
        name_map[sorted_indices[1]] = "Bull"
    elif n_regimes == 3:
        name_map[sorted_indices[0]] = "Bear"
        name_map[sorted_indices[1]] = "Sideways"
        name_map[sorted_indices[2]] = "Bull"
    else:
        # Generic labeling for arbitrary regime counts
        for rank, state_idx in enumerate(sorted_indices):
            name_map[state_idx] = f"Regime_{rank}"

    regime_series = pd.Series(
        hidden_states,
        index=returns.index[-n_obs:],
        name="regime",
    )

    current_state = int(hidden_states[-1])

    log.info(
        "regimes_detected",
        n_regimes=n_regimes,
        n_obs=n_obs,
        current_regime=name_map[current_state],
        regime_means=[round(float(m), 6) for m in means],
    )

    return {
        "regimes": regime_series,
        "regime_names": name_map,
        "transition_matrix": model.transmat_,
        "regime_means": means,
        "regime_vols": np.sqrt(model.covars_.flatten()),
        "current_regime": current_state,
        "current_regime_name": name_map[current_state],
    }
