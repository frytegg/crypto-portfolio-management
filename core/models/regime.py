"""HMM regime detection via hmmlearn.

CRITICAL: State labels are arbitrary between runs.
Always identify bull/bear by comparing means:
    bull_state = np.argmax(model.means_.flatten())
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass
class RegimeResult:
    """Result of HMM regime detection."""

    current_regime: str  # "Bull", "Bear", or "Sideways"
    regime_history: pd.Series  # DatetimeIndex, values = regime labels
    transition_matrix: np.ndarray
    state_means: dict[str, float]  # {"Bull": 0.001, "Bear": -0.002, ...}
    state_vols: dict[str, float]
    regime_probs: pd.DataFrame  # columns = regime labels, values = probabilities


def detect_regimes(
    returns: pd.DataFrame,
    n_regimes: int = 3,
    n_iter: int = 100,
) -> RegimeResult:
    """Fit GaussianHMM to portfolio returns and label states."""
    raise NotImplementedError
