"""Equal Weight (1/N) benchmark allocation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.optimization._base import PortfolioResult


def optimize_equal_weight(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """Compute 1/N equal weight portfolio."""
    raise NotImplementedError
