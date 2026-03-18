"""Expected returns estimation methods."""
from __future__ import annotations

from typing import Literal

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

ReturnMethod = Literal["hist", "ewma", "james_stein", "capm"]


def estimate_returns(
    returns: pd.DataFrame,
    method: ReturnMethod = "hist",
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Estimate expected returns vector (annualized).

    Returns pd.Series with index=asset names, values=annualized expected returns.
    """
    raise NotImplementedError
