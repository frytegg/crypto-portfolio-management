"""Covariance matrix estimation via riskfolio-lib.

Methods: Ledoit-Wolf shrinkage, Gerber statistic, denoised (Marchenko-Pastur).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

CovMethod = Literal["hist", "ledoit", "gerber", "denoised"]


def estimate_covariance(
    returns: pd.DataFrame,
    method: CovMethod = "ledoit",
) -> np.ndarray:
    """Estimate NxN covariance matrix from returns DataFrame.

    Returns a positive semi-definite numpy array (N, N).
    """
    raise NotImplementedError
