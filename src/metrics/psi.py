"""Population Stability Index (PSI) for feature drift.

PSI measures how much a variable's distribution has shifted between a reference
(training) window and a current (production) window. It is the symmetric
Jeffreys divergence between the two binned distributions:

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

where ``e_i`` and ``a_i`` are the reference and current mass in bin ``i``.
Equivalently, PSI = KL(actual || expected) + KL(expected || actual).

Conventional interpretation bands (industry standard, e.g. credit-risk
monitoring): < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
"""
from __future__ import annotations

import numpy as np

STABLE = 0.10
SIGNIFICANT = 0.25


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Return the PSI between a reference and a current sample.

    Bin edges are derived from the reference range and the outer edges are
    extended to +/- infinity, so current-window mass that falls outside the
    reference range is still counted (this matters precisely when the feature
    has drifted, and is the correction over a naive fixed-range histogram).

    A feature with (near) zero variance in either window carries no
    distributional information, so PSI is defined as 0.0 in that case.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0
    if np.std(expected) < eps or np.std(actual) < eps:
        return 0.0

    edges = np.linspace(expected.min(), expected.max(), n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    e = np.clip(np.histogram(expected, bins=edges)[0] / expected.size, eps, None)
    a = np.clip(np.histogram(actual, bins=edges)[0] / actual.size, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def interpret_psi(value: float) -> str:
    """Map a PSI value to its conventional stability band."""
    if value < STABLE:
        return "stable"
    if value < SIGNIFICANT:
        return "moderate_shift"
    return "significant_shift"
