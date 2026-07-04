"""Distribution-distance metrics for feature drift.

Three complementary views of how far a current sample has moved from a
reference sample:

* ``kl_divergence``      - KL(actual || expected), unbounded, asymmetric.
* ``jensen_shannon``     - symmetric, bounded in [0, 1] (log base 2).
* ``wasserstein``        - 1-Wasserstein / earth-mover's distance, in the
                           original units of the feature, so it stays
                           interpretable and is robust to binning.

KL and JS are computed on a shared histogram of the two samples; Wasserstein is
computed directly on the raw samples via ``scipy.stats.wasserstein_distance``.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


def _shared_hist(expected, actual, n_bins, eps):
    """Bin both samples on shared edges and return normalised, clipped mass."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, bins=edges)[0] / expected.size, eps, None)
    a = np.clip(np.histogram(actual, bins=edges)[0] / actual.size, eps, None)
    e = e / e.sum()
    a = a / a.sum()
    return e, a


def kl_divergence(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    """KL(actual || expected) in nats. 0.0 when the samples match."""
    e, a = _shared_hist(expected, actual, n_bins, eps)
    return float(np.sum(a * np.log(a / e)))


def jensen_shannon(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Jensen-Shannon divergence in [0, 1] (log base 2), symmetric."""
    e, a = _shared_hist(expected, actual, n_bins, eps)
    m = 0.5 * (e + a)
    kl = lambda p, q: np.sum(p * (np.log(p / q) / np.log(2.0)))
    return float(0.5 * kl(e, m) + 0.5 * kl(a, m))


def wasserstein(expected, actual) -> float:
    """1-Wasserstein (earth-mover's) distance in the feature's own units."""
    return float(wasserstein_distance(np.asarray(expected, dtype=float),
                                      np.asarray(actual, dtype=float)))
