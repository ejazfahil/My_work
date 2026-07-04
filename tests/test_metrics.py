"""Tests for the distribution-distance metrics."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.metrics.psi import compute_psi, interpret_psi
from src.metrics.distances import kl_divergence, jensen_shannon, wasserstein

RNG = np.random.default_rng(0)


def test_all_metrics_zero_on_identical_samples():
    x = RNG.normal(0, 1, 10000)
    assert compute_psi(x, x) < 1e-6
    assert kl_divergence(x, x) < 1e-6
    assert jensen_shannon(x, x) < 1e-6
    assert wasserstein(x, x) < 1e-6


def test_metrics_increase_with_shift():
    ref = RNG.normal(0, 1, 10000)
    small = RNG.normal(0.5, 1, 10000)
    large = RNG.normal(2.0, 1, 10000)
    assert compute_psi(ref, small) < compute_psi(ref, large)
    assert wasserstein(ref, small) < wasserstein(ref, large)


def test_wasserstein_of_translation_equals_shift():
    # W1 between N(0,1) and N(s,1) is exactly s; check to within sampling noise
    ref = RNG.normal(0, 1, 200000)
    cur = RNG.normal(1.5, 1, 200000)
    assert abs(wasserstein(ref, cur) - 1.5) < 0.05


def test_zero_variance_feature_is_stable():
    const = np.full(1000, 3.0)
    assert compute_psi(const, const) == 0.0


def test_interpret_bands():
    assert interpret_psi(0.05) == "stable"
    assert interpret_psi(0.15) == "moderate_shift"
    assert interpret_psi(0.30) == "significant_shift"
