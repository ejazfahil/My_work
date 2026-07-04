"""Behavioural tests for the feature, concept and streaming detectors."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.detectors.feature_drift import FeatureDriftDetector
from src.detectors.concept_drift import PageHinkleyDetector
from src.detectors.streaming_detector import StreamingDriftMonitor

RNG = np.random.default_rng(42)


def _df(mu=0.0, sigma=1.0, n=500):
    return pd.DataFrame({"f": RNG.normal(mu, sigma, n)})


# ---- feature drift -------------------------------------------------------- #
def test_no_drift_same_distribution():
    ref = _df()
    assert not FeatureDriftDetector(ref).detect(_df())["overall"]


def test_drift_large_mean_shift():
    report = FeatureDriftDetector(_df(), threshold=0.1).detect(_df(mu=10))
    assert report["overall"]
    assert report["features"]["f"]["status"] == "significant_shift"


def test_report_includes_all_metrics():
    report = FeatureDriftDetector(_df()).detect(_df(mu=1))
    stats = report["features"]["f"]
    assert {"psi", "kl", "wasserstein", "status"} <= stats.keys()
    assert stats["wasserstein"] > 0


# ---- concept drift (Page-Hinkley) ---------------------------------------- #
def test_page_hinkley_flags_increase():
    ph = PageHinkleyDetector(delta=0.01, threshold=5.0)
    stream = np.concatenate([RNG.normal(0.1, 0.02, 500),
                             RNG.normal(0.3, 0.02, 500)])
    detected_at = next((i for i, v in enumerate(stream) if ph.update(v).drift_detected), None)
    assert detected_at is not None and detected_at >= 500


def test_page_hinkley_quiet_on_stationary_stream():
    ph = PageHinkleyDetector(delta=0.01, threshold=5.0)
    assert not any(ph.update(v).drift_detected for v in RNG.normal(0.1, 0.02, 2000))


def test_page_hinkley_running_mean_tracks_signal():
    # regression guard for the frozen-mean bug: mean must follow the stream
    ph = PageHinkleyDetector()
    for v in RNG.normal(5.0, 0.1, 200):
        ph.update(v)
    assert abs(ph._mean - 5.0) < 0.1


# ---- streaming monitor ---------------------------------------------------- #
def test_streaming_monitor_counts_and_rate():
    seen = []
    mon = StreamingDriftMonitor(ph_threshold=5.0, delta=0.01,
                                on_drift=lambda p: seen.append(p))
    for v in np.concatenate([RNG.normal(0.1, 0.02, 500), RNG.normal(0.4, 0.02, 500)]):
        mon.update(v)
    assert mon.drift_count >= 1 and len(seen) == mon.drift_count
    assert 0.0 <= mon.drift_rate <= 1.0
