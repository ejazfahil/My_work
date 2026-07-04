"""Bounded-memory streaming wrapper around the Page-Hinkley detector.

Feeds a live error/loss stream through Page-Hinkley, fires an optional callback
on each detected change point, then resets the detector so it can catch the
next one. Keeps a fixed-size window of recent values (for dashboards) and
exposes a running drift rate for the Prometheus exporter.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

from src.detectors.concept_drift import PageHinkleyDetector


class StreamingDriftMonitor:
    def __init__(self, window: int = 1000, ph_threshold: float = 50.0,
                 delta: float = 0.005, on_drift: Callable[[dict], None] | None = None):
        self.window: deque = deque(maxlen=window)
        self.ph = PageHinkleyDetector(delta=delta, threshold=ph_threshold)
        self.on_drift = on_drift or (lambda payload: None)
        self.drift_count = 0
        self.total = 0

    def update(self, error: float) -> bool:
        self.window.append(error)
        self.total += 1
        result = self.ph.update(error)
        if result.drift_detected:
            self.drift_count += 1
            self.on_drift({"t": result.t, "drift_number": self.drift_count})
            self.ph.reset()
            return True
        return False

    @property
    def drift_rate(self) -> float:
        """Fraction of observed samples that triggered a change point."""
        return self.drift_count / self.total if self.total else 0.0
