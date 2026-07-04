"""Page-Hinkley sequential change-point test for concept drift.

Monitors a stream of a scalar signal (typically per-sample error or loss) and
raises an alarm when its mean shifts *upward* in a sustained way - i.e. the
model is getting worse. It keeps only running scalars, so memory is O(1)
regardless of stream length.

At each step the cumulative running mean ``m_t`` is updated, then

    sum_t = sum_{i<=t} (x_i - m_i - delta)
    PH_t  = sum_t - min_{i<=t} sum_i

and drift is flagged when ``PH_t > threshold`` (after a short warm-up).

``delta`` is the allowed slack (magnitude of change considered non-drift);
``threshold`` (lambda) is the alarm level. Both are scaled to the signal being
monitored - see ``benchmarks/run_benchmark.py`` for how they are chosen.

Note: the previous implementation used ``m = alpha*m + (1-alpha)*v`` with a
default ``alpha=1.0``, which froze the running mean at 0 and never updated it.
This version tracks the true cumulative mean.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PHResult:
    drift_detected: bool
    t: int
    statistic: float
    threshold: float


class PageHinkleyDetector:
    def __init__(self, delta: float = 0.005, threshold: float = 50.0,
                 min_instances: int = 30):
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._min_sum = 0.0
        self._mean = 0.0
        self._t = 0

    def update(self, value: float) -> PHResult:
        self._t += 1
        # cumulative running mean: m_t = m_{t-1} + (x_t - m_{t-1}) / t
        self._mean += (value - self._mean) / self._t
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        detected = self._t >= self.min_instances and ph > self.threshold
        return PHResult(detected, self._t, ph, self.threshold)
