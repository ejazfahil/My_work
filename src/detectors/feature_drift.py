"""Batch feature-drift detection across the numeric columns of a dataframe.

The reference (training) dataframe is captured once; each incoming batch is
compared against it. PSI drives the drift decision (via its conventional 0.25
band), while KL and Wasserstein are reported alongside for context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.psi import compute_psi, interpret_psi
from src.metrics.distances import kl_divergence, wasserstein


class FeatureDriftDetector:
    def __init__(self, reference: pd.DataFrame, threshold: float = 0.25,
                 n_bins: int = 10):
        self.threshold = threshold
        self.n_bins = n_bins
        self._cols = reference.select_dtypes(include=[np.number]).columns.tolist()
        self._ref = {c: reference[c].dropna().to_numpy() for c in self._cols}

    def detect(self, current: pd.DataFrame) -> dict:
        report: dict = {"features": {}, "drifted": [], "overall": False}
        for c in self._cols:
            if c not in current.columns:
                continue
            ref = self._ref[c]
            cur = current[c].dropna().to_numpy()
            psi = compute_psi(ref, cur, self.n_bins)
            report["features"][c] = {
                "psi": round(psi, 4),
                "kl": round(kl_divergence(ref, cur, self.n_bins), 4),
                "wasserstein": round(wasserstein(ref, cur), 4),
                "status": interpret_psi(psi),
            }
            if psi > self.threshold:
                report["drifted"].append(c)
        report["overall"] = bool(report["drifted"])
        report["rate"] = round(
            len(report["drifted"]) / max(len(report["features"]), 1), 4
        )
        return report
