"""Render a feature-drift report as Prometheus text-exposition metrics.

Emits one gauge per feature plus an overall drift rate, ready to be scraped
from a ``/metrics`` endpoint. Alerting (e.g. ``ml_drift_feature_psi > 0.25``
for 5m) lives in the Prometheus rules - see ``docs/DEPLOYMENT.md``.
"""
from __future__ import annotations


class PrometheusExporter:
    def export(self, report: dict) -> str:
        lines = [
            "# HELP ml_drift_feature_psi Population Stability Index per feature.",
            "# TYPE ml_drift_feature_psi gauge",
        ]
        for feature, stats in report.get("features", {}).items():
            lines.append(f'ml_drift_feature_psi{{feature="{feature}"}} {stats["psi"]}')
        lines.append("# HELP ml_drift_overall_rate Fraction of features drifting.")
        lines.append("# TYPE ml_drift_overall_rate gauge")
        lines.append(f"ml_drift_overall_rate {report.get('rate', 0.0)}")
        return "\n".join(lines) + "\n"
