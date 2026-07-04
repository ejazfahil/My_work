# Deployment

## Container

The repository ships a [`Dockerfile`](../Dockerfile) that installs the pinned
dependencies, runs the test suite at build time, and regenerates the benchmark
artifacts:

```bash
docker build -t mlops-drift-detector .
docker run --rm mlops-drift-detector        # runs benchmarks/run_benchmark.py
```

## Exporting metrics

`PrometheusExporter` turns a drift report into text-exposition metrics:

```
# HELP ml_drift_feature_psi Population Stability Index per feature.
# TYPE ml_drift_feature_psi gauge
ml_drift_feature_psi{feature="amount"} 0.31
ml_drift_overall_rate 0.14
```

Serving these from a live `/metrics` endpoint (a small `src/monitor.py` that runs
batch detection on a schedule and exposes the exporter output) is the planned
next step — see **Next steps** in the README. Today the exporter is a pure
function you call from your own service.

## Prometheus alert rule

```yaml
groups:
  - name: ml-drift
    rules:
      - alert: HighFeatureDrift
        expr: ml_drift_feature_psi > 0.25
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "Feature {{ $labels.feature }} PSI above 0.25 (significant shift)"
```
