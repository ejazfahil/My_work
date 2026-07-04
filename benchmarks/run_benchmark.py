"""Reproducible drift-detection benchmark.

Single entry point. Regenerates every table and plot in ``results/`` from fixed
seeds. Nothing here is hand-tuned to a target number: parameters are set once,
declared in ``results/config.json``, and whatever the detectors produce is what
gets reported.

    python benchmarks/run_benchmark.py            # default config
    python benchmarks/run_benchmark.py --seed 7   # override

Two scenarios:

  A. Distribution-distance sensitivity - how PSI / KL / JS / Wasserstein grow as
     a Gaussian feature is shifted in mean and inflated in variance.
  B. Page-Hinkley concept-drift detection - detection delay and false-alarm
     behaviour on simulated error streams, averaged over many seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.metrics.psi import compute_psi, interpret_psi, STABLE, SIGNIFICANT  # noqa: E402
from src.metrics.distances import kl_divergence, jensen_shannon, wasserstein  # noqa: E402
from src.detectors.concept_drift import PageHinkleyDetector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")


# --------------------------------------------------------------------------- #
# Scenario A: distribution-distance sensitivity
# --------------------------------------------------------------------------- #
def scenario_distance_sensitivity(rng, n=5000, n_bins=10):
    ref = rng.normal(0.0, 1.0, n)
    rows = []

    for shift in np.round(np.arange(0.0, 2.01, 0.1), 2):
        cur = rng.normal(shift, 1.0, n)
        psi = compute_psi(ref, cur, n_bins)
        rows.append({
            "scenario": "mean_shift", "magnitude": float(shift),
            "psi": round(psi, 4),
            "kl": round(kl_divergence(ref, cur, n_bins), 4),
            "js": round(jensen_shannon(ref, cur, n_bins), 4),
            "wasserstein": round(wasserstein(ref, cur), 4),
            "psi_status": interpret_psi(psi),
        })

    for scale in np.round(np.arange(1.0, 3.01, 0.25), 2):
        cur = rng.normal(0.0, scale, n)
        psi = compute_psi(ref, cur, n_bins)
        rows.append({
            "scenario": "variance_shift", "magnitude": float(scale),
            "psi": round(psi, 4),
            "kl": round(kl_divergence(ref, cur, n_bins), 4),
            "js": round(jensen_shannon(ref, cur, n_bins), 4),
            "wasserstein": round(wasserstein(ref, cur), 4),
            "psi_status": interpret_psi(psi),
        })

    # headline: mean shift (sigma units) at which PSI first crosses each band
    mean_rows = [r for r in rows if r["scenario"] == "mean_shift"]
    def first_cross(level):
        for r in mean_rows:
            if r["psi"] >= level:
                return r["magnitude"]
        return None

    headline = {
        "mean_shift_to_moderate_psi": first_cross(STABLE),
        "mean_shift_to_significant_psi": first_cross(SIGNIFICANT),
    }
    return rows, headline


def plot_distance_sensitivity(rows, path):
    mr = [r for r in rows if r["scenario"] == "mean_shift"]
    x = [r["magnitude"] for r in mr]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(x, [r["psi"] for r in mr], "o-", color="#1f77b4", label="PSI")
    ax1.plot(x, [r["kl"] for r in mr], "s-", color="#2ca02c", label="KL")
    ax1.plot(x, [r["js"] for r in mr], "^-", color="#9467bd", label="JS")
    ax1.axhline(STABLE, ls="--", color="#888", lw=1)
    ax1.axhline(SIGNIFICANT, ls="--", color="#d62728", lw=1)
    ax1.text(0.02, SIGNIFICANT + 0.03, "PSI 0.25 (significant)", color="#d62728", fontsize=8)
    ax1.text(0.02, STABLE + 0.03, "PSI 0.10 (moderate)", color="#666", fontsize=8)
    ax1.set_xlabel("Mean shift (in reference sigma)")
    ax1.set_ylabel("PSI / KL / JS")
    ax2 = ax1.twinx()
    ax2.plot(x, [r["wasserstein"] for r in mr], "d-", color="#ff7f0e", label="Wasserstein")
    ax2.set_ylabel("Wasserstein distance (feature units)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    lines = ax1.get_lines()[:3] + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    ax1.set_title("Drift metrics vs. covariate mean shift (N=5000)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Scenario B: Page-Hinkley concept-drift detection
# --------------------------------------------------------------------------- #
def make_error_stream(rng, n, change_point, base=0.10, shifted=0.30, noise=0.02):
    err = rng.normal(base, noise, n)
    if change_point is not None:
        err[change_point:] = rng.normal(shifted, noise, n - change_point)
    return err


def run_ph(stream, delta, threshold, min_instances):
    ph = PageHinkleyDetector(delta=delta, threshold=threshold, min_instances=min_instances)
    stat = np.empty(len(stream))
    first = None
    for i, v in enumerate(stream):
        r = ph.update(v)
        stat[i] = r.statistic
        if first is None and r.drift_detected:
            first = i
    return first, stat


def scenario_concept_drift(rng, n=4000, change_point=2000, n_seeds=30,
                           delta=0.01, threshold=5.0, min_instances=30):
    delays, detected, pre_change_false = [], 0, 0
    clean_alarm_runs, clean_alarm_total = 0, 0
    example = None

    for s in range(n_seeds):
        srng = np.random.default_rng(1000 + s)
        stream = make_error_stream(srng, n, change_point)
        first, stat = run_ph(stream, delta, threshold, min_instances)
        if first is not None and first >= change_point:
            detected += 1
            delays.append(first - change_point)
        elif first is not None and first < change_point:
            pre_change_false += 1
        if s == 0:
            example = {"stream": stream, "stat": stat, "detected_at": first}

        clean = make_error_stream(srng, n, None)
        cfirst, _ = run_ph(clean, delta, threshold, min_instances)
        if cfirst is not None:
            clean_alarm_runs += 1
            clean_alarm_total += 1

    delays = np.array(delays, dtype=float)
    result = {
        "params": {"n": n, "change_point": change_point, "n_seeds": n_seeds,
                   "delta": delta, "threshold": threshold,
                   "min_instances": min_instances,
                   "base_error": 0.10, "shifted_error": 0.30, "noise_std": 0.02},
        "detection_rate": round(detected / n_seeds, 4),
        "pre_change_false_alarm_runs": pre_change_false,
        "detection_delay_samples": {
            "mean": round(float(delays.mean()), 2) if delays.size else None,
            "median": round(float(np.median(delays)), 2) if delays.size else None,
            "std": round(float(delays.std()), 2) if delays.size else None,
            "min": int(delays.min()) if delays.size else None,
            "max": int(delays.max()) if delays.size else None,
        },
        "clean_stream_false_alarm_rate": round(clean_alarm_runs / n_seeds, 4),
        "clean_stream_false_alarms_total": clean_alarm_total,
    }
    return result, delays, example


def plot_ph_run(example, change_point, path):
    stream, stat = example["stream"], example["stat"]
    detected = example["detected_at"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5), sharex=True)
    ax1.plot(stream, color="#1f77b4", lw=0.6, alpha=0.8)
    ax1.axvline(change_point, color="#d62728", ls="--", lw=1.2, label="true change point")
    if detected is not None:
        ax1.axvline(detected, color="#2ca02c", ls="-", lw=1.2, label=f"detected (t={detected})")
    ax1.set_ylabel("error / loss")
    ax1.set_title("Page-Hinkley on a simulated error stream (0.10 -> 0.30)")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.plot(stat, color="#7f4fbf", lw=0.9, label="PH statistic")
    ax2.axhline(5.0, color="#888", ls="--", lw=1, label="threshold")
    ax2.axvline(change_point, color="#d62728", ls="--", lw=1.2)
    if detected is not None:
        ax2.axvline(detected, color="#2ca02c", ls="-", lw=1.2)
    ax2.set_xlabel("sample index")
    ax2.set_ylabel("PH statistic")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_delay_hist(delays, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    if delays.size:
        ax.hist(delays, bins=12, color="#2ca02c", alpha=0.85, edgecolor="white")
        ax.axvline(np.median(delays), color="#d62728", ls="--",
                   label=f"median = {np.median(delays):.0f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("detection delay (samples after change point)")
    ax.set_ylabel("count (seeds)")
    ax.set_title("Page-Hinkley detection delay distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Drift-detection benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--ph-delta", type=float, default=0.01)
    ap.add_argument("--ph-threshold", type=float, default=5.0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Scenario A: distribution-distance sensitivity ...")
    dist_rows, dist_head = scenario_distance_sensitivity(rng)
    _write_csv(os.path.join(RESULTS, "feature_drift_sensitivity.csv"), dist_rows)
    plot_distance_sensitivity(dist_rows, os.path.join(RESULTS, "psi_vs_shift.png"))

    print("Scenario B: Page-Hinkley concept-drift detection ...")
    cd, delays, example = scenario_concept_drift(
        rng, n_seeds=args.n_seeds, delta=args.ph_delta, threshold=args.ph_threshold)
    with open(os.path.join(RESULTS, "concept_drift_detection.json"), "w") as f:
        json.dump(cd, f, indent=2)
    plot_ph_run(example, cd["params"]["change_point"],
                os.path.join(RESULTS, "page_hinkley_run.png"))
    plot_delay_hist(delays, os.path.join(RESULTS, "detection_delay_hist.png"))

    import numpy as _np
    import pandas as _pd
    import scipy as _sp
    import matplotlib as _mpl
    config = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed, "python": sys.version.split()[0],
        "numpy": _np.__version__, "pandas": _pd.__version__,
        "scipy": _sp.__version__, "matplotlib": _mpl.__version__,
        "ph_delta": args.ph_delta, "ph_threshold": args.ph_threshold,
        "n_seeds": args.n_seeds,
    }
    with open(os.path.join(RESULTS, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    summary = {
        "distribution_distance": dist_head,
        "concept_drift": {
            "detection_rate": cd["detection_rate"],
            "median_detection_delay_samples": cd["detection_delay_samples"]["median"],
            "mean_detection_delay_samples": cd["detection_delay_samples"]["mean"],
            "clean_stream_false_alarm_rate": cd["clean_stream_false_alarm_rate"],
        },
        "config": config,
    }
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== HEADLINE (real, from this run) =====")
    print(json.dumps(summary["distribution_distance"], indent=2))
    print(json.dumps(summary["concept_drift"], indent=2))
    print("artifacts written to results/")


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
