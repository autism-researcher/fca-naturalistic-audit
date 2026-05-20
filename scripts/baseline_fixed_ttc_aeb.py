"""Direct baseline: fixed-TTC AEB engagement rate per dataset.

Computes the fraction of trajectories with at least one tick of
raw TTC < threshold_s on each of NGSIM, HighD, Waymo. The result
is the natural fixed-TTC AEB engagement rate baseline, directly
comparable to the FCA supervisor's realized rate tau_hat at B_sim.

Run from paper3_pipeline/ root:
    python scripts/baseline_fixed_ttc_aeb.py

Outputs a small table to stdout. Paste it back to fill the
placeholder values in the manuscript's §V-B baseline subsection.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_S = [1.5, 2.0, 3.0]  # commonly-used AEB cutoffs


def trajectory_min_ttc(t):
    """Minimum raw TTC across the trajectory's ticks (or +inf if none)."""
    ttc_raw = t.get("ttc_raw")
    if ttc_raw is None:
        return float("inf")
    arr = np.asarray(ttc_raw, dtype=float)
    # NaN/inf compare False under <, so they don't contribute to engagement
    valid = arr[np.isfinite(arr)]
    return float(valid.min()) if valid.size else float("inf")


def aeb_engagement_rate(feature_json_path, threshold_s):
    """Fraction of trajectories with min(TTC) < threshold_s."""
    with open(feature_json_path) as f:
        d = json.load(f)
    trajs = d["trajectories"]
    n = len(trajs)
    if n == 0:
        return 0.0, 0, 0
    n_engaged = sum(
        1 for t in trajs if trajectory_min_ttc(t) < threshold_s
    )
    return n_engaged / n, n_engaged, n


def main():
    results = {}
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        ds = Path(fpath).stem.replace("_features", "")
        results[ds] = {}
        for thr in THRESHOLDS_S:
            rate, n_eng, n_tot = aeb_engagement_rate(fpath, thr)
            results[ds][thr] = (rate, n_eng, n_tot)

    # Print a compact LaTeX-ready table
    print()
    print("Fixed-TTC AEB engagement rate per dataset")
    print("=" * 60)
    print(f"{'Dataset':<8} {'TTC<1.5s':>12} {'TTC<2.0s':>12} {'TTC<3.0s':>12}")
    print("-" * 60)
    for ds in sorted(results):
        row = results[ds]
        cells = []
        for thr in THRESHOLDS_S:
            rate, n_eng, n_tot = row[thr]
            cells.append(f"{rate:>6.3f} ({n_eng}/{n_tot})")
        print(f"{ds:<8} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")
    print("=" * 60)
    print()
    print("Paste the TTC<2.0s column back to fill the manuscript placeholder.")
    print()
    print("Compact JSON output for record:")
    print(json.dumps({ds: {str(thr): r[0] for thr, r in row.items()}
                      for ds, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
