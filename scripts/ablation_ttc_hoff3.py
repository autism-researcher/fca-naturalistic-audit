"""Post-hoc TTC-ablation robustness check for H_OFF3.

Question answered
-----------------
H_OFF3 tests whether the composite risk R(x) discriminates trajectories
that contain low-TTC (< 2 s) events. Because raw TTC is itself feature 6
of R (the largest single weight, 0.25), a reviewer can object that the
test is partly circular. This script removes TTC from R (sets w_ttc = 0
and renormalizes the remaining seven weights to the original sum 0.94),
then re-runs the H_OFF3 discrimination test. If the seven NON-TTC
features still separate crossing from non-crossing trajectories, the
discrimination is not merely an artifact of TTC being inside R.

Design (matched comparison)
---------------------------
  Outcome (DV)      : per-trajectory count of ticks with raw TTC < 2 s
                      (UNCHANGED -- the same outcome as confirmatory H_OFF3).
  Grouping (full)   : R_max(full)   > (1-tau)-quantile of R_max(full)
  Grouping (ablated): R_max(no-TTC) > (1-tau)-quantile of R_max(no-TTC)
  Test              : src.tests.h_off3 (one-sided Mann-Whitney U,
                      rank-biserial r), identical to the confirmatory code.

Both groupings use each metric's own (1-tau)-quantile so the comparison
is apples-to-apples; the full-R column therefore will NOT exactly equal
Table II (which partitions by B_sim) -- it is the matched baseline for
the ablation, not a reproduction of Table II.

STATUS: POST-HOC / EXPLORATORY. Not in the OSF pre-registration. Does
NOT change the confirmatory H_OFF3 verdict. Log one line in
deviations_log.md classified as 'post-hoc addition'.

Run from paper3_pipeline/ root:
    python scripts/ablation_ttc_hoff3.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_weights          # noqa: E402
from src.risk import composite_risk          # noqa: E402
from src.tests import h_off3                  # noqa: E402

TAUS = [0.10, 0.15, 0.20]
TTC_THRESH_S = 2.0
# feature order: speed, accel, jerk, steer_var, lane_offset, ttc, headway, density
TTC_INDEX = 5


def low_ttc_count(traj):
    """Per-trajectory count of ticks with raw TTC < 2 s (finite only)."""
    arr = np.asarray(traj.get("ttc_raw", []), dtype=float)
    arr = arr[np.isfinite(arr)]
    return int(np.sum(arr < TTC_THRESH_S))


def rmax_per_traj(trajs, weights):
    return np.array([
        float(np.max(composite_risk(np.asarray(t["features"], dtype=float), weights)))
        for t in trajs
    ])


def main():
    w_full, _ = load_weights()
    w_full = np.asarray(w_full, dtype=float)
    w_abl = w_full.copy()
    w_abl[TTC_INDEX] = 0.0
    w_abl *= w_full.sum() / w_abl.sum()   # renormalize back to sum(w_full)=0.94

    print(f"Full weights:    {np.round(w_full, 4).tolist()}  sum={w_full.sum():.4f}")
    print(f"Ablated weights: {np.round(w_abl, 4).tolist()}  sum={w_abl.sum():.4f}")
    print("(TTC weight zeroed, remaining seven renormalized to 0.94)")
    print()

    header = (f"{'Dataset':<8}{'tau':>6}{'r_full':>9}{'r_abl':>9}"
              f"{'p_abl':>12}{'n_cross':>9}{'survives':>10}")
    print(header)
    print("-" * len(header))

    out = {}
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        ds = Path(fpath).stem.replace("_features", "")
        with open(fpath) as f:
            trajs = json.load(f)["trajectories"]
        dv = np.array([low_ttc_count(t) for t in trajs])
        rmax_full = rmax_per_traj(trajs, w_full)
        rmax_abl = rmax_per_traj(trajs, w_abl)
        out[ds] = {}
        for tau in TAUS:
            b_full = float(np.quantile(rmax_full, 1 - tau))
            b_abl = float(np.quantile(rmax_abl, 1 - tau))
            c_full = rmax_full > b_full
            c_abl = rmax_abl > b_abl
            res_full = h_off3(dv[c_full].tolist(), dv[~c_full].tolist())
            res_abl = h_off3(dv[c_abl].tolist(), dv[~c_abl].tolist())
            r_full = res_full["effect_size_r"]
            r_abl = res_abl["effect_size_r"]
            survives = bool(
                res_abl["pass"] and r_abl is not None and abs(r_abl) >= 0.10
            )
            print(f"{ds:<8}{tau:>6.2f}{r_full:>9.3f}{r_abl:>9.3f}"
                  f"{res_abl['p']:>12.2e}{int(c_abl.sum()):>9d}"
                  f"{('YES' if survives else 'no'):>10}")
            out[ds][f"{tau:.2f}"] = {
                "r_full_matched": r_full,
                "r_ablated": r_abl,
                "p_ablated": res_abl["p"],
                "pass_ablated": res_abl["pass"],
                "n_crossing": int(c_abl.sum()),
                "survives": survives,
            }
    print()
    print("survives = ablated test passes Bonferroni AND |r_ablated| >= 0.10.")
    print("r_full here partitions by the full-R (1-tau)-quantile (matched"
          " baseline), not by B_sim, so it need not equal Table II.")
    print()
    print("Compact JSON (paste back to the manuscript conversation):")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
