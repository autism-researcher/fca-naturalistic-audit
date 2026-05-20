r"""H_OFF3 sensitivity check: HighD-provided TTC vs our recomputed TTC.

Motivation
----------
The TTC sanity-check diagnostic (results/diagnostics/highd_ttc_check.json)
shows that our recomputed TTC agrees with HighD's provided `ttc` column for
most trajectories (median-of-median |diff| ~= 0.18 s) but disagrees by
several seconds on a long tail of trajectories. Per pre-reg deviations
policy this is a "within-protocol refinement" -- we re-run H_OFF3 using
HighD's provided TTC column to confirm the verdict is robust to the TTC
definition.

What this script does
---------------------
1. Loads results/per_dataset/highd_features.json (existing extraction).
2. Loads results/boundaries/highd.json (existing B_sim, kept unchanged).
3. For each trajectory in the JSON, re-fetches the PROVIDED `ttc` column
   from the corresponding NN_tracks.csv file.
4. Re-runs ONLY H_OFF3, keeping the composite-risk crossing partition
   (based on B_sim and our composite-risk function) exactly as in
   results/verdicts/highd.json. Only the TTC<2s tick counts change.
5. Writes results/verdicts/highd_sensitivity_provided_ttc.json. The
   original highd.json is NOT overwritten.

What this script does NOT do
----------------------------
- It does not re-extract features.
- It does not recompute boundaries (B_d, tau_hat_d) -- those are unchanged.
- It does not change the H_OFF1 or H_OFF2 verdicts.
- It does not modify highd_features.json or highd.json.

Reading the output
------------------
If H_OFF3 still PASSES for all three tau values, the original H_OFF3
PASS is robust to the TTC measurement choice; report this as a
sensitivity check in the paper.

If H_OFF3 FAILS for any tau under the provided-TTC variant, the original
result depends on our TTC recomputation -- you'll need to discuss this
in the paper and possibly investigate the closing-speed convention.

Usage
-----
    python scripts\run_highd_sensitivity_ttc.py ^
        --highd-dir "D:\New Paper3\paper3_pipeline\data\highd\highd-dataset-v1.0\data"

(The --highd-dir flag MUST point to the same data directory used by
run_highd_experiment.py, otherwise the trajectory_id -> CSV lookup fails.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_weights        # noqa: E402
from src.risk import composite_risk       # noqa: E402
from src.tests import h_off3              # noqa: E402

TAUS = [0.10, 0.15, 0.20]
TTC_THRESH_S = 2.0
N_COMPARISONS = 9  # pre-registered Bonferroni factor

FEATURES_PATH = ROOT / "results" / "per_dataset" / "highd_features.json"
BOUNDARIES_PATH = ROOT / "results" / "boundaries" / "highd.json"
OUTPUT_PATH = ROOT / "results" / "verdicts" / "highd_sensitivity_provided_ttc.json"
ORIGINAL_VERDICTS_PATH = ROOT / "results" / "verdicts" / "highd.json"


# ----------------------------------------------------------------------
# Loading the provided TTC column
# ----------------------------------------------------------------------
def load_provided_ttc_by_recording(highd_dir: Path, recording_ids: list[int]) -> dict:
    """Pre-load each recording's tracks.csv ONCE; build {rid: {vid: ttc_array}}.

    Loading each NN_tracks.csv on every trajectory lookup would be O(N*60);
    this is O(60) plus one groupby per recording. HighD's `ttc` column
    convention: 0 (or sometimes -1) means "not applicable" (no leader, not
    closing). We convert these to +inf so they cannot count as below the
    2-second threshold.
    """
    index: dict[int, dict[int, np.ndarray]] = {}
    for rid in sorted(set(recording_ids)):
        csv_path = highd_dir / f"{rid:02d}_tracks.csv"
        if not csv_path.exists():
            print(f"  [warn] missing {csv_path.name}; skipping (its trajectories will be dropped)")
            index[rid] = {}
            continue
        print(f"  [load] {csv_path.name}")
        df = pd.read_csv(csv_path, usecols=["frame", "id", "ttc"])
        per_vid: dict[int, np.ndarray] = {}
        for vid, g in df.groupby("id"):
            g = g.sort_values("frame")
            ttc = g["ttc"].to_numpy(dtype=float)
            ttc = np.where((ttc > 0) & np.isfinite(ttc), ttc, np.inf)
            per_vid[int(vid)] = ttc
        index[rid] = per_vid
    return index


# ----------------------------------------------------------------------
# H_OFF3 re-test
# ----------------------------------------------------------------------
def _partition_with_provided_ttc(
    trajs: list[dict], weights, B_sim_tau: float, ttc_index: dict
) -> tuple[list[int], list[int], dict]:
    """Same crossing-vs-non-crossing partition as the original H_OFF3, but
    counts TTC<2s ticks using HighD's PROVIDED ttc column.

    Returns (crossing_counts, noncrossing_counts, audit_dict).
    """
    crossing: list[int] = []
    noncrossing: list[int] = []
    audit = {"n_trajectories": 0, "n_with_provided_ttc": 0, "n_skipped_no_lookup": 0}

    for t in trajs:
        audit["n_trajectories"] += 1
        feats = np.array(t["features"], dtype=float)
        Rm = float(np.max(composite_risk(feats, weights)))

        # Look up the provided ttc by (recording_id, vehicle_id)
        try:
            rid_str, vid_str = t["trajectory_id"].split("__")
            rid, vid = int(rid_str), int(vid_str)
        except (ValueError, KeyError):
            audit["n_skipped_no_lookup"] += 1
            continue
        provided_ttc = ttc_index.get(rid, {}).get(vid)
        if provided_ttc is None:
            audit["n_skipped_no_lookup"] += 1
            continue

        audit["n_with_provided_ttc"] += 1
        cnt = int(np.sum(provided_ttc < TTC_THRESH_S))
        (crossing if Rm > B_sim_tau else noncrossing).append(cnt)

    return crossing, noncrossing, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H_OFF3 sensitivity check using HighD's provided ttc column.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for what this does and does not change.",
    )
    parser.add_argument(
        "--highd-dir", type=Path, required=True,
        help="Directory containing NN_tracks.csv files (must match the directory "
             "used by run_highd_experiment.py).",
    )
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        sys.exit(f"[error] Missing {FEATURES_PATH}. Run run_highd_experiment.py first.")
    if not BOUNDARIES_PATH.exists():
        sys.exit(f"[error] Missing {BOUNDARIES_PATH}. Run run_highd_experiment.py first.")
    if not args.highd_dir.exists():
        sys.exit(f"[error] HighD data directory not found: {args.highd_dir}")

    print(f"[load] features:   {FEATURES_PATH}")
    with open(FEATURES_PATH) as fh:
        features = json.load(fh)
    print(f"[load] boundaries: {BOUNDARIES_PATH}")
    with open(BOUNDARIES_PATH) as fh:
        boundaries = json.load(fh)
    if ORIGINAL_VERDICTS_PATH.exists():
        with open(ORIGINAL_VERDICTS_PATH) as fh:
            original_verdicts = json.load(fh)
    else:
        original_verdicts = None

    n_traj = len(features["trajectories"])
    print(f"[info] {n_traj} trajectories in features file")

    # Group trajectory ids by recording so we load each CSV exactly once.
    rids_needed: set[int] = set()
    for t in features["trajectories"]:
        try:
            rid = int(t["trajectory_id"].split("__")[0])
            rids_needed.add(rid)
        except (ValueError, KeyError):
            continue
    print(f"[info] need to load {len(rids_needed)} HighD recording(s)")

    t0 = time.time()
    ttc_index = load_provided_ttc_by_recording(args.highd_dir, sorted(rids_needed))
    print(f"[info] loaded provided-TTC index in {time.time() - t0:.1f}s")

    # ---- Re-run H_OFF3 per tau with provided TTC ----
    weights, _ = load_weights()
    bsim_norm = {f"{float(k):.2f}": v for k, v in boundaries["B_sim"].items()}

    sensitivity = {
        "dataset": "highd",
        "variant": "provided_ttc_for_H_OFF3",
        "N": features["n"],
        "ttc_threshold_s": TTC_THRESH_S,
        "n_comparisons_bonferroni": N_COMPARISONS,
        "tests": {},
    }

    print("\n=== Sensitivity H_OFF3 (HighD-provided ttc column) ===")
    for tau in TAUS:
        key = f"{tau:.2f}"
        B_sim = bsim_norm[key]
        crossing, noncrossing, audit = _partition_with_provided_ttc(
            features["trajectories"], weights, B_sim, ttc_index,
        )
        v3 = h_off3(crossing, noncrossing, n_comparisons=N_COMPARISONS)
        v3["n_crossing"] = len(crossing)
        v3["n_noncrossing"] = len(noncrossing)
        v3["audit"] = audit
        sensitivity["tests"][key] = {"H_OFF3_provided_ttc": v3}

        verdict = "PASS" if v3.get("pass") is True else "FAIL"
        print(f"  tau={tau:.2f}  B_sim={B_sim:.4f}  "
              f"n_cross={len(crossing)}  n_noncross={len(noncrossing)}  "
              f"p={v3.get('p')}  alpha_bonf={v3.get('alpha_bonf')}  -> {verdict}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(sensitivity, fh, indent=2)
    print(f"\n[write] {OUTPUT_PATH}")

    # ---- Compare to the original H_OFF3 verdict if available ----
    if original_verdicts is not None:
        print("\n=== Comparison: original vs sensitivity ===")
        print(f"{'tau':>6}  {'orig H_OFF3':>14}  {'sens H_OFF3':>14}  agree?")
        all_agree = True
        for tau in TAUS:
            key = f"{tau:.2f}"
            orig = original_verdicts["tests"][key]["H_OFF3"].get("pass")
            sens = sensitivity["tests"][key]["H_OFF3_provided_ttc"].get("pass")
            orig_label = "PASS" if orig is True else "FAIL"
            sens_label = "PASS" if sens is True else "FAIL"
            agree = "yes" if orig == sens else "NO"
            if orig != sens:
                all_agree = False
            print(f"{tau:>6.2f}  {orig_label:>14}  {sens_label:>14}  {agree}")
        print()
        if all_agree:
            print("[robustness] OK -- H_OFF3 verdict is robust to TTC definition. "
                  "Report this sensitivity check in the paper's robustness section.")
        else:
            print("[robustness] WARNING -- H_OFF3 verdict CHANGES when using HighD's "
                  "provided ttc column. Investigate before finalizing the paper.")


if __name__ == "__main__":
    main()
