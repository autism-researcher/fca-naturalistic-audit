r"""End-to-end HighD experiment runner (Paper 3).

This single script orchestrates the full HighD-only pipeline that the numbered
scripts (03 -> 04 -> 05) would otherwise run separately. It mirrors
`run_waymo_experiment.py` exactly: same boundary computation, same hypothesis
tests, same output format. Nothing here re-implements logic - it composes the
existing modules.

Unlike Waymo, HighD has no TensorFlow / protobuf dependency, so it runs
directly in the slim project env (`requirements.txt`). No conda env, no WSL.

==============================================================================
STEP-BY-STEP USAGE
==============================================================================

PREREQUISITES (one-time):

  1. The HighD dataset (v1.0) should already be unpacked. The script expects
     a directory containing files named like:

         01_tracks.csv          01_tracksMeta.csv          01_recordingMeta.csv
         02_tracks.csv          02_tracksMeta.csv          02_recordingMeta.csv
         ...                    (up to 60 recordings)

     Default search path: paper3_pipeline/data/highd/
     Override with --highd-dir if your data lives elsewhere.

  2. Install pipeline dependencies (once, in any Python env):

         pip install -r requirements.txt

  3. Lock the Git tag `pipeline-frozen-pre-confirmatory` per pre-registration
     (the script refuses to run otherwise). Override only with
     --skip-gate-check and document the override in deviations_log.md.

RUN (from paper3_pipeline/ root, on Windows in cmd / PowerShell, or any shell):

  # Small pilot first to confirm everything works:
  python scripts\run_highd_experiment.py ^
      --highd-dir "D:\New Paper3\paper3_pipeline\data\highd\highd-dataset-v1.0\data" ^
      --n 100

  # Full pre-registered run:
  python scripts\run_highd_experiment.py ^
      --highd-dir "D:\New Paper3\paper3_pipeline\data\highd\highd-dataset-v1.0\data" ^
      --n 5000

  (On macOS / Linux replace ^ line continuations with \ and use forward slashes.)

WHAT GETS WRITTEN:

  results/per_dataset/highd_features.json   (raw features + TTC per trajectory)
  results/boundaries/highd.json             (B_d, tau_hat_d, DKW eps, bootstrap CIs)
  results/verdicts/highd.json               (H_OFF1, H_OFF2, H_OFF3 per tau)
  results/diagnostics/highd_ttc_check.json  (sanity check: our TTC vs HighD-provided)

  Plus a one-page PASS/FAIL summary printed to stdout at the end.

==============================================================================
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import SEED, load_weights, load_B_sim                    # noqa: E402
from src.risk import composite_risk                                     # noqa: E402
from src.boundary import (                                              # noqa: E402
    boundary as B_quantile,
    realized_rate,
    dkw_epsilon,
    bootstrap_ci,
)
from src.tests import h_off1, h_off2, h_off3                            # noqa: E402
from src.features.highd import (                                        # noqa: E402
    list_recordings,
    load_highd_recording,
    build_frame_index,
    trajectories as highd_trajectories,
    extract_features as highd_extract,
    compare_ttc_with_provided,
)

TAUS = [0.10, 0.15, 0.20]
TTC_THRESH_S = 2.0
N_COMPARISONS = 9  # 3 datasets x 3 hypotheses, pre-registered Bonferroni factor
REQUIRED_GIT_TAG = "pipeline-frozen-pre-confirmatory"
DEFAULT_HIGHD_DIR = ROOT / "data" / "highd"


# ----------------------------------------------------------------------
# Stage 0: pre-flight checks
# ----------------------------------------------------------------------
def check_gate(skip: bool) -> None:
    """Refuse to run unless the pre-confirmatory Git tag exists."""
    if skip:
        print("[gate] WARNING: --skip-gate-check passed. Document this in deviations_log.md.")
        return
    try:
        out = subprocess.check_output(
            ["git", "tag", "--list", REQUIRED_GIT_TAG],
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.exit(
            f"[gate] Could not query git tags ({exc}). Run this from inside a "
            f"git working tree, or pass --skip-gate-check if appropriate."
        )
    if out != REQUIRED_GIT_TAG:
        sys.exit(
            f"[gate] Required tag '{REQUIRED_GIT_TAG}' not found. Per "
            f"pre-registration, you may not compute boundaries on naturalistic "
            f"data before Gate 2 is locked. Tag the frozen pipeline first:\n"
            f"    git tag {REQUIRED_GIT_TAG}"
        )
    print(f"[gate] OK ({REQUIRED_GIT_TAG} present)")


def check_data(highd_dir: Path) -> list[int]:
    """Locate HighD recordings; fail with a clear message if none found."""
    if not highd_dir.exists():
        sys.exit(
            f"[data] Directory does not exist: {highd_dir}\n"
            f"Pass --highd-dir to point at the folder containing the\n"
            f"NN_tracks.csv / NN_tracksMeta.csv / NN_recordingMeta.csv files."
        )
    rids = list_recordings(highd_dir)
    if not rids:
        sys.exit(
            f"[data] No complete HighD recordings under {highd_dir}.\n"
            f"Each recording needs all three of:\n"
            f"    NN_tracks.csv, NN_tracksMeta.csv, NN_recordingMeta.csv"
        )
    print(f"[data] Found {len(rids)} HighD recording(s): "
          f"{rids[0]:02d}..{rids[-1]:02d}")
    return rids


# ----------------------------------------------------------------------
# Stage 1: feature extraction (mirrors scripts/03_extract_features_at_scale.py)
# ----------------------------------------------------------------------
def extract_features(
    highd_dir: Path, recording_ids: list[int], n_max: int, output_path: Path,
    diagnostics_path: Path,
) -> dict:
    """Walk recordings in SEED-shuffled order, extract up to n_max eligible vehicles.

    Also writes a TTC-vs-HighD-provided sanity-check report.
    """
    rng = np.random.default_rng(SEED)
    rid_order = list(recording_ids)
    rng.shuffle(rid_order)

    out: list[dict] = []
    n_excluded = {
        "too_short": 0,
        "non_finite_position": 0,
        "accel_above_10": 0,
        "other": 0,
    }
    ttc_diffs: list[dict] = []
    t_start = time.time()

    for ri, rid in enumerate(rid_order):
        if len(out) >= n_max:
            break
        print(f"  [extract] recording {ri + 1}/{len(rid_order)}: "
              f"{rid:02d} (running tally: {len(out)})")
        try:
            tracks, _meta, rmeta = load_highd_recording(highd_dir, rid)
        except Exception as exc:
            print(f"  [extract] load error on recording {rid:02d}: {exc}; skipping")
            continue
        frame_index = build_frame_index(tracks)

        # SEED-deterministic intra-recording vehicle ordering
        vids = [vid for vid, _ in highd_trajectories(tracks)]
        idx = np.arange(len(vids))
        rng.shuffle(idx)

        for j in idx:
            if len(out) >= n_max:
                break
            vid = vids[j]
            g = tracks[tracks["id"] == vid].sort_values("frame").reset_index(drop=True)
            feats, ttc, ok, reason = highd_extract(
                g, recording_meta=rmeta, frame_index=frame_index,
            )
            if not ok or feats is None:
                key = reason if reason in n_excluded else "other"
                n_excluded[key] = n_excluded.get(key, 0) + 1
                continue
            out.append({
                "trajectory_id": f"{int(rid):02d}__{int(vid)}",
                "T": int(feats.shape[0]),
                "features": feats.tolist(),
                "ttc_raw": ttc.tolist() if ttc is not None else None,
            })

            # TTC sanity-check sampled lightly to keep the diagnostics file small
            if ttc is not None and len(ttc_diffs) < 200:
                cmp = compare_ttc_with_provided(g, ttc)
                if cmp is not None:
                    cmp["trajectory_id"] = f"{int(rid):02d}__{int(vid)}"
                    ttc_diffs.append(cmp)

    elapsed = time.time() - t_start
    print(f"[extract] eligible: {len(out)}; excluded: {n_excluded}; "
          f"elapsed: {elapsed:.0f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "highd",
        "seed": SEED,
        "n": len(out),
        "excluded": n_excluded,
        "trajectories": out,
    }
    with open(output_path, "w") as fh:
        json.dump(payload, fh)
    print(f"[extract] wrote {output_path}")

    # ---- TTC-vs-provided diagnostic summary ----
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    if ttc_diffs:
        med = float(np.median([d["median_abs_diff_s"]
                               for d in ttc_diffs if "median_abs_diff_s" in d]))
        mean = float(np.mean([d["mean_abs_diff_s"]
                              for d in ttc_diffs if "mean_abs_diff_s" in d]))
        diag = {
            "n_trajectories_checked": len(ttc_diffs),
            "median_of_median_abs_diff_s": med,
            "mean_of_mean_abs_diff_s": mean,
            "per_trajectory_sample": ttc_diffs[:50],
        }
    else:
        diag = {"n_trajectories_checked": 0, "note": "no overlap with HighD ttc column"}
    with open(diagnostics_path, "w") as fh:
        json.dump(diag, fh, indent=2)
    print(f"[extract] wrote {diagnostics_path}")

    return payload


# ----------------------------------------------------------------------
# Stage 2: boundaries + realized rate (mirrors scripts/04_compute_boundaries.py)
# ----------------------------------------------------------------------
def compute_boundaries(features: dict, output_path: Path) -> dict:
    weights, _ = load_weights()
    B_sim_map = load_B_sim()
    n = features["n"]

    r_max = np.array([
        float(np.max(composite_risk(np.array(t["features"], dtype=float), weights)))
        for t in features["trajectories"]
    ])

    result: dict = {
        "dataset": "highd",
        "N": n,
        "DKW_epsilon": dkw_epsilon(n, delta=0.05),
        "underpowered": n < 738,
        "B_sim": {f"{k:.2f}": v for k, v in B_sim_map.items()},
        "B_d": {},
        "tau_hat_d": {},
        "ci": {},
    }
    for tau in TAUS:
        B_d = B_quantile(r_max, tau)
        B_sim_tau = B_sim_map[tau]
        result["B_d"][f"{tau:.2f}"] = B_d
        result["tau_hat_d"][f"{tau:.2f}"] = realized_rate(r_max, B_sim_tau)
        B_lo, B_hi = bootstrap_ci(r_max, lambda x: float(np.quantile(x, 1 - tau)))
        rate_lo, rate_hi = bootstrap_ci(r_max, lambda x: float(np.mean(x > B_sim_tau)))
        result["ci"][f"{tau:.2f}"] = {
            "B_d": [B_lo, B_hi],
            "tau_hat_d": [rate_lo, rate_hi],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[boundaries] N={n}  DKW eps={result['DKW_epsilon']:.4f}  -> {output_path}")
    for tau in TAUS:
        key = f"{tau:.2f}"
        print(f"  tau={tau:.2f}  B_sim={B_sim_map[tau]:.4f}  "
              f"B_d={result['B_d'][key]:.4f}  "
              f"tau_hat={result['tau_hat_d'][key]:.4f}")
    return result


# ----------------------------------------------------------------------
# Stage 3: pre-registered tests (mirrors scripts/05_run_tests.py)
# ----------------------------------------------------------------------
def _ttc_counts_partitioned(trajs: list[dict], weights, B_sim_tau: float):
    crossing, noncrossing = [], []
    for t in trajs:
        feats = np.array(t["features"], dtype=float)
        ttc = np.array(t["ttc_raw"], dtype=float)
        Rm = float(np.max(composite_risk(feats, weights)))
        cnt = int(np.sum(ttc < TTC_THRESH_S))
        (crossing if Rm > B_sim_tau else noncrossing).append(cnt)
    return crossing, noncrossing


def run_tests(features: dict, boundaries: dict, output_path: Path) -> dict:
    weights, _ = load_weights()
    bsim_norm = {f"{float(k):.2f}": v for k, v in boundaries["B_sim"].items()}

    verdicts: dict = {"dataset": "highd", "N": features["n"], "tests": {}}
    for tau in TAUS:
        key = f"{tau:.2f}"
        B_sim = bsim_norm[key]
        B_d = boundaries["B_d"][key]
        tau_hat = boundaries["tau_hat_d"][key]

        v1 = h_off1(B_sim, B_d)
        v2 = h_off2(tau_hat, tau)
        crossing, noncrossing = _ttc_counts_partitioned(
            features["trajectories"], weights, B_sim
        )
        v3 = h_off3(crossing, noncrossing, n_comparisons=N_COMPARISONS)
        v3["n_crossing"] = len(crossing)
        v3["n_noncrossing"] = len(noncrossing)

        verdicts["tests"][key] = {"H_OFF1": v1, "H_OFF2": v2, "H_OFF3": v3}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(verdicts, fh, indent=2)
    print(f"[tests] wrote {output_path}")
    return verdicts


def print_summary(verdicts: dict) -> None:
    print("\n" + "=" * 70)
    print(f"HIGHD RESULTS SUMMARY (N = {verdicts['N']})")
    print("=" * 70)
    print(f"{'tau':>6}  {'H_OFF1':>10}  {'H_OFF2':>10}  {'H_OFF3':>10}")
    print("-" * 70)
    for tau in TAUS:
        v = verdicts["tests"][f"{tau:.2f}"]
        h1 = "PASS" if v["H_OFF1"]["pass"] else "FAIL"
        h2 = "PASS" if v["H_OFF2"]["pass"] else "FAIL"
        h3 = "PASS" if v["H_OFF3"].get("pass") is True else "FAIL"
        print(f"{tau:>6.2f}  {h1:>10}  {h2:>10}  {h3:>10}")
    print("=" * 70)
    print("Detailed JSON: results/verdicts/highd.json")
    print("Disjunctive criterion (handled in script 09): an overall PASS for "
          "H_OFF1/H_OFF2 requires that at least ONE of {NGSIM, HighD, Waymo} "
          "passes for all three tau values.\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full HighD validation pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for prerequisites and step-by-step setup.",
    )
    parser.add_argument("--n", type=int, default=5000,
                        help="Max trajectories to extract (default: 5000, pre-registered).")
    parser.add_argument("--highd-dir", type=Path, default=DEFAULT_HIGHD_DIR,
                        help=f"Directory containing NN_tracks.csv / NN_tracksMeta.csv / "
                             f"NN_recordingMeta.csv files. Default: {DEFAULT_HIGHD_DIR}")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip stage 1 if highd_features.json already exists.")
    parser.add_argument("--skip-gate-check", action="store_true",
                        help="Bypass the Gate 2 git-tag check (document in deviations_log.md).")
    args = parser.parse_args()

    # --- Pre-flight ---
    check_gate(args.skip_gate_check)
    rids = check_data(args.highd_dir)

    features_path = ROOT / "results" / "per_dataset" / "highd_features.json"
    boundaries_path = ROOT / "results" / "boundaries" / "highd.json"
    verdicts_path = ROOT / "results" / "verdicts" / "highd.json"
    diagnostics_path = ROOT / "results" / "diagnostics" / "highd_ttc_check.json"

    # --- Stage 1: extract ---
    if args.skip_extract and features_path.exists():
        print(f"[extract] --skip-extract; reusing {features_path}")
        with open(features_path) as fh:
            features = json.load(fh)
    else:
        features = extract_features(
            args.highd_dir, rids, args.n, features_path, diagnostics_path,
        )

    if features["n"] < 738:
        print(f"[warn] N = {features['n']} < 738 (DKW underpowered). Results still "
              f"valid but report epsilon explicitly per pre-reg.")

    # --- Stage 2: boundaries ---
    boundaries = compute_boundaries(features, boundaries_path)

    # --- Stage 3: tests ---
    verdicts = run_tests(features, boundaries, verdicts_path)

    # --- Summary ---
    print_summary(verdicts)


if __name__ == "__main__":
    main()
