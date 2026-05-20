"""End-to-end Waymo Open Motion experiment runner (Paper 3).

This single script orchestrates the full Waymo-only pipeline that the existing
numbered scripts (03 -> 04 -> 05) would otherwise run separately. It is built
to mirror the NGSIM and HighD flows exactly: same feature extractor signature,
same boundary computation, same hypothesis tests, same output format. Nothing
here re-implements logic - it composes the existing modules.

==============================================================================
STEP-BY-STEP USAGE
==============================================================================

PREREQUISITES (one-time):

  1. Register at https://waymo.com/open/ and accept the dataset license.
  2. Download Waymo Open Motion Dataset v1.2 TFRecord shards (any subset).
     The script walks --waymo-dir recursively for *.tfrecord* files.
  3. Create the Waymo conda env (TensorFlow is heavy; keep it isolated):
         conda create -n waymo python=3.10 -y
         conda activate waymo
         pip install waymo-open-dataset-tf-2-12-0 numpy pandas scipy
  4. Confirm the pre-registration gate: this script will REFUSE to run unless
     the Git tag `pipeline-frozen-pre-confirmatory` exists. Override only with
     --skip-gate-check and document any override in deviations_log.md.

RUN (from paper3_pipeline/ root, inside the waymo conda env):

  # Small pilot first to confirm everything works:
  python scripts/run_waymo_experiment.py --n 100

  # Full pre-registered run:
  python scripts/run_waymo_experiment.py --n 5000

  # Custom data path (if shards live outside ./data/waymo/):
  python scripts/run_waymo_experiment.py --waymo-dir /path/to/shards --n 5000

WHAT GETS WRITTEN:

  results/per_dataset/waymo_features.json   (raw features + TTC per trajectory)
  results/boundaries/waymo.json             (B_d, tau_hat_d, DKW eps, CIs)
  results/verdicts/waymo.json               (H_OFF1, H_OFF2, H_OFF3 per tau)

  Plus a one-page summary printed to stdout at the end.
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

from src.utils import SEED, load_weights, load_B_sim          # noqa: E402
from src.risk import composite_risk                            # noqa: E402
from src.boundary import (                                     # noqa: E402
    boundary as B_quantile,
    realized_rate,
    dkw_epsilon,
    bootstrap_ci,
)
from src.tests import h_off1, h_off2, h_off3                   # noqa: E402

TAUS = [0.10, 0.15, 0.20]
TTC_THRESH_S = 2.0
N_COMPARISONS = 9
REQUIRED_GIT_TAG = "pipeline-frozen-pre-confirmatory"


# ----------------------------------------------------------------------
# Stage 0: pre-flight checks
# ----------------------------------------------------------------------
def check_gate(skip: bool) -> None:
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


def check_waymo_env() -> None:
    try:
        import tensorflow  # noqa: F401
        from waymo_open_dataset.protos import scenario_pb2  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"[env] Cannot import Waymo SDK ({exc}).\n"
            f"Activate the dedicated env and try again:\n"
            f"    conda activate waymo\n"
            f"    pip install waymo-open-dataset-tf-2-12-0"
        )
    print("[env] Waymo SDK OK")


def check_data(waymo_dir: Path) -> list[Path]:
    if not waymo_dir.exists():
        sys.exit(
            f"[data] Directory does not exist: {waymo_dir}\n"
            f"Pass --waymo-dir to point at the folder containing the TFRecord shards."
        )
    shards = sorted(waymo_dir.rglob("*.tfrecord*"))
    if not shards:
        sys.exit(
            f"[data] No TFRecord shards found under {waymo_dir}.\n"
            f"Download from gs://waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/"
        )
    print(f"[data] Found {len(shards)} TFRecord shard(s) under {waymo_dir}")
    return shards


# ----------------------------------------------------------------------
# Stage 1: feature extraction
# ----------------------------------------------------------------------
def extract_features(shards: list[Path], n_max: int, output_path: Path) -> dict:
    from src.features.waymo import scenarios_from_shard, extract_features as waymo_extract

    rng = np.random.default_rng(SEED)
    order = list(range(len(shards)))
    rng.shuffle(order)

    out: list[dict] = []
    n_excluded = {"no_ego": 0, "too_short": 0, "valid_too_short": 0,
                  "accel_above_10": 0, "other": 0}
    t_start = time.time()

    for shard_idx in order:
        if len(out) >= n_max:
            break
        shard = shards[shard_idx]
        print(f"  [extract] shard {shard_idx + 1}/{len(shards)}: {shard.name} "
              f"(running tally: {len(out)})")
        try:
            for scenario in scenarios_from_shard(shard):
                if len(out) >= n_max:
                    break
                feats, ttc, ok, reason = waymo_extract(scenario)
                if not ok or feats is None:
                    key = reason if reason in n_excluded else "other"
                    n_excluded[key] = n_excluded.get(key, 0) + 1
                    continue
                out.append({
                    "trajectory_id": scenario.scenario_id,
                    "T": int(feats.shape[0]),
                    "features": feats.tolist(),
                    "ttc_raw": ttc.tolist() if ttc is not None else None,
                })
        except Exception as exc:
            print(f"  [extract] shard parse error on {shard.name}: {exc}; skipping")
            continue

    elapsed = time.time() - t_start
    print(f"[extract] eligible: {len(out)}; excluded: {n_excluded}; elapsed: {elapsed:.0f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "waymo", "seed": SEED, "n": len(out),
               "excluded": n_excluded, "trajectories": out}
    with open(output_path, "w") as fh:
        json.dump(payload, fh)
    print(f"[extract] wrote {output_path}")
    return payload


# ----------------------------------------------------------------------
# Stage 2: boundaries + realized rate
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
        "dataset": "waymo", "N": n,
        "DKW_epsilon": dkw_epsilon(n, delta=0.05),
        "underpowered": n < 738,
        "B_sim": {f"{k:.2f}": v for k, v in B_sim_map.items()},
        "B_d": {}, "tau_hat_d": {}, "ci": {},
    }
    for tau in TAUS:
        B_d = B_quantile(r_max, tau)
        B_sim_tau = B_sim_map[tau]
        result["B_d"][f"{tau:.2f}"] = B_d
        result["tau_hat_d"][f"{tau:.2f}"] = realized_rate(r_max, B_sim_tau)
        B_lo, B_hi = bootstrap_ci(r_max, lambda x: float(np.quantile(x, 1 - tau)))
        rate_lo, rate_hi = bootstrap_ci(r_max, lambda x: float(np.mean(x > B_sim_tau)))
        result["ci"][f"{tau:.2f}"] = {"B_d": [B_lo, B_hi], "tau_hat_d": [rate_lo, rate_hi]}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[boundaries] N={n}  DKW eps={result['DKW_epsilon']:.4f}  -> {output_path}")
    for tau in TAUS:
        key = f"{tau:.2f}"
        print(f"  tau={tau:.2f}  B_sim={B_sim_map[tau]:.4f}  "
              f"B_d={result['B_d'][key]:.4f}  tau_hat={result['tau_hat_d'][key]:.4f}")
    return result


# ----------------------------------------------------------------------
# Stage 3: pre-registered tests
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

    verdicts: dict = {"dataset": "waymo", "N": features["n"], "tests": {}}
    for tau in TAUS:
        key = f"{tau:.2f}"
        B_sim = bsim_norm[key]
        B_d = boundaries["B_d"][key]
        tau_hat = boundaries["tau_hat_d"][key]

        v1 = h_off1(B_sim, B_d)
        v2 = h_off2(tau_hat, tau)
        crossing, noncrossing = _ttc_counts_partitioned(
            features["trajectories"], weights, B_sim)
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
    print(f"WAYMO RESULTS SUMMARY (N = {verdicts['N']})")
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
    print("Detailed JSON: results/verdicts/waymo.json")
    print("Disjunctive criterion (handled in script 09): an overall PASS for "
          "H_OFF1/H_OFF2 requires that at least ONE of {NGSIM, HighD, Waymo} "
          "passes for all three tau values.\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Waymo Open Motion validation pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for prerequisites and setup.",
    )
    parser.add_argument("--n", type=int, default=5000,
                        help="Max trajectories to extract (default: 5000, pre-registered).")
    parser.add_argument("--waymo-dir", type=Path,
                        default=ROOT / "data" / "waymo",
                        help="Directory containing Waymo TFRecord shards (walks recursively).")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip stage 1 if waymo_features.json already exists.")
    parser.add_argument("--skip-gate-check", action="store_true",
                        help="Bypass the Gate 2 git-tag check.")
    args = parser.parse_args()

    check_gate(args.skip_gate_check)
    check_waymo_env()
    shards = check_data(args.waymo_dir)

    features_path = ROOT / "results" / "per_dataset" / "waymo_features.json"
    boundaries_path = ROOT / "results" / "boundaries" / "waymo.json"
    verdicts_path = ROOT / "results" / "verdicts" / "waymo.json"

    if args.skip_extract and features_path.exists():
        print(f"[extract] --skip-extract; reusing {features_path}")
        with open(features_path) as fh:
            features = json.load(fh)
    else:
        features = extract_features(shards, args.n, features_path)

    if features["n"] < 738:
        print(f"[warn] N = {features['n']} < 738 (DKW underpowered). "
              f"Results valid but report epsilon explicitly per pre-reg.")

    boundaries = compute_boundaries(features, boundaries_path)
    verdicts = run_tests(features, boundaries, verdicts_path)
    print_summary(verdicts)


if __name__ == "__main__":
    main()
