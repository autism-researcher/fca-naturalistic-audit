"""Option 2: Out-of-sample recalibration demonstration.

For each dataset, randomly split the N=5000 eligible trajectories
into a 70% calibration set (train) and a 30% held-out set (test)
using the pre-registered SEED. Fit B_d^{train} as the (1-tau)
quantile of R_max on the train set; evaluate the realized rate
tau_hat^{test} = fraction of test trajectories with R_max > B_d^{train}.

If per-distribution recalibration works on naturalistic data,
|tau_hat^{test} - tau| should fall within the DKW band
eps(N_test, delta=0.05) on every (dataset, tau) cell.

Run from paper3_pipeline/ root:
    python scripts/recalibration_70_30_split.py

Paste the output table back to the manuscript-revision conversation
so the actual numbers can be filled into the new §VI-D subsection.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import SEED, load_weights          # noqa: E402
from src.risk import composite_risk                # noqa: E402

TAUS = [0.10, 0.15, 0.20]
SPLIT_FRAC = 0.7       # 70% train, 30% test
DELTA = 0.05           # DKW confidence parameter


def per_trajectory_rmax(features_path, weights):
    """Return an array of R_max values, one per trajectory."""
    with open(features_path) as f:
        data = json.load(f)
    trajs = data["trajectories"]
    return np.array([
        float(np.max(composite_risk(np.array(t["features"], dtype=float), weights)))
        for t in trajs
    ])


def dkw_epsilon(n, delta=DELTA):
    return float(np.sqrt(np.log(2 / delta) / (2 * n)))


def split_and_recalibrate(r_max, tau):
    n = len(r_max)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(n * SPLIT_FRAC)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    b_d_train = float(np.quantile(r_max[train_idx], 1 - tau))
    tau_test = float(np.mean(r_max[test_idx] > b_d_train))
    n_test = len(test_idx)
    eps_test = dkw_epsilon(n_test)

    return {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "B_d_train": b_d_train,
        "tau_test": tau_test,
        "tau_target": tau,
        "diff": abs(tau_test - tau),
        "dkw_eps_test": eps_test,
        "within_dkw": abs(tau_test - tau) <= eps_test,
    }


def main():
    weights, _ = load_weights()
    print(f"Frozen weights loaded; sum(w) = {weights.sum():.4f}")
    print()

    print("Per-distribution recalibration: 70/30 split, SEED =", SEED)
    print("=" * 84)
    print(f"{'Dataset':<8} {'tau':>5} {'N_train':>8} {'N_test':>7} "
          f"{'B_d^train':>10} {'tau_hat^test':>13} {'|diff|':>7} "
          f"{'eps_DKW':>8} {'in-band':>8}")
    print("-" * 84)

    results = {}
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        ds = Path(fpath).stem.replace("_features", "")
        r_max = per_trajectory_rmax(fpath, weights)
        results[ds] = {}
        for tau in TAUS:
            r = split_and_recalibrate(r_max, tau)
            results[ds][tau] = r
            check = "PASS" if r["within_dkw"] else "FAIL"
            print(f"{ds:<8} {tau:>5.2f} {r['n_train']:>8d} {r['n_test']:>7d} "
                  f"{r['B_d_train']:>10.4f} {r['tau_test']:>13.4f} "
                  f"{r['diff']:>7.4f} {r['dkw_eps_test']:>8.4f} {check:>8}")
    print("=" * 84)
    print()
    print("PASS means |tau_hat^test - tau| <= eps_DKW(N_test, delta=0.05).")
    print()

    print("Compact JSON record (paste back):")
    out = {
        ds: {f"{tau:.2f}": {k: v for k, v in r.items() if k != "within_dkw"}
             for tau, r in row.items()}
        for ds, row in results.items()
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
