"""
compute_B.py  --  Paper 3 (naturalistic audit), hypothesis H_OFF1
=================================================================
SELF-CONTAINED. No dependency on any other paper. Both boundaries use ONE rule.

THE RULE (identical for both):
    B = the (1 - tau) quantile of the per-trajectory peak risk R_max.

Only the DATA differs:
    B_sim = rule applied to the SIMULATOR's NORMAL-driving peak risks.
            Deployed values live in  supervisor_spec.json  and are reproducible
            from  data/sim_calibration/sim_normal_peak_risk.csv.
    B_d   = rule applied to the REAL dataset's 5000 trajectories
            in data/<dataset>/  (HighD / NGSIM / Waymo).

H_OFF1 verdict per (dataset, tau):  PASS if |B_sim - B_d| < 0.03, else FAIL.

Run:
    python compute_B.py               # B_sim from supervisor_spec.json (deployed)
    python compute_B.py --recompute   # B_sim recomputed live from the shipped sample
    python compute_B.py --real        # use real data/<dataset>/ for B_d (when added)
"""

import argparse
import json
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# THE RULE
# ----------------------------------------------------------------------
def boundary(r_max_list, tau):
    """B = (1 - tau) quantile of peak risk."""
    return float(np.quantile(np.asarray(r_max_list, dtype=float), 1.0 - tau))

def realized_rate(r_max_list, b_value):
    """Fraction of trajectories whose peak risk exceeds B (should be ~tau)."""
    return float(np.mean(np.asarray(r_max_list, dtype=float) > b_value))

# ----------------------------------------------------------------------
# COMPOSITE RISK  R = sum_i w_i * f_i   (this repo's own risk model)
# ----------------------------------------------------------------------
WEIGHTS = np.array([0.08, 0.10, 0.10, 0.08, 0.12, 0.25, 0.15, 0.06])  # sum = 0.94

def r_max_from_features(features_TxF):
    R = np.asarray(features_TxF, dtype=float) @ WEIGHTS
    return float(np.max(R))

# ----------------------------------------------------------------------
# 1)  B_sim
# ----------------------------------------------------------------------
SPEC = "supervisor_spec.json"
SIM_SAMPLE = "data/sim_calibration/sim_normal_peak_risk.csv"

def load_B_sim():
    """Deployed calibrated boundary of the supervisor under audit."""
    cfg = json.load(open(SPEC))["B_sim"]
    return {float(k): float(v) for k, v in cfg.items()}

def recompute_B_sim():
    """(1 - tau) quantile of the shipped simulator NORMAL peak risks,
    averaged across calibration seeds. Reproduces load_B_sim() within 0.003."""
    df = pd.read_csv(SIM_SAMPLE)                 # columns: seed, episode_id, R_max
    out = {}
    for tau in (0.10, 0.15, 0.20):
        per_seed = [boundary(g.R_max.values, tau) for _, g in df.groupby("seed")]
        out[tau] = float(np.mean(per_seed))
    return out

# ----------------------------------------------------------------------
# 2)  B_d
# ----------------------------------------------------------------------
def get_rmax_list(dataset, real):
    if real:
        raise NotImplementedError(
            f"Put raw {dataset} data in data/{dataset}/ and load features here.")
    rng = np.random.default_rng(42)
    centre = {"highd": 0.30, "ngsim": 0.80, "waymo": 0.35}[dataset]
    return [r_max_from_features(np.clip(rng.normal(centre, 0.12, (30, 8)), 0, 1))
            for _ in range(5000)]

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main(real=False, recompute=False):
    TAUS = [0.10, 0.15, 0.20]
    DATASETS = ["highd", "ngsim", "waymo"]
    B_sim = recompute_B_sim() if recompute else load_B_sim()

    src = f"recomputed from {SIM_SAMPLE}" if recompute else f"deployed values in {SPEC}"
    print(f"B_sim ({src}):", {f"{t:.2f}": round(B_sim[t], 4) for t in TAUS})
    print()
    print(f"{'dataset':7} {'tau':4} {'B_sim':7} {'B_d':7} {'|dB|':7} {'tau_hat':7} verdict")
    print("-" * 56)
    overall_pass = False
    for d in DATASETS:
        rmax = get_rmax_list(d, real)
        dataset_all_pass = True
        for tau in TAUS:
            B_d = boundary(rmax, tau)
            B_s = B_sim[tau]
            dB = abs(B_s - B_d)
            tau_hat = realized_rate(rmax, B_s)
            ok = dB < 0.03
            dataset_all_pass &= ok
            print(f"{d:7} {tau:<4} {B_s:<7.4f} {B_d:<7.4f} {dB:<7.4f} "
                  f"{tau_hat:<7.4f} {'PASS' if ok else 'FAIL'}")
        overall_pass |= dataset_all_pass
    print("-" * 56)
    print("H_OFF1 overall:", "PASS" if overall_pass else "FAIL",
          "(passes only if some dataset is within 0.03 at ALL three tau)")
    if not real:
        print("\n[demo mode: naturalistic B_d numbers are fabricated. Use --real with data/ filled in.]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real data/ for B_d")
    ap.add_argument("--recompute", action="store_true", help="recompute B_sim from shipped sample")
    a = ap.parse_args()
    main(real=a.real, recompute=a.recompute)
