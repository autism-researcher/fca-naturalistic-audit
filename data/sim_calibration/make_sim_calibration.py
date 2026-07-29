"""
make_sim_calibration.py
=======================
Regenerate  sim_normal_peak_risk.csv  — the simulator NORMAL-driving peak-risk
sample that Paper 3 uses to (re)compute B_sim.

WHAT IT DOES (pure extraction, no values are altered):
    1. read the raw CARLA episode log
    2. keep only the NORMAL-driving controller episodes  (controller_label == "normal")
    3. take three existing columns: seed0, episode_id, max_R
    4. rename to seed, episode_id, R_max ; sort ; save

Each R_max is the peak composite risk of one NORMAL episode, already present in
the source log — nothing is computed or fabricated here.

USAGE (self-contained — runs with no arguments):
    python make_sim_calibration.py

The default --source is the simulator log shipped alongside this script
(simulator_episode_log.csv: the NORMAL-driving calibration episodes only —
columns controller_label, seed0, episode_id, max_R). Override with --source to
point at your own CARLA log if you prefer.
This is a one-off provenance/build tool; Paper 3 does NOT need it at runtime
(the resulting CSV is already shipped in the repo).
"""

import argparse
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.join(HERE, "simulator_episode_log.csv")
DEFAULT_OUT = os.path.join(HERE, "sim_normal_peak_risk.csv")


def main(source, out):
    src = pd.read_csv(source)
    if "controller_label" not in src.columns:
        raise SystemExit("source file has no 'controller_label' column — wrong file?")

    # 1-2. NORMAL-driving episodes only
    nrm = src[src["controller_label"] == "normal"].copy()

    # 3-4. select + rename + sort (no numeric transformation)
    nrm = nrm[["seed0", "episode_id", "max_R"]].copy()
    nrm.columns = ["seed", "episode_id", "R_max"]
    nrm = nrm.sort_values(["seed", "episode_id"]).reset_index(drop=True)

    nrm.to_csv(out, index=False)
    print(f"wrote {out}: {len(nrm)} NORMAL peak-risk rows, seeds {sorted(nrm.seed.unique())}")

    # quick provenance report: the (1-tau) quantiles this sample yields
    import numpy as np
    for tau in (0.10, 0.15, 0.20):
        per_seed = [np.quantile(g.R_max.values, 1 - tau) for _, g in nrm.groupby("seed")]
        print(f"  tau={tau:.2f}  B_sim (per-seed mean) = {np.mean(per_seed):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="raw simulator log (default: simulator_episode_log.csv beside this script)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    main(a.source, a.out)
