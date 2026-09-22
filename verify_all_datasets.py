"""
Verify ALL THREE processed datasets (NGSIM, HighD, Waymo) against the paper.

Reads results/per_dataset/*_features.json and, for each dataset x tau,
recomputes B_d and tau_hat straight from the processed data, then prints
them next to the paper's Table III values with a MATCH/CHECK flag.

This answers one question directly: does the processed data on disk
reproduce the numbers printed in the paper?

Run from the repo root (after the feature files exist):
    python verify_all_datasets.py

Nothing is written or modified. Empty datasets are reported and skipped.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

# --- risk config (same as the pipeline; self-contained) -------------------
WEIGHTS = np.array([0.08, 0.10, 0.10, 0.08, 0.12, 0.25, 0.15, 0.06])
TAUS = [0.10, 0.15, 0.20]
B_SIM = {0.10: 0.5116, 0.15: 0.4885, 0.20: 0.4733}

# --- the paper's Table III (B_d, tau_hat) per dataset x tau ----------------
PAPER = {
    "ngsim": {0.10: (0.778, 0.734), 0.15: (0.746, 0.794), 0.20: (0.718, 0.825)},
    "highd": {0.10: (0.314, 0.002), 0.15: (0.301, 0.004), 0.20: (0.290, 0.005)},
    "waymo": {0.10: (0.375, 0.011), 0.15: (0.346, 0.020), 0.20: (0.326, 0.026)},
}
TOL = 0.01          # a reproduction should land within 0.01 of the printed value

def rmax_of(path):
    d = json.load(open(path))
    ds = str(d["dataset"]).lower()
    trajs = d.get("trajectories", [])
    if not trajs:
        return ds, None
    rm = np.array([float(np.max(np.asarray(t["features"], float) @ WEIGHTS)) for t in trajs])
    return ds, rm

def main():
    files = sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json")))
    if not files:
        sys.exit("No results/per_dataset/*_features.json found.")

    print("=" * 84)
    print("VERIFY PROCESSED DATA vs PAPER TABLE III   (B_sim frozen: "
          "0.5116 / 0.4885 / 0.4733)")
    print("=" * 84)

    worst = 0.0
    checked = 0
    missing = []
    for f in files:
        ds, rm = rmax_of(f)
        if rm is None:
            print(f"\n[{ds.upper()}]  EMPTY (0 trajectories) — SKIPPED  <-- process this dataset")
            missing.append(ds)
            continue
        print(f"\n[{ds.upper()}]  N = {len(rm)} trajectories")
        print(f"  {'tau':>4} | {'B_d(yours)':>10} {'B_d(paper)':>10} {'d':>7} |"
              f" {'tauhat(yours)':>13} {'tauhat(paper)':>13} {'d':>7} | match")
        print("  " + "-" * 80)
        ref = PAPER.get(ds, {})
        for tau in TAUS:
            bd = float(np.quantile(rm, 1 - tau))
            th = float(np.mean(rm > B_SIM[tau]))
            pbd, pth = ref.get(tau, (float("nan"), float("nan")))
            dbd, dth = abs(bd - pbd), abs(th - pth)
            ok = (dbd <= TOL) and (dth <= TOL)
            worst = max(worst, dbd, dth)
            checked += 1
            flag = "MATCH" if ok else "CHECK"
            print(f"  {tau:>4.2f} | {bd:>10.4f} {pbd:>10.3f} {dbd:>7.4f} |"
                  f" {th:>13.4f} {pth:>13.3f} {dth:>7.4f} | {flag}")

    print("\n" + "=" * 84)
    if missing:
        print(f"NOT VERIFIED (no processed data): {', '.join(m.upper() for m in missing)}")
    if checked:
        allok = worst <= TOL and not missing
        print(f"Largest deviation from the paper over {checked} checked cells: {worst:.4f}"
              f"   (tolerance {TOL})")
        print("VERDICT:",
              "PASS — every processed dataset reproduces Table III within tolerance."
              if allok else
              ("PARTIAL — checked cells match; process the missing dataset(s) above."
               if worst <= TOL else
               "REVIEW — one or more cells deviate beyond tolerance; see CHECK rows."))
    print("=" * 84)

if __name__ == "__main__":
    main()
