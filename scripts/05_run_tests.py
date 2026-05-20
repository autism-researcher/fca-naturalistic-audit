"""Stage 5c: run the three pre-registered tests in locked order.

Reads results/per_dataset/{d}_features.json (for raw TTC) and
results/boundaries/{d}.json (for B_d, B_sim, tau_hat_d).
Writes results/verdicts/{d}.json.

Run AFTER 04_compute_boundaries.py.  Locked order: H_OFF1 -> H_OFF2 -> H_OFF3.
"""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_weights
from src.risk import composite_risk
from src.tests import h_off1, h_off2, h_off3

TAUS = [0.10, 0.15, 0.20]
TTC_THRESH_S = 2.0
N_COMPARISONS = 9

def ttc_below_2s_counts(trajs, weights, B_sim_tau):
    """Per-trajectory raw-TTC<2s tick count; partitioned by crossing/non-crossing."""
    crossing, noncrossing = [], []
    for t in trajs:
        feats = np.array(t["features"], dtype=float)
        ttc = np.array(t["ttc_raw"], dtype=float)
        R = composite_risk(feats, weights)
        Rm = float(np.max(R))
        cnt = int(np.sum(ttc < TTC_THRESH_S))
        (crossing if Rm > B_sim_tau else noncrossing).append(cnt)
    return crossing, noncrossing

def process(features_path, boundaries_path, out_dir):
    with open(features_path) as f:
        feat = json.load(f)
    with open(boundaries_path) as f:
        bnd = json.load(f)
    weights, _ = load_weights()

    # Normalize B_sim keys (JSON serialization may have stripped "0.10" -> "0.1").
    bsim_norm = {}
    for k, v in bnd["B_sim"].items():
        try:
            bsim_norm[f"{float(k):.2f}"] = v
        except (TypeError, ValueError):
            bsim_norm[k] = v

    verdicts = {"dataset": feat["dataset"], "N": feat["n"], "tests": {}}
    for tau in TAUS:
        key = f"{tau:.2f}"
        B_sim = bsim_norm[key]
        B_d   = bnd["B_d"][key]
        tau_hat = bnd["tau_hat_d"][key]

        v1 = h_off1(B_sim, B_d)
        v2 = h_off2(tau_hat, tau)
        crossing, noncrossing = ttc_below_2s_counts(feat["trajectories"], weights, B_sim)
        v3 = h_off3(crossing, noncrossing, n_comparisons=N_COMPARISONS)
        v3["n_crossing"] = len(crossing); v3["n_noncrossing"] = len(noncrossing)
        verdicts["tests"][key] = {"H_OFF1": v1, "H_OFF2": v2, "H_OFF3": v3}

    out_path = out_dir / f"{feat['dataset']}.json"
    with open(out_path, "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"[{feat['dataset']}] -> {out_path}")
    for tau in TAUS:
        v = verdicts["tests"][f"{tau:.2f}"]
        h1 = "PASS" if v["H_OFF1"]["pass"] else "FAIL"
        h2 = "PASS" if v["H_OFF2"]["pass"] else "FAIL"
        h3 = "PASS" if (v["H_OFF3"].get("pass") is True) else "FAIL"
        print(f"  tau={tau:.2f}  H_OFF1:{h1}  H_OFF2:{h2}  H_OFF3:{h3}")

if __name__ == "__main__":
    out_dir = ROOT / "results/verdicts"; out_dir.mkdir(parents=True, exist_ok=True)
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        ds = Path(fpath).stem.replace("_features", "")
        bpath = ROOT / "results/boundaries" / f"{ds}.json"
        if not bpath.exists():
            print(f"[{ds}] boundaries missing; run 04 first.")
            continue
        process(Path(fpath), bpath, out_dir)
