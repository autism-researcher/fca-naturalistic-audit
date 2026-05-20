"""Stage 6: 3x3 cross-dataset transfer matrix per tau.

Cell (d1, d2) = realized rate of B_{d1, tau} when evaluated on R_max of d2.
Bootstrap CI per cell.  Writes results/transfer/matrix.json.
"""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_weights
from src.risk import composite_risk
from src.boundary import bootstrap_ci

TAUS = [0.10, 0.15, 0.20]

def rmax_for_dataset(features_path, weights):
    with open(features_path) as f:
        data = json.load(f)
    out = []
    for t in data["trajectories"]:
        feats = np.array(t["features"], dtype=float)
        out.append(float(np.max(composite_risk(feats, weights))))
    return data["dataset"], np.array(out)

if __name__ == "__main__":
    weights, _ = load_weights()
    rmax = {}
    bnd = {}
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        ds, r = rmax_for_dataset(fpath, weights)
        rmax[ds] = r
        bpath = ROOT / "results/boundaries" / f"{ds}.json"
        with open(bpath) as f:
            bnd[ds] = json.load(f)
    datasets = sorted(rmax.keys())
    print(f"Datasets: {datasets}")

    out = {"datasets": datasets, "by_tau": {}}
    for tau in TAUS:
        cells = {}
        for d1 in datasets:
            B_d1 = bnd[d1]["B_d"][f"{tau:.2f}"]
            for d2 in datasets:
                r = rmax[d2]
                realized = float(np.mean(r > B_d1))
                lo, hi = bootstrap_ci(r, lambda x: float(np.mean(x > B_d1)))
                cells[f"{d1}__on__{d2}"] = {
                    "B_used": B_d1, "realized_rate": realized,
                    "ci_lo": lo, "ci_hi": hi, "tau_target": tau,
                    "abs_diff_from_target": abs(realized - tau),
                }
        out["by_tau"][f"{tau:.2f}"] = cells

    out_dir = ROOT / "results/transfer"; out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "matrix.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {out_dir / 'matrix.json'}")
