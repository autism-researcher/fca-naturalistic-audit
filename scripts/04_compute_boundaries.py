"""Stage 5b: composite risk, R_max, and boundary B_d per (dataset, tau).

Reads results/per_dataset/{dataset}_features.json.
Writes results/boundaries/{dataset}.json with B_d, tau_hat_d, DKW epsilon.

The first computation of R(x) on naturalistic data happens here.
Pre-registration MUST be locked before running.
"""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_weights, load_B_sim
from src.risk import composite_risk
from src.boundary import boundary as B_quantile, realized_rate, dkw_epsilon, bootstrap_ci

TAUS = [0.10, 0.15, 0.20]

def process(feat_path, out_dir):
    with open(feat_path) as f:
        data = json.load(f)
    dataset = data["dataset"]
    trajs = data["trajectories"]
    n = len(trajs)
    weights, _ = load_weights()
    B_sim_map = load_B_sim()

    r_max_list = []
    for t in trajs:
        feats = np.array(t["features"], dtype=float)
        R = composite_risk(feats, weights)
        r_max_list.append(float(np.max(R)))
    r_max = np.array(r_max_list)

    result = {
        "dataset": dataset, "N": n,
        "DKW_epsilon": dkw_epsilon(n, delta=0.05),
        "underpowered": n < 738,
        "B_sim": {f"{k:.2f}": v for k, v in B_sim_map.items()},
        "B_d": {}, "tau_hat_d": {}, "ci": {}
    }
    for tau in TAUS:
        B_d = B_quantile(r_max, tau)
        result["B_d"][f"{tau:.2f}"] = B_d
        B_sim_tau = B_sim_map[tau]
        result["tau_hat_d"][f"{tau:.2f}"] = realized_rate(r_max, B_sim_tau)
        B_lo, B_hi = bootstrap_ci(r_max, lambda x: float(np.quantile(x, 1 - tau)))
        rate_lo, rate_hi = bootstrap_ci(r_max, lambda x: float(np.mean(x > B_sim_tau)))
        result["ci"][f"{tau:.2f}"] = {"B_d": [B_lo, B_hi], "tau_hat_d": [rate_lo, rate_hi]}
    out_path = out_dir / f"{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{dataset}] N={n}  DKW eps={result['DKW_epsilon']:.4f}  -> {out_path}")
    for tau in TAUS:
        print(f"  tau={tau:.2f}  B_sim={B_sim_map[tau]:.4f}  "
              f"B_d={result['B_d'][f'{tau:.2f}']:.4f}  "
              f"tau_hat={result['tau_hat_d'][f'{tau:.2f}']:.4f}")

if __name__ == "__main__":
    out_dir = ROOT / "results/boundaries"; out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        process(Path(f), out_dir)
