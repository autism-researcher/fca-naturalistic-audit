"""Stage 7: feature-bound sensitivity (+-10%) and sample-size sensitivity.

EXPLORATORY only.  Results do not affect the confirmatory verdicts.
"""
import glob, json, sys, copy
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_weights
from src.risk import composite_risk
from src.boundary import boundary as B_quantile, realized_rate, dkw_epsilon

TAUS = [0.10, 0.15, 0.20]
PERTURBATIONS = [-0.10, +0.10]
FEATURE_NAMES = ["speed","accel","jerk","steer_var","lane_off","ttc","headway","density"]
SAMPLE_SIZES = [1000, 2000, 3000, 5000]

def collect_rmax_per_traj_with_bounds(features_path, weights):
    """For sensitivity, we DON'T renormalize from raw; we use per-feature scaling shifts.
    Simpler proxy: scale each column by (1 + p) where p in PERTURBATIONS; clip to [0,1]."""
    with open(features_path) as f:
        data = json.load(f)
    return data["dataset"], data["trajectories"]

def sensitivity_for(features_path, B_sim_map):
    weights, _ = load_weights()
    dataset, trajs = collect_rmax_per_traj_with_bounds(features_path, weights)
    out = {"dataset": dataset, "feature_bound": {}, "sample_size": {}}

    # ---- feature-bound +-10% (acts on the normalized column) ----
    base_feats = [np.array(t["features"], dtype=float) for t in trajs]
    base_rmax = np.array([float(np.max(composite_risk(f, weights))) for f in base_feats])
    for i, fname in enumerate(FEATURE_NAMES):
        for p in PERTURBATIONS:
            new_rmax = []
            scale = 1.0 / (1.0 + p)  # bound +10% -> column / 1.1
            for f in base_feats:
                f2 = f.copy()
                f2[:, i] = np.clip(f2[:, i] * scale, 0.0, 1.0)
                new_rmax.append(float(np.max(composite_risk(f2, weights))))
            new_rmax = np.array(new_rmax)
            for tau in TAUS:
                key = f"{fname}_{'pos' if p > 0 else 'neg'}{int(abs(p)*100)}__tau{tau:.2f}"
                B_new = B_quantile(new_rmax, tau)
                out["feature_bound"][key] = {
                    "B_d": B_new, "delta_B": B_new - B_quantile(base_rmax, tau),
                    "tau_hat_d": realized_rate(new_rmax, B_sim_map[tau]),
                }

    # ---- sample-size sensitivity ----
    rng = np.random.default_rng(42)
    for n in SAMPLE_SIZES:
        if n > len(base_rmax):
            continue
        idx = rng.choice(len(base_rmax), n, replace=False)
        sub = base_rmax[idx]
        for tau in TAUS:
            out["sample_size"][f"n{n}__tau{tau:.2f}"] = {
                "B_d": B_quantile(sub, tau),
                "tau_hat_d": realized_rate(sub, B_sim_map[tau]),
                "DKW_eps": dkw_epsilon(n),
            }
    return out

if __name__ == "__main__":
    from src.utils import load_B_sim
    B_sim = load_B_sim()
    out_dir = ROOT / "results/sensitivity"; out_dir.mkdir(parents=True, exist_ok=True)
    for fpath in sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json"))):
        ds_results = sensitivity_for(Path(fpath), B_sim)
        outp = out_dir / f"{ds_results['dataset']}.json"
        with open(outp, "w") as f:
            json.dump(ds_results, f, indent=2)
        print(f"[{ds_results['dataset']}] sensitivity -> {outp}")

        # ---- printable summary ----
        print(f"\n=== {ds_results['dataset'].upper()} SENSITIVITY SUMMARY ===")

        # Feature-bound: rank by max |delta_B| across taus, per feature
        per_feature = {}
        for key, val in ds_results["feature_bound"].items():
            # key format: "<feature>_<pos10|neg10>__tau<x>"
            feat = key.split("__")[0].rsplit("_", 1)[0]
            per_feature.setdefault(feat, []).append(abs(val["delta_B"]))
        ranked = sorted(per_feature.items(), key=lambda kv: -max(kv[1]))
        print("\nFeature-bound (+-10%) sensitivity, ranked by max |delta_B|:")
        print(f"  {'feature':<14s}  {'max |dB|':>10s}")
        for feat, dlist in ranked:
            print(f"  {feat:<14s}  {max(dlist):>10.4f}")

        # Sample-size sensitivity at tau=0.20
        print("\nSample-size sensitivity (tau=0.20):")
        print(f"  {'N':>6s}  {'B_d':>8s}  {'tau_hat':>8s}  {'DKW_eps':>8s}")
        for key, val in sorted(ds_results["sample_size"].items()):
            if "tau0.20" not in key:
                continue
            n = key.split("__")[0].lstrip("n")
            print(f"  {int(n):>6d}  {val['B_d']:>8.4f}  {val['tau_hat_d']:>8.4f}  {val['DKW_eps']:>8.4f}")
        print()
