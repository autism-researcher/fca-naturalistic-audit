"""
Forensic check ON YOUR REAL EXTRACTED DATA.

Cross-validates the consolidated pipeline against the repo's OWN
authoritative functions (src/boundary.py, src/risk.py) — the exact code
the paper uses — on whatever is in results/per_dataset/*_features.json.

It proves three things, per dataset, to machine precision:
  1. R_max computed my way  ==  R_max via src.risk.composite_risk
  2. B_d and tau_hat my way  ==  src.boundary.boundary / realized_rate
  3. the cross-dataset transfer cells match the repo's formula
and it reports the baseline engagement and H_OFF3 group sizes so you can
eyeball them.

Nothing is written or changed. Run from the repo root:

    python verify_on_real_data.py

Empty datasets (e.g. waymo with 0 trajectories) are skipped with a notice.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---- the repo's OWN authoritative code = the reference -----------------------
try:
    from src.risk import composite_risk as REPO_risk
    from src.boundary import boundary as REPO_boundary, realized_rate as REPO_realized
    from src.utils import load_weights as REPO_load_weights
except Exception as e:
    sys.exit(f"Must be run from the repo root (needs src/). Import error: {e}")

# ---- the consolidated pipeline's config (must equal the repo's) --------------
MY_WEIGHTS = np.array([0.08, 0.10, 0.10, 0.08, 0.12, 0.25, 0.15, 0.06])
TTC_INDEX = 5
TAUS = [0.10, 0.15, 0.20]
B_SIM = {0.10: 0.5116, 0.15: 0.4885, 0.20: 0.4733}

def my_composite(feats, w):            # consolidated pipeline's risk
    return np.asarray(feats, float) @ np.asarray(w, float)


def main():
    files = sorted(glob.glob(str(ROOT / "results/per_dataset/*_features.json")))
    if not files:
        sys.exit("No results/per_dataset/*_features.json — run scripts/03 first.")

    # sanity: the weights the repo loads must match the ones baked into the
    # consolidated pipeline, else every downstream number would silently differ.
    repo_w, _ = REPO_load_weights()
    repo_w = np.asarray(repo_w, float)
    if not np.allclose(repo_w, MY_WEIGHTS, atol=1e-12):
        print("!!! WEIGHT MISMATCH  repo:", repo_w.tolist(), " mine:", MY_WEIGHTS.tolist())
    else:
        print(f"weights match repo exactly (sum={repo_w.sum():.2f})\n")

    w_abl = MY_WEIGHTS.copy(); w_abl[TTC_INDEX] = 0.0
    w_abl *= MY_WEIGHTS.sum() / w_abl.sum()

    RF = {}   # dataset -> R_max_full array (used later for transfer)
    max_rmax = max_bd = max_th = 0.0
    print("per-dataset checks (mine vs repo's own functions):")
    print("-" * 74)
    for f in files:
        d = json.load(open(f))
        ds = str(d["dataset"]).lower()
        trajs = d["trajectories"]
        if not trajs:
            print(f"[{ds}]  EMPTY (0 trajectories) — skipped\n"); continue

        rf_mine, rf_repo, ra_mine, mt, lt = [], [], [], [], []
        for t in trajs:
            feats = np.asarray(t["features"], float)
            rf_mine.append(float(np.max(my_composite(feats, MY_WEIGHTS))))
            rf_repo.append(float(np.max(REPO_risk(feats, repo_w))))    # repo's function
            ra_mine.append(float(np.max(my_composite(feats, w_abl))))
            ttc = np.asarray(t.get("ttc_raw") or [], float); ttc = ttc[np.isfinite(ttc)]
            mt.append(float(ttc.min()) if ttc.size else np.inf)
            lt.append(int(np.sum(ttc < 2.0)))
        rf_mine = np.array(rf_mine); rf_repo = np.array(rf_repo)
        RF[ds] = rf_mine

        d_rmax = float(np.max(np.abs(rf_mine - rf_repo)))
        max_rmax = max(max_rmax, d_rmax)
        print(f"[{ds}]  N={len(trajs)}   R_max  mine-vs-repo  max|diff|={d_rmax:.2e}")
        for tau in TAUS:
            bd_mine = float(np.quantile(rf_mine, 1 - tau))
            bd_repo = REPO_boundary(rf_repo, tau)               # repo's function
            th_mine = float(np.mean(rf_mine > B_SIM[tau]))
            th_repo = REPO_realized(rf_repo, B_SIM[tau])        # repo's function
            max_bd = max(max_bd, abs(bd_mine - bd_repo))
            max_th = max(max_th, abs(th_mine - th_repo))
            gate = "PASS" if abs(B_SIM[tau] - bd_mine) < 0.03 else "FAIL"
            print(f"        tau={tau:.2f}  B_d={bd_mine:.6f} (repo {bd_repo:.6f})"
                  f"  tau_hat={th_mine:.6f} (repo {th_repo:.6f})"
                  f"  H_OFF1={gate}")
        # baseline + H_OFF3 group sizes (informational)
        mt = np.array(mt); lt = np.array(lt)
        base2 = float(np.mean(mt < 2.0))
        cross = rf_mine > B_SIM[0.10]
        print(f"        baseline frac(min_ttc<2s)={base2:.4f}   "
              f"H_OFF3@tau0.10: {int(cross.sum())} crossing / {int((~cross).sum())} non   "
              f"mean low-ttc ticks {lt[cross].mean() if cross.any() else 0:.3f} vs "
              f"{lt[~cross].mean() if (~cross).any() else 0:.3f}\n")

    # ---- transfer cross-check (repo formula: mean(R_max_d2 > B_d(d1,tau))) ----
    ds_list = sorted(RF.keys())
    max_tr = 0.0
    if len(ds_list) >= 2:
        print("transfer 3x3 cross-check (mine vs repo formula):")
        for tau in TAUS:
            for d1 in ds_list:
                bd1_mine = float(np.quantile(RF[d1], 1 - tau))
                bd1_repo = REPO_boundary(RF[d1], tau)
                for d2 in ds_list:
                    cell_mine = float(np.mean(RF[d2] > bd1_mine))
                    cell_repo = REPO_realized(RF[d2], bd1_repo)
                    max_tr = max(max_tr, abs(cell_mine - cell_repo))
        print(f"  transfer max|diff| over all {len(ds_list)**2 * len(TAUS)} cells = {max_tr:.2e}\n")
    else:
        print(f"transfer skipped (only {len(ds_list)} non-empty dataset)\n")

    print("=" * 74)
    print(f"R_max     max|diff| = {max_rmax:.2e}")
    print(f"B_d       max|diff| = {max_bd:.2e}")
    print(f"tau_hat   max|diff| = {max_th:.2e}")
    print(f"transfer  max|diff| = {max_tr:.2e}")
    ok = max(max_rmax, max_bd, max_th, max_tr) < 1e-9
    print("\nVERDICT:", "PASS — consolidated pipeline reproduces the repo's own"
          " functions on your REAL data (machine epsilon)."
          if ok else "FAIL — differences above tolerance; investigate.")


if __name__ == "__main__":
    main()
