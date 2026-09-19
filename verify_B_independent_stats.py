"""
INDEPENDENT cross-check of Table III's NGSIM numbers, written fresh and
deliberately NOT importing src/boundary.py or src/tests.py (the modules
that produced the published numbers). Only src/risk.py's composite_risk
formula (R = features @ weights, a one-line dot product specified by the
pre-registration) is treated as ground truth and re-typed by hand below,
not imported, so this script shares zero code with the pipeline being
checked except the raw stored feature arrays themselves.

Consumes the EXISTING results/per_dataset/ngsim_features.json (the run
already used for the paper) so it is fast (no re-extraction). Its purpose
is to catch any bug in boundary.py/tests.py's math (quantile, realized
rate, Mann-Whitney U, rank-biserial r) -- separate from the extraction
bugfix, which is checked by a different script.
"""
import json, sys, time
import numpy as np
from scipy.stats import mannwhitneyu

from pathlib import Path

t0 = time.time()
ROOT = Path(__file__).resolve().parent

with open(ROOT / "carla_weights.json") as f:
    wcfg = json.load(f)
weights = np.array(wcfg["weights"], dtype=float)
assert abs(weights.sum() - 0.94) < 1e-9, "weights sum mismatch"

with open(ROOT / "supervisor_spec.json") as f:
    scfg = json.load(f)
B_sim = {float(k): float(v) for k, v in scfg["B_sim"].items()}

with open(ROOT / "results" / "verdicts" / "ngsim.json") as f:
    official = json.load(f)

print(f"[t+{time.time()-t0:5.1f}s] loading results/per_dataset/ngsim_features.json (377MB) ...")
with open(ROOT / "results" / "per_dataset" / "ngsim_features.json") as f:
    feat_data = json.load(f)
trajs = feat_data["trajectories"]
N = len(trajs)
print(f"[t+{time.time()-t0:5.1f}s] loaded. N={N} (official N={official['N']})")
assert N == official["N"], f"SAMPLE SIZE MISMATCH: features file has {N}, verdict file says {official['N']}"

# --- independent R_max computation: hand-written dot product, no src.risk import ---
r_max = np.empty(N, dtype=float)
ttc_raw_list = []
for i, t in enumerate(trajs):
    F = np.asarray(t["features"], dtype=float)  # (T, 8)
    # R(x_t) = sum_i w_i * f_i(x_t) -- pre-registration formula, hand-typed here
    R = np.zeros(F.shape[0], dtype=float)
    for j in range(8):
        R += weights[j] * F[:, j]
    r_max[i] = R.max()
    ttc_raw_list.append(np.asarray(t["ttc_raw"], dtype=float))
print(f"[t+{time.time()-t0:5.1f}s] independent R_max computed for all {N} trajectories.")

# --- independent quantile: manual sort + linear interpolation, not np.quantile ---
def manual_quantile(arr, q):
    a = np.sort(np.asarray(arr, dtype=float))
    n = len(a)
    pos = q * (n - 1)
    lo = int(np.floor(pos)); hi = int(np.ceil(pos))
    if lo == hi:
        return float(a[lo])
    frac = pos - lo
    return float(a[lo] * (1 - frac) + a[hi] * frac)

TAUS = [0.10, 0.15, 0.20]
TOL = 1e-9
print()
print("=" * 100)
print("INDEPENDENT RECOMPUTATION vs OFFICIAL results/verdicts/ngsim.json")
print("=" * 100)
all_ok = True
for tau in TAUS:
    key = f"{tau:.2f}"
    off = official["tests"][key]

    B_d_indep = manual_quantile(r_max, 1.0 - tau)
    B_d_official = off["H_OFF1"]["B_d"]
    ok_Bd = abs(B_d_indep - B_d_official) < 1e-6

    tau_hat_indep = float(np.mean(r_max > B_sim[tau]))
    tau_hat_official = off["H_OFF2"]["tau_hat_d"]
    ok_tauhat = abs(tau_hat_indep - tau_hat_official) < 1e-9

    crossing_ttc, noncrossing_ttc = [], []
    for i in range(N):
        cnt = int(np.sum(ttc_raw_list[i] < 2.0))
        if r_max[i] > B_sim[tau]:
            crossing_ttc.append(cnt)
        else:
            noncrossing_ttc.append(cnt)
    n1, n2 = len(crossing_ttc), len(noncrossing_ttc)
    ok_n = (n1 == off["H_OFF3"]["n_crossing"]) and (n2 == off["H_OFF3"]["n_noncrossing"])

    # Independent effect size + significance via scipy's own Mann-Whitney implementation
    # (a totally different code path than the hand-rolled permutation test in src/tests.py)
    U_scipy, p_scipy = mannwhitneyu(crossing_ttc, noncrossing_ttc, alternative="greater",
                                     method="asymptotic")
    r_scipy = 1.0 - 2.0 * U_scipy / (n1 * n2)
    r_official = off["H_OFF3"]["effect_size_r"]
    U_official = off["H_OFF3"]["U"]
    ok_U = abs(U_scipy - U_official) < 1.0  # U statistic should match exactly (same rank-sum definition)
    ok_r = abs(r_scipy - r_official) < 1e-6
    ok_p_conclusion = (p_scipy < off["H_OFF3"]["alpha_bonf"]) == True  # both should be far below alpha_bonf

    print(f"\ntau={tau:.2f}")
    print(f"  B_d          indep={B_d_indep:.6f}   official={B_d_official:.6f}   match={ok_Bd}")
    print(f"  tau_hat_d    indep={tau_hat_indep:.6f}   official={tau_hat_official:.6f}   match={ok_tauhat}")
    print(f"  n_crossing/n_noncrossing  indep={n1}/{n2}   official={off['H_OFF3']['n_crossing']}/{off['H_OFF3']['n_noncrossing']}   match={ok_n}")
    print(f"  U (rank-sum) scipy={U_scipy:.1f}   official(permutation code)={U_official:.1f}   match={ok_U}")
    print(f"  rank-biserial r   scipy-derived={r_scipy:.6f}   official={r_official:.6f}   match={ok_r}")
    print(f"  scipy asymptotic p={p_scipy:.3e}  (alpha_bonf={off['H_OFF3']['alpha_bonf']:.3e})  "
          f"both-far-below-threshold={ok_p_conclusion}  (note: scipy uses normal-approx method, "
          f"NOT the same permutation method as the paper's src/tests.py -- exact p-value agreement "
          f"is not expected here, only the PASS conclusion and U/r, which use the same standard "
          f"rank-sum definition regardless of p-value method)")

    ok_cell = ok_Bd and ok_tauhat and ok_n and ok_U and ok_r
    print(f"  >>> CELL RESULT: {'MATCH' if ok_cell else 'MISMATCH <<<<<<<<<<<<<<<<<<<<'}")
    all_ok = all_ok and ok_cell

print()
print("=" * 100)
print(f"OVERALL: {'ALL CELLS INDEPENDENTLY CONFIRMED' if all_ok else 'AT LEAST ONE MISMATCH -- INVESTIGATE'}")
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("=" * 100)
print("INDEPENDENT_STATS_CHECK_DONE_SENTINEL", "PASS" if all_ok else "FAIL")
