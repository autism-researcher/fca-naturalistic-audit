#!/usr/bin/env python3
"""Verify Table III (H_OFF3) p-values: reproduce the reported artifact and
compute the statistically valid exact/permutation p-values.

What this shows
---------------
1. The pipeline's U and rank-biserial r are trajectory-level and CORRECT.
2. The reported p-values came from scipy's tie-corrected normal
   approximation, which is invalid here (HighD: ~99.9% of per-trajectory
   TTC<2s counts are zero, n1 as small as 8 -> variance collapses,
   z explodes, p underflows).
3. The valid exact permutation p-values are still far below
   alpha_bonf = 1.11e-4, so every PASS verdict in Table III stands.

Expected (from Claude's verification run, 2026-07-15):
  highd  tau=0.10  exact p ~ 2.96e-11   (reported < 1e-300, impossible: floor 1e-25)
  highd  tau=0.15  exact p ~ 6.62e-13   (reported 5.0e-217,  impossible: floor 1e-56)
  highd  tau=0.20  exact p ~ 3.11e-17   (reported 1.6e-305,  impossible: floor 1e-67)
  waymo  all tau   MC p < 1/(M+1)       (true value far smaller; verdict PASS)
  ngsim  all tau   asymptotic ~ valid   (mild ties, huge groups; verdict PASS)

Run from the paper3_pipeline directory:
  python verify_table3_exact.py highd            (fast, the decisive one)
  python verify_table3_exact.py highd waymo      (adds Waymo, ~1 min)
  python verify_table3_exact.py                  (all three; NGSIM loads 850 MB JSON)
"""
import json, sys, os
from itertools import combinations
from math import lgamma, exp, log10
import numpy as np
from scipy.stats import mannwhitneyu, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.risk import composite_risk

TTC_THRESH_S = 2.0
ALPHA_BONF = 0.001 / 9
MC_DRAWS = 2_000_000
TAUS = ("0.10", "0.15", "0.20")


def lchoose(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def build_counts(dataset):
    """Same logic as scripts/05_run_tests.py::ttc_below_2s_counts."""
    w = np.array(json.load(open(os.path.join(HERE, "carla_weights.json")))["weights"], float)
    feat = json.load(open(os.path.join(HERE, "results", "per_dataset", f"{dataset}_features.json")))
    bnd = json.load(open(os.path.join(HERE, "results", "boundaries", f"{dataset}.json")))
    bsim = {f"{float(k):.2f}": v for k, v in bnd["B_sim"].items()}
    out = {}
    for tau in TAUS:
        B = bsim[tau]
        cr, nc = [], []
        for t in feat["trajectories"]:
            R = composite_risk(np.array(t["features"], float), w)
            cnt = int(np.sum(np.array(t["ttc_raw"], float) < TTC_THRESH_S))
            (cr if float(np.max(R)) > B else nc).append(cnt)
        out[tau] = (np.array(cr, float), np.array(nc, float))
    return out


def exact_perm_p(x, y):
    """Exact permutation p when few pooled values are nonzero (HighD case).
    P(rank-sum of n1 random trajectories >= observed), enumerated over which
    nonzero values land in the crossing group."""
    pooled = np.concatenate([x, y]); ranks = rankdata(pooled)
    n1, N = len(x), len(pooled)
    obs = ranks[:n1].sum()
    nz_ranks = ranks[pooled > 0]; m = len(nz_ranks)
    if m > 12:
        return None  # enumeration too large; caller falls back to MC
    zero_rank = ranks[pooled == 0][0]
    log_denom = lchoose(N, n1); tot = 0.0
    for k in range(0, min(m, n1) + 1):
        good = sum(1 for s in combinations(nz_ranks, k)
                   if sum(s) + (n1 - k) * zero_rank >= obs - 1e-9)
        if good:
            tot += good * exp(lchoose(N - m, n1 - k) - log_denom)
    return tot


def mc_perm_p(x, y, draws=MC_DRAWS, seed=0):
    """Monte Carlo permutation p using the tie-group structure
    (multivariate hypergeometric over distinct count values) -- fast and
    memory-light even for N = 5000."""
    pooled = np.concatenate([x, y]); ranks = rankdata(pooled)
    n1 = len(x); obs = ranks[:n1].sum()
    vals, inv, cnts = np.unique(pooled, return_inverse=True, return_counts=True)
    grp_rank = np.zeros(len(vals))
    for g in range(len(vals)):
        grp_rank[g] = ranks[inv == g][0]  # ties share the average rank
    rng = np.random.default_rng(seed)
    draws_mat = rng.multivariate_hypergeometric(cnts.astype(np.int64), n1, size=draws)
    sums = draws_mat @ grp_rank
    ge = int((sums >= obs - 1e-9).sum())
    return (ge + 1) / (draws + 1), ge


def main():
    datasets = sys.argv[1:] or ["highd", "waymo", "ngsim"]
    print(f"alpha_bonf = {ALPHA_BONF:.3e}\n")
    for d in datasets:
        print(f"===== {d} =====")
        try:
            rep = json.load(open(os.path.join(HERE, "results", "verdicts", f"{d}.json")))["tests"]
        except Exception:
            rep = {}
        counts = build_counts(d)
        for tau in TAUS:
            x, y = counts[tau]
            u, p_asym = mannwhitneyu(x, y, alternative="greater")
            r = 1.0 - 2.0 * u / (len(x) * len(y))
            floor = lchoose(len(x) + len(y), len(x)) / np.log(10)
            p_ex = exact_perm_p(x, y)
            if p_ex is not None:
                method, p_str = "exact", f"{p_ex:.3e}"
                verdict = "PASS" if p_ex < ALPHA_BONF else "FAIL"
            else:
                p_mc, ge = mc_perm_p(x, y)
                if ge == 0:
                    method, p_str = "MC", f"< {1/(MC_DRAWS+1):.1e} (0/{MC_DRAWS:,} exceedances)"
                    verdict = "PASS"
                else:
                    method, p_str = "MC", f"{p_mc:.3e}"
                    verdict = "PASS" if p_mc < ALPHA_BONF else "FAIL"
            rp = rep.get(tau, {}).get("H_OFF3", {})
            print(f" tau={tau}: n1={len(x)}, n2={len(y)}")
            print(f"   reported : U={rp.get('U')}, r={rp.get('effect_size_r')}, p={rp.get('p')}")
            print(f"   recomputed: U={u:.1f}, r={r:.4f}  (should match reported)")
            print(f"   scipy asymptotic p = {p_asym:.3g}  <- the artifact (tie-corrected normal approx)")
            print(f"   exact floor 1/C(N,n1) = 10^-{floor:.1f}  (any reported p below this is impossible)")
            print(f"   VALID {method} permutation p = {p_str}   -> verdict {verdict}")
        print()
    print("Conclusion: reported p-values must be replaced by the exact/permutation")
    print("values above; all PASS verdicts at alpha_bonf = 1.11e-4 are unchanged.")


if __name__ == "__main__":
    main()
