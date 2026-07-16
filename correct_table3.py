#!/usr/bin/env python3
"""Recompute ALL nine Table III (H_OFF3) p-values with a uniform, valid
permutation method and emit the corrected LaTeX rows + JSON.

Method (uniform across all cells):
  - Exact permutation enumeration when the pooled sample has <= 12 nonzero
    per-trajectory TTC<2s counts (all three HighD cells).
  - Monte Carlo permutation with 10,000,000 resamples otherwise (NGSIM,
    Waymo), sampled from the tie-group structure via the multivariate
    hypergeometric distribution (statistically identical to full
    permutation of the count vector). Cells where no resample reaches the
    observed statistic are reported as upper bounds: p < 1.0e-7.

All entries are compared against alpha_bonf = 0.001/9 = 1.11e-4.
U and rank-biserial r are recomputed and must equal the released
results/verdicts/*.json values (they will: those were always correct).

Run from the paper3_pipeline directory (NGSIM loads ~850 MB, allow RAM):
  python correct_table3.py
Writes:
  results/figures/table_verdicts_corrected.tex
  results/verdicts/hoff3_corrected.json
"""
import json, sys, os
from itertools import combinations
from math import lgamma, exp
import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.risk import composite_risk

TTC_THRESH_S = 2.0
ALPHA_BONF = 0.001 / 9
MC_DRAWS = 10_000_000
TAUS = ("0.10", "0.15", "0.20")
DATASETS = ("highd", "ngsim", "waymo")


def lchoose(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def build_counts(dataset):
    """Identical logic to scripts/05_run_tests.py::ttc_below_2s_counts."""
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
    pooled = np.concatenate([x, y]); ranks = rankdata(pooled)
    n1, N = len(x), len(pooled)
    obs = ranks[:n1].sum()
    nz_ranks = ranks[pooled > 0]; m = len(nz_ranks)
    if m > 12:
        return None
    zero_rank = ranks[pooled == 0][0]
    log_denom = lchoose(N, n1); tot = 0.0
    for k in range(0, min(m, n1) + 1):
        good = sum(1 for s in combinations(nz_ranks, k)
                   if sum(s) + (n1 - k) * zero_rank >= obs - 1e-9)
        if good:
            tot += good * exp(lchoose(N - m, n1 - k) - log_denom)
    return tot


def mc_perm_p(x, y, draws=MC_DRAWS, seed=0, chunk=1_000_000):
    pooled = np.concatenate([x, y]); ranks = rankdata(pooled)
    n1 = len(x); obs = ranks[:n1].sum()
    vals, inv, cnts = np.unique(pooled, return_inverse=True, return_counts=True)
    grp_rank = np.array([ranks[inv == g][0] for g in range(len(vals))])
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    while done < draws:
        n = min(chunk, draws - done)
        mat = rng.multivariate_hypergeometric(cnts.astype(np.int64), n1, size=n)
        ge += int(((mat @ grp_rank) >= obs - 1e-9).sum())
        done += n
    return ge, done


def fmt_p(entry):
    if entry["method"] == "exact":
        p = entry["p"]
        mant, e = f"{p:.1e}".split("e")
        return f"${mant}\\times10^{{{int(e)}}}$"
    if entry["exceedances"] == 0:
        return f"$<10^{{-7}}$"
    p = entry["p"]
    mant, e = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(e)}}}$"


def main():
    results = {}
    for d in DATASETS:
        print(f"[{d}] building per-trajectory counts ...")
        counts = build_counts(d)
        results[d] = {}
        for tau in TAUS:
            x, y = counts[tau]
            pooled = np.concatenate([x, y]); ranks = rankdata(pooled)
            n1 = len(x)
            u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
            r = 1.0 - 2.0 * u / (n1 * len(y))
            p_ex = exact_perm_p(x, y)
            if p_ex is not None:
                entry = {"method": "exact", "p": p_ex, "exceedances": None}
            else:
                ge, done = mc_perm_p(x, y)
                entry = {"method": f"MC({done:,})", "p": (ge + 1) / (done + 1),
                         "exceedances": ge}
            entry.update({"n1": int(n1), "n2": int(len(y)),
                          "U": float(u), "r": float(r),
                          "pass": bool((entry["p"] if entry["exceedances"] not in (0,)
                                        else 1 / (MC_DRAWS + 1)) < ALPHA_BONF)})
            results[d][tau] = entry
            print(f"  tau={tau}: n1={n1}, U={u:.1f}, r={r:.4f}, "
                  f"{entry['method']} p={entry['p']:.3e}"
                  f"{' (0 exceedances -> report as <1e-7)' if entry['exceedances']==0 else ''}"
                  f" -> {'PASS' if entry['pass'] else 'FAIL'}")

    out_json = os.path.join(HERE, "results", "verdicts", "hoff3_corrected.json")
    json.dump(results, open(out_json, "w"), indent=1)

    # corrected LaTeX rows (same row layout as table_verdicts.tex, p column replaced)
    tex_lines = []
    old = open(os.path.join(HERE, "results", "figures", "table_verdicts.tex")).read().strip().splitlines()
    for line in old:
        parts = [c.strip() for c in line.replace("\\\\", "").split("&")]
        d, tau = parts[0], parts[1]
        entry = results[d][f"{float(tau):.2f}"]
        parts[7] = fmt_p(entry)
        tex_lines.append(" & ".join(parts) + " \\\\")
    out_tex = os.path.join(HERE, "results", "figures", "table_verdicts_corrected.tex")
    open(out_tex, "w").write("\n".join(tex_lines) + "\n")
    print(f"\nwrote {out_json}\nwrote {out_tex}")
    print("All nine cells " +
          ("PASS" if all(e["pass"] for dd in results.values() for e in dd.values())
           else "-- CHECK: at least one FAIL --") +
          f" at alpha_bonf = {ALPHA_BONF:.3e}")


if __name__ == "__main__":
    main()
