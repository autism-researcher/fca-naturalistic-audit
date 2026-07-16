"""H_OFF1, H_OFF2, H_OFF3 implementations.  Outputs are JSON-safe.

2026-07 correction (deviations register): H_OFF3's p-value is now a valid
permutation probability instead of scipy's tie-corrected normal
approximation. The asymptotic approximation is invalid for these data:
per-trajectory TTC<2s counts are heavily tied at zero, and with small
crossing groups (n1 as low as 8) the tie correction collapses the variance
estimate and grossly understates p (printed values fell below the exact
floor 1/C(N, n1)). U and the rank-biserial effect size are unchanged.

Method: exact permutation enumeration when the pooled sample has <= 12
nonzero values; otherwise Monte Carlo permutation (default 10,000,000
resamples) drawn from the tie-group structure via the multivariate
hypergeometric distribution (distributionally identical to permuting the
count vector). When no resample reaches the observed statistic, `p` holds
the conservative bound 1/(draws+1) and `p_is_upper_bound` is True.
"""
from itertools import combinations
from math import lgamma, exp
import numpy as np
from scipy.stats import rankdata

MC_DRAWS_DEFAULT = 10_000_000
EXACT_MAX_NONZERO = 12


def h_off1(B_sim, B_d, tol=0.03):
    diff = float(abs(B_sim - B_d))
    return {"B_sim": float(B_sim), "B_d": float(B_d), "diff": diff,
            "tol": float(tol), "pass": bool(diff < tol)}


def h_off2(tau_hat_d, tau, tol=0.03):
    diff = float(abs(tau_hat_d - tau))
    return {"tau_hat_d": float(tau_hat_d), "tau": float(tau), "diff": diff,
            "tol": float(tol), "pass": bool(diff <= tol)}


def rank_biserial(u, n1, n2):
    """Effect size for Mann-Whitney U (one-sided 'greater')."""
    return float(1.0 - 2.0 * u / (n1 * n2))


def _lchoose(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _exact_perm_p(ranks, n1, obs, pooled):
    """Exact P(rank-sum of n1 random draws >= obs); feasible when few pooled
    values are nonzero (enumerates which nonzero values land in group 1)."""
    nz_ranks = ranks[pooled > 0]
    m = len(nz_ranks)
    if m > EXACT_MAX_NONZERO:
        return None
    N = len(pooled)
    zero_rank = ranks[pooled == 0][0] if (pooled == 0).any() else None
    if zero_rank is None:  # no zeros at all -> nothing gained; fall back to MC
        return None
    log_denom = _lchoose(N, n1)
    tot = 0.0
    for k in range(0, min(m, n1) + 1):
        good = sum(1 for s in combinations(nz_ranks, k)
                   if sum(s) + (n1 - k) * zero_rank >= obs - 1e-9)
        if good:
            tot += good * exp(_lchoose(N - m, n1 - k) - log_denom)
    return float(tot)


def _mc_perm_p(ranks, n1, obs, pooled, draws, seed=0, chunk=1_000_000):
    """Monte Carlo permutation via multivariate hypergeometric over tie
    groups -- fast and memory-light. Returns (exceedances, draws)."""
    vals, inv, cnts = np.unique(pooled, return_inverse=True, return_counts=True)
    grp_rank = np.array([ranks[inv == g][0] for g in range(len(vals))])
    rng = np.random.default_rng(seed)
    ge, done = 0, 0
    while done < draws:
        n = min(chunk, draws - done)
        mat = rng.multivariate_hypergeometric(cnts.astype(np.int64), n1, size=n)
        ge += int(((mat @ grp_rank) >= obs - 1e-9).sum())
        done += n
    return ge, done


def h_off3(crossing_ttc_counts, noncrossing_ttc_counts,
           alpha=0.001, n_comparisons=9, mc_draws=MC_DRAWS_DEFAULT, seed=0):
    alpha_bonf = float(alpha / n_comparisons)
    if len(crossing_ttc_counts) < 2 or len(noncrossing_ttc_counts) < 2:
        return {"U": None, "p": None, "alpha_bonf": alpha_bonf,
                "effect_size_r": None, "pass": False,
                "note": "insufficient_samples"}
    x = np.asarray(crossing_ttc_counts, dtype=float)
    y = np.asarray(noncrossing_ttc_counts, dtype=float)
    n1, n2 = len(x), len(y)
    pooled = np.concatenate([x, y])
    ranks = rankdata(pooled)
    obs = float(ranks[:n1].sum())
    u = obs - n1 * (n1 + 1) / 2.0
    r = rank_biserial(u, n1, n2)

    p_exact = _exact_perm_p(ranks, n1, obs, pooled)
    if p_exact is not None:
        p, method, upper = p_exact, "exact_permutation", False
    else:
        ge, done = _mc_perm_p(ranks, n1, obs, pooled, mc_draws, seed=seed)
        p = (ge + 1) / (done + 1)
        method = f"mc_permutation_{done}"
        upper = (ge == 0)

    return {"U": float(u), "p": float(p), "p_method": method,
            "p_is_upper_bound": bool(upper), "alpha_bonf": alpha_bonf,
            "effect_size_r": r, "pass": bool(p < alpha_bonf)}
