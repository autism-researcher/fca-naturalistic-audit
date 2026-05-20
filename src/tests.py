"""H_OFF1, H_OFF2, H_OFF3 implementations.  Outputs are JSON-safe."""
import numpy as np
from scipy.stats import mannwhitneyu

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

def h_off3(crossing_ttc_counts, noncrossing_ttc_counts,
           alpha=0.001, n_comparisons=9):
    if len(crossing_ttc_counts) < 2 or len(noncrossing_ttc_counts) < 2:
        return {"U": None, "p": None,
                "alpha_bonf": float(alpha / n_comparisons),
                "effect_size_r": None, "pass": False,
                "note": "insufficient_samples"}
    u, p = mannwhitneyu(crossing_ttc_counts, noncrossing_ttc_counts,
                        alternative="greater")
    alpha_bonf = float(alpha / n_comparisons)
    r = rank_biserial(u, len(crossing_ttc_counts), len(noncrossing_ttc_counts))
    return {"U": float(u), "p": float(p), "alpha_bonf": alpha_bonf,
            "effect_size_r": r, "pass": bool(p < alpha_bonf)}
