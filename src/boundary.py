"""Boundary B_{N,tau} = (1-tau)-quantile; realized rate; DKW epsilon."""
import numpy as np

def boundary(r_max_list, tau):
    return float(np.quantile(np.asarray(r_max_list, dtype=float), 1.0 - tau))

def realized_rate(r_max_list, b_value):
    arr = np.asarray(r_max_list, dtype=float)
    return float(np.mean(arr > b_value))

def dkw_epsilon(n, delta=0.05):
    return float(np.sqrt(np.log(2.0 / delta) / (2.0 * n)))

def bootstrap_ci(values, statistic, n_boot=5000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for a 1-arg statistic over a list of values."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = statistic(arr[rng.integers(0, n, n)])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi
