"""Shared utilities: eligibility, normalization helpers, EMA, Butterworth."""
import json
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED = 42  # locked in pre-registration

def load_weights():
    with open(REPO_ROOT / "carla_weights.json") as f:
        cfg = json.load(f)
    w = np.array(cfg["weights"], dtype=float)
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights sum={w.sum()}"
    return w, cfg["bounds"]

def load_B_sim():
    with open(REPO_ROOT / "paper2_constants.json") as f:
        cfg = json.load(f)
    return {float(k): float(v) for k, v in cfg["B_sim"].items()}

def normalize_upper(x, x_max):
    return np.clip(np.abs(x) / x_max, 0.0, 1.0)

def normalize_inverse_range(x, x_min, x_max):
    """For TTC/headway: small value = high risk."""
    x = np.asarray(x, dtype=float)
    out = np.where(np.isnan(x), 0.0, (x_max - x) / (x_max - x_min))
    return np.clip(out, 0.0, 1.0)

def ema_filter(x, tau_s, dt):
    """Exponential moving average with time constant tau_s."""
    alpha = dt / (tau_s + dt)
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y

def butterworth_position(x, fs_hz, cutoff_hz=2.0, order=3):
    """Third-order low-pass Butterworth on position; NGSIM preprocessing."""
    if len(x) < 4 * order:
        return x  # too short to filter
    b, a = butter(order, cutoff_hz / (0.5 * fs_hz), btype="low")
    return filtfilt(b, a, x)

def rolling_variance(x, window):
    """Centered rolling variance."""
    n = len(x)
    half = window // 2
    out = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.var(x[lo:hi]) if hi - lo > 1 else 0.0
    return out

def is_eligible(traj_features, min_length_s, fs_hz):
    """Length >= min_length_s, finite values."""
    if traj_features is None or len(traj_features) == 0:
        return False, "empty"
    if len(traj_features) < min_length_s * fs_hz:
        return False, "too_short"
    if not np.all(np.isfinite(traj_features)):
        return False, "non_finite"
    return True, "ok"
