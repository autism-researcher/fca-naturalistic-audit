"""Composite risk and per-trajectory R_max."""
import numpy as np

def composite_risk(features_TxF, weights_F):
    """R(x_t) = sum_i w_i * f_i(x_t), per timestep. Output in [0,1]."""
    features_TxF = np.asarray(features_TxF, dtype=float)
    weights_F = np.asarray(weights_F, dtype=float)
    return features_TxF @ weights_F

def r_max(features_TxF, weights_F):
    return float(np.max(composite_risk(features_TxF, weights_F)))
