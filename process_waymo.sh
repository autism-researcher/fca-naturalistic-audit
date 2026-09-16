#!/usr/bin/env bash
# ==========================================================================
# process_waymo.sh — REPRODUCE the Waymo processing from raw shards, in WSL.
#
# Runs the SAME program that produced the paper's Waymo results
# (run_waymo_experiment.py), with the same flags (--n 5000 --skip-gate-check),
# reading the raw TFRecord shards directly via --waymo-dir. Then it checks the
# new output against your original cached file for reproducibility.
#
# Run from the Ubuntu (WSL) terminal, AFTER the one-time env setup:
#     bash /mnt/d/<PROJECT_DIR>/process_waymo.sh
#
# Writes: results/per_dataset/waymo_features.json  (+ boundaries + verdicts)
# ==========================================================================
set -e

REPO="/mnt/d/<PROJECT_DIR>"
WAYMO_RAW="/path/to/waymo/raw/shards"
CACHED="/path/to/your/original/cached/waymo_features.json"

# --- activate the waymo conda env -----------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate waymo

cd "$REPO"

# --- confirm the Waymo SDK imports ----------------------------------------
echo ">> checking Waymo SDK ..."
python -c "from waymo_open_dataset.protos import scenario_pb2; import tensorflow as tf; print('   Waymo SDK OK, TF', tf.__version__)"

# --- confirm the raw shards are reachable ---------------------------------
echo ">> raw shards under --waymo-dir:"
find "$WAYMO_RAW" -name '*.tfrecord*' | head -3
echo "   ($(find "$WAYMO_RAW" -name '*.tfrecord*' | wc -l) shards total)"

# --- THE ORIGINAL PROGRAM: exact command from your WAYMO_WSL_SETUP.md ------
echo ">> reproducing Waymo run (n=5000, pre-registered SEED) ..."
python scripts/run_waymo_experiment.py \
    --n 5000 \
    --skip-gate-check \
    --waymo-dir "$WAYMO_RAW"

# --- reproducibility check vs your original cached file -------------------
echo ">> reproducibility check vs your original cached waymo_features.json:"
python - <<'PY'
import json, numpy as np
W=np.array([0.08,0.10,0.10,0.08,0.12,0.25,0.15,0.06]); BSIM={0.10:0.5116,0.15:0.4885,0.20:0.4733}
def summ(p):
    d=json.load(open(p)); rm=np.array([float(np.max(np.asarray(t['features'],float)@W)) for t in d['trajectories']])
    return d.get('n',len(d['trajectories'])), {tau:(float(np.quantile(rm,1-tau)),float(np.mean(rm>BSIM[tau]))) for tau in (0.10,0.15,0.20)}
n_new,s_new = summ("results/per_dataset/waymo_features.json")
print(f"   NEW run:      n={n_new}")
for tau in (0.10,0.15,0.20):
    print(f"      tau={tau:.2f}  B_d={s_new[tau][0]:.4f}  tau_hat={s_new[tau][1]:.4f}")
try:
    n_old,s_old = summ("/path/to/your/original/cached/waymo_features.json")
    print(f"   ORIGINAL:     n={n_old}   (match: {n_new==n_old})")
except Exception as e:
    print("   (original cached file not readable here:", e, ")")
print("   Paper Table II Waymo: B_d 0.375/0.346/0.326  tau_hat 0.011/0.020/0.026")
PY
ls -la results/per_dataset/waymo_features.json
