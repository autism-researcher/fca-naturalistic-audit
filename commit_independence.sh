#!/usr/bin/env bash
# Commit full independence of the Paper 3 repo from Paper 2.
# Usage (Git Bash / WSL):  cd /d/<PROJECT_DIR>/fca-naturalistic-audit && bash commit_independence.sh
set -e

# 1) remove the orphaned Paper 2-named file (nothing reads it anymore)
git rm --ignore-unmatch paper2_constants.json

# 2) stage the self-contained files
git add supervisor_spec.json compute_B.py \
        carla_weights.json src/utils.py README.md \
        deviations_log.md scripts/10_generate_flow_diagram.py \
        data/sim_calibration/sim_normal_peak_risk.csv \
        data/sim_calibration/simulator_episode_log.csv \
        data/sim_calibration/make_sim_calibration.py

# 3) commit
git commit -m "Make Paper 3 fully self-contained: ship raw simulator log + regeneration script; recompute B_sim locally; remove all Paper 2 references"

echo
echo "Done. Verify zero Paper 2 refs:  git grep -i 'paper.\?2' -- . ':(exclude).git'"
echo "Review:  git show --stat HEAD"
echo "Push (if remote):  git push"
