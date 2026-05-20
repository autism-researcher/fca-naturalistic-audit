"""Stage 3: hand-validate 10 trajectories per dataset.
Print raw-to-normalized feature ranges; flag impossible values.
Does NOT compute R(x) or any boundary."""
import sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.ngsim import extract_features as ngsim_extract, trajectories as ngsim_trajs

def summarise(features, ttc_raw, tag):
    if features is None:
        print(f"  {tag}: REJECTED")
        return
    fnames = ["speed","accel","jerk","steer_var","lane_off","ttc","headway","density"]
    print(f"  {tag}: T={features.shape[0]}")
    for i, name in enumerate(fnames):
        col = features[:, i]
        warn = " CLIPPED" if (col.min() == 0 and col.max() == 1) else ""
        print(f"    {name:9s}: [{col.min():.3f}, {col.max():.3f}]{warn}")

def validate_ngsim(n_to_pick=10):
    csvs = sorted(glob.glob(str(ROOT / "data/ngsim/*.csv")))
    if not csvs:
        print("[NGSIM] no data, skipping")
        return
    df = pd.read_csv(csvs[0])
    # heuristic: first n_to_pick distinct vehicles with at least 5 s of data
    picked = []
    for tid, g in ngsim_trajs(df):
        if len(g) < 50:  # 5 s at 10 Hz
            continue
        picked.append((tid, g))
        if len(picked) >= n_to_pick:
            break
    print(f"[NGSIM] Validating {len(picked)} trajectories from {csvs[0]}:")
    for tid, g in picked:
        loc, vid = tid
        feats, ttc, ok, reason = ngsim_extract(g, all_frames_df=None)
        tag = f"{loc} veh {vid} ({reason})"
        summarise(feats, ttc, tag)

if __name__ == "__main__":
    validate_ngsim()
    # Waymo and HighD: same pattern; uncomment once Waymo env / HighD data is ready.
