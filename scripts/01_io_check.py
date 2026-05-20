"""Stage 1: I/O sanity check. Load one trajectory per dataset and print arrays.
NO feature computation. NO risk values."""
import sys, glob
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def check_ngsim():
    csvs = sorted(glob.glob(str(ROOT / "data/ngsim/*.csv")))
    if not csvs:
        print("[NGSIM] No CSVs in data/ngsim/.  Skipping.")
        return
    df = pd.read_csv(csvs[0])
    print(f"[NGSIM] File: {csvs[0]}  rows: {len(df)}")
    print(f"[NGSIM] Columns: {df.columns.tolist()}")
    vid = df["Vehicle_ID"].iloc[0]
    g = df[df["Vehicle_ID"] == vid].sort_values("Frame_ID")
    print(f"[NGSIM] First trajectory: Vehicle_ID={vid}  length={len(g)} frames")
    print(g[["Frame_ID","Local_X","Local_Y","v_Vel","v_Acc","Lane_ID"]].head())

def check_waymo():
    shards = sorted((ROOT / "data/waymo").rglob("*.tfrecord*"))
    shards = [str(x) for x in shards]
    if not shards:
        print("[Waymo] No TFRecord shards in data/waymo/.  Skipping.")
        return
    print(f"[Waymo] Found {len(shards)} shard(s); first: {shards[0]}")
    try:
        from src.features.waymo import scenarios_from_shard, pick_ego_track
        for sc in scenarios_from_shard(shards[0]):
            ego = pick_ego_track(sc)
            print(f"[Waymo] First scenario id={sc.scenario_id}  tracks={len(sc.tracks)}")
            if ego is not None:
                print(f"[Waymo] Ego id={ego.id}  states={len(ego.states)}  "
                      f"obj_type={ego.object_type}")
            break
    except ImportError as e:
        print(f"[Waymo] waymo-open-dataset library not installed in this env.")
        print(f"[Waymo] Path-walking works (good); install the library in a")
        print(f"[Waymo] Python 3.10 env to actually parse shards.")

def check_highd():
    csvs = sorted(glob.glob(str(ROOT / "data/highd/*_tracks.csv")))
    if not csvs:
        print("[HighD] No tracks CSVs in data/highd/.  (Awaiting license.)")
        return
    df = pd.read_csv(csvs[0])
    print(f"[HighD] File: {csvs[0]}  rows: {len(df)}")
    print(f"[HighD] Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    check_ngsim()
    print()
    check_waymo()
    print()
    check_highd()
