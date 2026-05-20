"""Stage 5a: feature extraction at scale, NGSIM + Waymo.

Samples up to --n eligible trajectories per dataset using the pre-registered
SEED, runs the dataset-specific extractor, and dumps per-trajectory features
+ raw TTC + IDs to results/per_dataset/{dataset}_features.json.

ONLY RUN AFTER GATE 2 (Git tag `pipeline-frozen-pre-confirmatory`).
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import SEED

# --------------------------- NGSIM ---------------------------
def extract_ngsim_at_scale(n_max, output_path):
    from src.features.ngsim import (extract_features as ngsim_extract, trajectories as ngsim_trajs, build_frame_index as ngsim_build_index)
    csvs = sorted(glob.glob(str(ROOT / "data/ngsim/*.csv")))
    if not csvs:
        print("[NGSIM] no data, skipping"); return
    df = pd.read_csv(csvs[0])
    print("[NGSIM] building per-frame spatial index for density (~30 s) ...")
    frame_index = ngsim_build_index(df)
    print(f"[NGSIM] frame_index has {len(frame_index)} (Location, Frame_ID) entries")
    rng = np.random.default_rng(SEED)
    all_tids = [tid for tid, g in ngsim_trajs(df) if len(g) >= 50]
    print(f"[NGSIM] eligible-by-length pool: {len(all_tids)} trajectories")
    # rng.shuffle accepts numpy arrays of objects; use list-shuffle for tuples.
    idx = np.arange(len(all_tids)); rng.shuffle(idx)
    out = []
    n_excluded = {"too_short":0, "non_finite_position":0, "accel_above_10":0, "other":0}
    for i in idx:
        if len(out) >= n_max: break
        tid = all_tids[i]
        loc, vid = tid
        g = df[(df["Location"] == loc) & (df["Vehicle_ID"] == vid)].sort_values("Frame_ID").reset_index(drop=True)
        feats, ttc, ok, reason = ngsim_extract(g, frame_index=frame_index)
        if not ok or feats is None:
            n_excluded[reason if reason in n_excluded else "other"] = (
                n_excluded.get(reason, 0) + 1
            ); continue
        out.append({
            "trajectory_id": f"{loc}__{int(vid)}",
            "T": int(feats.shape[0]),
            "features": feats.tolist(),
            "ttc_raw": ttc.tolist() if ttc is not None else None,
        })
    print(f"[NGSIM] eligible after extraction: {len(out)}; excluded: {n_excluded}")
    with open(output_path, "w") as f:
        json.dump({"dataset": "ngsim", "seed": SEED, "n": len(out),
                   "excluded": n_excluded, "trajectories": out}, f)
    print(f"[NGSIM] wrote {output_path}")

# --------------------------- Waymo ---------------------------
def extract_waymo_at_scale(n_max, output_path):
    """Walks data/waymo/**/*.tfrecord* (handles v1.2/training and v1.2/validation)."""
    shards = sorted((ROOT / "data/waymo").rglob("*.tfrecord*"))
    if not shards:
        print("[Waymo] no shards in data/waymo/, skipping"); return
    print(f"[Waymo] found {len(shards)} shards across the v1.2 tree")
    try:
        from src.features.waymo import scenarios_from_shard, extract_features as waymo_extract
    except ImportError as e:
        print(f"[Waymo] cannot import waymo extractor: {e}")
        print("[Waymo] activate the conda env with waymo-open-dataset-tf-2-12-0 installed.")
        return
    rng = np.random.default_rng(SEED)
    shard_order = list(range(len(shards)))
    rng.shuffle(shard_order)
    out = []
    n_excluded = {"no_ego":0, "too_short":0, "valid_too_short":0, "accel_above_10":0, "other":0}
    t0 = time.time()
    for si in shard_order:
        if len(out) >= n_max: break
        shard = shards[si]
        print(f"  [Waymo] shard {si+1}/{len(shards)}: {shard.name}  (running tally: {len(out)})")
        try:
            for sc in scenarios_from_shard(shard):
                if len(out) >= n_max: break
                feats, ttc, ok, reason = waymo_extract(sc)
                if not ok or feats is None:
                    n_excluded[reason if reason in n_excluded else "other"] = (
                        n_excluded.get(reason, 0) + 1
                    ); continue
                out.append({
                    "trajectory_id": sc.scenario_id,
                    "T": int(feats.shape[0]),
                    "features": feats.tolist(),
                    "ttc_raw": ttc.tolist() if ttc is not None else None,
                })
        except Exception as e:
            print(f"  [Waymo] shard parse error ({e}); skipping")
            continue
    print(f"[Waymo] eligible after extraction: {len(out)}; excluded: {n_excluded}; "
          f"elapsed: {time.time()-t0:.0f}s")
    with open(output_path, "w") as f:
        json.dump({"dataset": "waymo", "seed": SEED, "n": len(out),
                   "excluded": n_excluded, "trajectories": out}, f)
    print(f"[Waymo] wrote {output_path}")

# --------------------------- HighD ---------------------------
def extract_highd_at_scale(n_max, output_path, highd_dir=None):
    """Walk recordings in data/highd/ (or --highd-dir) and extract eligible vehicles.

    Each HighD recording = three CSVs (NN_tracks.csv, NN_tracksMeta.csv,
    NN_recordingMeta.csv). SEED-shuffled order at both recording and intra-
    recording level for reproducibility.
    """
    from src.features.highd import (
        list_recordings, load_highd_recording, build_frame_index,
        trajectories as highd_trajs, extract_features as highd_extract,
    )
    base = Path(highd_dir) if highd_dir else (ROOT / "data" / "highd")
    if not base.exists():
        print(f"[HighD] directory not found: {base}; skipping"); return
    rids = list_recordings(base)
    if not rids:
        print(f"[HighD] no recordings under {base}; skipping"); return
    print(f"[HighD] found {len(rids)} recordings: {rids[0]:02d}..{rids[-1]:02d}")

    rng = np.random.default_rng(SEED)
    rid_order = list(rids); rng.shuffle(rid_order)
    out = []
    n_excluded = {"too_short":0, "non_finite_position":0, "accel_above_10":0, "other":0}
    t0 = time.time()
    for ri, rid in enumerate(rid_order):
        if len(out) >= n_max: break
        print(f"  [HighD] recording {ri+1}/{len(rid_order)}: {rid:02d}  (running tally: {len(out)})")
        try:
            tracks, _meta, rmeta = load_highd_recording(base, rid)
        except Exception as e:
            print(f"  [HighD] load error on {rid:02d} ({e}); skipping")
            continue
        frame_index = build_frame_index(tracks)
        vids = [vid for vid, _ in highd_trajs(tracks)]
        idx = np.arange(len(vids)); rng.shuffle(idx)
        for j in idx:
            if len(out) >= n_max: break
            vid = vids[j]
            g = tracks[tracks["id"] == vid].sort_values("frame").reset_index(drop=True)
            feats, ttc, ok, reason = highd_extract(g, recording_meta=rmeta, frame_index=frame_index)
            if not ok or feats is None:
                n_excluded[reason if reason in n_excluded else "other"] = (
                    n_excluded.get(reason, 0) + 1
                ); continue
            out.append({
                "trajectory_id": f"{int(rid):02d}__{int(vid)}",
                "T": int(feats.shape[0]),
                "features": feats.tolist(),
                "ttc_raw": ttc.tolist() if ttc is not None else None,
            })
    print(f"[HighD] eligible after extraction: {len(out)}; excluded: {n_excluded}; "
          f"elapsed: {time.time()-t0:.0f}s")
    with open(output_path, "w") as f:
        json.dump({"dataset": "highd", "seed": SEED, "n": len(out),
                   "excluded": n_excluded, "trajectories": out}, f)
    print(f"[HighD] wrote {output_path}")

# --------------------------- main ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="max trajectories per dataset")
    ap.add_argument("--datasets", nargs="+", default=["ngsim"],
                    choices=["ngsim","waymo","highd"])
    ap.add_argument("--highd-dir", type=str, default=None,
                    help="Override location of HighD CSVs (default: data/highd/).")
    args = ap.parse_args()
    out_dir = ROOT / "results/per_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    if "ngsim" in args.datasets:
        extract_ngsim_at_scale(args.n, out_dir / "ngsim_features.json")
    if "waymo" in args.datasets:
        extract_waymo_at_scale(args.n, out_dir / "waymo_features.json")
    if "highd" in args.datasets:
        extract_highd_at_scale(args.n, out_dir / "highd_features.json",
                               highd_dir=args.highd_dir)
