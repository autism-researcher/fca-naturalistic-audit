"""HighD feature extractor.

Per pre-reg §7: native 25 Hz; velocity / acceleration provided directly
(no Butterworth needed); steering proxy via yaw rate from atan2(vy, vx);
lane offset via recordingMeta lane markings; TTC recomputed from dhw and
closing speed (cross-validated against the provided ttc column as a
sanity check).

Each HighD recording ships as three CSVs:
    NN_tracks.csv          one row per (vehicle, frame)
    NN_tracksMeta.csv      one row per vehicle (class, drivingDirection, ...)
    NN_recordingMeta.csv   one row per recording (frameRate, lane markings, ...)

tracks.csv columns (release v1):
    frame, id, x, y, width, height,
    xVelocity, yVelocity, xAcceleration, yAcceleration,
    frontSightDistance, backSightDistance,
    dhw, thw, ttc, precedingXVelocity,
    precedingId, followingId,
    leftPrecedingId, leftAlongsideId, leftFollowingId,
    rightPrecedingId, rightAlongsideId, rightFollowingId,
    laneId

Function signature MUST match the NGSIM and Waymo extractors:
    extract_features(traj_df, recording_meta=None, frame_index=None)
        -> (features [T,8], raw_ttc [T], ok_flag, reason)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from ..utils import (load_weights, ema_filter, rolling_variance,
                     normalize_upper, normalize_inverse_range, is_eligible)

FS_HZ = 25.0
DT = 1.0 / FS_HZ
DENSITY_RADIUS_M = 35.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def list_recordings(highd_dir):
    """Return sorted list of recording IDs (as ints) found under highd_dir.

    A recording is "present" if all three files exist:
    NN_tracks.csv, NN_tracksMeta.csv, NN_recordingMeta.csv.
    """
    p = Path(highd_dir)
    rids = []
    for tracks_csv in sorted(p.glob("*_tracks.csv")):
        rid_str = tracks_csv.name.split("_", 1)[0]
        try:
            rid = int(rid_str)
        except ValueError:
            continue
        meta = p / f"{rid_str}_tracksMeta.csv"
        rec = p / f"{rid_str}_recordingMeta.csv"
        if meta.exists() and rec.exists():
            rids.append(rid)
    return rids


def load_highd_recording(highd_dir, recording_id):
    """Load (tracks, tracks_meta, recording_meta_dict) for one HighD recording."""
    p = Path(highd_dir)
    rid = f"{int(recording_id):02d}"
    tracks = pd.read_csv(p / f"{rid}_tracks.csv")
    tracks_meta = pd.read_csv(p / f"{rid}_tracksMeta.csv")
    rmeta_df = pd.read_csv(p / f"{rid}_recordingMeta.csv")
    rmeta = rmeta_df.iloc[0].to_dict()
    return tracks, tracks_meta, rmeta


def parse_lane_markings(recording_meta):
    """Return (upper_markings, lower_markings) as sorted float arrays.

    HighD stores lane markings as semicolon-separated y-coordinates,
    e.g., "16.62;20.05;23.45;26.95".
    """
    upper = np.array(sorted(
        float(v) for v in str(recording_meta["upperLaneMarkings"]).split(";") if v.strip()
    ))
    lower = np.array(sorted(
        float(v) for v in str(recording_meta["lowerLaneMarkings"]).split(";") if v.strip()
    ))
    return upper, lower


def build_frame_index(tracks_df):
    """Pre-build frame -> Nx2 array of (x, y) for fast density lookup.

    Mirrors the NGSIM `build_frame_index` helper. Call once per recording
    and pass the result as `frame_index` to extract_features.
    """
    out = {}
    for frm, g in tracks_df.groupby("frame"):
        out[int(frm)] = np.column_stack([
            g["x"].to_numpy(float),
            g["y"].to_numpy(float),
        ])
    return out


def trajectories(tracks_df, min_frames=None):
    """Yield (vehicle_id, traj_df) for each vehicle, sorted by frame.

    min_frames defaults to 5 seconds at 25 Hz (= 125 frames), matching
    the pre-reg eligibility floor.
    """
    if min_frames is None:
        min_frames = int(5 * FS_HZ)
    for vid, g in tracks_df.groupby("id"):
        g = g.sort_values("frame").reset_index(drop=True)
        if len(g) >= min_frames:
            yield int(vid), g


# ---------------------------------------------------------------------------
# Lane-offset helper
# ---------------------------------------------------------------------------
def _lane_center_for_y(y_arr, upper, lower):
    """For each y_t, locate the lane segment it falls into and return its center.

    Upper lanes (lower y-values in HighD) and lower lanes have separate
    marking arrays. We pick the set that contains (or is closest to) each
    point, then take the midpoint of the bracketing pair of markings.
    """
    y_arr = np.asarray(y_arr, dtype=float)
    out = np.zeros_like(y_arr)

    have_upper = upper.size >= 2
    have_lower = lower.size >= 2
    if not (have_upper or have_lower):
        return out  # no marking info; lane offset stays 0

    for i, yi in enumerate(y_arr):
        if have_upper and upper.min() - 1.0 <= yi <= upper.max() + 1.0:
            markings = upper
        elif have_lower and lower.min() - 1.0 <= yi <= lower.max() + 1.0:
            markings = lower
        else:
            # Outside both bands; assign to whichever set the point is closer to.
            d_u = abs(yi - upper.mean()) if have_upper else np.inf
            d_l = abs(yi - lower.mean()) if have_lower else np.inf
            markings = upper if d_u < d_l else lower
        idx = int(np.clip(np.searchsorted(markings, yi) - 1, 0, len(markings) - 2))
        out[i] = 0.5 * (markings[idx] + markings[idx + 1])
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(traj_df, recording_meta=None, frame_index=None):
    """Compute (T, 8) normalized features and (T,) raw TTC for one HighD trajectory.

    Parameters
    ----------
    traj_df : DataFrame
        Rows of NN_tracks.csv for a single vehicle, sorted by frame.
    recording_meta : dict or None
        Row of NN_recordingMeta.csv as a dict; needs `upperLaneMarkings`,
        `lowerLaneMarkings`. If None, lane offset is filled with 0.
    frame_index : dict or None
        `frame -> Nx2 (x, y)` array, built via `build_frame_index`.
        If None, density stays 0.

    Returns
    -------
    features : (T, 8) array in [0, 1] or None
    raw_ttc  : (T,) array in seconds or None
    ok       : bool
    reason   : str
    """
    weights, bounds = load_weights()

    n = len(traj_df)
    if n < 5 * FS_HZ:
        return None, None, False, "too_short"

    x = traj_df["x"].to_numpy(float)
    y = traj_df["y"].to_numpy(float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None, None, False, "non_finite_position"

    # Velocity / acceleration are provided directly; pre-reg explicitly
    # waives Butterworth pre-filtering for HighD.
    vx = traj_df["xVelocity"].to_numpy(float)
    vy = traj_df["yVelocity"].to_numpy(float)
    ax = traj_df["xAcceleration"].to_numpy(float)
    ay = traj_df["yAcceleration"].to_numpy(float)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    a = np.sqrt(ax ** 2 + ay ** 2)

    # Pre-reg exclusion: trajectories with >10% of ticks above 10 m/s^2 are
    # unphysical and excluded (matches NGSIM rule).
    if np.mean(np.abs(a) > 10.0) > 0.10:
        return None, None, False, "accel_above_10"

    jerk_raw = np.gradient(a, DT)
    jerk = ema_filter(np.abs(jerk_raw), tau_s=0.40, dt=DT)

    # Steer proxy: yaw rate from instantaneous heading
    heading = np.arctan2(vy, vx)
    yaw_rate = np.gradient(np.unwrap(heading), DT)
    steer_var = rolling_variance(yaw_rate, window=int(FS_HZ))  # 1 s window

    # Lane offset via recordingMeta lane markings
    if recording_meta is not None and "upperLaneMarkings" in recording_meta:
        upper, lower = parse_lane_markings(recording_meta)
        lane_center = _lane_center_for_y(y, upper, lower)
        lane_offset = np.abs(y - lane_center)
    else:
        lane_offset = np.zeros(n)

    # Headway: dhw is in meters; <=0 means "no valid leader at this tick".
    dhw = traj_df["dhw"].to_numpy(float)
    headway = np.where(dhw > 0, dhw, 60.0)

    # ---- TTC: recompute from dhw + closing speed (mirrors NGSIM/Waymo) ----
    preceding = traj_df["precedingId"].to_numpy(int)
    v_lead_x = traj_df["precedingXVelocity"].to_numpy(float)

    # Vehicles in the same lane travel in the same direction, so taking
    # |vx_ego| - |vx_lead| gives a sign-correct closing speed regardless
    # of whether the vehicle is in an upper (leftward) or lower (rightward) lane.
    closing_speed = np.abs(vx) - np.abs(v_lead_x)
    closing_speed = np.where(closing_speed > 0.05, closing_speed, np.nan)

    # Dilated leader-transition mask matches NGSIM: any frame within +-5 of a
    # leader-ID change is excluded from TTC computation.
    leader_change = np.concatenate([[True], preceding[1:] != preceding[:-1]])
    DILATE = 5
    transition_dilated = np.zeros(n, dtype=bool)
    for shift in range(-DILATE, DILATE + 1):
        transition_dilated |= np.roll(leader_change, shift)

    ttc_raw = np.full(n, np.inf, dtype=float)
    closing_mask = (
        (preceding > 0) & (~np.isnan(closing_speed)) & (~transition_dilated)
    )
    ttc_raw[closing_mask] = (
        dhw[closing_mask] / np.maximum(closing_speed[closing_mask], 0.05)
    )
    ttc_raw = np.where(np.isfinite(ttc_raw), np.clip(ttc_raw, 0.0, 1e6), np.inf)

    # ---- Density: count vehicles in 35 m radius at each frame ----
    density = np.zeros(n)
    if frame_index is not None:
        for i, frm in enumerate(traj_df["frame"].to_numpy()):
            key = int(frm)
            if key in frame_index:
                pts = frame_index[key]
                d = np.linalg.norm(pts - np.array([x[i], y[i]]), axis=1)
                # subtract 1 for the ego itself, which is also in the frame index
                density[i] = max(0, int(np.sum(d <= DENSITY_RADIUS_M)) - 1)

    # ---- Normalize to [0, 1] ----
    f1 = normalize_upper(speed,       bounds["speed"]["max"])
    f2 = normalize_upper(a,           bounds["acceleration"]["max"])
    f3 = normalize_upper(jerk,        bounds["jerk"]["max"])
    f4 = normalize_upper(steer_var,   bounds["steer_var"]["max"])
    f5 = normalize_upper(lane_offset, bounds["lane_offset"]["max"])
    f6 = normalize_inverse_range(ttc_raw, bounds["ttc"]["min"],     bounds["ttc"]["max"])
    f7 = normalize_inverse_range(headway, bounds["headway"]["min"], bounds["headway"]["max"])
    f8 = normalize_upper(density,     bounds["density"]["max"])

    features = np.stack([f1, f2, f3, f4, f5, f6, f7, f8], axis=1)
    ok, reason = is_eligible(features, min_length_s=5.0, fs_hz=FS_HZ)
    return features, ttc_raw, ok, reason


# ---------------------------------------------------------------------------
# Optional sanity check: compare our TTC to HighD's provided column
# ---------------------------------------------------------------------------
def compare_ttc_with_provided(traj_df, our_ttc_raw):
    """Return summary stats comparing our recomputed TTC to HighD's `ttc` column.

    HighD stores `ttc = 0` when there is no leader or no closing. We compare
    only on rows where both sources are finite and positive.
    """
    if "ttc" not in traj_df.columns:
        return None
    provided = traj_df["ttc"].to_numpy(float)
    mask = (
        np.isfinite(our_ttc_raw) & (our_ttc_raw > 0) & (our_ttc_raw < 1e5)
        & np.isfinite(provided) & (provided > 0) & (provided < 1e5)
    )
    if mask.sum() < 5:
        return {"n_overlap": int(mask.sum()), "note": "insufficient_overlap"}
    diff = our_ttc_raw[mask] - provided[mask]
    return {
        "n_overlap": int(mask.sum()),
        "mean_abs_diff_s": float(np.mean(np.abs(diff))),
        "median_abs_diff_s": float(np.median(np.abs(diff))),
        "max_abs_diff_s": float(np.max(np.abs(diff))),
    }
