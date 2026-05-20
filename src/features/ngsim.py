"""NGSIM feature extractor.

Schema (US-101 / I-80 release): Vehicle_ID, Frame_ID, Total_Frames, Global_Time,
Local_X, Local_Y, Global_X, Global_Y, v_Length, v_Width, v_Class, v_Vel, v_Acc,
Lane_ID, Preceding, Following, Space_Headway, Time_Headway, Location.

NGSIM is 10 Hz. Lane width on US-101 ~= 3.66 m.
Pre-reg §7 mandates: third-order Butterworth (2 Hz cutoff) on position before
differentiation; exclude trajectories with >10% missing values or |a|>10 m/s^2.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from ..utils import (load_weights, butterworth_position, ema_filter,
                     rolling_variance, normalize_upper,
                     normalize_inverse_range, is_eligible)

FS_HZ = 10.0
DT = 1.0 / FS_HZ
LANE_WIDTH_M = 3.66
DENSITY_RADIUS_M = 35.0
TTC_FORWARD_CONE_DEG = 5.0

def load_ngsim(csv_path):
    """Load the full NGSIM CSV. Lazy: caller filters by Location/Vehicle_ID."""
    return pd.read_csv(csv_path)

def build_frame_index(df):
    """Pre-build a (Location, Frame_ID) -> Nx2 array of (Local_X_m, Local_Y_m).

    Speeds up density lookup from O(n_rows) per frame to O(1).  Call once
    in your driver script and pass the result as `frame_index` to extract_features.
    """
    FT2M = 0.3048
    out = {}
    for (loc, frm), g in df.groupby(["Location", "Frame_ID"]):
        out[(loc, int(frm))] = np.column_stack([
            g["Local_X"].to_numpy(float) * FT2M,
            g["Local_Y"].to_numpy(float) * FT2M,
        ])
    return out


def trajectories(df, location=None):
    """Yield ((Location, Vehicle_ID), traj_df) sorted by Frame_ID.

    IMPORTANT: NGSIM Vehicle_ID is unique only WITHIN a Location.
    US-101 vehicle 1 and I-80 vehicle 1 are different cars.
    We always group on (Location, Vehicle_ID) to avoid merging trajectories.
    """
    if location is not None:
        df = df[df["Location"] == location]
    for (loc, vid), g in df.groupby(["Location", "Vehicle_ID"]):
        yield (loc, vid), g.sort_values("Frame_ID").reset_index(drop=True)

def extract_features(traj_df, all_frames_df=None, frame_index=None):
    """Compute (T, 8) normalized features and (T,) raw TTC for one NGSIM trajectory.

    all_frames_df is the full DataFrame for the same Location (needed for density
    and leader lookup); if None, density and TTC are filled with 0/inf respectively.
    Returns (features [T,8] in [0,1], raw_ttc [T] in seconds, ok_flag, reason).
    """
    weights, bounds = load_weights()

    # Dedupe duplicate Frame_IDs (NGSIM occasionally has these as a data-cleaning artifact)
    traj_df = traj_df.drop_duplicates(subset="Frame_ID", keep="first").reset_index(drop=True)
    n = len(traj_df)
    if n < 5 * FS_HZ:
        return None, None, False, "too_short" 

    # NGSIM units: feet to meters (1 ft = 0.3048 m); v_Vel and v_Acc are in ft/s and ft/s^2.
    # Convert immediately.
    FT2M = 0.3048
    x = traj_df["Local_X"].to_numpy(float) * FT2M
    y = traj_df["Local_Y"].to_numpy(float) * FT2M
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return None, None, False, "non_finite_position"

    # Per pre-reg: Butterworth filter on position before differentiating
    x_f = butterworth_position(x, FS_HZ, cutoff_hz=2.0, order=3)
    y_f = butterworth_position(y, FS_HZ, cutoff_hz=2.0, order=3)

    # Use Frame_ID-based time array so any non-contiguous gaps inside a single
    # vehicle's record still yield correct derivatives.
    t = traj_df["Frame_ID"].to_numpy(float) * DT
    if len(t) < 3:
        return None, None, False, "too_short"
    vx = np.gradient(x_f, t)
    vy = np.gradient(y_f, t)
    speed = np.sqrt(vx ** 2 + vy ** 2)

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    a = np.sqrt(ax ** 2 + ay ** 2)

    # Pre-reg exclusion: trajectories with >10% of ticks above 10 m/s^2 are
    # unphysical and excluded.  Single-tick spikes from filter boundary effects
    # are tolerated; persistent unphysical accel is not.
    if np.mean(np.abs(a) > 10.0) > 0.10:
        return None, None, False, "accel_above_10"

    jerk_raw = np.gradient(a, DT)
    jerk = ema_filter(np.abs(jerk_raw), tau_s=0.40, dt=DT)

    # Steer proxy: yaw rate from heading
    heading = np.arctan2(vy, vx)
    yaw_rate = np.gradient(np.unwrap(heading), DT)
    steer_var = rolling_variance(yaw_rate, window=int(FS_HZ))  # 1 s window

    # Lane offset: Local_X (m) - lane_center; lane_center = (Lane_ID - 0.5) * LANE_WIDTH_M
    lane_id = traj_df["Lane_ID"].to_numpy(float)
    lane_center = (lane_id - 0.5) * LANE_WIDTH_M
    lane_offset = np.abs(x - lane_center)

    # Headway (m): provided as Space_Headway in ft
    headway = traj_df["Space_Headway"].to_numpy(float) * FT2M
    headway = np.where(headway > 0, headway, 60.0)  # 0 means no leader -> max

    # TTC computation, NGSIM-specific guards:
    #   - Smooth headway with causal EMA (no ringing across discontinuities).
    #   - Mask leader-identity transitions with +-5 frame dilation (filter settling).
    #   - Use pre-reg threshold of 0.05 m/s for "is closing".
    # When no leader OR not closing -> TTC = +inf (normalized to 0 = safe).
    preceding = traj_df["Preceding"].to_numpy(int)
    headway_smooth = ema_filter(headway, tau_s=0.50, dt=DT)
    closing_speed = -np.gradient(headway_smooth, DT)     # positive when closing
    closing_speed = np.where(closing_speed > 0.05, closing_speed, np.nan)

    # Dilated leader-transition mask: any frame within +-5 of a leader change
    # has filter settling artifacts and is excluded from TTC computation.
    leader_change = np.concatenate([[True], preceding[1:] != preceding[:-1]])
    DILATE = 5
    transition_dilated = np.zeros(n, dtype=bool)
    for shift in range(-DILATE, DILATE + 1):
        transition_dilated |= np.roll(leader_change, shift)

    ttc_raw = np.full(n, np.inf, dtype=float)
    closing_mask = (preceding > 0) & (~np.isnan(closing_speed)) & (~transition_dilated)
    ttc_raw[closing_mask] = (
        headway[closing_mask] / np.maximum(closing_speed[closing_mask], 0.05)
    )
    ttc_raw = np.where(np.isfinite(ttc_raw), np.clip(ttc_raw, 0.0, 1e6), np.inf)

    # Density: count vehicles in 35 m radius at each frame.  Two paths:
    # (a) frame_index dict from build_frame_index() -- O(1) per frame, used in script 03.
    # (b) None -- skipped (density stays 0); used by validation script 02 for speed.
    density = np.zeros(n)
    if frame_index is not None:
        ego_loc = traj_df["Location"].iloc[0] if "Location" in traj_df.columns else None
        for i, frame_id in enumerate(traj_df["Frame_ID"].to_numpy()):
            key = (ego_loc, int(frame_id))
            if key in frame_index:
                pts = frame_index[key]  # (M, 2) array of (x, y) in meters
                d = np.linalg.norm(pts - np.array([x[i], y[i]]), axis=1)
                # subtract 1 for the ego itself, which is also in the frame index
                density[i] = max(0, int(np.sum(d <= DENSITY_RADIUS_M)) - 1)

    # Normalize to [0,1]
    f1 = normalize_upper(speed,        bounds["speed"]["max"])
    f2 = normalize_upper(a,            bounds["acceleration"]["max"])
    f3 = normalize_upper(jerk,         bounds["jerk"]["max"])
    f4 = normalize_upper(steer_var,    bounds["steer_var"]["max"])
    f5 = normalize_upper(lane_offset,  bounds["lane_offset"]["max"])
    f6 = normalize_inverse_range(ttc_raw,  bounds["ttc"]["min"],     bounds["ttc"]["max"])
    f7 = normalize_inverse_range(headway,  bounds["headway"]["min"], bounds["headway"]["max"])
    f8 = normalize_upper(density,      bounds["density"]["max"])

    features = np.stack([f1, f2, f3, f4, f5, f6, f7, f8], axis=1)
    ok, reason = is_eligible(features, min_length_s=5.0, fs_hz=FS_HZ)
    return features, ttc_raw, ok, reason
