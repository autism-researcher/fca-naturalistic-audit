"""Waymo Open Motion feature extractor.

Per pre-reg §7: ego = AV track if labelled, else longest-tracked vehicle
(deterministic tiebreak by lowest track ID). Lane offset via road-graph
projection. TTC/headway: nearest vehicle in +-5 deg forward cone.

Requires waymo-open-dataset (install in a SEPARATE conda env):
    pip install waymo-open-dataset-tf-2-12-0

Each TFRecord shard contains many Scenario protos. Each Scenario contains
tracks (vehicles, pedestrians, cyclists), map features (lanes, road edges,
etc.), and dynamic map states (traffic lights).
"""
import numpy as np
from pathlib import Path
from ..utils import (load_weights, ema_filter, rolling_variance,
                     normalize_upper, normalize_inverse_range, is_eligible)

FS_HZ = 10.0
DT = 1.0 / FS_HZ
DENSITY_RADIUS_M = 35.0
TTC_FORWARD_CONE_RAD = np.deg2rad(5.0)

# ---------- TFRecord loading ----------
def scenarios_from_shard(shard_path):
    """Yield Scenario protos from a Waymo Open Motion TFRecord shard.

    Requires `pip install waymo-open-dataset-tf-2-12-0` in the active env.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from waymo_open_dataset.protos import scenario_pb2
    except ImportError as exc:
        raise ImportError(
            "Install waymo-open-dataset-tf-2-12-0 in a separate conda env "
            "and run scripts that touch Waymo from that env."
        ) from exc

    dataset = tf.data.TFRecordDataset([str(shard_path)], compression_type="")
    for raw in dataset:
        s = scenario_pb2.Scenario()
        s.ParseFromString(raw.numpy())
        yield s

def pick_ego_track(scenario):
    """AV track if labelled (scenario.sdc_track_index), else longest-tracked vehicle."""
    if scenario.sdc_track_index >= 0 and scenario.sdc_track_index < len(scenario.tracks):
        return scenario.tracks[scenario.sdc_track_index]
    # Fallback: longest-tracked vehicle (object_type == 1 is TYPE_VEHICLE), tiebreak lowest ID
    cands = [(t, sum(1 for s in t.states if s.valid))
             for t in scenario.tracks if t.object_type == 1]
    if not cands:
        return None
    max_valid = max(c[1] for c in cands)
    longest = sorted([t for t, n in cands if n == max_valid], key=lambda t: t.id)
    return longest[0] if longest else None

# ---------- Feature extraction ----------
def extract_features(scenario):
    """Returns (features [T,8], raw_ttc [T], ok_flag, reason) for one scenario."""
    weights, bounds = load_weights()
    ego = pick_ego_track(scenario)
    if ego is None:
        return None, None, False, "no_ego"

    states = list(ego.states)
    n = len(states)
    if n < 5 * FS_HZ:
        return None, None, False, "too_short"

    valid = np.array([s.valid for s in states])
    x = np.array([s.center_x for s in states])
    y = np.array([s.center_y for s in states])
    heading = np.array([s.heading for s in states])
    speed_raw = np.array([np.sqrt(s.velocity_x ** 2 + s.velocity_y ** 2) for s in states])

    if valid.sum() < 5 * FS_HZ:
        return None, None, False, "valid_too_short"

    # Validity: enforced by the valid-count length check above and the strict
    # any-tick |a|>10 exclusion below; no gap filling is applied.
    speed = speed_raw.copy()
    a = np.gradient(speed, DT)
    if np.any(np.abs(a) > 10.0):
        return None, None, False, "accel_above_10"
    jerk_raw = np.gradient(a, DT)
    jerk = ema_filter(np.abs(jerk_raw), tau_s=0.40, dt=DT)

    yaw_rate = np.gradient(np.unwrap(heading), DT)
    steer_var = rolling_variance(yaw_rate, window=int(FS_HZ))

    # Lane offset via road-graph projection
    lane_offset = _project_to_nearest_lane(scenario, x, y)

    # Density: count vehicles within radius at each step
    density = _density_per_step(scenario, ego.id, x, y, n)

    # TTC/headway via +-5 deg forward cone
    ttc_raw, headway = _ttc_and_headway(scenario, ego.id, x, y, heading, n)

    f1 = normalize_upper(speed,       bounds["speed"]["max"])
    f2 = normalize_upper(a,           bounds["acceleration"]["max"])
    f3 = normalize_upper(jerk,        bounds["jerk"]["max"])
    f4 = normalize_upper(steer_var,   bounds["steer_var"]["max"])
    f5 = normalize_upper(lane_offset, bounds["lane_offset"]["max"])
    f6 = normalize_inverse_range(ttc_raw,  bounds["ttc"]["min"],     bounds["ttc"]["max"])
    f7 = normalize_inverse_range(headway,  bounds["headway"]["min"], bounds["headway"]["max"])
    f8 = normalize_upper(density,     bounds["density"]["max"])

    features = np.stack([f1, f2, f3, f4, f5, f6, f7, f8], axis=1)
    ok, reason = is_eligible(features, min_length_s=5.0, fs_hz=FS_HZ)
    return features, ttc_raw, ok, reason

# ---------- helpers (road-graph + neighbours) ----------
def _project_to_nearest_lane(scenario, x, y):
    """Project (x, y) onto nearest lane centerline polyline from scenario.map_features."""
    lane_polylines = []
    for mf in scenario.map_features:
        if mf.HasField("lane"):
            pts = np.array([(p.x, p.y) for p in mf.lane.polyline])
            if len(pts) >= 2:
                lane_polylines.append(pts)
    if not lane_polylines:
        return np.zeros(len(x))
    out = np.zeros(len(x))
    for i, (xi, yi) in enumerate(zip(x, y)):
        min_d = np.inf
        for pts in lane_polylines:
            d = np.min(np.linalg.norm(pts - np.array([xi, yi]), axis=1))
            if d < min_d:
                min_d = d
        out[i] = min_d
    return out

def _density_per_step(scenario, ego_id, x, y, n):
    """Count vehicles within DENSITY_RADIUS_M of ego at each step."""
    others = [t for t in scenario.tracks if t.id != ego_id and t.object_type == 1]
    density = np.zeros(n)
    for i in range(n):
        cnt = 0
        for t in others:
            if i < len(t.states) and t.states[i].valid:
                d = np.hypot(t.states[i].center_x - x[i], t.states[i].center_y - y[i])
                if d <= DENSITY_RADIUS_M:
                    cnt += 1
        density[i] = cnt
    return density

def _ttc_and_headway(scenario, ego_id, x, y, heading, n):
    """Nearest vehicle in +-5 deg forward cone; TTC via closing speed."""
    others = [t for t in scenario.tracks if t.id != ego_id and t.object_type == 1]
    headway = np.full(n, 60.0)
    ttc = np.full(n, np.inf)
    for i in range(n):
        h = heading[i]
        forward = np.array([np.cos(h), np.sin(h)])
        best_d = np.inf
        best_lead = None
        for t in others:
            if i >= len(t.states) or not t.states[i].valid:
                continue
            dx = t.states[i].center_x - x[i]
            dy = t.states[i].center_y - y[i]
            d = np.hypot(dx, dy)
            if d < 0.5:
                continue
            cos_angle = (dx * forward[0] + dy * forward[1]) / d
            if cos_angle > np.cos(TTC_FORWARD_CONE_RAD) and d < best_d:
                best_d = d
                best_lead = t
        if best_lead is not None:
            headway[i] = best_d
            # closing speed = (-d/dt of distance) along forward axis
            if i > 0 and i < len(best_lead.states) - 1 and best_lead.states[i - 1].valid:
                prev_d = np.hypot(best_lead.states[i - 1].center_x - x[max(i - 1, 0)],
                                  best_lead.states[i - 1].center_y - y[max(i - 1, 0)])
                closing = (prev_d - best_d) / DT
                if closing > 0.05:
                    ttc[i] = best_d / closing
    return np.clip(ttc, 0.0, 1e6), headway
