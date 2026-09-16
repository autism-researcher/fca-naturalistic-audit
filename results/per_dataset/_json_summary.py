# _json_summary.py - one-glance overview of every *_features.json here.
# For each file writes <dataset>_summary.csv : ONE ROW PER TRAJECTORY
#   (id, frames, duration, R_max, R_mean, when the peak happened,
#    min raw TTC, ticks with TTC<2s, and each feature's max)
# Also prints dataset-level statistics and, if matplotlib exists,
# saves <dataset>_overview.png (R_max histogram with B_sim lines).
import json, os, glob, csv

FEAT = ["speed","accel","jerk","steer_var","lane_offset","ttc","headway","density"]
W    = [0.08, 0.10, 0.10, 0.08, 0.12, 0.25, 0.15, 0.06]
HZ   = {"highd": 25.0, "ngsim": 10.0, "waymo": 10.0}
BSIM = {0.10: 0.5116, 0.15: 0.4885, 0.20: 0.4733}

def quant(sorted_x, q):
    # linear-interpolation quantile (matches numpy default) on a sorted list
    n = len(sorted_x)
    if n == 1: return sorted_x[0]
    pos = q * (n - 1); lo = int(pos); hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_x[lo] * (1 - frac) + sorted_x[hi] * frac

for fname in sorted(glob.glob("*_features.json")):
    print("=" * 66)
    print("Loading {} ({:.0f} MB) ...".format(fname, os.path.getsize(fname)/1e6))
    d = json.load(open(fname, encoding="utf-8-sig"))
    ds = str(d.get("dataset","data")).lower()
    hz = HZ.get(ds, 10.0)
    trajs = d["trajectories"]

    rows, rmax_all = [], []
    for k, t in enumerate(trajs):
        tid = t.get("trajectory_id", t.get("id", "#{}".format(k)))
        feats = t["features"]; T = len(feats)
        R = [sum(w*x for w, x in zip(W, row)) for row in feats]
        imax = max(range(T), key=lambda i: R[i])
        ttc = [v for v in (t.get("ttc_raw") or []) if isinstance(v,(int,float))]
        row = dict(idx=k+1, trajectory_id=tid, frames=T,
                   duration_s=round(T/hz,2),
                   R_max=round(R[imax],6), R_mean=round(sum(R)/T,6),
                   peak_at_s=round(imax/hz,2),
                   min_ttc_raw=(round(min(ttc),3) if ttc else ""),
                   ticks_ttc_lt2=sum(1 for v in ttc if v < 2.0))
        for j, name in enumerate(FEAT):
            row["max_"+name] = round(max(r[j] for r in feats), 4)
        rows.append(row); rmax_all.append(R[imax])

    out = ds + "_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    s = sorted(rmax_all); n = len(s)
    print("  -> wrote {}  ({} rows - open in Excel)".format(out, n))
    print("  R_max at a glance:  min={:.3f}  median={:.3f}  max={:.3f}".format(
        s[0], quant(s,0.5), s[-1]))
    for tau in (0.10, 0.15, 0.20):
        bd = quant(s, 1.0-tau)
        crossing = sum(1 for v in s if v > BSIM[tau])
        print("    tau={:.2f}:  B_d={:.4f}   crossing B_sim({:.4f}) = {} traj ({:.2%})".format(
            tau, bd, BSIM[tau], crossing, crossing/n))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(rmax_all, bins=60, color="#5a9", edgecolor="white")
        for tau, c in zip((0.10,0.15,0.20), ("#c33","#d80","#39c")):
            ax.axvline(BSIM[tau], color=c, ls="--", lw=1.2,
                       label="B_sim tau={:.2f} = {:.4f}".format(tau, BSIM[tau]))
        ax.set_xlabel("per-trajectory peak risk R_max"); ax.set_ylabel("trajectories")
        ax.set_title("{}: distribution of R_max (n={})".format(ds, n)); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(ds + "_overview.png", dpi=120)
        print("  -> wrote {}_overview.png".format(ds))
    except Exception:
        print("  (matplotlib not installed - skipped the picture; CSV is complete)")
print("=" * 66)
print("Done. Open the *_summary.csv files in Excel: one row = one trajectory.")
