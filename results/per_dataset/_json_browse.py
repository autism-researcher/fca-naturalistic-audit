# _json_browse.py - interactively browse trajectories inside a *_features.json
# Commands: [Enter]=next  p=previous  <number>=jump  e=export to CSV  q=quit
import json, os, glob, sys, csv

FEAT = ["speed","accel","jerk","steer_var","lane_offset","ttc","headway","density"]
W    = [0.08, 0.10, 0.10, 0.08, 0.12, 0.25, 0.15, 0.06]
HZ   = {"highd": 25.0, "ngsim": 10.0, "waymo": 10.0}

def pick_file():
    files = sorted(glob.glob("*_features.json")) or sorted(glob.glob("*.json"))
    if not files:
        print("No .json files here."); sys.exit(0)
    print("Files in this folder:")
    for i, f in enumerate(files, 1):
        print("  {}. {}  ({:.1f} MB)".format(i, f, os.path.getsize(f)/1e6))
    s = input("Open which file? [1-{}]: ".format(len(files))).strip() or "1"
    try:
        return files[int(s)-1]
    except Exception:
        return files[0]

def show(d, k, ds, hz):
    trajs = d["trajectories"]
    t = trajs[k]
    tid = t.get("trajectory_id", t.get("id", "#{}".format(k)))
    T = int(t.get("T", len(t["features"])))
    print("=" * 70)
    print("Trajectory {} of {}   id = {}".format(k+1, len(trajs), tid))
    print("frames T = {}   duration = {:.1f} s  (at {:.0f} Hz)".format(T, T/hz, hz))
    feats = t["features"]
    # per-feature min/mean/max
    print("-" * 70)
    print("{:<12}{:>9}{:>9}{:>9}   (normalized 0..1)".format("feature","min","mean","max"))
    for j, name in enumerate(FEAT):
        col = [row[j] for row in feats]
        print("{:<12}{:>9.3f}{:>9.3f}{:>9.3f}".format(name, min(col), sum(col)/len(col), max(col)))
    # composite risk
    R = [sum(w*x for w, x in zip(W, row)) for row in feats]
    imax = max(range(len(R)), key=lambda i: R[i])
    print("-" * 70)
    print("R_max = {:.4f}   at frame {}  (t = {:.1f} s)".format(R[imax], imax, imax/hz))
    print("R mean = {:.4f}".format(sum(R)/len(R)))
    ttc = [v for v in (t.get("ttc_raw") or []) if isinstance(v, (int, float))]
    if ttc:
        lt2 = sum(1 for v in ttc if v < 2.0)
        print("raw TTC: min = {:.2f} s   ticks with TTC<2s = {}".format(min(ttc), lt2))
    else:
        print("raw TTC: none recorded")
    print("first 3 frames (8 features):")
    for row in feats[:3]:
        print("   [" + ", ".join("{:.3f}".format(x) for x in row) + "]")
    print("=" * 70)
    return tid, feats, t

def export(tid, feats, t, ds):
    safe = str(tid).replace("/", "_").replace("\\", "_")
    out = "traj_{}_{}.csv".format(ds, safe)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + FEAT + ["R", "ttc_raw"])
        ttc = t.get("ttc_raw") or []
        for i, row in enumerate(feats):
            R = sum(wi*x for wi, x in zip(W, row))
            tv = ttc[i] if i < len(ttc) else ""
            w.writerow([i] + ["{:.6f}".format(x) for x in row] + ["{:.6f}".format(R), tv])
    print("  exported -> {}   (open in Excel/Notepad)".format(out))

def main():
    fname = pick_file()
    print("\nLoading {} ... (the 847 MB NGSIM file can take 1-2 minutes)".format(fname))
    d = json.load(open(fname, encoding="utf-8-sig"))
    ds = str(d.get("dataset", "data")).lower()
    hz = HZ.get(ds, 10.0)
    n = len(d["trajectories"])
    print("Loaded: {} trajectories.\n".format(n))
    print("Commands:  [Enter] next   p previous   <number> jump   e export CSV   q quit\n")
    k = 0
    while True:
        tid, feats, t = show(d, k, ds, hz)
        cmd = input("[{} / {}]  command > ".format(k+1, n)).strip().lower()
        if cmd == "q":
            break
        elif cmd == "p":
            k = (k - 1) % n
        elif cmd == "e":
            export(tid, feats, t, ds)
        elif cmd.isdigit():
            k = min(max(int(cmd) - 1, 0), n - 1)
        else:
            k = (k + 1) % n

if __name__ == "__main__":
    main()
