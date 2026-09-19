"""
Two independent checks, run via CLI arg:

  --hash   : sha256 every HighD/Waymo results file (verdicts, boundaries,
             per_dataset features). Run once BEFORE the second NGSIM run
             and once AFTER; the two hash sets must be byte-identical,
             proving the re-run touched NGSIM only.

  --diff <run1_dir> <run2_dir> : numerically compare run1 vs run2's NGSIM
             verdicts/boundaries JSON (two independent end-to-end
             executions of the same pre-registered, seed-locked pipeline).
             Exact equality is expected since SEED=42 is fixed and every
             step is deterministic; any difference would indicate hidden
             nondeterminism (race condition, unordered dict/set iteration,
             filesystem enumeration order, etc.) that the paper's
             reproducibility claim depends on being absent.
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def do_hash():
    targets = [
        ROOT / "results/verdicts/highd.json",
        ROOT / "results/verdicts/waymo.json",
        ROOT / "results/boundaries/highd.json",
        ROOT / "results/boundaries/waymo.json",
        ROOT / "results/per_dataset/highd_features.json",
        ROOT / "results/per_dataset/waymo_features.json",
    ]
    print("=" * 100)
    for t in targets:
        if t.exists():
            print(f"{t.name:40s} sha256={sha256_of(t)}  size={t.stat().st_size}")
        else:
            print(f"{t.name:40s} MISSING")
    print("=" * 100)
    print("HASH_CHECK_DONE_SENTINEL")

def _walk_compare(a, b, path, diffs, tol=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a) | set(b)
        for k in keys:
            if k not in a:
                diffs.append(f"{path}.{k}: missing in run1")
            elif k not in b:
                diffs.append(f"{path}.{k}: missing in run2")
            else:
                _walk_compare(a[k], b[k], f"{path}.{k}", diffs, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > tol:
            diffs.append(f"{path}: run1={a} run2={b} |diff|={abs(a-b):.3e}")
    else:
        if a != b:
            diffs.append(f"{path}: run1={a!r} run2={b!r}")

def do_diff(run1_verdict, run2_verdict):
    with open(run1_verdict) as f:
        a = json.load(f)
    with open(run2_verdict) as f:
        b = json.load(f)
    diffs = []
    _walk_compare(a, b, "ngsim_verdict", diffs)
    print("=" * 100)
    print(f"Comparing {run1_verdict} vs {run2_verdict}")
    if diffs:
        print(f"!!! {len(diffs)} DIFFERENCES FOUND:")
        for d in diffs:
            print("   ", d)
    else:
        print("NO DIFFERENCES: run1 and run2 verdict JSON are numerically identical "
              "(tolerance 1e-9).")
    print("=" * 100)
    print("DIFF_CHECK_DONE_SENTINEL", "IDENTICAL" if not diffs else f"DIFFERS({len(diffs)})")

if __name__ == "__main__":
    if sys.argv[1] == "--hash":
        do_hash()
    elif sys.argv[1] == "--diff":
        do_diff(sys.argv[2], sys.argv[3])
    else:
        print("usage: --hash | --diff run1.json run2.json")
