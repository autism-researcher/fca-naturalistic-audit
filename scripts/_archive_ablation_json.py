"""Archive the TTC-free ablation script's console output as a results file.

`scripts/ablation_ttc_hoff3.py` (logged 2026-05-20 in the deviations register)
computes the full per-cell TTC-free ablation but only prints its JSON summary
to stdout -- it was never wired up to write a committed artifact, unlike the
confirmatory H_OFF3 results (results/verdicts/hoff3_corrected.json). That gap
was flagged in a 2026-09-19 deviations-register entry.

This helper does NOT re-implement or modify the ablation logic -- it only
captures the *unmodified* script's own stdout and extracts the "Compact
JSON:" block it already prints, writing it to results/verdicts/hoff3_ttc_ablation.json
so the reported ablation numbers are independently checkable from the repo.

Usage:
    python scripts/ablation_ttc_hoff3.py > ablation_stdout.log 2>&1
    python scripts/_archive_ablation_json.py ablation_stdout.log
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/_archive_ablation_json.py <stdout_log_path>")
        sys.exit(1)
    log_path = Path(sys.argv[1])
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = "Compact JSON:"
    idx = text.find(marker)
    if idx == -1:
        print(f"ERROR: marker {marker!r} not found in {log_path}; "
              f"ablation script may have failed. See log for details.")
        sys.exit(1)
    payload = text[idx + len(marker):].strip()
    data = json.loads(payload)
    out_dir = ROOT / "results" / "verdicts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hoff3_ttc_ablation.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    datasets = sorted(data.keys())
    print(f"[archive] wrote {out_path}")
    print(f"[archive] datasets covered: {datasets}")
    for ds in datasets:
        for tau_key, cell in sorted(data[ds].items()):
            print(f"  {ds:8s} tau={tau_key}  r_ablated={cell['r_ablated']:.4f}  "
                  f"p_ablated={cell['p_ablated']:.3e}  survives={cell['survives']}")

if __name__ == "__main__":
    main()
