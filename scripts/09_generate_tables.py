"""Stage 8 (cont): emit the verdict table as Markdown and a TeX snippet.

Produces results/figures/table_verdicts.md and table_verdicts.tex.
"""
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAUS = [0.10, 0.15, 0.20]

def cell(v, ok_field="pass"):
    return "PASS" if (isinstance(v, dict) and v.get(ok_field)) else "FAIL"

def main():
    rows_md = ["| d | τ | B_sim | B_d | |ΔB| | τ̂ | |τ̂−τ| | MW p | H_OFF1 | H_OFF2 | H_OFF3 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows_tex = []
    overall = {"H_OFF1": [], "H_OFF2": [], "H_OFF3": []}
    for vpath in sorted(glob.glob(str(ROOT / "results/verdicts/*.json"))):
        with open(vpath) as f:
            v = json.load(f)
        ds = v["dataset"]
        bpath = ROOT / "results/boundaries" / f"{ds}.json"
        with open(bpath) as fb:
            b = json.load(fb)
        for tau in TAUS:
            key = f"{tau:.2f}"
            t = v["tests"][key]
            bsim_norm = {f"{float(k):.2f}": v for k, v in b["B_sim"].items()}
            B_sim = bsim_norm[key]
            B_d   = b["B_d"][key]
            tau_hat = b["tau_hat_d"][key]
            p = t["H_OFF3"].get("p")
            rows_md.append(
                f"| {ds} | {tau} | {B_sim:.4f} | {B_d:.4f} | {abs(B_sim - B_d):.4f} | "
                f"{tau_hat:.4f} | {abs(tau_hat - tau):.4f} | "
                f"{('%.2e' % p) if p else '—'} | "
                f"{cell(t['H_OFF1'])} | {cell(t['H_OFF2'])} | {cell(t['H_OFF3'])} |"
            )
            rows_tex.append(
                f"{ds} & {tau:.2f} & {B_sim:.3f} & {B_d:.3f} & {abs(B_sim-B_d):.3f} & "
                f"{tau_hat:.3f} & {abs(tau_hat-tau):.3f} & "
                f"{('%.2g' % p) if p else '---'} & "
                f"{cell(t['H_OFF1'])} & {cell(t['H_OFF2'])} & {cell(t['H_OFF3'])} \\\\"
            )
            overall["H_OFF1"].append(t["H_OFF1"]["pass"])
            overall["H_OFF2"].append(t["H_OFF2"]["pass"])
            overall["H_OFF3"].append(bool(t["H_OFF3"].get("pass")))
    summary = " | ".join(
        f"{k}: {'PASS' if any(v) else 'FAIL'}" for k, v in overall.items()
    )
    out_md = ROOT / "results/figures/table_verdicts.md"
    out_md.write_text(f"**Overall (disjunctive):** {summary}\n\n" + "\n".join(rows_md), encoding="utf-8")
    out_tex = ROOT / "results/figures/table_verdicts.tex"
    out_tex.write_text("\n".join(rows_tex), encoding="utf-8")
    print(f"-> {out_md}\n-> {out_tex}")
    print(f"Overall: {summary}")

if __name__ == "__main__":
    main()
