"""Stage 8 (cont): emit the verdict table as Markdown and a TeX snippet.

Produces results/figures/table_verdicts.md and table_verdicts.tex.
"""
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAUS = [0.10, 0.15, 0.20]

def cell(v, ok_field="pass"):
    return "PASS" if (isinstance(v, dict) and v.get(ok_field)) else "FAIL"

def disjunctive_pass(per_dataset_flags):
    """Pre-registered disjunctive criterion: PASS iff there exists at least
    one dataset for which EVERY tau in TAUS passes -- NOT merely "any cell
    anywhere passes". `per_dataset_flags` is {dataset: [pass_tau1, pass_tau2, ...]}.

    2026-09 bugfix (deviations register): the previous implementation took
    any(<flat list of all dataset*tau cells>), which is equivalent to "does
    any single cell pass anywhere" -- a strictly weaker (easier-to-satisfy)
    condition than the pre-registered "all three tau on at least one
    dataset". On this study's actual results every hypothesis is unanimous
    (H_OFF1/H_OFF2 fail all 9 cells, H_OFF3 passes all 9), so the two
    formulations happened to agree numerically and no reported verdict was
    wrong -- but the old code was incorrect for any non-unanimous result
    pattern. See tests/test_pipeline.py::test_disjunctive_pass_grouping for
    a synthetic case where the two formulations disagree.
    """
    return any(
        len(flags) == len(TAUS) and all(flags)
        for flags in per_dataset_flags.values()
    )


def main():
    rows_md = ["| d | τ | B_sim | B_d | |ΔB| | τ̂ | |τ̂−τ| | MW p | H_OFF1 | H_OFF2 | H_OFF3 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows_tex = []
    # Per-hypothesis, per-dataset lists of tau-level pass flags (grouped, not flat)
    # -- required so the disjunctive criterion below can check "all three tau
    # pass for the SAME dataset" rather than "any cell anywhere passes".
    per_dataset = {"H_OFF1": {}, "H_OFF2": {}, "H_OFF3": {}}
    # 2026-09 bugfix (deviations register): results/verdicts/*.json also
    # matches unrelated files written by other scripts into the same
    # directory (hoff3_corrected.json, highd_sensitivity_provided_ttc.json,
    # and -- if archived -- hoff3_ttc_ablation.json), none of which share
    # this file's {"dataset": ..., "tests": {...}} schema. The old glob
    # picked those up too and crashed with KeyError the first time this
    # script was re-run after those files existed. Restrict the glob to the
    # three actual per-dataset confirmatory-verdict files by name.
    DATASETS = ("highd", "ngsim", "waymo")
    for vpath in sorted(
        str(ROOT / "results/verdicts" / f"{ds}.json") for ds in DATASETS
    ):
        if not Path(vpath).exists():
            continue
        with open(vpath) as f:
            v = json.load(f)
        ds = v["dataset"]
        bpath = ROOT / "results/boundaries" / f"{ds}.json"
        with open(bpath) as fb:
            b = json.load(fb)
        for hyp in per_dataset:
            per_dataset[hyp].setdefault(ds, [])
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
            per_dataset["H_OFF1"][ds].append(bool(t["H_OFF1"]["pass"]))
            per_dataset["H_OFF2"][ds].append(bool(t["H_OFF2"]["pass"]))
            per_dataset["H_OFF3"][ds].append(bool(t["H_OFF3"].get("pass")))
    summary = " | ".join(
        f"{hyp}: {'PASS' if disjunctive_pass(per_ds) else 'FAIL'}"
        for hyp, per_ds in per_dataset.items()
    )
    out_md = ROOT / "results/figures/table_verdicts.md"
    out_md.write_text(f"**Overall (disjunctive):** {summary}\n\n" + "\n".join(rows_md), encoding="utf-8")
    out_tex = ROOT / "results/figures/table_verdicts.tex"
    out_tex.write_text("\n".join(rows_tex), encoding="utf-8")
    print(f"-> {out_md}\n-> {out_tex}")
    print(f"Overall: {summary}")

if __name__ == "__main__":
    main()
