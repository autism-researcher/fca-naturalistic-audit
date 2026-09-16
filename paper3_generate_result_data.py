"""
Paper 3 — ONE consolidated experiment.

Replaces the six separate scripts (H_OFF1, H_OFF2, H_OFF3, baseline,
ablation, transfer) with a SINGLE pass over the real corpora that emits
ONE result-data table: one row per trajectory. Every hypothesis is then
a plain calculation on that table (done live by Excel formulas in the
'Proofs' sheet) — no second run over the data is ever needed.

What one row of RESULT DATA contains
------------------------------------
  traj_id       identifier
  dataset       highd | ngsim | waymo
  length_s      trajectory duration in seconds
  eligible      length_s >= 5.0  (pre-registration minimum)
  R_max_full    peak composite risk over the trajectory (all 8 features)
  R_max_noTTC   peak composite risk with the TTC weight zeroed+renormalized
  min_ttc       smallest raw time-to-collision (seconds) on the trajectory
  ticks_ttc_lt2 number of ticks with raw TTC < 2 s

From those columns the 'Proofs' sheet computes, by formula only:
  - H_OFF1  boundary:  B_d = (1-tau) quantile of R_max_full   -> |B_sim-B_d|<0.03
  - H_OFF2  coverage:  tau_hat = fraction R_max_full > B_sim   -> |tau_hat-tau|<=0.03
  - Transfer:          B_d learned on dataset A applied to dataset B (3x3 grid)
  - Baseline (opt.):   fraction min_ttc < 2 s
  - Ablation (opt.):   same coverage math on R_max_noTTC

Run from the repo root (needs results/per_dataset/*_features.json,
produced by scripts/03_extract_features_at_scale.py):

    python paper3_generate_result_data.py

Output: Paper3_result_data_master.xlsx
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.worksheet.formula import ArrayFormula
except ImportError:
    sys.exit("Install openpyxl:  pip install openpyxl")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---- risk configuration (self-contained; no Paper 2 dependency) -------------
# feature order: speed, accel, jerk, steer_var, lane_offset, ttc, headway, density
WEIGHTS = np.array([0.08, 0.10, 0.10, 0.08, 0.12, 0.25, 0.15, 0.06], dtype=float)
TTC_INDEX = 5
TAUS = [0.10, 0.15, 0.20]
B_SIM = {0.10: 0.5116, 0.15: 0.4885, 0.20: 0.4733}   # frozen simulator boundary
HZ = {"highd": 25.0, "ngsim": 10.0, "waymo": 10.0}
TTC_THRESH_S = 2.0
DATASETS = ["highd", "ngsim", "waymo"]


def composite_risk(features_TxF, weights_F):
    return np.asarray(features_TxF, dtype=float) @ np.asarray(weights_F, dtype=float)


def extract_rows(per_dataset_glob):
    """One dict per trajectory across all datasets."""
    w_abl = WEIGHTS.copy()
    w_abl[TTC_INDEX] = 0.0
    w_abl *= WEIGHTS.sum() / w_abl.sum()          # renormalize to 0.94

    rows = []
    for fpath in sorted(glob.glob(per_dataset_glob)):
        data = json.load(open(fpath))
        ds = str(data["dataset"]).lower()
        trajs = data.get("trajectories", [])
        if not trajs:                       # e.g. Waymo not yet extracted
            print(f"[{ds}] 0 trajectories — SKIPPED (nothing to add)")
            continue
        hz = HZ.get(ds, 10.0)
        for i, t in enumerate(trajs):
            feats = np.asarray(t["features"], dtype=float)
            n = feats.shape[0]
            length_s = n / hz
            R_full = float(np.max(composite_risk(feats, WEIGHTS)))
            R_abl = float(np.max(composite_risk(feats, w_abl)))
            ttc = np.asarray(t.get("ttc_raw", []), dtype=float)
            ttc = ttc[np.isfinite(ttc)]
            min_ttc = float(ttc.min()) if ttc.size else float("inf")
            ticks_lt2 = int(np.sum(ttc < TTC_THRESH_S))
            rows.append({
                # scripts/03 writes the id under "trajectory_id"; fall back gracefully
                "traj_id": t.get("trajectory_id") or t.get("id") or f"{ds}_{i}",
                "dataset": ds,
                "length_s": round(length_s, 3),   # display only; not used in any proof
                "eligible": length_s >= 5.0,
                # FULL precision on every column that feeds a proof — no rounding,
                # so B_d / tau_hat / transfer match the reference pipeline to
                # machine epsilon and no trajectory can flip across a threshold.
                "R_max_full": R_full,
                "R_max_noTTC": R_abl,
                "min_ttc": min_ttc if np.isfinite(min_ttc) else "",
                "ticks_ttc_lt2": ticks_lt2,
            })
        print(f"[{ds}] {len(data['trajectories'])} trajectories")
    return rows


# --------------------------------------------------------------------------- #
#  Workbook builder — used for both the real run and the DEMO preview.
# --------------------------------------------------------------------------- #
COLS = ["traj_id", "dataset", "length_s", "eligible",
        "R_max_full", "R_max_noTTC", "min_ttc", "ticks_ttc_lt2"]


def build_workbook(rows, out_path, demo=False):
    # Only emit proofs for datasets that actually have rows, in canonical order.
    # A missing/empty dataset (e.g. Waymo not yet extracted) is silently omitted
    # so no formula can divide by zero.
    present = [d for d in DATASETS if any(r["dataset"] == d for r in rows)]
    extra = sorted({r["dataset"] for r in rows} - set(DATASETS))
    present += extra                       # tolerate unexpected dataset names too
    if not present:
        raise SystemExit("build_workbook: no non-empty datasets in rows.")

    wb = openpyxl.Workbook()
    hdr = PatternFill("solid", fgColor="1F3864")
    hf = Font(bold=True, color="FFFFFF")
    tf = Font(bold=True, size=13, color="1F3864")
    warn = PatternFill("solid", fgColor="FFF2CC")
    warnf = Font(bold=True, color="9C6500")
    thin = Side(style="thin", color="CCCCCC")
    bd = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap = Alignment(wrap_text=True, vertical="top")

    # ---- Sheet 1: trajectories (the RESULT DATA) ----
    ws = wb.active
    ws.title = "trajectories"
    r0 = 1
    if demo:
        ws.cell(1, 1, "DEMO DATA — synthetic rows to show structure. Run "
                      "paper3_generate_result_data.py on your real corpora to replace.")
        ws.cell(1, 1).fill = warn; ws.cell(1, 1).font = warnf
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(COLS))
        r0 = 2
    for j, c in enumerate(COLS, 1):
        cell = ws.cell(r0, j, c)
        cell.fill = hdr; cell.font = hf; cell.border = bd
    for k, row in enumerate(rows, r0 + 1):
        for j, c in enumerate(COLS, 1):
            ws.cell(k, j, row[c])
    data_first = r0 + 1
    data_last = r0 + len(rows)
    for w_, c_ in zip([16, 10, 10, 9, 12, 13, 10, 14], "ABCDEFGH"):
        ws.column_dimensions[c_].width = w_
    ws.freeze_panes = ws.cell(data_first, 1)

    # Excel ranges (generous upper bound so real 15k rows fit).
    # Sheet-qualified, so they work from the Proofs sheet. COUNTIFS +
    # classic CSE array PERCENTILE(IF()) are used (portable to every
    # Excel version and LibreOffice; no FILTER / dynamic arrays).
    LAST = 60000
    DS = f"trajectories!$B${data_first}:$B${LAST}"
    RF = f"trajectories!$E${data_first}:$E${LAST}"
    RA = f"trajectories!$F${data_first}:$F${LAST}"
    MT = f"trajectories!$G${data_first}:$G${LAST}"

    # ---- Sheet 2: Proofs (formulas only; derive from sheet 1) ----
    ps = wb.create_sheet("Proofs")
    ps["A1"] = "All proofs — calculated live from the 'trajectories' sheet"
    ps["A1"].font = tf
    ps["A2"] = ("These are formulas, not typed numbers. Fill 'trajectories' with real rows "
                "(run the script) and every value below recomputes. Works in any Excel version "
                "and LibreOffice (COUNTIFS + array PERCENTILE(IF())).")
    ps["A2"].font = Font(italic=True, color="555555")

    row = 4
    # B_sim reference block
    ps.cell(row, 1, "B_sim (frozen simulator boundary)").font = Font(bold=True)
    row += 1
    ps.cell(row, 1, "tau"); ps.cell(row, 2, "B_sim")
    for c in (1, 2):
        ps.cell(row, c).fill = hdr; ps.cell(row, c).font = hf
    bsim_cell = {}
    for tau in TAUS:
        row += 1
        ps.cell(row, 1, tau)
        ps.cell(row, 2, B_SIM[tau])
        bsim_cell[tau] = f"$B${row}"

    # H_OFF1 + H_OFF2 table
    row += 2
    h1row = row
    ps.cell(row, 1, "H_OFF1 boundary consistency  &  H_OFF2 coverage").font = Font(bold=True, color="1F3864")
    row += 1
    heads = ["dataset", "tau", "B_sim", "B_d = (1-tau) quantile of R_max_full",
             "|B_sim - B_d|", "H_OFF1 (<0.03)",
             "tau_hat = frac R_max>B_sim", "|tau_hat - tau|", "H_OFF2 (<=0.03)"]
    for j, h in enumerate(heads, 1):
        ps.cell(row, j, h); ps.cell(row, j).fill = hdr; ps.cell(row, j).font = hf
        ps.cell(row, j).alignment = wrap; ps.cell(row, j).border = bd
    bd_ref = {}   # (dataset, tau) -> cell of B_d, for the transfer grid
    for ds in present:
        for tau in TAUS:
            row += 1
            bs = bsim_cell[tau]
            ps.cell(row, 1, ds)
            ps.cell(row, 2, tau)
            ps.cell(row, 3, f"={bs}")
            ps.cell(row, 4).value = ArrayFormula(
                f"D{row}", f'=PERCENTILE(IF({DS}="{ds}",{RF}),1-B{row})')
            ps.cell(row, 5, f"=ABS(C{row}-D{row})")
            ps.cell(row, 6, f'=IF(E{row}<0.03,"PASS","FAIL")')
            ps.cell(row, 7, f'=COUNTIFS({DS},"{ds}",{RF},">"&C{row})/COUNTIFS({DS},"{ds}")')
            ps.cell(row, 8, f"=ABS(G{row}-B{row})")
            ps.cell(row, 9, f'=IF(H{row}<=0.03,"PASS","FAIL")')
            for j in range(1, 10):
                ps.cell(row, j).border = bd
            bd_ref[(ds, tau)] = f"$D${row}"

    # Transfer 3x3 grids (one per tau)
    for tau in TAUS:
        row += 2
        ps.cell(row, 1, f"Cross-dataset transfer — tau = {tau:.2f}  "
                        f"(cell = frac of COLUMN dataset with R_max > B_d of ROW dataset)"
                ).font = Font(bold=True, color="1F3864")
        row += 1
        ps.cell(row, 1, "calibrated on \\ eval on")
        for j, ds in enumerate(present, 2):
            ps.cell(row, j, ds)
        for c in range(1, 5):
            ps.cell(row, c).fill = hdr; ps.cell(row, c).font = hf; ps.cell(row, c).border = bd
        for ds_a in present:
            row += 1
            ps.cell(row, 1, ds_a); ps.cell(row, 1).font = Font(bold=True)
            for j, ds_b in enumerate(present, 2):
                bda = bd_ref[(ds_a, tau)]
                ps.cell(row, j, f'=COUNTIFS({DS},"{ds_b}",{RF},">"&{bda})/COUNTIFS({DS},"{ds_b}")')
                ps.cell(row, j).border = bd
            ps.cell(row, 1).border = bd

    # Optional robustness block.
    #  - Baseline (frac min_ttc<2s) IS a genuine, correct result -> formula.
    #  - The ablation B_d_noTTC is shown ONLY as a reference boundary. It is NOT
    #    the ablation verdict: the real ablation asks whether R_noTTC still
    #    DISCRIMINATES low-TTC trajectories (a Mann-Whitney effect size), which
    #    is not a spreadsheet formula. A self-coverage rate on R_noTTC would be
    #    ~tau by construction and prove nothing, so it is deliberately omitted.
    row += 2
    ps.cell(row, 1, "Optional robustness (demoted from main results)").font = Font(bold=True, color="808080")
    row += 1
    ps.cell(row, 1, "dataset")
    ps.cell(row, 2, "baseline: frac min_ttc<2s  (REAL result)")
    ps.cell(row, 3, "B_d_noTTC @tau=0.10  (reference only, NOT the ablation verdict)")
    for j in range(1, 4):
        ps.cell(row, j).fill = hdr; ps.cell(row, j).font = hf; ps.cell(row, j).alignment = wrap; ps.cell(row, j).border = bd
    for ds in present:
        row += 1
        ps.cell(row, 1, ds)
        ps.cell(row, 2, f'=COUNTIFS({DS},"{ds}",{MT},"<2")/COUNTIFS({DS},"{ds}")')
        ps.cell(row, 3).value = ArrayFormula(
            f"C{row}", f'=PERCENTILE(IF({DS}="{ds}",{RA}),0.9)')
        for j in range(1, 4):
            ps.cell(row, j).border = bd
    ps.cell(row + 2, 1,
            "The ablation and H_OFF3 verdicts are Mann-Whitney U tests, NOT spreadsheet formulas. "
            "Both read columns straight from 'trajectories': H_OFF3 uses ticks_ttc_lt2 split by "
            "(R_max_full > B_sim); the ablation uses ticks_ttc_lt2 split by (R_max_noTTC > its own "
            "(1-tau) quantile). If you keep either, feed those two groups to src/tests.py::h_off3 — "
            "the exact confirmatory function. Do NOT read a coverage rate as the ablation result.")
    ps.cell(row + 2, 1).font = Font(italic=True, color="555555")
    ps.cell(row + 2, 1).alignment = wrap
    for w_, c_ in zip([26, 26, 30, 30, 16, 16, 24, 16, 16], "ABCDEFGHI"):
        ps.column_dimensions[c_].width = w_

    # ---- Sheet 3: README ----
    rs = wb.create_sheet("README")
    lines = [
        ("Paper 3 — one experiment, one result-data table", tf),
        ("", None),
        ("WHY ONE SCRIPT.  The six original scripts all read the same extracted features and all "
         "reduce to per-trajectory numbers. This script measures those numbers ONCE (the "
         "'trajectories' sheet). Every hypothesis is then a calculation on that sheet, done live "
         "in 'Proofs'. Nothing is measured twice.", None),
        ("", None),
        ("KEPT AS NECESSARY (core distribution-shift story):", Font(bold=True)),
        ("  H_OFF1  boundary consistency   -> Proofs, column D/E/F", None),
        ("  H_OFF2  coverage               -> Proofs, column G/H/I", None),
        ("  Transfer 3x3                   -> Proofs, transfer grids", None),
        ("", None),
        ("REMOVED / DEMOTED as redundant (derivable from the same table, kept optional):", Font(bold=True)),
        ("  Baseline fixed-TTC AEB   -> one formula: frac(min_ttc<2s)", None),
        ("  TTC-free ablation        -> reuse coverage math on R_max_noTTC", None),
        ("  H_OFF3 tail exposure     -> two ready columns; run one Mann-Whitney if you keep it", None),
        ("", None),
        ("HOW TO RUN.  From the repo root, after scripts/03 has produced "
         "results/per_dataset/*_features.json:", None),
        ("      python paper3_generate_result_data.py", Font(name="Consolas")),
        ("  -> writes Paper3_result_data_master.xlsx with real rows; open it and every "
         "proof is already computed.", None),
    ]
    r = 1
    for text, font in lines:
        rs.cell(r, 1, text)
        if font:
            rs.cell(r, 1).font = font
        rs.cell(r, 1).alignment = wrap
        r += 1
    rs.column_dimensions["A"].width = 110

    wb.save(out_path)
    return data_last - data_first + 1


def main():
    per = str(ROOT / "results/per_dataset/*_features.json")
    files = glob.glob(per)
    if not files:
        sys.exit(f"No feature files at {per}\n"
                 f"Run scripts/03_extract_features_at_scale.py first.")
    rows = extract_rows(per)
    n = build_workbook(rows, ROOT / "Paper3_result_data_master.xlsx", demo=False)
    print(f"\nWrote Paper3_result_data_master.xlsx  ({n} trajectory rows). "
          f"Open the 'Proofs' sheet — all values are already computed.")


if __name__ == "__main__":
    main()
