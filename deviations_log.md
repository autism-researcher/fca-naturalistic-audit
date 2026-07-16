# Deviations Register

Per §13 of the pre-registration, classify each entry:

- **within-protocol refinement** (does not change scope)
- **post-hoc addition** (analysis not in this registration)
- **verdict-changing** (alters pass/fail conclusion of a confirmatory test)

Verdict-changing deviations require explicit pre-disclosure before the affected analysis is run.

| Date | Item | Class | Impact |
|---|---|---|---|
| 2026-05-19 | `--skip-gate-check` on `run_waymo_experiment.py` (N=5000): the `/mnt/d` copy is not a git working tree, so the Gate 2 tag cannot be queried. Pipeline frozen state recorded by tree sha256 = `6c57385f90ff33c37c12cc65319e90d73e770578d1f8bc214e870e73659c6399` over `src/`, `scripts/`, `carla_weights.json`, `paper2_constants.json`. | within-protocol refinement | none — same code as OSF supplement; gate intent (freeze before confirmatory) satisfied by hash anchor |
| 2026-05-20 | TTC-ablation robustness check for H_OFF3: recompute `R` with `w_ttc = 0` and the remaining seven weights renormalized to 0.94; crossing group redefined by the (1−τ)-quantile of the TTC-free risk. Script: `scripts/ablation_ttc_hoff3.py`. | post-hoc addition | none — exploratory; does not alter the confirmatory H_OFF1/H_OFF2/H_OFF3 verdicts. Reported in §VII-B (threats to validity). |
| 2026-07-15 | Mann–Whitney p-value computation corrected: the originally reported p-values (Table III and the provided-TTC sensitivity check) came from SciPy's tie-corrected normal approximation, which is invalid for the heavily tied per-trajectory TTC<2s counts at small crossing-group sizes (n1 as low as 8) and produced p-values below the exact-test floor 1/C(N, n1). All affected cells recomputed with exact permutation enumeration (HighD; feasible because at most seven pooled counts are nonzero) or 10^7-resample Monte Carlo permutation (NGSIM, Waymo). `src/tests.py::h_off3` now implements the permutation method; verification: `verify_table3_exact.py`; corrected values: `results/verdicts/hoff3_corrected.json`. | within-protocol refinement (correction of the test's numerical method, not its design) | none on conclusions — U statistics, rank-biserial effect sizes, and all PASS/FAIL outcomes unchanged; only the reported p-values changed (e.g., HighD τ=0.10: <10⁻³⁰⁰ → 2.96×10⁻¹¹). Detected during internal audit. |
| 2026-07-15 | Deferred logistic-regression weight variant (listed as exploratory in the registration) formally logged as a deferred pre-registered analysis; results postponed to future work on adaptive recalibration. | within-protocol refinement (deferral) | none — exploratory item; no confirmatory verdict depends on it. |
