# Paper 3 — Offline Validation Pipeline

Pre-registered, multi-dataset validation of a quantile-calibrated FCA supervisor.
Locked OSF pre-registration: 2026-05-11. Companion CARLA pre-reg: https://osf.io/sfdj2.

---

## Project layout

```
paper3_pipeline/
├── carla_weights.json         # 8 weights from Paper 2 (frozen, sum 0.94)
├── paper2_constants.json      # B_sim values per τ from Paper 2 Table V
├── deviations_log.md
├── requirements.txt
├── data/{ngsim,waymo,highd}/  # raw data (gitignored)
├── src/
│   ├── utils.py               # weights, Butterworth, EMA, eligibility
│   ├── risk.py                # composite risk R(x)
│   ├── boundary.py            # B = (1-τ) quantile, realized rate, DKW, bootstrap CI
│   ├── tests.py               # H_OFF1, H_OFF2, H_OFF3
│   └── features/{ngsim,waymo,highd}.py   # per-dataset extractors
├── scripts/                   # numbered driver scripts 01-09
├── tests/test_pipeline.py     # smoke tests
└── results/                   # populated by scripts 03-09
```

---

## Setup (once)

```bash
cd paper3_pipeline
python3 -m venv .venv
source .venv/bin/activate           # Linux/Mac
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt
python3 tests/test_pipeline.py      # smoke check
```

For Waymo only (heavy TensorFlow dependency), create a separate conda env:

```bash
conda create -n waymo python=3.10 -y
conda activate waymo
pip install waymo-open-dataset-tf-2-12-0 numpy pandas scipy
```

Place data in:
- `data/ngsim/<file>.csv`   (US-101 + I-80 combined CSV from FHWA)
- `data/waymo/<shard>.tfrecord-NNNNN-of-NNNNN`
- `data/highd/<NN>_tracks.csv`  (+ `*_tracksMeta.csv`, `*_recordingMeta.csv` when approval arrives)

---

## Run order

The pipeline maps directly onto the 9 conceptual stages of the workflow. Run in this order:

| # | Stage | Script | Output |
|---|---|---|---|
| 0 | Setup | (manual: weights frozen, repo committed) | — |
| 1 | I/O check | `python scripts/01_io_check.py` | console |
| 2 | Feature extractors | (implementation in `src/features/`) | — |
| 3 | Hand validation | `python scripts/02_validate_pipeline.py` | console |
| **4** | **Gate 2: code review + `git tag pipeline-frozen-pre-confirmatory`** | | |
| 5a | Extract at scale | `python scripts/03_extract_features_at_scale.py --datasets ngsim waymo` | `results/per_dataset/*.json` |
| 5b | Boundaries | `python scripts/04_compute_boundaries.py` | `results/boundaries/*.json` |
| 5c | Confirmatory tests | `python scripts/05_run_tests.py` | `results/verdicts/*.json` |
| 6 | Cross-dataset transfer | `python scripts/06_cross_dataset_transfer.py` | `results/transfer/matrix.json` |
| 7 | Sensitivity | `python scripts/07_sensitivity.py` | `results/sensitivity/*.json` |
| 8a | Figures | `python scripts/08_generate_figures.py` | `results/figures/fig*.png` |
| 8b | Tables | `python scripts/09_generate_tables.py` | `results/figures/table_verdicts.{md,tex}` |

**Hard rule:** scripts 04 onward must not run before step 4 (Gate 2). Pipeline freezes at the Git tag.

---

## Smoke test (data-free)

```bash
python tests/test_pipeline.py
```

Confirms weights sum to 0.94, composite risk is bounded, DKW floor at N=738 gives ε≈0.05, boundary and realized-rate logic are correct, H_OFF1 / H_OFF3 behave as expected on synthetic data.

---

## Outputs and what to look at

After scripts 03-05 run cleanly per dataset, look in:

- `results/per_dataset/{d}_features.json` — every trajectory's (T, 8) feature array + raw TTC + ID
- `results/boundaries/{d}.json` — per-τ values of B_d, τ̂_d, bootstrap CIs, DKW ε at realized N
- `results/verdicts/{d}.json` — H_OFF1 / H_OFF2 / H_OFF3 verdicts per (d, τ)
- `results/transfer/matrix.json` — full 3×3 transfer matrix per τ
- `results/figures/table_verdicts.md` — overall PASS/FAIL summary in one glance

The disjunctive criterion is built into `09_generate_tables.py`: each hypothesis passes overall if any dataset passes for all three τ values.

---

## CLI examples

Smaller pilot run before full N=5000:

```bash
python scripts/03_extract_features_at_scale.py --n 500 --datasets ngsim
python scripts/04_compute_boundaries.py
python scripts/05_run_tests.py
python scripts/08_generate_figures.py
```

Once the pilot looks right, scale up to N=5000 and re-run scripts 03-08.

---

## Hard rules (from pre-registration §3, §13)

1. No boundary computation on naturalistic data before the OSF pre-reg is locked (it is — 2026-05-11) and before Gate 2 (Git tag `pipeline-frozen-pre-confirmatory`).
2. Composite-risk weights are frozen at `carla_weights.json`; no retuning permitted.
3. Sampling uses `SEED = 42` from `src/utils.py`; do not re-roll.
4. Save raw TTC alongside normalized TTC — H_OFF3 needs the raw 2-second threshold.
5. Any pipeline change after Gate 2 goes into `deviations_log.md` with a class label (within-protocol refinement / post-hoc addition / verdict-changing).
