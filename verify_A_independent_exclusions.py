"""
INDEPENDENT cross-check of the missing-tick exclusion bugfix, run directly
against the raw NGSIM CSV.

For every one of the 8,310 length-eligible candidates, this script:
  1. Computes frac_missing with its OWN hand-written formula (not calling
     src/features/ngsim.py::extract_features's internal check), reading
     only the raw Frame_ID column.
  2. Independently decides too_short (n_deduped < 50).
  3. Calls the REAL, deployed extract_features() (with frame_index=None,
     which only zeroes the density feature -- density plays no role in
     eligibility, only in R -- so this does not change any accept/reject
     decision, only makes the run fast by skipping the O(large) per-frame
     spatial lookup).
  4. Cross-tabulates: wherever this script's own independent verdict is
     decisive (too_short or missing_frames_above_10pct), the real
     function's returned reason must match exactly. Any mismatch is
     printed in full.
  5. Reports the full, corrected exclusion-reason breakdown (this is also
     the first time the CORRECTED tally code has been exercised against
     the complete candidate pool rather than just the accepted subsample).
"""
import glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

t0 = time.time()
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import SEED
from src.features.ngsim import extract_features as real_extract, trajectories as ngsim_trajs

csvs = sorted(glob.glob(str(ROOT / "data/ngsim/*.csv")))
assert csvs, "no NGSIM csv found"
print(f"[t+{time.time()-t0:6.1f}s] loading raw NGSIM csv: {csvs[0]}")
df = pd.read_csv(csvs[0])
print(f"[t+{time.time()-t0:6.1f}s] loaded, {len(df)} rows")

rng = np.random.default_rng(SEED)
all_tids = [tid for tid, g in ngsim_trajs(df) if len(g) >= 50]
print(f"[t+{time.time()-t0:6.1f}s] length-eligible pool (raw len>=50): {len(all_tids)} "
      f"(pipeline's own printed value was 8310)")

mismatches = []
indep_reason_counts = {"too_short_indep": 0, "missing_frames_indep": 0, "passed_both_checks": 0}
real_reason_counts = {}
n_processed = 0
n_accepted_real = 0

for tid in all_tids:
    loc, vid = tid
    g = df[(df["Location"] == loc) & (df["Vehicle_ID"] == vid)].sort_values("Frame_ID").reset_index(drop=True)

    # --- independent computation, fresh code, not calling extract_features internals ---
    g_dedup = g.drop_duplicates(subset="Frame_ID", keep="first")
    n_dedup = len(g_dedup)
    if n_dedup < 50:
        indep_verdict = "too_short_indep"
    else:
        fids = g_dedup["Frame_ID"].to_numpy(dtype=np.int64)
        n_expected_indep = int(fids.max() - fids.min()) + 1
        frac_missing_indep = 1.0 - (n_dedup / n_expected_indep)
        if frac_missing_indep > 0.10:
            indep_verdict = "missing_frames_indep"
        else:
            indep_verdict = "passed_both_checks"
    indep_reason_counts[indep_verdict] += 1

    # --- call the real, deployed function (frame_index=None: skips density only) ---
    feats, ttc, ok, real_reason = real_extract(g, frame_index=None)
    real_reason_counts[real_reason] = real_reason_counts.get(real_reason, 0) + 1
    if ok:
        n_accepted_real += 1

    # --- cross-check: wherever our independent verdict is decisive, it must match ---
    if indep_verdict == "too_short_indep" and real_reason != "too_short":
        mismatches.append((tid, "indep=too_short", f"real={real_reason}"))
    if indep_verdict == "missing_frames_indep" and real_reason != "missing_frames_above_10pct":
        mismatches.append((tid, "indep=missing_frames", f"real={real_reason}"))
    if indep_verdict == "passed_both_checks" and real_reason == "missing_frames_above_10pct":
        mismatches.append((tid, "indep=passed_missing_check", f"real={real_reason} (DISAGREEMENT)"))
    if indep_verdict == "passed_both_checks" and real_reason == "too_short":
        mismatches.append((tid, "indep=passed_length_check", f"real={real_reason} (DISAGREEMENT)"))

    n_processed += 1
    if n_processed % 1000 == 0:
        print(f"[t+{time.time()-t0:6.1f}s] processed {n_processed}/{len(all_tids)} "
              f"(accepted so far: {n_accepted_real})")

print()
print("=" * 100)
print(f"Total pool processed: {n_processed}")
print(f"Independent verdict breakdown: {indep_reason_counts}")
print(f"REAL (deployed) function reason breakdown (corrected tally): {real_reason_counts}")
print(f"REAL function total accepted (ok=True): {n_accepted_real}  "
      f"(with density on, the real pipeline run reported N=2979 -- density never affects "
      f"eligibility, only the R value, so this count should equal 2979 exactly)")
print(f"Arithmetic check: accepted + sum(rejections) = "
      f"{n_accepted_real + sum(v for k, v in real_reason_counts.items() if k is not None or True)} "
      f"vs pool {len(all_tids)}")
print()
if mismatches:
    print(f"!!! {len(mismatches)} MISMATCHES between independent and real logic:")
    for m in mismatches[:50]:
        print("   ", m)
else:
    print("NO MISMATCHES: independent frac_missing/too_short logic agrees with the deployed "
          "extract_features() on every one of the 8,310 candidates.")
print("=" * 100)
print(f"Total elapsed: {time.time()-t0:.1f}s")
print("INDEPENDENT_EXCLUSION_CHECK_DONE_SENTINEL", "PASS" if (not mismatches and n_accepted_real == 2979) else "FAIL")
