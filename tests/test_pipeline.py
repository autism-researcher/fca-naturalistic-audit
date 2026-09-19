"""Smoke tests for the core pipeline modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

def test_weights_sum_094():
    from src.utils import load_weights
    w, _ = load_weights()
    assert abs(w.sum() - 0.94) < 1e-6

def test_composite_risk_bounded():
    from src.risk import composite_risk
    from src.utils import load_weights
    w, _ = load_weights()
    f = np.ones((20, 8))
    R = composite_risk(f, w)
    assert R.max() <= 1.0 and R.min() >= 0.0

def test_dkw_floor():
    from src.boundary import dkw_epsilon
    eps = dkw_epsilon(738, delta=0.05)
    assert abs(eps - 0.05) < 5e-3, f"DKW eps at N=738: {eps}"

def test_boundary_and_realized():
    from src.boundary import boundary, realized_rate
    np.random.seed(0)
    r = np.random.uniform(0, 1, 1000)
    B = boundary(r, 0.10)
    rate = realized_rate(r, B)
    assert abs(rate - 0.10) < 0.02

def test_h_off1():
    from src.tests import h_off1
    assert h_off1(0.50, 0.51)["pass"]
    assert not h_off1(0.50, 0.54)["pass"]

def test_h_off3_with_separation():
    from src.tests import h_off3
    crossing = [10]*50; noncrossing = [1]*50
    v = h_off3(crossing, noncrossing)
    assert v["pass"]

def _load_disjunctive_pass():
    """scripts/09_generate_tables.py is not a package (filename starts with a
    digit), so load it by path rather than `import`."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / "09_generate_tables.py"
    spec = importlib.util.spec_from_file_location("generate_tables_09", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.disjunctive_pass

def test_disjunctive_pass_grouping():
    """2026-09 regression test (deviations register) for the fix to the
    overall-verdict aggregation bug: PASS requires ALL three tau to pass for
    the SAME dataset, not merely any single (dataset, tau) cell anywhere.

    On this study's actual data every hypothesis is unanimous (all 9 cells
    FAIL, or all 9 PASS), so the old flat `any()` bug happened to agree with
    the correct grouped criterion and no published verdict was wrong. This
    test uses a deliberately MIXED synthetic pattern -- one that the old
    code would have gotten wrong -- to pin the fix down.
    """
    disjunctive_pass = _load_disjunctive_pass()

    # Case 1: no single dataset passes all three tau (mixed, 4 of 9 cells
    # pass) -> must be FAIL under the pre-registered criterion, even though
    # "any cell passes" is True. This is exactly the case the old buggy
    # `any(<flat list>)` implementation would have gotten wrong (it would
    # have reported PASS).
    mixed_no_full_dataset = {
        "highd": [True, True, False],   # 2/3 tau pass, not all three
        "ngsim": [False, False, False],
        "waymo": [False, True, False],
    }
    assert disjunctive_pass(mixed_no_full_dataset) is False

    # Case 2: exactly one dataset passes all three tau -> PASS.
    one_full_dataset = {
        "highd": [True, True, True],
        "ngsim": [False, False, False],
        "waymo": [False, True, False],
    }
    assert disjunctive_pass(one_full_dataset) is True

    # Case 3: unanimous all-FAIL (this study's actual H_OFF1/H_OFF2 pattern)
    # and unanimous all-PASS (this study's actual H_OFF3 pattern) -- the
    # cases where the old buggy code happened to still be right.
    assert disjunctive_pass({"highd": [False]*3, "ngsim": [False]*3, "waymo": [False]*3}) is False
    assert disjunctive_pass({"highd": [True]*3, "ngsim": [True]*3, "waymo": [True]*3}) is True

def test_ngsim_missing_frames_exclusion():
    """2026-09 fix (deviations register): NGSIM trajectories with >10%
    missing ticks (gaps in the Frame_ID sequence) must now be excluded, per
    the pre-registration. Previously only |a|>10 m/s^2 and an all-or-nothing
    non-finite-position check were enforced, so a vehicle with real tracking
    gaps but >=5 s of *recorded* ticks passed through uncaught.
    """
    import pandas as pd
    from src.features.ngsim import extract_features

    def make_traj(frame_ids, n_rows):
        return pd.DataFrame({
            "Frame_ID": frame_ids,
            "Local_X": np.linspace(0, 50, n_rows),
            "Local_Y": np.full(n_rows, 10.0),
            "Lane_ID": np.full(n_rows, 2),
            "Space_Headway": np.full(n_rows, 30.0),
            "Preceding": np.zeros(n_rows, dtype=int),
        })

    # Contiguous 60 ticks (6 s at 10 Hz, no gaps) -> must NOT be rejected on
    # missing-data grounds.
    contig = make_traj(np.arange(1, 61), 60)
    _, _, ok, reason = extract_features(contig)
    assert ok, f"contiguous trajectory wrongly rejected: {reason}"

    # 60 recorded ticks spanning Frame_ID 1..200 (140/200 = 70% missing over
    # its own span) -> must be rejected. This is the case the pre-fix code
    # missed (n=60 >= the 50-tick length floor, nothing else catches it).
    rng = np.random.RandomState(0)
    sparse_ids = np.sort(rng.choice(np.arange(1, 201), size=60, replace=False))
    sparse = make_traj(sparse_ids, 60)
    _, _, ok2, reason2 = extract_features(sparse)
    assert not ok2 and reason2 == "missing_frames_above_10pct", (
        f"gappy trajectory not rejected as expected: ok={ok2} reason={reason2}")

if __name__ == "__main__":
    test_weights_sum_094()
    test_composite_risk_bounded()
    test_dkw_floor()
    test_boundary_and_realized()
    test_h_off1()
    test_h_off3_with_separation()
    test_disjunctive_pass_grouping()
    test_ngsim_missing_frames_exclusion()
    print("All smoke tests passed.")
