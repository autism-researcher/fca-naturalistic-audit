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

if __name__ == "__main__":
    test_weights_sum_094()
    test_composite_risk_bounded()
    test_dkw_floor()
    test_boundary_and_realized()
    test_h_off1()
    test_h_off3_with_separation()
    print("All smoke tests passed.")
