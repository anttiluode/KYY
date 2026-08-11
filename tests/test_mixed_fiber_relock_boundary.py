import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_relock_boundary_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_relock_boundary.py")
assert SPEC is not None and SPEC.loader is not None
relock = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = relock
SPEC.loader.exec_module(relock)
base = relock.base
noise = relock.noise


def test_interval_one_matches_every_step_reification_contract():
    model = base.SoftFiberTracker(0)
    tok, q0, a0, qy, ay = noise.generate_eval(32, 64, 0.05, 0.08, 123)
    logits, analog, n = relock.periodic_relock_runtime(model, tok, q0, a0, 0.01, 1, 9)
    m = noise.metrics(logits.argmax(-1), analog, qy, ay)
    assert n == 64
    assert m["q_accuracy"] == 1.0


def test_never_relock_performs_no_projection():
    model = base.SoftFiberTracker(1)
    tok, q0, a0, _, _ = noise.generate_eval(8, 16, 0.05, 0.08, 456)
    _, _, n = relock.periodic_relock_runtime(model, tok, q0, a0, 0.01, None, 7)
    assert n == 0
