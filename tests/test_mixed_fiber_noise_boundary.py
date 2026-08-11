import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_noise_boundary_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_noise_boundary.py")
assert SPEC is not None and SPEC.loader is not None
noise = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = noise
SPEC.loader.exec_module(noise)
base = noise.base


def test_zero_noise_all_exact_and_analog_numerical_floor():
    model = base.SoftFiberTracker(0)
    tok, q0, a0, qy, ay = noise.generate_eval(32, 32, 0.05, 0.08, 123)
    lc, ac = noise.compiled_noisy_runtime(model, tok, q0, a0, 0.0, reify=False, seed=1)
    lr, ar = noise.compiled_noisy_runtime(model, tok, q0, a0, 0.0, reify=True, seed=1)
    qh, ah = noise.explicit_hybrid_runtime(tok, q0, a0, 0.0, seed=1)
    mc = noise.metrics(lc.argmax(-1), ac, qy, ay)
    mr = noise.metrics(lr.argmax(-1), ar, qy, ay)
    mh = noise.metrics(qh, ah, qy, ay)
    assert mc["q_accuracy"] == 1.0
    assert mr["q_accuracy"] == 1.0
    assert mh["q_accuracy"] == 1.0
    assert mc["analog_rmse"] < 1e-6
    assert mr["analog_rmse"] < 1e-6
    assert mh["analog_rmse"] < 1e-12


def test_explicit_hybrid_digital_state_is_immune_to_continuous_noise():
    tok, q0, a0, qy, ay = noise.generate_eval(32, 64, 0.05, 0.08, 456)
    qh, ah = noise.explicit_hybrid_runtime(tok, q0, a0, 0.1, seed=7)
    result = noise.metrics(qh, ah, qy, ay)
    assert result["q_accuracy"] == 1.0
    assert result["analog_rmse"] > 0.0
