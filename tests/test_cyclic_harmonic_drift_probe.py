from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_harmonic_drift_probe"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_harmonic_drift_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_common_phase_defect_distance_formula():
    import math
    import numpy as np

    n = 31
    freqs = [1, 3, 7, 11]
    eta = 1e-3
    proto = probe.harmonic.harmonic_prototypes(n, freqs)
    A_bad = probe.defect_operator(n, freqs, eta)
    z = proto[0].copy()
    for t in range(1, 20):
        z = A_bad @ z
        ideal = proto[t % n]
        measured = float(np.linalg.norm(z - ideal))
        expected = probe.common_defect_distance(t, eta)
        assert math.isclose(measured, expected, rel_tol=1e-10, abs_tol=1e-12)


def test_single_phase_safe_horizon_shrinks_with_modulus():
    eta = 1e-4
    horizons = []
    for n in (31, 101, 1009):
        radius = probe.harmonic.geometric_metrics(n, [1])["nearest_prototype_noise_radius"]
        horizons.append(probe.safe_steps_from_radius(float(radius), eta))
    assert horizons[0] > horizons[1] > horizons[2]
    assert horizons[-1] < horizons[0] / 20


def test_harmonic_margin_buys_order_of_magnitude_more_defect_runway():
    n = 31
    eta = 1e-3
    single_fail = probe.first_decoder_failure(n, [1], eta, max_steps=2000)
    freqs, metrics = probe.harmonic.random_search(n, 8, trials=500, seed=0)
    harmonic_fail = probe.first_decoder_failure(n, freqs, eta, max_steps=2000)

    assert single_fail is not None
    assert harmonic_fail is not None
    assert float(metrics["nearest_prototype_noise_radius"]) > 0.6
    assert harmonic_fail > 8 * single_fail
