import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "metacircuit_nonideality_calibration_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "metacircuit_nonideality_calibration_boundary.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def banks():
    unconstrained = probe.design.greedy_design(101, 8, None).frequencies
    conditioned = probe.design.greedy_design(101, 8, 2.0).frequencies
    return unconstrained, conditioned


def test_ideal_companion_body_is_exactly_periodic_over_winding_histories():
    _, conditioned = banks()
    x, _ = probe.physical_features(101, conditioned, cycles=3)
    # q=0 appears at rows k=0,1,2 because ordering is q outer, winding k inner.
    assert np.allclose(x[0], x[1], atol=1e-9)
    assert np.allclose(x[1], x[2], atol=1e-9)


def test_static_sensor_coordinate_distortion_is_a_port_problem():
    _, conditioned = banks()
    sensor = probe.orthogonal_sensor(16, 5500)
    xt, yt = probe.physical_features(101, conditioned, cycles=16)
    x, y = probe.physical_features(101, conditioned, cycles=128)
    original = probe.exact_companion_port(101, conditioned)
    uncalibrated, _ = probe.accuracy_margin(original, x @ sensor.T, y)
    calibrated = probe.fit_affine_port(xt @ sensor.T, yt, 101)
    repaired, _ = probe.accuracy_margin(calibrated, x @ sensor.T, y)
    assert uncalibrated < 0.2
    assert repaired == 1.0


def test_backend_conditioned_bank_survives_same_small_ratio_error_longer():
    unconstrained, conditioned = banks()
    sensor = probe.orthogonal_sensor(16, 5500)
    z = np.random.default_rng(4600).normal(size=8)
    eps = 1e-5 * z

    scores = {}
    for name, frequencies in [("unconstrained", unconstrained), ("conditioned", conditioned)]:
        xt, yt = probe.physical_features(101, frequencies, 16, eps, np.zeros(8))
        port = probe.fit_affine_port(xt @ sensor.T, yt, 101)
        x, y = probe.physical_features(101, frequencies, 1024, eps, np.zeros(8))
        scores[name] = probe.accuracy_margin(port, x @ sensor.T, y)[0]

    assert scores["conditioned"] > 0.99
    assert scores["unconstrained"] < 0.90
    assert scores["conditioned"] > scores["unconstrained"]


def test_exact_body_trim_plus_port_calibration_is_exact_at_long_horizon():
    unconstrained, conditioned = banks()
    sensor = probe.orthogonal_sensor(16, 5500)
    for frequencies in [unconstrained, conditioned]:
        acc, _ = probe.trimmed_control(101, frequencies, 16, 1024, sensor)
        assert acc == 1.0
