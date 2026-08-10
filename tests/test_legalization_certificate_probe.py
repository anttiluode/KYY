from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "legalization_certificate_probe"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "legalization_certificate_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def prototype_readout(n: int, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = probe.phase_prototypes(n, angles).numpy()
    return z.copy(), np.zeros(n, dtype=np.float64)


def test_zero_snap_is_certified_for_exact_distinct_orbit():
    n = 7
    angles = np.asarray([2.0 * math.pi / n, 6.0 * math.pi / n], dtype=np.float64)
    W, b = prototype_readout(n, angles)
    cert = probe.decoder_preservation_certificate(n, angles, angles, W, b)
    assert cert.pre_prototype_accuracy == 1.0
    assert cert.projected_prototype_accuracy == 1.0
    assert cert.exhaustive_orbit_certified
    assert cert.cauchy_certified
    assert cert.max_snap_distance == 0.0
    assert cert.projected_min_true_margin > 0.0
    assert cert.cauchy_min_slack > 0.0


def test_large_snap_can_break_decoder_and_certificate():
    n = 4
    learned = np.asarray([math.pi / 2.0], dtype=np.float64)
    collapsed = np.asarray([0.0], dtype=np.float64)
    W, b = prototype_readout(n, learned)
    cert = probe.decoder_preservation_certificate(n, learned, collapsed, W, b)
    assert cert.pre_prototype_accuracy == 1.0
    assert cert.projected_prototype_accuracy < 1.0
    assert not cert.exhaustive_orbit_certified
    assert not cert.cauchy_certified
    assert cert.projected_min_true_margin <= 0.0


def test_cauchy_certificate_is_sufficient_not_necessary():
    n = 5
    learned = np.asarray([2.0 * math.pi / n], dtype=np.float64)
    # A tiny phase perturbation keeps the prototype classifier correct.
    projected = learned + 1e-3
    W, b = prototype_readout(n, learned)
    cert = probe.decoder_preservation_certificate(n, learned, projected, W, b)
    if cert.cauchy_certified:
        assert cert.exhaustive_orbit_certified
    # Regardless of whether the conservative bound fires, direct exhaustive
    # finite-orbit verification is the ground truth for this toy compiler.
    assert cert.projected_prototype_accuracy == 1.0
    assert cert.exhaustive_orbit_certified
