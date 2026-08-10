from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_prototype_certificate_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_prototype_certificate.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_gcd_one_certifies_unique_prototype_decode() -> None:
    ok, kernel = probe.prototype_correctness_certificate(12, [2, 5])
    assert ok
    assert kernel == 1
    report = probe.audit(12, [2, 5])
    assert report.exhaustive_zero_margin_displacements == []
    assert report.exhaustive_min_nonzero_margin > 0.0


def test_nontrivial_gcd_predicts_exact_collisions() -> None:
    ok, kernel = probe.prototype_correctness_certificate(12, [2, 4])
    assert not ok
    assert kernel == 2
    report = probe.audit(12, [2, 4])
    assert report.predicted_orbit_size == 6
    assert report.exhaustive_zero_margin_displacements == [6]


def test_kernel_size_matches_number_of_zero_margin_group_elements() -> None:
    for n, frequencies in [(15, [3, 6]), (18, [4, 10]), (20, [5, 10]), (21, [6, 9])]:
        report = probe.audit(n, frequencies)
        # The identity plus zero-margin nonidentity displacements are the kernel.
        assert 1 + len(report.exhaustive_zero_margin_displacements) == report.character_gcd
        assert report.predicted_orbit_size * report.character_gcd == n


def test_zero_amplitude_modes_do_not_fake_faithfulness() -> None:
    # f=5 alone on C15 has kernel size 5. The inactive f=2 mode must not repair it.
    ok, kernel = probe.prototype_correctness_certificate(
        15, [5, 2], squared_amplitudes=np.array([1.0, 0.0])
    )
    assert not ok
    assert kernel == 5
