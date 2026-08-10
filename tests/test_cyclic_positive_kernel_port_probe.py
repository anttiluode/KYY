from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_positive_kernel_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_positive_kernel_port_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_positive_kernel_certificate_accepts_faithful_active_characters() -> None:
    ok, g = probe.positive_kernel_certificate(
        12,
        np.asarray([2, 5]),
        np.asarray([0.7, 0.2]),
    )
    assert ok
    assert g == 1


def test_positive_kernel_certificate_ignores_zero_weight_mode() -> None:
    ok, g = probe.positive_kernel_certificate(
        15,
        np.asarray([5, 2]),
        np.asarray([1.0, 0.0]),
    )
    assert not ok
    assert g == 5


def test_positive_kernel_certificate_rejects_negative_weight() -> None:
    ok, _ = probe.positive_kernel_certificate(
        11,
        np.asarray([1, 3]),
        np.asarray([0.5, -0.1]),
    )
    assert not ok
