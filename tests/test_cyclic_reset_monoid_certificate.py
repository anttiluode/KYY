from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_reset_monoid_certificate_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_reset_monoid_certificate.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_word_reduction_keeps_only_suffix_after_last_reset() -> None:
    n = 11
    assert probe.reduce_word(n, [3, 4, 8]).kind == "rotation"
    assert probe.reduce_word(n, [3, 4, 8]).index == 4
    a = probe.reduce_word(n, [7, 5, "R", 3, 9])
    b = probe.reduce_word(n, [1, "R", 3, 9])
    assert a == b == probe.NormalForm("constant", 1)


def test_faithful_characters_certify_two_n_transformations() -> None:
    report = probe.certificate(15, [2, 5])
    assert report.certified
    assert report.character_gcd == 1
    assert report.group_normal_forms == 15
    assert report.constant_normal_forms == 15
    assert report.predicted_transformation_monoid_size == 30


def test_nonfaithful_characters_collapse_rotation_and_constant_orbits() -> None:
    report = probe.certificate(12, [2, 4])
    assert not report.certified
    assert report.character_gcd == 2
    assert report.group_normal_forms == 6
    assert report.constant_normal_forms == 6
    assert report.predicted_transformation_monoid_size == 12
