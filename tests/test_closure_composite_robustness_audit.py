from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "closure_composite_robustness_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "closure_composite_robustness_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_composite_gcd_controls_hit_both_outcomes() -> None:
    rows = probe.certificate_controls()
    assert [r["measured_gcd"] for r in rows] == [2, 1, 3, 1]
    assert [r["certified"] for r in rows] == [False, True, False, True]
    assert all(r["measured_gcd"] == r["expected_gcd"] for r in rows)
