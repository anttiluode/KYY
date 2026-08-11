import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "metacircuit_frequency_design_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "metacircuit_frequency_design.py")
assert SPEC is not None and SPEC.loader is not None
design = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = design
SPEC.loader.exec_module(design)


def test_condition_cap_trades_small_symbolic_margin_for_large_physical_improvement():
    free = design.greedy_design(101, 8, None)
    capped = design.greedy_design(101, 8, 2.0)
    assert free.certified_faithful
    assert capped.certified_faithful
    assert capped.max_phase_map_condition <= 2.0 + 1e-12
    assert capped.max_phase_map_condition < free.max_phase_map_condition
    assert capped.min_positive_relative_stability_headroom > free.min_positive_relative_stability_headroom
    assert capped.equal_weight_min_margin > 0.95 * free.equal_weight_min_margin


def test_composite_modulus_design_remains_faithful():
    row = design.greedy_design(100, 8, 2.0)
    assert row.character_gcd == 1
    assert row.certified_faithful
