import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "physical_cycle_port_boundary_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "physical_cycle_port_boundary.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_exact_body_remains_exact_at_all_winding_histories():
    port = probe.exact_prototype_port(4)
    for cycles in [1, 16, 256]:
        x, y = probe.states(4, 0.0, cycles)
        acc, margin = probe.accuracy_margin(port, x, y)
        assert acc == 1.0
        assert margin > 0.9


def test_finite_horizon_port_calibration_does_not_repair_body_relation():
    _, rows = probe.run(4, 1e-3, 16, [16, 1024])
    assert rows[0].calibrated_port_accuracy > 0.99
    assert rows[1].calibrated_port_accuracy < 0.9
    assert rows[1].min_interclass_distance < rows[0].min_interclass_distance
    assert rows[1].legalized_accuracy == 1.0
