import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_relock_frontier_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_relock_frontier.py")
assert SPEC is not None and SPEC.loader is not None
frontier = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = frontier
SPEC.loader.exec_module(frontier)


def test_frontier_selects_largest_interval_meeting_both_thresholds():
    rows = [
        frontier.FrontierRow(0.01, 4, 1.0, 1.0, 0.1),
        frontier.FrontierRow(0.01, 8, 0.9995, 0.995, 0.1),
        frontier.FrontierRow(0.01, 16, 0.998, 1.0, 0.1),
    ]
    out = frontier.summarize(rows, 0.999, 0.99)
    assert out[0]["largest_tested_safe_interval"] == 8
