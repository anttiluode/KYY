import importlib.util
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME="shil_c4_quotient_for_tests"
SPEC=importlib.util.spec_from_file_location(NAME,ROOT/"map"/"shil_c4_quotient_backend.py")
assert SPEC and SPEC.loader
m=importlib.util.module_from_spec(SPEC);sys.modules[NAME]=m;SPEC.loader.exec_module(m)


def test_analytic_merge_phase_maximizes_symmetric_basin_margin():
    a,margin=m.best_alpha_grid(20001)
    assert abs(a-math.pi/8) < 1e-4
    assert abs(margin-math.pi/8) < 1e-4
    assert m.deterministic_merge_margin(math.pi/8) > m.deterministic_merge_margin(math.pi/16)
    assert m.deterministic_merge_margin(math.pi/8) > m.deterministic_merge_margin(3*math.pi/16)


def test_compiled_alpha_merges_all_four_states_without_noise():
    r=m.one_merge_trial(math.pi/8,0.0,200,1)
    assert r["accuracy"] == 1.0


def test_midpoint_alpha_is_not_a_safe_c4_reentry_geometry():
    assert m.deterministic_merge_margin(math.pi/4) == 0.0


def test_compiled_alpha_beats_midpoint_under_small_phase_diffusion():
    good=m.one_merge_trial(math.pi/8,0.005,2000,10)
    bad=m.one_merge_trial(math.pi/4,0.005,2000,10)
    assert good["accuracy"] > 0.99
    assert bad["accuracy"] < 0.65
