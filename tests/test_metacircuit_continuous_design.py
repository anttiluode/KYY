import importlib.util
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME="metacircuit_continuous_design_for_tests"
SPEC=importlib.util.spec_from_file_location(NAME,ROOT/"map"/"metacircuit_continuous_design.py")
assert SPEC and SPEC.loader
m=importlib.util.module_from_spec(SPEC);sys.modules[NAME]=m;SPEC.loader.exec_module(m)


def test_exhaustive_physical_bank_is_well_conditioned_and_strong():
    c=m.contributions(101)
    row,candidates=m.exhaustive_constrained(101,8,2.0,31,c)
    assert row["frequencies"] == [16,18,19,20,25,28,30,31]
    assert row["symbolic_margin"] > 5.70
    assert row["max_phase_map_condition"] < 1.85
    assert row["max_relative_phase_sensitivity"] < 0.97
    assert len(candidates) >= 8


def test_physical_bank_keeps_most_margin_of_strong_digital_heuristic():
    c=m.contributions(101)
    physical,_=m.exhaustive_constrained(101,8,2.0,31,c)
    digital=m.local_digital_search(101,8,c,seed=1,random_samples=20000,restarts=20)
    assert physical["symbolic_margin"] > 0.95*digital["symbolic_margin"]
    assert physical["max_phase_map_condition"] < digital["max_phase_map_condition"]
    assert physical["max_relative_phase_sensitivity"] < digital["max_relative_phase_sensitivity"]


def test_physical_bank_has_longer_bounded_static_tolerance_certificate():
    c=m.contributions(101)
    physical,_=m.exhaustive_constrained(101,8,2.0,31,c)
    digital=m.local_digital_search(101,8,c,seed=1,random_samples=20000,restarts=20)
    pc=m.cont.max_certified_cycles(101,physical["frequencies"],1e-5)
    dc=m.cont.max_certified_cycles(101,digital["frequencies"],1e-5)
    assert pc > dc
