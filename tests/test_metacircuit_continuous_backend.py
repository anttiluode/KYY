import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAME="metacircuit_continuous_backend_for_tests"
SPEC=importlib.util.spec_from_file_location(NAME,ROOT/"map"/"metacircuit_continuous_backend.py")
assert SPEC is not None and SPEC.loader is not None
m=importlib.util.module_from_spec(SPEC);sys.modules[NAME]=m;SPEC.loader.exec_module(m)


def test_continuous_ratio_differs_from_central_difference_except_small_angle_limit():
    theta=2*math.pi*25/101
    continuous=m.continuous_ratio(theta)
    discrete=m.discrete.required_admittance_over_fdnr(theta)
    assert abs(continuous-discrete) > 0.4


def test_continuous_ratio_samples_to_exact_character_recurrence():
    row=m.lower_mode(101,25)
    assert row.sampled_relation_defect < 1e-10
    assert abs(row.natural_frequency-row.theta) < 1e-12
    assert abs(row.admittance_over_fdnr-row.theta**2) < 1e-12


def test_no_fake_discrete_stability_cliff_near_nyquist():
    row=m.lower_mode(101,50)
    assert row.admittance_over_fdnr > 9.0
    assert row.sampled_relation_defect < 1e-9


def test_continuous_mismatch_advantage_is_modest_not_order_of_magnitude():
    u=m.symbolic.greedy_design(101,8,None).frequencies
    c=m.symbolic.greedy_design(101,8,2.0).frequencies
    eps=np.random.default_rng(4600).normal(size=8)*1e-5
    ud=m.relation_defect(101,u,eps)
    cd=m.relation_defect(101,c,eps)
    assert cd < ud
    assert ud/cd < 2.0


def test_continuous_bounded_tolerance_horizon_is_longer_for_conditioned_bank():
    u=m.symbolic.greedy_design(101,8,None).frequencies
    c=m.symbolic.greedy_design(101,8,2.0).frequencies
    assert 700 <= m.max_certified_cycles(101,u,1e-5) <= 720
    assert 890 <= m.max_certified_cycles(101,c,1e-5) <= 910
