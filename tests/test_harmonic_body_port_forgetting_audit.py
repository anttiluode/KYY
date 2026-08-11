import math

from map.harmonic_body_port_forgetting_audit import (
    analytic_gap,
    audit_pair,
    c4_characters,
    time_to_gap,
)


def test_alternating_pairs_are_identical_at_quotient_port():
    for a, b in [(0, 2), (1, 3)]:
        r = audit_pair(a, b, epsilon=0.1, gamma=1.0, time=0.0)
        assert r.quotient_port_gap < 1e-12


def test_port_only_future_mixer_exposes_residual_fundamental():
    r = audit_pair(0, 2, epsilon=0.1, gamma=1.0, time=0.0)
    assert abs(r.future_mixer_gap_port_only - 0.2) < 1e-12


def test_damping_gap_matches_closed_form():
    for t in [0.0, 0.25, 1.0, 2.5, 8.0]:
        r = audit_pair(0, 2, epsilon=0.07, gamma=1.3, time=t)
        assert abs(r.future_mixer_gap_damped - analytic_gap(0.07, 1.3, t)) < 1e-12


def test_hard_isolation_gives_behavioral_forgetting_without_erasing_hidden_carrier():
    r = audit_pair(0, 2, epsilon=0.3, gamma=0.0, time=0.0)
    assert r.future_mixer_gap_isolated < 1e-12
    z1a, _ = c4_characters(0)
    z1b, _ = c4_characters(2)
    assert abs(z1a - z1b) > 1.9  # hidden physical distinction still exists


def test_hard_erase_also_gives_zero_future_gap():
    r = audit_pair(1, 3, epsilon=0.3, gamma=1.0, time=0.0)
    assert r.future_mixer_gap_erased < 1e-12


def test_time_to_target_gap_certificate():
    t = time_to_gap(epsilon=0.1, gamma=1.0, target_gap=1e-3)
    assert abs(t - math.log(200.0)) < 1e-12
    assert analytic_gap(0.1, 1.0, t) <= 1e-3 * (1 + 1e-12)
