import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "metacircuit_tolerance_certificate_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "metacircuit_tolerance_certificate.py"
)
assert SPEC is not None and SPEC.loader is not None
cert = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = cert
SPEC.loader.exec_module(cert)


def test_zero_tolerance_recovers_exact_symbolic_margin():
    d = cert.design.greedy_design(101, 8, 2.0)
    margin = cert.robust_phase_margin(101, d.frequencies, 0.0, 1024)
    assert abs(margin - d.equal_weight_min_margin) < 1e-10


def test_conditioned_bank_has_smaller_worst_step_phase_error():
    u = cert.design.greedy_design(101, 8, None)
    c = cert.design.greedy_design(101, 8, 2.0)
    eta = 1e-5
    uerr = max(cert.worst_mode_phase_errors(101, u.frequencies, eta))
    cerr = max(cert.worst_mode_phase_errors(101, c.frequencies, eta))
    assert cerr < uerr / 3.0


def test_conditioned_bank_certifies_longer_static_mismatch_free_run():
    u = cert.design.greedy_design(101, 8, None)
    c = cert.design.greedy_design(101, 8, 2.0)
    ucycles, ucensored = cert.max_certified_cycles(101, u.frequencies, 1e-5, 5000)
    ccycles, ccensored = cert.max_certified_cycles(101, c.frequencies, 1e-5, 5000)
    assert not ucensored
    assert not ccensored
    assert 250 <= ucycles <= 270
    assert 640 <= ccycles <= 670
    assert ccycles > 2.4 * ucycles


def test_static_tolerance_horizon_is_approximately_inverse_in_eta():
    c = cert.design.greedy_design(101, 8, 2.0)
    t1, _ = cert.max_certified_cycles(101, c.frequencies, 1e-5, 5000)
    t2, _ = cert.max_certified_cycles(101, c.frequencies, 2e-5, 5000)
    assert 1.8 < t1 / t2 < 2.2
