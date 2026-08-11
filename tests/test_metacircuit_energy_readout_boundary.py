import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = "metacircuit_energy_readout_for_tests"
SPEC = importlib.util.spec_from_file_location(NAME, ROOT / "map" / "metacircuit_energy_readout_boundary.py")
assert SPEC is not None and SPEC.loader is not None
m = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = m
SPEC.loader.exec_module(m)

FREQS = [16,18,19,20,25,28,30,31]


def test_current_state_phase_prototypes_distinguish_all_c101_states():
    assert m.instantaneous_accuracy(101, FREQS) == 1.0


def test_any_tested_fixed_projection_has_full_period_energy_invariant_to_phase_offset():
    for seed in range(5):
        w = np.random.default_rng(seed).normal(size=2 * len(FREQS))
        e = m.full_period_energies(101, FREQS, w)
        assert e.max() - e.min() < 1e-10


def test_partial_energy_window_can_retain_phase_but_full_period_erases_it():
    w = np.random.default_rng(9).normal(size=2 * len(FREQS))
    assert m.partial_window_spread(101, FREQS, w, 16) > 1e-3
    assert m.partial_window_spread(101, FREQS, w, 101) < 1e-10
