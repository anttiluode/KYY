from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "harmonic_training_probe"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "harmonic_training_probe.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_exact_character_schedules_have_zero_group_relation_defect():
    n, k = 31, 8
    for f in (
        probe.prime_frequencies(n, k),
        probe.geometric_character_frequencies(n, k),
        probe.low_coherence_frequencies(n, k, trials=50, seed=0),
    ):
        angles = 2.0 * math.pi * f / n
        assert probe.character_defect(n, angles) < 1e-12


def test_standard_rope_is_not_an_exact_mod31_character_bank():
    defect = probe.character_defect(31, probe.standard_rope_angles(8))
    assert defect > 1e-2


def test_low_coherence_search_beats_single_phase_margin_for_mod31():
    f = probe.low_coherence_frequencies(31, 8, trials=100, seed=31)
    radius, _ = probe.character_margin(31, f)
    assert radius > 0.60
    assert radius > 5.0 * math.sin(math.pi / 31)


def test_rotary_tracker_shapes():
    n = 7
    f = np.asarray([1, 2, 3], dtype=np.int64)
    model = probe.RotaryModTracker(n, 2.0 * math.pi * f / n, learn_angles=False)
    x = torch.tensor([[1, 2, 0, 3], [0, 1, 1, 1]], dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, 4, n)
