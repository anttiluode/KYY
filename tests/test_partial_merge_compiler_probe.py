from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "partial_merge_compiler_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "partial_merge_compiler_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_exact_pinch_has_required_behavioral_kernel() -> None:
    angle = math.pi / 2
    cert = probe.merge_generator_certificate(angle)
    assert cert["certified"]
    assert cert["behavioral_rank"] == 2
    assert cert["behavioral_kernel_blocks"] == [[0, 1], [2, 3]]
    assert cert["behavioral_image_states"] == [0, 2]
    assert cert["merge_continuous_rank"] == 1
    assert cert["merge_legal_orbit_action_defect"] < 1e-10


def test_exact_pinch_collapses_only_the_required_square_direction() -> None:
    M = probe.EXACT_MERGE
    killed = torch.tensor([1.0, -1.0], dtype=torch.float64)
    kept = torch.tensor([1.0, 1.0], dtype=torch.float64)
    assert torch.linalg.vector_norm(M @ killed).item() < 1e-12
    assert torch.linalg.vector_norm(M @ kept).item() > 0.0


def test_compiled_runtime_matches_cycle_plus_partial_merge_task() -> None:
    angle = math.pi / 2
    orbit = probe.exact_orbit(angle)
    W, b = probe.eq.prototype_decoder(orbit)
    # state: 0 --+1-->1 --M-->0 --+2-->2 --+1-->3 --M-->2 --+3-->1
    tokens = torch.tensor([[1, probe.MERGE_TOKEN, 2, 1, probe.MERGE_TOKEN, 3]], dtype=torch.long)
    expected = torch.tensor([[1, 0, 2, 3, 2, 1]], dtype=torch.long)
    pred = probe.compiled_runtime(tokens, angle, W, b)[0].argmax(dim=-1)
    assert torch.equal(pred, expected)


def test_compiled_merge_makes_paired_histories_exactly_identical_forever() -> None:
    angle = math.pi / 2
    orbit = probe.exact_orbit(angle)
    W, b = probe.eq.prototype_decoder(orbit)
    leak = probe.compiled_leakage(angle, W, b, batch_size=32, continuation_length=12)
    assert leak["hidden_difference_at_merge"] < 1e-12
    assert leak["hidden_difference_max_future"] < 1e-12
    assert leak["probability_tv_max_future"] < 1e-12
    assert leak["prediction_mismatch_max_rate"] == 0.0
