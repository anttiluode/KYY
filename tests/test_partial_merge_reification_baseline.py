from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "partial_merge_reification_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "partial_merge_reification_baseline.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_nearest_prototype_returns_c4_state() -> None:
    p = probe.legal_prototypes()
    h = torch.tensor([[0.91, 0.08], [-0.1, -0.95]], dtype=torch.float32)
    snapped, idx = probe.nearest_prototype(h, p)
    assert idx.tolist() == [0, 3]
    assert torch.equal(snapped, p[idx])


def test_exact_merge_then_reify_collides_zero_and_one() -> None:
    p = probe.legal_prototypes(dtype=torch.float64)
    merged = p[:2] @ probe.pm.EXACT_MERGE.T
    snapped, idx = probe.nearest_prototype(merged, p)
    assert idx.tolist() == [0, 0]
    assert torch.allclose(snapped[0], snapped[1])
