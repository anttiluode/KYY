from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from kyy import generate_batch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exact_perm3_local_oracle", ROOT / "map" / "exact_perm3_local_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def test_group_relations_and_local_factorizations_are_exact():
    checks = oracle.exhaustive_relation_checks()
    assert all(checks.values())


def test_oracle_tracks_perm3_exactly_far_beyond_training_lengths():
    torch.manual_seed(123)
    x, target = generate_batch("perm3", batch_size=16, length=2048, device="cpu")
    pred = oracle.run_tokens(x)
    assert torch.equal(pred, target)


def test_oracle_resource_point():
    _, _, words, prototypes, _ = oracle.build_oracle()
    assert prototypes.shape == (6, 3)
    assert [len(w) for w in words] == [0, 1, 2]
