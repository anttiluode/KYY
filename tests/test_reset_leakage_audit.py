from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "reset_leakage_audit"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "reset_leakage_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = audit
SPEC.loader.exec_module(audit)


def test_reset_pairs_have_different_prefix_state_and_same_post_reset_behavior():
    torch.manual_seed(0)
    a, b, reset_index = audit.make_reset_pairs(64, 12, 20)
    audit.validate_reset_pairs(a, b, reset_index)
    ya = audit.permreset3_targets(a)
    yb = audit.permreset3_targets(b)
    assert torch.all(ya[:, reset_index - 1] != yb[:, reset_index - 1])
    assert torch.equal(ya[:, reset_index:], yb[:, reset_index:])


def test_exact_behavioral_logits_have_zero_post_reset_leakage():
    torch.manual_seed(1)
    a, b, reset_index = audit.make_reset_pairs(32, 8, 16)
    ya = audit.permreset3_targets(a)
    yb = audit.permreset3_targets(b)
    logits_a = torch.nn.functional.one_hot(ya, num_classes=3).float() * 8.0
    logits_b = torch.nn.functional.one_hot(yb, num_classes=3).float() * 8.0
    curve = audit.leakage_curve_from_logits(logits_a, logits_b, reset_index)
    for row in curve:
        assert row["mean_logit_l2"] == 0.0
        assert row["mean_probability_tv"] == 0.0
        assert row["prediction_mismatch_rate"] == 0.0


def test_port_metric_detects_hidden_history_if_it_leaks_after_reset():
    torch.manual_seed(2)
    a, b, reset_index = audit.make_reset_pairs(8, 6, 4)
    y = audit.permreset3_targets(a)
    logits_a = torch.nn.functional.one_hot(y, num_classes=3).float() * 4.0
    logits_b = logits_a.clone()
    # Inject a visible difference one step after reset without changing the
    # benchmark target.  The metric should detect it even if argmax stays equal.
    logits_b[:, reset_index + 1, 1] += 0.5
    curve = audit.leakage_curve_from_logits(logits_a, logits_b, reset_index)
    assert curve[1]["mean_logit_l2"] > 0.0
    assert curve[1]["mean_probability_tv"] > 0.0
