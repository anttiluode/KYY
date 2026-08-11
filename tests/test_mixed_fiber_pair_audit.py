import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_pair_audit_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_pair_audit.py")
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = audit
SPEC.loader.exec_module(audit)
base = audit.base


def test_reification_can_collapse_digital_identity_but_preserve_tangent_history():
    model = base.SoftFiberTracker(0)
    with torch.no_grad():
        model.eps_logit.fill_(-4.0)
        model.r0.fill_(0.12)
        model.r1.fill_(-0.08)
        model.gain.fill_(1.0)
    tokens = torch.full((8, 1), base.MERGE, dtype=torch.long)
    qa = torch.zeros(8, dtype=torch.long)
    qb = torch.ones(8, dtype=torch.long)
    a0 = torch.linspace(-1.0, 1.0, 8)
    result = audit.reified_pair_audit(model, tokens, qa, qb, a0)
    assert result["q_mismatch_at_merge"] == 0.0
    assert abs(result["analog_gap_at_merge_mean"] - 0.20) < 1e-5


def test_exact_compiler_erases_only_forbidden_history_and_preserves_shared_analog():
    model = base.SoftFiberTracker(1)
    tokens = torch.tensor([[base.MERGE, 1, base.SCALE, 3]] * 8, dtype=torch.long)
    qa = torch.zeros(8, dtype=torch.long)
    qb = torch.ones(8, dtype=torch.long)
    a0 = torch.linspace(-1.0, 1.0, 8)
    result = audit.compiled_pair_audit(model, tokens, qa, qb, a0)
    assert result["q_output_mismatch_at_merge"] == 0.0
    assert result["q_output_mismatch_max_future_rate"] == 0.0
    assert result["analog_gap_at_merge_mean"] < 1e-12
    assert result["analog_gap_max_future"] < 1e-12
