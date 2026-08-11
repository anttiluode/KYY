import torch

from map.mixed_fiber_compiler_probe import (
    SoftFiberTracker,
    exact_ports,
    fiber_basis,
    nearest_fiber,
    synthesize_exact_operators,
)


def test_exact_operators_preserve_fiber_and_merge_digital_generators():
    model = SoftFiberTracker(0)
    c0, c1, v = fiber_basis(model)
    cycle, merge, scale = synthesize_exact_operators(model)
    assert torch.linalg.vector_norm(cycle @ c0 - c1) < 1e-10
    assert torch.linalg.vector_norm(cycle @ c1 + c0) < 1e-10
    assert torch.linalg.vector_norm(cycle @ v - v) < 1e-10
    assert torch.linalg.vector_norm(merge @ c0 - c0) < 1e-10
    assert torch.linalg.vector_norm(merge @ c1 - c0) < 1e-10
    assert torch.linalg.vector_norm(merge @ v - v) < 1e-10
    assert torch.linalg.vector_norm(scale @ v - 0.9 * v) < 1e-10


def test_exact_analog_port_reads_fiber_coordinate():
    model = SoftFiberTracker(1)
    c0, c1, v = fiber_basis(model)
    _, _, wa = exact_ports(model)
    assert abs(float(c0 @ wa)) < 1e-10
    assert abs(float(c1 @ wa)) < 1e-10
    assert abs(float(v @ wa) - 1.0) < 1e-10


def test_nearest_fiber_snaps_transverse_error_but_keeps_tangent_coordinate():
    model = SoftFiberTracker(2)
    c0, _, v = fiber_basis(model)
    h = (c0 + 0.37 * v + 0.04 * fiber_basis(model)[1]).view(1, 3).to(torch.float32)
    snapped, q, a = nearest_fiber(model, h)
    assert int(q.item()) == 0
    assert abs(float(a.item()) - 0.37) < 1e-5
    assert torch.linalg.vector_norm(snapped.to(torch.float64)[0] - (c0 + 0.37 * v)) < 1e-5
