import torch

from kyy.models import GeometricWaveSSM, build_model, weighted_laplacian_apply
from kyy.tasks import TASKS, generate_batch


def test_task_shapes():
    for task, spec in TASKS.items():
        x, y = generate_batch(task, 7, 13)
        assert x.shape == (7, 13)
        assert y.shape == (7, 13)
        assert int(x.max()) < spec.vocab_size
        assert int(y.max()) < spec.n_classes


def test_known_parity():
    x = torch.tensor([[1, 1, 0, 1]])
    y = torch.cumsum(x, dim=1).remainder(2)
    assert y.tolist() == [[1, 0, 0, 1]]


def test_weighted_laplacian_annihilates_constant():
    q = torch.ones(3, 5)
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    w = torch.rand(3, 5)
    out = weighted_laplacian_apply(q, src, dst, w)
    assert torch.allclose(out, torch.zeros_like(out))


def test_models_forward_shape():
    tokens = torch.randint(0, 2, (4, 11))
    for name in ("diag_signed", "complex_diag", "householder2", "geom_wave", "geom_scatter", "gru"):
        model = build_model(name, vocab_size=2, n_classes=3, state_dim=16)
        out = model(tokens)
        assert out.shape == (4, 11, 3)
        assert torch.isfinite(out).all()


def test_geometric_transition_is_finite():
    model = GeometricWaveSSM(vocab_size=2, n_classes=2, state_dim=16, topology="ring")
    A = model.dense_transition(0)
    eig = torch.linalg.eigvals(A)
    assert A.shape == (16, 16)
    assert torch.isfinite(eig).all()


def test_geometric_scatter_transition_is_orthogonal():
    from kyy.models import GeometricScatterSSM
    model = GeometricScatterSSM(vocab_size=3, n_classes=2, state_dim=16, topology="ring")
    A = model.dense_transition(1)
    assert torch.allclose(A.T @ A, torch.eye(16), atol=2e-5, rtol=2e-5)


def test_matching_topology_has_no_cross_pair_edges():
    from kyy.models import make_graph
    g = make_graph(8, "matching")
    assert list(zip(g.src.tolist(), g.dst.tolist())) == [(0, 1), (2, 3), (4, 5), (6, 7)]
