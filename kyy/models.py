from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class SequenceStateModel(nn.Module):
    name = "base"

    def __init__(self, vocab_size: int, n_classes: int, state_dim: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.state_dim = int(state_dim)

    def operator_summary(self) -> Dict[str, object]:
        return {}


class DiagonalSignedSSM(SequenceStateModel):
    """Token-selective signed diagonal linear recurrence."""

    name = "diag_signed"

    def __init__(self, vocab_size: int, n_classes: int, state_dim: int):
        super().__init__(vocab_size, n_classes, state_dim)
        self.a_raw = nn.Parameter(torch.zeros(vocab_size, state_dim))
        self.drive = nn.Parameter(torch.randn(vocab_size, state_dim) * 0.03)
        self.h0 = nn.Parameter(torch.randn(state_dim) * 0.03)
        self.readout = nn.Linear(state_dim, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1)
        outs = []
        for t in range(length):
            tok = tokens[:, t]
            a = 0.999 * torch.tanh(self.a_raw[tok])
            h = a * h + self.drive[tok]
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def operator_summary(self) -> Dict[str, object]:
        a = (0.999 * torch.tanh(self.a_raw)).detach().cpu()
        return {
            "transition": "token-selective signed diagonal",
            "eigenvalue_min": float(a.min()),
            "eigenvalue_max": float(a.max()),
        }


class ComplexDiagonalSSM(SequenceStateModel):
    """Token-selective independent 2D rotary blocks (complex diagonal state)."""

    name = "complex_diag"

    def __init__(self, vocab_size: int, n_classes: int, state_dim: int):
        if state_dim % 2:
            raise ValueError("complex_diag requires even state_dim")
        super().__init__(vocab_size, n_classes, state_dim)
        modes = state_dim // 2
        self.radius_raw = nn.Parameter(torch.full((vocab_size, modes), 2.0))
        self.angle_raw = nn.Parameter(torch.zeros(vocab_size, modes))
        nn.init.uniform_(self.angle_raw, -0.25, 0.25)
        self.drive = nn.Parameter(torch.randn(vocab_size, modes, 2) * 0.03)
        self.h0 = nn.Parameter(torch.randn(modes, 2) * 0.03)
        self.readout = nn.Linear(state_dim, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        modes = self.state_dim // 2
        h = self.h0.unsqueeze(0).expand(bsz, -1, -1)
        outs = []
        for t in range(length):
            tok = tokens[:, t]
            r = 0.999 * torch.sigmoid(self.radius_raw[tok])
            theta = math.pi * torch.tanh(self.angle_raw[tok])
            c, s = torch.cos(theta), torch.sin(theta)
            x, y = h[..., 0], h[..., 1]
            xr = r * (c * x - s * y)
            yr = r * (s * x + c * y)
            h = torch.stack((xr, yr), dim=-1) + self.drive[tok]
            outs.append(self.readout(h.reshape(bsz, modes * 2)))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def operator_summary(self) -> Dict[str, object]:
        r = (0.999 * torch.sigmoid(self.radius_raw)).detach().cpu()
        theta = (math.pi * torch.tanh(self.angle_raw)).detach().cpu()
        return {
            "transition": "token-selective complex diagonal / rotary blocks",
            "radius_min": float(r.min()),
            "radius_max": float(r.max()),
            "angle_abs_max_rad": float(theta.abs().max()),
        }


@dataclass(frozen=True)
class GraphSpec:
    n_nodes: int
    topology: str
    src: torch.Tensor
    dst: torch.Tensor


def make_graph(n_nodes: int, topology: str = "ring") -> GraphSpec:
    if n_nodes < 2:
        raise ValueError("need at least two graph nodes")
    edges: list[tuple[int, int]] = []
    if topology == "ring":
        if n_nodes == 2:
            edges = [(0, 1)]
        else:
            edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    elif topology == "path":
        edges = [(i, i + 1) for i in range(n_nodes - 1)]
    elif topology == "matching":
        edges = [(i, i + 1) for i in range(0, n_nodes - 1, 2)]
    elif topology == "disconnected":
        split = n_nodes // 2
        edges = [(i, i + 1) for i in range(0, max(0, split - 1))]
        edges += [(i, i + 1) for i in range(split, n_nodes - 1)]
    else:
        raise ValueError("topology must be ring, path, matching, or disconnected")
    src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    return GraphSpec(n_nodes=n_nodes, topology=topology, src=src, dst=dst)


def weighted_laplacian_apply(q: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Apply B^T diag(w) B without constructing a dense matrix."""
    diff = q[:, src] - q[:, dst]
    force = diff * w
    out = torch.zeros_like(q)
    out.index_add_(1, src, force)
    out.index_add_(1, dst, -force)
    return out


class GeometricWaveSSM(SequenceStateModel):
    """Sparse token-conditioned second-order wave recurrence on a graph."""

    name = "geom_wave"

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        state_dim: int,
        topology: str = "ring",
        dt: float = 0.30,
    ):
        if state_dim % 2:
            raise ValueError("geom_wave requires even state_dim")
        super().__init__(vocab_size, n_classes, state_dim)
        self.n_nodes = state_dim // 2
        graph = make_graph(self.n_nodes, topology)
        self.topology = topology
        self.register_buffer("src", graph.src)
        self.register_buffer("dst", graph.dst)
        self.n_edges = int(graph.src.numel())
        self.dt = float(dt)

        self.edge_base_raw = nn.Parameter(torch.zeros(self.n_edges))
        self.edge_gate_raw = nn.Parameter(torch.zeros(vocab_size, self.n_edges))
        nn.init.normal_(self.edge_gate_raw, std=0.08)
        self.self_k_raw = nn.Parameter(torch.tensor(-1.2))
        self.damping_raw = nn.Parameter(torch.full((vocab_size,), 2.2))
        self.drive = nn.Parameter(torch.randn(vocab_size, self.n_nodes) * 0.03)
        self.q0 = nn.Parameter(torch.randn(self.n_nodes) * 0.03)
        self.p0 = nn.Parameter(torch.randn(self.n_nodes) * 0.03)
        self.readout = nn.Linear(state_dim, n_classes)

    def _edge_weights(self, tok: torch.Tensor) -> torch.Tensor:
        base = 0.05 + 1.20 * torch.sigmoid(self.edge_base_raw)
        gate = 0.25 + 1.50 * torch.sigmoid(self.edge_gate_raw[tok])
        return base.unsqueeze(0) * gate

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        q = self.q0.unsqueeze(0).expand(bsz, -1)
        p = self.p0.unsqueeze(0).expand(bsz, -1)
        outs = []
        k0 = 0.02 + 0.58 * torch.sigmoid(self.self_k_raw)
        dt = self.dt

        for t in range(length):
            tok = tokens[:, t]
            w = self._edge_weights(tok)
            lap_q = weighted_laplacian_apply(q, self.src, self.dst, w)
            rho = 0.70 + 0.295 * torch.sigmoid(self.damping_raw[tok])
            p = rho.unsqueeze(-1) * p - dt * (k0 * q + lap_q) + self.drive[tok]
            q = q + dt * p
            outs.append(self.readout(torch.cat((q, p), dim=-1)))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def dense_transition(self, token: int) -> torch.Tensor:
        device = self.edge_base_raw.device
        n = self.n_nodes
        tok = torch.tensor([token], device=device, dtype=torch.long)
        w = self._edge_weights(tok)[0]
        L = torch.zeros((n, n), device=device, dtype=self.edge_base_raw.dtype)
        for e in range(self.n_edges):
            i, j = int(self.src[e]), int(self.dst[e])
            we = w[e]
            L[i, i] += we
            L[j, j] += we
            L[i, j] -= we
            L[j, i] -= we
        k0 = 0.02 + 0.58 * torch.sigmoid(self.self_k_raw)
        K = k0 * torch.eye(n, device=device) + L
        rho = 0.70 + 0.295 * torch.sigmoid(self.damping_raw[token])
        I = torch.eye(n, device=device)
        top_left = I - (self.dt**2) * K
        top_right = self.dt * rho * I
        bottom_left = -self.dt * K
        bottom_right = rho * I
        return torch.cat(
            (torch.cat((top_left, top_right), dim=1), torch.cat((bottom_left, bottom_right), dim=1)),
            dim=0,
        )

    @torch.no_grad()
    def operator_summary(self) -> Dict[str, object]:
        max_abs = 0.0
        radii = []
        complex_fracs = []
        for tok in range(self.vocab_size):
            eig = torch.linalg.eigvals(self.dense_transition(tok)).cpu()
            radii.append(float(eig.abs().max()))
            complex_fracs.append(float((eig.imag.abs() > 1e-6).float().mean()))
            max_abs = max(max_abs, float(eig.abs().max()))
        return {
            "transition": "token-modulated sparse second-order graph wave",
            "topology": self.topology,
            "nodes": self.n_nodes,
            "edges": self.n_edges,
            "spectral_radius_max": max_abs,
            "complex_eigenvalue_fraction_mean": sum(complex_fracs) / len(complex_fracs),
            "token_spectral_radii": radii,
        }


class GeometricScatterSSM(SequenceStateModel):
    """Product of local symmetric orthogonal two-port scatterers.

    Each token selects one angle per declared physical edge. Disjoint edges are
    updated in checkerboard phases, so global state mixing is generated by local
    geometry while recurrent work remains O(E)=O(state_dim).
    """

    name = "geom_scatter"

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        state_dim: int,
        topology: str = "ring",
        sweeps: int = 2,
    ):
        if state_dim < 4 or state_dim % 2:
            raise ValueError("geom_scatter requires an even state_dim >= 4")
        super().__init__(vocab_size, n_classes, state_dim)
        self.topology = topology
        self.sweeps = int(sweeps)
        graph = make_graph(state_dim, topology)
        self.register_buffer("src", graph.src)
        self.register_buffer("dst", graph.dst)
        self.n_edges = int(graph.src.numel())

        phase0, phase1 = [], []
        for e, (i, j) in enumerate(zip(graph.src.tolist(), graph.dst.tolist())):
            if {i, j} == {0, state_dim - 1}:
                bucket = phase1
            else:
                bucket = phase0 if min(i, j) % 2 == 0 else phase1
            bucket.append(e)
        self.register_buffer("phase0", torch.tensor(phase0, dtype=torch.long))
        self.register_buffer("phase1", torch.tensor(phase1, dtype=torch.long))

        self.angle_raw = nn.Parameter(torch.empty(vocab_size, self.n_edges))
        nn.init.uniform_(self.angle_raw, -0.35, 0.35)
        self.h0 = nn.Parameter(torch.randn(state_dim) * 0.05)
        self.readout = nn.Linear(state_dim, n_classes)

    def _scatter_phase(self, h: torch.Tensor, tok: torch.Tensor, edge_ids: torch.Tensor) -> torch.Tensor:
        if edge_ids.numel() == 0:
            return h
        src = self.src[edge_ids]
        dst = self.dst[edge_ids]
        theta = math.pi * torch.tanh(self.angle_raw[tok][:, edge_ids])
        c, s = torch.cos(theta), torch.sin(theta)
        a = h[:, src]
        b = h[:, dst]
        ap = c * a + s * b
        bp = s * a - c * b
        out = h.clone()
        out[:, src] = ap
        out[:, dst] = bp
        return out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1)
        outs = []
        for t in range(length):
            tok = tokens[:, t]
            for _ in range(self.sweeps):
                h = self._scatter_phase(h, tok, self.phase0)
                h = self._scatter_phase(h, tok, self.phase1)
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def dense_transition(self, token: int) -> torch.Tensor:
        eye = torch.eye(self.state_dim, device=self.angle_raw.device, dtype=self.angle_raw.dtype)
        toks = torch.full((self.state_dim, 1), token, device=self.angle_raw.device, dtype=torch.long)
        h = eye.clone()
        tokv = toks[:, 0]
        for _ in range(self.sweeps):
            h = self._scatter_phase(h, tokv, self.phase0)
            h = self._scatter_phase(h, tokv, self.phase1)
        return h.T

    @torch.no_grad()
    def operator_summary(self) -> Dict[str, object]:
        radii = []
        neg_real = []
        complex_frac = []
        for tok in range(self.vocab_size):
            eig = torch.linalg.eigvals(self.dense_transition(tok)).cpu()
            radii.append(float(eig.abs().max()))
            neg_real.append(float((eig.real < -1e-6).float().mean()))
            complex_frac.append(float((eig.imag.abs() > 1e-6).float().mean()))
        return {
            "transition": "token-selective product of local symmetric orthogonal scatterers",
            "topology": self.topology,
            "state_channels": self.state_dim,
            "edges": self.n_edges,
            "sweeps": self.sweeps,
            "token_spectral_radii": radii,
            "negative_real_eigenvalue_fraction_mean": sum(neg_real) / len(neg_real),
            "complex_eigenvalue_fraction_mean": sum(complex_frac) / len(complex_frac),
        }


class DenseHouseholderSSM(SequenceStateModel):
    """Token-selective product of dense Householder reflections.

    This is a control, not a KYY novelty: modern state-tracking work already uses
    Householder-product transitions. It asks whether local geometric support buys
    anything beyond the generic reflector algebra.
    """

    name = "householder2"

    def __init__(self, vocab_size: int, n_classes: int, state_dim: int, n_reflectors: int = 2):
        super().__init__(vocab_size, n_classes, state_dim)
        self.n_reflectors = int(n_reflectors)
        self.v = nn.Parameter(torch.randn(vocab_size, self.n_reflectors, state_dim) * 0.2)
        self.h0 = nn.Parameter(torch.randn(state_dim) * 0.05)
        self.readout = nn.Linear(state_dim, n_classes)

    def _householder(self, h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        denom = (v * v).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        proj = (h * v).sum(dim=-1, keepdim=True) / denom
        return h - 2.0 * proj * v

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1)
        outs = []
        for t in range(length):
            tok = tokens[:, t]
            vecs = self.v[tok]
            for r in range(self.n_reflectors):
                h = self._householder(h, vecs[:, r, :])
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def dense_transition(self, token: int) -> torch.Tensor:
        h = torch.eye(self.state_dim, device=self.v.device, dtype=self.v.dtype)
        for r in range(self.n_reflectors):
            vv = self.v[token, r].unsqueeze(0).expand(self.state_dim, -1)
            h = self._householder(h, vv)
        return h.T

    @torch.no_grad()
    def operator_summary(self) -> Dict[str, object]:
        complex_frac = []
        for tok in range(self.vocab_size):
            eig = torch.linalg.eigvals(self.dense_transition(tok)).cpu()
            complex_frac.append(float((eig.imag.abs() > 1e-6).float().mean()))
        return {
            "transition": f"token-selective product of {self.n_reflectors} dense Householder reflections",
            "reflectors": self.n_reflectors,
            "complex_eigenvalue_fraction_mean": sum(complex_frac) / len(complex_frac),
        }


class GRUReference(SequenceStateModel):
    name = "gru"

    def __init__(self, vocab_size: int, n_classes: int, state_dim: int):
        super().__init__(vocab_size, n_classes, state_dim)
        self.embedding = nn.Embedding(vocab_size, state_dim)
        self.cell = nn.GRUCell(state_dim, state_dim)
        self.h0 = nn.Parameter(torch.zeros(state_dim))
        self.readout = nn.Linear(state_dim, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1)
        outs = []
        for t in range(length):
            h = self.cell(self.embedding(tokens[:, t]), h)
            outs.append(self.readout(h))
        return torch.stack(outs, dim=1)

    def operator_summary(self) -> Dict[str, object]:
        return {"transition": "dense nonlinear GRU reference"}


def build_model(
    name: str,
    vocab_size: int,
    n_classes: int,
    state_dim: int,
    topology: str = "ring",
) -> SequenceStateModel:
    if name == "diag_signed":
        return DiagonalSignedSSM(vocab_size, n_classes, state_dim)
    if name == "complex_diag":
        return ComplexDiagonalSSM(vocab_size, n_classes, state_dim)
    if name == "householder2":
        return DenseHouseholderSSM(vocab_size, n_classes, state_dim, n_reflectors=2)
    if name == "geom_wave":
        return GeometricWaveSSM(vocab_size, n_classes, state_dim, topology=topology)
    if name == "geom_scatter":
        return GeometricScatterSSM(vocab_size, n_classes, state_dim, topology=topology)
    if name == "gru":
        return GRUReference(vocab_size, n_classes, state_dim)
    raise KeyError(f"unknown model {name!r}")


MODEL_NAMES = ("diag_signed", "complex_diag", "householder2", "geom_wave", "geom_scatter", "gru")
