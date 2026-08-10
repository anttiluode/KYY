from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_joint_legalization_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_joint_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_nearest_reflection_is_exact_reflection() -> None:
    m = np.array([[1.12, 0.19], [-0.08, -0.91]], dtype=np.float64)
    q = probe.nearest_reflection(m)
    eye = np.eye(2)
    assert np.allclose(q.T @ q, eye, atol=1e-12)
    assert np.linalg.det(q) < 0.0
    assert np.allclose(q @ q, eye, atol=1e-12)


def test_joint_projection_enforces_all_dihedral_relations() -> None:
    n = 31
    angles = np.array([0.23, -1.1, 2.4], dtype=np.float64)
    refs = np.array([
        [[1.1, 0.1], [0.03, -0.9]],
        [[0.7, 0.6], [0.5, -0.8]],
        [[0.9, -0.2], [-0.1, -1.05]],
    ])
    projected_angles, _ = probe.base.project_angles_to_dn_characters(n, angles)
    projected_refs = probe.project_reflections(refs)
    post = probe.relation_defects(n, projected_angles, projected_refs)
    assert max(post) < 1e-10


def test_perturbed_reflection_has_nonzero_relation_defect() -> None:
    n = 11
    angle = np.array([2.0 * math.pi / n])
    ref = np.array([[[1.0, 0.12], [0.05, -0.91]]])
    _, inv, conj, orth = probe.relation_defects(n, angle, ref)
    assert inv > 1e-3
    assert conj > 1e-3
    assert orth > 1e-3


def test_quotient_block_port_recovers_known_coset_transforms() -> None:
    g = torch.Generator().manual_seed(5)
    n = 17
    d = 6
    source = torch.randn(2*n, d, generator=g, dtype=torch.float64)
    q0_blocks=[]; q1_blocks=[]
    for a,b in [(0.2,-0.4),(0.7,0.1),(-0.3,0.5)]:
        ca,sa=math.cos(a),math.sin(a); cb,sb=math.cos(b),math.sin(b)
        q0_blocks.append(torch.tensor([[ca,sa],[-sa,ca]],dtype=torch.float64))
        q1_blocks.append(torch.tensor([[cb,sb],[-sb,cb]],dtype=torch.float64))
    q0=torch.block_diag(*q0_blocks); q1=torch.block_diag(*q1_blocks)
    target=torch.cat((source[:n]@q0, source[n:]@q1),dim=0)
    p0,p1=probe.quotient_block_port(source,target,n)
    out=probe.apply_quotient_port(source,n,p0,p1)
    assert torch.allclose(out,target,atol=1e-10,rtol=1e-10)
