from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
MODULE_NAME="dihedral_joint_runtime_for_tests"
SPEC=importlib.util.spec_from_file_location(MODULE_NAME,ROOT/"map"/"dihedral_joint_runtime_probe.py")
assert SPEC is not None and SPEC.loader is not None
probe=importlib.util.module_from_spec(SPEC);sys.modules[MODULE_NAME]=probe;SPEC.loader.exec_module(probe)


def test_exact_step_obeys_reflection_square() -> None:
    n=7
    angles=torch.tensor([2.0*math.pi/n],dtype=torch.float64)
    refs=torch.tensor([[[1.0,0.0],[0.0,-1.0]]],dtype=torch.float64)
    h=torch.tensor([[[0.3,0.8]]],dtype=torch.float64)
    s=torch.tensor([n],dtype=torch.long)
    h2=probe.exact_step(probe.exact_step(h,s,n=n,angles=angles,reflections=refs),s,n=n,angles=angles,reflections=refs)
    assert torch.allclose(h2,h,atol=1e-12,rtol=1e-12)


def test_quotient_port_switches_between_two_maps() -> None:
    h=torch.tensor([[[1.0,2.0]],[[1.0,2.0]]],dtype=torch.float64)
    q0=torch.eye(2,dtype=torch.float64)
    q1=-torch.eye(2,dtype=torch.float64)
    branch=torch.tensor([0,1],dtype=torch.long)
    out=probe.apply_quotient_block_port(h,branch,q0,q1)
    assert torch.allclose(out[0],torch.tensor([1.0,2.0],dtype=torch.float64))
    assert torch.allclose(out[1],torch.tensor([-1.0,-2.0],dtype=torch.float64))
