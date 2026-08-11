from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_joint_for_runtime"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_joint_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
joint = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = joint
SPEC.loader.exec_module(joint)
base = joint.base


def exact_step(
    h: torch.Tensor,
    token: torch.Tensor,
    *,
    n: int,
    angles: torch.Tensor,
    reflections: torch.Tensor,
    angle_error: float = 0.0,
) -> torch.Tensor:
    ref = token == n
    inc = torch.where(ref, torch.zeros_like(token), token).to(h.dtype).unsqueeze(-1)
    theta = inc * (angles + float(angle_error)).unsqueeze(0)
    c, s = torch.cos(theta), torch.sin(theta)
    x, y = h[..., 0], h[..., 1]
    rot = torch.stack((c*x-s*y, s*x+c*y), dim=-1)
    refl = torch.einsum("kij,bkj->bki", reflections, h)
    return torch.where(ref[:,None,None], refl, rot)


def apply_quotient_block_port(
    h: torch.Tensor,
    branch: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
) -> torch.Tensor:
    flat = h.reshape(h.shape[0], -1)
    z0 = flat @ q0
    z1 = flat @ q1
    return torch.where(branch[:,None] == 0, z0, z1)


def compiled_forward(
    model: joint.ApproxDihedralTracker,
    tokens: torch.Tensor,
    *,
    projected_angles: np.ndarray,
    projected_reflections: np.ndarray,
    q0: torch.Tensor,
    q1: torch.Tensor,
    angle_error: float = 0.0,
) -> torch.Tensor:
    """Streaming exact D_n operator plus one-bit quotient-conditioned port."""
    dtype = model.h0.dtype
    angles = torch.as_tensor(projected_angles, dtype=dtype, device=tokens.device)
    refs = torch.as_tensor(projected_reflections, dtype=dtype, device=tokens.device)
    q0d = q0.to(dtype=dtype, device=tokens.device)
    q1d = q1.to(dtype=dtype, device=tokens.device)
    h = model.h0.unsqueeze(0).expand(tokens.shape[0], -1, -1)
    branch = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
    outs: list[torch.Tensor] = []
    for t in range(tokens.shape[1]):
        tok = tokens[:,t]
        h = exact_step(
            h, tok, n=model.n, angles=angles, reflections=refs, angle_error=angle_error
        )
        branch = torch.where(tok == model.n, 1-branch, branch)
        ported = apply_quotient_block_port(h, branch, q0d, q1d)
        outs.append(model.readout(ported))
    return torch.stack(outs,dim=1)


def raw_exact_forward(
    model: joint.ApproxDihedralTracker,
    tokens: torch.Tensor,
    *,
    projected_angles: np.ndarray,
    projected_reflections: np.ndarray,
    angle_error: float = 0.0,
) -> torch.Tensor:
    dtype=model.h0.dtype
    angles=torch.as_tensor(projected_angles,dtype=dtype,device=tokens.device)
    refs=torch.as_tensor(projected_reflections,dtype=dtype,device=tokens.device)
    h=model.h0.unsqueeze(0).expand(tokens.shape[0],-1,-1)
    outs=[]
    for t in range(tokens.shape[1]):
        h=exact_step(h,tokens[:,t],n=model.n,angles=angles,reflections=refs,angle_error=angle_error)
        outs.append(model.readout(h.reshape(tokens.shape[0],-1)))
    return torch.stack(outs,dim=1)


def evaluate(
    model: joint.ApproxDihedralTracker,
    *,
    projected_angles: np.ndarray,
    projected_reflections: np.ndarray,
    q0: torch.Tensor,
    q1: torch.Tensor,
    n: int,
    lengths: list[int],
    batch_size: int,
    max_increment: int,
    reflection_probability: float,
    random_start: bool,
    angle_error: float,
) -> tuple[dict[str,float],dict[str,float]]:
    raw={}; compiled={}
    model.eval()
    with torch.no_grad():
        for length in lengths:
            x,y=base.generate_batch(
                n,batch_size,length,max_increment,reflection_probability,random_start=random_start
            )
            p0=raw_exact_forward(
                model,x,projected_angles=projected_angles,
                projected_reflections=projected_reflections,angle_error=angle_error
            ).argmax(dim=-1)
            p1=compiled_forward(
                model,x,projected_angles=projected_angles,
                projected_reflections=projected_reflections,q0=q0,q1=q1,
                angle_error=angle_error
            ).argmax(dim=-1)
            raw[str(length)]=float((p0==y).float().mean().item())
            compiled[str(length)]=float((p1==y).float().mean().item())
    return raw,compiled


@dataclass
class RuntimeRun:
    seed:int
    pre_relation_defects:dict[str,float]
    post_relation_defects:dict[str,float]
    raw_orbit_accuracy:float
    compiled_orbit_accuracy:float
    compiled_orbit_min_margin:float
    raw_clean_accuracy:dict[str,float]
    compiled_clean_accuracy:dict[str,float]
    raw_eta_1e3_accuracy:dict[str,float]
    compiled_eta_1e3_accuracy:dict[str,float]
    sidecar_bits:int
    port_family:str


def train_and_probe(
    *,n:int,modes:int,seed:int,train_length:int,train_steps:int,batch_size:int,
    eval_batch_size:int,max_increment:int,reflection_probability:float,lr:float,
    random_start:bool,reflection_scale:float,lengths:list[int]
)->RuntimeRun:
    model=joint.train_model(
        n=n,modes=modes,seed=seed,train_length=train_length,train_steps=train_steps,
        batch_size=batch_size,max_increment=max_increment,
        reflection_probability=reflection_probability,lr=lr,random_start=random_start,
        reflection_scale=reflection_scale
    )
    learned_angles=model.angles.detach().cpu().numpy().astype(np.float64)
    learned_ref=model.reflection_matrices().detach().cpu().numpy().astype(np.float64)
    projected_angles,_=base.project_angles_to_dn_characters(n,learned_angles)
    projected_ref=joint.project_reflections(learned_ref)
    pre=joint.relation_defects(n,learned_angles,learned_ref)
    post=joint.relation_defects(n,projected_angles,projected_ref)

    h0=model.h0.detach().cpu()
    z_learned=joint.canonical_orbit(n,learned_angles,learned_ref,h0)
    z_exact=joint.canonical_orbit(n,projected_angles,projected_ref,h0)
    q0,q1=joint.quotient_block_port(z_exact,z_learned,n)
    W=model.readout.weight.detach().cpu(); b=model.readout.bias.detach().cpu()
    raw_orbit,_,_=joint.metrics(z_exact,W,b)
    z_comp=joint.apply_quotient_port(z_exact,n,q0,q1)
    comp_orbit,comp_margin,_=joint.metrics(z_comp,W,b)

    raw_clean,comp_clean=evaluate(
        model,projected_angles=projected_angles,projected_reflections=projected_ref,
        q0=q0,q1=q1,n=n,lengths=lengths,batch_size=eval_batch_size,
        max_increment=max_increment,reflection_probability=reflection_probability,
        random_start=random_start,angle_error=0.0
    )
    raw_eta,comp_eta=evaluate(
        model,projected_angles=projected_angles,projected_reflections=projected_ref,
        q0=q0,q1=q1,n=n,lengths=lengths,batch_size=eval_batch_size,
        max_increment=max_increment,reflection_probability=reflection_probability,
        random_start=random_start,angle_error=1e-3
    )
    names=("rotation_order","reflection_involution","conjugation","reflection_orthogonality")
    return RuntimeRun(
        seed=seed,
        pre_relation_defects=dict(zip(names,map(float,pre))),
        post_relation_defects=dict(zip(names,map(float,post))),
        raw_orbit_accuracy=float(raw_orbit),compiled_orbit_accuracy=float(comp_orbit),
        compiled_orbit_min_margin=float(comp_margin),raw_clean_accuracy=raw_clean,
        compiled_clean_accuracy=comp_clean,raw_eta_1e3_accuracy=raw_eta,
        compiled_eta_1e3_accuracy=comp_eta,sidecar_bits=1,
        port_family="two quotient-conditioned block-diagonal O(2)^m maps"
    )


def main()->None:
    p=argparse.ArgumentParser(description="Live jointly legalized D_n operator plus quotient block port")
    p.add_argument("--n",type=int,default=101);p.add_argument("--modes",type=int,default=8)
    p.add_argument("--seeds",nargs="+",type=int,default=list(range(10)))
    p.add_argument("--train-length",type=int,default=16);p.add_argument("--train-steps",type=int,default=2200)
    p.add_argument("--batch-size",type=int,default=128);p.add_argument("--eval-batch-size",type=int,default=64)
    p.add_argument("--max-increment",type=int,default=4);p.add_argument("--reflection-probability",type=float,default=0.25)
    p.add_argument("--reflection-scale",type=float,default=0.25);p.add_argument("--lr",type=float,default=3e-3)
    p.add_argument("--random-start",action="store_true");p.add_argument("--lengths",nargs="+",type=int,default=[16,64,256,1024])
    p.add_argument("--json",action="store_true");args=p.parse_args()
    rows=[train_and_probe(
        n=args.n,modes=args.modes,seed=s,train_length=args.train_length,
        train_steps=args.train_steps,batch_size=args.batch_size,eval_batch_size=args.eval_batch_size,
        max_increment=args.max_increment,reflection_probability=args.reflection_probability,
        lr=args.lr,random_start=args.random_start,reflection_scale=args.reflection_scale,
        lengths=args.lengths
    ) for s in args.seeds]
    payload={"config":vars(args),"results":[asdict(x) for x in rows]}
    if args.json: print(json.dumps(payload,indent=2,sort_keys=True)); return
    print("seed raw-orbit compiled-orbit raw-L1024 compiled-L1024 eta-compiled-L1024 margin")
    for x in rows:
        print(f"{x.seed:4d} {x.raw_orbit_accuracy:9.3f} {x.compiled_orbit_accuracy:14.3f} "
              f"{x.raw_clean_accuracy['1024']:10.3f} {x.compiled_clean_accuracy['1024']:15.3f} "
              f"{x.compiled_eta_1e3_accuracy['1024']:18.3f} {x.compiled_orbit_min_margin:+8.3f}")

if __name__=="__main__": main()
