from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path); assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

cont=load_local("continuous_backend_for_design",ROOT/"map"/"metacircuit_continuous_backend.py")


def contributions(n:int):
    d=np.arange(1,n,dtype=np.float64)
    return {f:1.0-np.cos(2.0*np.pi*f*d/n) for f in range(1,n//2+1)}


def bank_margin(fs, contrib):
    acc=np.zeros_like(next(iter(contrib.values())))
    for f in fs: acc += contrib[int(f)]
    return float(acc.min())


def bank_metrics(n:int,fs:list[int],contrib):
    rows=[cont.lower_mode(n,f) for f in fs]
    return {
        "frequencies":list(map(int,fs)),
        "symbolic_margin":bank_margin(fs,contrib),
        "max_phase_map_condition":max(r.phase_map_condition for r in rows),
        "max_phase_map_norm":max(r.phase_map_norm for r in rows),
        "max_relative_phase_sensitivity":max(r.relative_phase_sensitivity_to_ratio for r in rows),
        "ratio_range":[min(r.admittance_over_fdnr for r in rows),max(r.admittance_over_fdnr for r in rows)],
    }


def exhaustive_constrained(n:int,modes:int,cond_cap:float,max_frequency:int,contrib):
    candidates=[]
    for f in range(1,min(max_frequency,n//2)+1):
        row=cont.lower_mode(n,f)
        if row.phase_map_condition <= cond_cap:
            candidates.append(f)
    if len(candidates)<modes: raise ValueError("not enough candidates")
    best=(-float("inf"),None)
    for combo in itertools.combinations(candidates,modes):
        mm=bank_margin(combo,contrib)
        if mm>best[0]: best=(mm,combo)
    return bank_metrics(n,list(best[1]),contrib),candidates


def local_digital_search(n:int,modes:int,contrib,seed:int=1,random_samples:int=100000,restarts:int=100):
    rng=random.Random(seed); candidates=list(range(1,n//2+1)); best=(-float("inf"),None)
    for _ in range(random_samples):
        fs=tuple(sorted(rng.sample(candidates,modes))); mm=bank_margin(fs,contrib)
        if mm>best[0]: best=(mm,fs)
    for restart in range(restarts):
        fs=set(best[1] if restart==0 else rng.sample(candidates,modes)); cur=bank_margin(fs,contrib)
        while True:
            move=None; move_score=cur
            for out in list(fs):
                for inn in candidates:
                    if inn in fs: continue
                    trial=(fs-{out})|{inn}; mm=bank_margin(trial,contrib)
                    if mm>move_score+1e-12:
                        move_score=mm; move=(out,inn)
            if move is None: break
            fs.remove(move[0]);fs.add(move[1]);cur=move_score
        if cur>best[0]: best=(cur,tuple(sorted(fs)))
    return bank_metrics(n,list(best[1]),contrib)


def mismatch_eval(n:int,fs:list[int],sigma:float,trials:int,train_cycles:int,test_cycles:int,seed:int):
    acc=[]; defects=[]
    for trial in range(trials):
        eps=np.random.default_rng(seed+trial).normal(size=len(fs))*sigma
        xt,yt=cont.physical_features(n,fs,train_cycles,eps); port=cont.fit_port(xt,yt,n)
        x,y=cont.physical_features(n,fs,test_cycles,eps)
        acc.append(cont.accuracy(port,x,y)); defects.append(cont.relation_defect(n,fs,eps))
    return {"sigma":sigma,"mean_accuracy":float(np.mean(acc)),"min_accuracy":float(np.min(acc)),"mean_max_relation_defect":float(np.mean(defects))}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--n",type=int,default=101);p.add_argument("--modes",type=int,default=8)
    p.add_argument("--cond-cap",type=float,default=2.0);p.add_argument("--max-frequency",type=int,default=31)
    p.add_argument("--random-samples",type=int,default=100000);p.add_argument("--restarts",type=int,default=100)
    p.add_argument("--sigmas",nargs="+",type=float,default=[1e-5,2e-5,5e-5])
    p.add_argument("--trials",type=int,default=8);p.add_argument("--seed",type=int,default=4600)
    p.add_argument("--json",action="store_true");args=p.parse_args()
    contrib=contributions(args.n)
    physical,candidates=exhaustive_constrained(args.n,args.modes,args.cond_cap,args.max_frequency,contrib)
    digital=local_digital_search(args.n,args.modes,contrib,1,args.random_samples,args.restarts)
    payload={
        "config":vars(args),
        "physical_constraint_candidates":candidates,
        "physically_constrained_exact_optimum":physical,
        "digital_only_strong_heuristic":digital,
        "margin_loss_fraction_vs_heuristic":1.0-physical["symbolic_margin"]/digital["symbolic_margin"],
        "mismatch_sweep":{
            "physically_constrained":[mismatch_eval(args.n,physical["frequencies"],s,args.trials,16,1024,args.seed) for s in args.sigmas],
            "digital_only_heuristic":[mismatch_eval(args.n,digital["frequencies"],s,args.trials,16,1024,args.seed) for s in args.sigmas],
        },
        "bounded_eta_1e-5":{
            "physically_constrained_certified_cycles":cont.max_certified_cycles(args.n,physical["frequencies"],1e-5),
            "digital_only_heuristic_certified_cycles":cont.max_certified_cycles(args.n,digital["frequencies"],1e-5),
        },
        "scope":"The constrained optimum is exhaustive only inside the declared cond/frequency candidate set. The digital-only bank is a strong deterministic heuristic, not a global optimum."
    }
    print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__":main()
