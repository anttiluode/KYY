from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


symbolic = load_local("metacircuit_symbolic_for_continuous", ROOT / "map" / "metacircuit_frequency_design.py")
discrete = load_local("metacircuit_discrete_for_continuous", ROOT / "map" / "metacircuit_cyclic_backend.py")


def continuous_ratio(theta: float, dt: float = 1.0) -> float:
    """Exact physical D^-1 Y for D u'' + Y u = 0 sampled with phase theta per dt."""
    return (float(theta) / float(dt)) ** 2


def sampled_companion(theta: float) -> np.ndarray:
    """Exact displacement/lag recurrence of a continuous harmonic oscillator sampled every dt."""
    c = math.cos(theta)
    return np.array([[2.0 * c, -1.0], [1.0, 0.0]], dtype=np.float64)


@dataclass
class ContinuousMode:
    n: int
    frequency: int
    theta: float
    natural_frequency: float
    admittance_over_fdnr: float
    sampled_relation_defect: float
    phase_map_condition: float
    phase_map_norm: float
    relative_phase_sensitivity_to_ratio: float
    one_mode_symbolic_margin: float


def lower_mode(n: int, frequency: int, dt: float = 1.0) -> ContinuousMode:
    f = int(frequency) % int(n)
    if f == 0 or f >= n / 2 + 1e-12:
        # Keep the canonical 0<theta<pi branch used by the current backend design.
        raise ValueError("continuous backend expects canonical character 0 < f < n/2")
    theta = 2.0 * math.pi * f / n
    lam = continuous_ratio(theta, dt)
    a = sampled_companion(theta)
    relation = float(np.linalg.norm(np.linalg.matrix_power(a, n) - np.eye(2), ord="fro"))
    t = discrete.phase_from_companion(theta)
    # theta = dt*sqrt(lambda), hence lambda*dtheta/dlambda = theta/2.
    rel_sens = theta / 2.0
    return ContinuousMode(
        n=int(n),
        frequency=f,
        theta=float(theta),
        natural_frequency=float(theta / dt),
        admittance_over_fdnr=float(lam),
        sampled_relation_defect=relation,
        phase_map_condition=float(np.linalg.cond(t)),
        phase_map_norm=float(np.linalg.norm(t, ord=2)),
        relative_phase_sensitivity_to_ratio=float(rel_sens),
        one_mode_symbolic_margin=discrete.prototype_margin(n, f),
    )


def physical_features(n: int, frequencies: list[int], cycles: int, relative_ratio_error: np.ndarray | None = None):
    """Exact samples of the continuous free resonators at integer dt=1 times."""
    m = len(frequencies)
    eps = np.zeros(m, dtype=np.float64) if relative_ratio_error is None else np.asarray(relative_ratio_error, dtype=np.float64)
    if eps.shape != (m,):
        raise ValueError("one ratio error per mode required")
    times = np.asarray([q + k * n for q in range(n) for k in range(cycles)], dtype=np.float64)
    x = np.empty((len(times), 2 * m), dtype=np.float64)
    for j, (f, e) in enumerate(zip(frequencies, eps)):
        theta = 2.0 * math.pi * f / n
        phi = theta * math.sqrt(1.0 + float(e))
        # Initialize with the ideal character state [u_0,u_-1]=[1,cos(theta)].
        # Under perturbed phi this becomes a phase/amplitude-adjusted physical orbit.
        beta = (math.cos(phi) - math.cos(theta)) / math.sin(phi)
        u = np.cos(times * phi) + beta * np.sin(times * phi)
        tm1 = times - 1.0
        um1 = np.cos(tm1 * phi) + beta * np.sin(tm1 * phi)
        x[:, 2*j] = u
        x[:, 2*j+1] = um1
    y = np.repeat(np.arange(n, dtype=np.int64), cycles)
    return x, y


def exact_companion_port(n: int, frequencies: list[int]) -> np.ndarray:
    m = len(frequencies)
    port = np.zeros((n, 2*m + 1), dtype=np.float64)
    for q in range(n):
        for j, f in enumerate(frequencies):
            theta = 2.0 * math.pi * f / n
            proto = np.asarray([math.cos(theta*q), math.sin(theta*q)])
            port[q,2*j:2*j+2] = proto @ discrete.phase_from_companion(theta)
    return port


def fit_port(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    xa = np.concatenate((x, np.ones((len(x),1))), axis=1)
    target = np.eye(n, dtype=np.float64)[y]
    return np.linalg.lstsq(xa, target, rcond=None)[0].T


def accuracy(port: np.ndarray, x: np.ndarray, y: np.ndarray, chunk: int = 16384) -> float:
    good = 0
    for start in range(0, len(x), chunk):
        xb = x[start:start+chunk]
        yb = y[start:start+chunk]
        xa = np.concatenate((xb, np.ones((len(xb),1))), axis=1)
        good += int(np.sum((xa @ port.T).argmax(axis=1) == yb))
    return float(good/len(x))


def relation_defect(n: int, frequencies: list[int], eps: np.ndarray) -> float:
    vals = []
    for f, e in zip(frequencies, eps):
        theta = 2.0 * math.pi * f/n
        phi = theta * math.sqrt(1.0 + float(e))
        # Rotation-coordinate Frobenius defect after n sampled steps.
        residual = n * (phi-theta)
        vals.append(2.0 * math.sqrt(2.0) * abs(math.sin(residual/2.0)))
    return float(max(vals))


def worst_phase_step_error(n: int, frequency: int, eta: float) -> float:
    theta = 2.0 * math.pi * int(frequency)/int(n)
    return max(
        abs(theta*math.sqrt(1.0+eta)-theta),
        abs(theta*math.sqrt(1.0-eta)-theta),
    )


def max_sin_interval(a: float, b: float) -> float:
    k0 = math.ceil((a-math.pi/2.0)/(2.0*math.pi))
    k1 = math.floor((b-math.pi/2.0)/(2.0*math.pi))
    if k0 <= k1:
        return 1.0
    return max(math.sin(a), math.sin(b))


def robust_phase_margin(n: int, frequencies: list[int], eta: float, cycles: int) -> float:
    tmax = n*cycles - 1
    ds = [worst_phase_step_error(n,f,eta) for f in frequencies]
    best = float("inf")
    for d in range(1,n):
        gap = 0.0
        for f,de in zip(frequencies,ds):
            Delta = (2.0*math.pi*f*d/n)%(2.0*math.pi)
            s = math.sin(Delta/2.0)
            E = tmax*de
            gap += -2.0*s*max_sin_interval(-E-Delta/2.0,E-Delta/2.0)
        best = min(best,gap)
    return float(best)


def max_certified_cycles(n: int, frequencies: list[int], eta: float, cap: int = 10000) -> int:
    lo = 0
    for cycles in range(1,cap+1):
        if robust_phase_margin(n,frequencies,eta,cycles) > 0.0:
            lo = cycles
        else:
            break
    return lo


def main():
    p = argparse.ArgumentParser(description="Correct KYY metacircuit lowering using the continuous physical resonator law")
    p.add_argument("--n",type=int,default=101)
    p.add_argument("--modes",type=int,default=8)
    p.add_argument("--condition-cap",type=float,default=2.0)
    p.add_argument("--trials",type=int,default=8)
    p.add_argument("--train-cycles",type=int,default=16)
    p.add_argument("--test-cycles",type=int,default=1024)
    p.add_argument("--sigmas",nargs="+",type=float,default=[1e-5,2e-5,5e-5,1e-4])
    p.add_argument("--tolerances",nargs="+",type=float,default=[5e-6,1e-5,2e-5,5e-5,1e-4])
    p.add_argument("--seed",type=int,default=4600)
    p.add_argument("--json",action="store_true")
    args=p.parse_args()

    u = symbolic.greedy_design(args.n,args.modes,None)
    c = symbolic.greedy_design(args.n,args.modes,args.condition_cap)
    banks={"unconstrained":u.frequencies,"conditioned":c.frequencies}
    payload={
        "config":vars(args),
        "discrete_surrogate_correction":"The older D^-1Y=2(1-cos(theta))/dt^2 mapping is exact for the central-difference recurrence, not the continuous physical circuit. The continuous free resonator requires D^-1Y=(theta/dt)^2 for exact sampled phase.",
        "banks":{},"mismatch_sweep":{},"bounded_tolerance":{}
    }
    for name,fs in banks.items():
        rows=[lower_mode(args.n,f) for f in fs]
        payload["banks"][name]={
            "frequencies":fs,
            "max_phase_map_condition":max(r.phase_map_condition for r in rows),
            "max_phase_map_norm":max(r.phase_map_norm for r in rows),
            "max_relative_phase_sensitivity":max(r.relative_phase_sensitivity_to_ratio for r in rows),
            "ratio_range":[min(r.admittance_over_fdnr for r in rows),max(r.admittance_over_fdnr for r in rows)],
        }
        sweep=[]
        for sigma in args.sigmas:
            accs=[]; defects=[]
            for trial in range(args.trials):
                eps=np.random.default_rng(args.seed+trial).normal(size=args.modes)*sigma
                xt,yt=physical_features(args.n,fs,args.train_cycles,eps)
                port=fit_port(xt,yt,args.n)
                x,y=physical_features(args.n,fs,args.test_cycles,eps)
                accs.append(accuracy(port,x,y))
                defects.append(relation_defect(args.n,fs,eps))
            sweep.append({"sigma":sigma,"mean_accuracy":float(np.mean(accs)),"min_accuracy":float(np.min(accs)),"mean_max_relation_defect":float(np.mean(defects))})
        payload["mismatch_sweep"][name]=sweep
        cert=[]
        for eta in args.tolerances:
            cert.append({
                "eta":eta,
                "max_step_phase_error":max(worst_phase_step_error(args.n,f,eta) for f in fs),
                "certified_cycles":max_certified_cycles(args.n,fs,eta),
                "margin_at_64":robust_phase_margin(args.n,fs,eta,64),
            })
        payload["bounded_tolerance"][name]=cert
    if args.json:
        print(json.dumps(payload,indent=2,sort_keys=True))
    else:
        print(json.dumps(payload,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
