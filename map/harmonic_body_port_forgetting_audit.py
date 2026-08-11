from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PairAudit:
    pair: list[int]
    quotient_port_gap: float
    future_mixer_gap_port_only: float
    future_mixer_gap_damped: float
    future_mixer_gap_isolated: float
    future_mixer_gap_erased: float


def c4_characters(q: int) -> tuple[complex, complex]:
    phi = 0.5 * math.pi * int(q)
    z1 = complex(np.exp(1j * phi))
    z2 = complex(np.exp(2j * phi))
    return z1, z2


def quotient_port(z1: complex, z2: complex) -> complex:
    """Port-only quotient: deliberately ignore the faithful fundamental."""
    del z1
    return z2


def future_mixer(z1: complex, z2: complex, epsilon: float, coupling_gate: float = 1.0) -> complex:
    """A generic later observable/path that can reveal residual faithful state.

    This is not claimed as a specific transistor mixer. It is the smallest
    linear future-coupling witness: if a forbidden carrier remains observable,
    epsilon*z1 can leak it back into the coarse observable.
    """
    return z2 + float(epsilon) * float(coupling_gate) * z1


def damp(z1: complex, gamma: float, time: float) -> complex:
    return z1 * math.exp(-float(gamma) * float(time))


def pair_gap(a: complex, b: complex) -> float:
    return float(abs(a - b))


def audit_pair(q_a: int, q_b: int, epsilon: float, gamma: float, time: float) -> PairAudit:
    a1, a2 = c4_characters(q_a)
    b1, b2 = c4_characters(q_b)

    port_gap = pair_gap(quotient_port(a1, a2), quotient_port(b1, b2))
    port_only = pair_gap(future_mixer(a1, a2, epsilon), future_mixer(b1, b2, epsilon))

    da1 = damp(a1, gamma, time)
    db1 = damp(b1, gamma, time)
    damped = pair_gap(future_mixer(da1, a2, epsilon), future_mixer(db1, b2, epsilon))

    isolated = pair_gap(
        future_mixer(a1, a2, epsilon, coupling_gate=0.0),
        future_mixer(b1, b2, epsilon, coupling_gate=0.0),
    )
    erased = pair_gap(future_mixer(0j, a2, epsilon), future_mixer(0j, b2, epsilon))

    return PairAudit(
        pair=[int(q_a), int(q_b)],
        quotient_port_gap=port_gap,
        future_mixer_gap_port_only=port_only,
        future_mixer_gap_damped=damped,
        future_mixer_gap_isolated=isolated,
        future_mixer_gap_erased=erased,
    )


def analytic_gap(epsilon: float, gamma: float, time: float) -> float:
    """For q and q+2, z1 differs by sign, so residual future gap is 2 eps exp(-gamma t)."""
    return 2.0 * abs(float(epsilon)) * math.exp(-float(gamma) * float(time))


def time_to_gap(epsilon: float, gamma: float, target_gap: float) -> float:
    eps = abs(float(epsilon))
    g = float(gamma)
    d = float(target_gap)
    if g <= 0:
        return math.inf
    if d <= 0:
        return math.inf
    initial = 2.0 * eps
    if d >= initial:
        return 0.0
    return math.log(initial / d) / g


def run_grid(epsilon: float = 0.1, gamma: float = 1.0) -> dict:
    times = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    rows = []
    for t in times:
        a = audit_pair(0, 2, epsilon, gamma, t)
        predicted = analytic_gap(epsilon, gamma, t)
        rows.append({
            "time": t,
            "measured_future_gap": a.future_mixer_gap_damped,
            "analytic_future_gap": predicted,
            "absolute_error": abs(a.future_mixer_gap_damped - predicted),
        })

    targets = [1e-1, 1e-2, 1e-3, 1e-6]
    thresholds = [
        {
            "target_gap": d,
            "minimum_damping_time": time_to_gap(epsilon, gamma, d),
        }
        for d in targets
    ]

    pairs = [
        asdict(audit_pair(0, 2, epsilon, gamma, 2.0)),
        asdict(audit_pair(1, 3, epsilon, gamma, 2.0)),
    ]

    return {
        "model": {
            "full_state": "z1=exp(i phi), z2=exp(i 2phi)",
            "desired_kernel": "q ~ q+2; keep z2 and forget z1",
            "future_witness": "y=z2 + epsilon*g*z1",
            "damping": "z1(t)=exp(-gamma t) z1(0)",
        },
        "parameters": {"epsilon": epsilon, "gamma": gamma},
        "pair_audits_at_t2": pairs,
        "damping_grid": rows,
        "threshold_times": thresholds,
        "analytic_law": "for merged histories q and q+2, future gap = 2|epsilon| exp(-gamma t)",
        "behavioral_forgetting_rule": (
            "Exact hidden-state erasure is sufficient but unnecessary. If every future coupling from the old "
            "faithful carrier is gated to zero, the old physical state is future-unobservable and the quotient is "
            "behaviorally exact even before its residual amplitude decays."
        ),
        "physical_interpretation": (
            "A pre-carried quotient harmonic avoids runtime phase multiplication, but body-level forgetting still "
            "requires retiring the faithful carrier by damping, isolation, disconnection, or another mechanism."
        ),
        "scope": (
            "Reduced modal audit only. It is not a transistor-level model and does not claim a particular parasitic "
            "coupling value for existing SHIL/Potts hardware."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit body versus port forgetting in a fundamental+harmonic C4 state code")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    print(json.dumps(run_grid(args.epsilon, args.gamma), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
