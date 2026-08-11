from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend = load_local("metacircuit_backend_certificate", ROOT / "map" / "metacircuit_cyclic_backend.py")
design = load_local("metacircuit_design_certificate", ROOT / "map" / "metacircuit_frequency_design.py")


def max_sin_on_interval(a: float, b: float) -> float:
    """Exact maximum of sin(x) on a closed interval."""
    if b < a:
        a, b = b, a
    # If an interval contains pi/2 + 2*pi*k, the maximum is exactly one.
    k_lo = math.ceil((a - math.pi / 2.0) / (2.0 * math.pi))
    k_hi = math.floor((b - math.pi / 2.0) / (2.0 * math.pi))
    if k_lo <= k_hi:
        return 1.0
    return max(math.sin(a), math.sin(b))


def worst_phase_step_error(n: int, frequency: int, relative_ratio_tolerance: float) -> float:
    """Worst |phi-theta| for lambda_actual=lambda*(1+eps), |eps|<=eta.

    Canonical characters are in 0<theta<pi.  acos is monotone in lambda on the
    stable oscillatory interior, so extrema occur at tolerance endpoints.
    """
    eta = float(relative_ratio_tolerance)
    if eta < 0.0:
        raise ValueError("tolerance must be nonnegative")
    theta = 2.0 * math.pi * int(frequency) / int(n)
    lam = backend.required_admittance_over_fdnr(theta)
    errors = []
    for eps in (-eta, eta):
        actual = lam * (1.0 + eps)
        cosine = 1.0 - actual / 2.0
        if cosine <= -1.0 or cosine >= 1.0:
            return float("inf")
        phi = math.acos(cosine)
        errors.append(abs(phi - theta))
    return max(errors)


def worst_mode_phase_errors(n: int, frequencies: list[int], eta: float) -> list[float]:
    return [worst_phase_step_error(n, f, eta) for f in frequencies]


def robust_phase_margin(n: int, frequencies: list[int], eta: float, cycles: int) -> float:
    """Certified positive-kernel margin in normalized phase coordinates.

    A symbolic state q reached after q+k*n increments can accumulate a per-mode
    phase error e_i=t*delta_i.  For all histories inside `cycles`,

        |e_i| <= (n*cycles-1) * delta_i,max.

    For competitor displacement d the one-mode score gap is

        cos(e) - cos(e-Delta)
        = -2 sin(Delta/2) sin(e-Delta/2).

    Component errors are independent, so a conservative worst-case total gap is
    obtained by minimizing each mode over its allowed phase-error interval and
    summing those exact scalar minima.  Positive output certifies separation for
    every bounded static ratio-error vector in the box.

    This certificate is for the normalized per-mode phase geometry after an
    invertible physical-coordinate calibration.  It is not a transistor/circuit
    interval proof of the whole metacircuit.
    """
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    tmax = int(n) * int(cycles) - 1
    step_errors = worst_mode_phase_errors(n, frequencies, eta)
    if any(not math.isfinite(x) for x in step_errors):
        return float("-inf")

    best = float("inf")
    for d in range(1, n):
        gap = 0.0
        for f, delta in zip(frequencies, step_errors):
            Delta = (2.0 * math.pi * int(f) * d / n) % (2.0 * math.pi)
            s = math.sin(Delta / 2.0)  # nonnegative for Delta in [0,2pi)
            E = tmax * delta
            z_lo = -E - Delta / 2.0
            z_hi = E - Delta / 2.0
            gap += -2.0 * s * max_sin_on_interval(z_lo, z_hi)
        best = min(best, gap)
    return float(best)


def max_certified_cycles(n: int, frequencies: list[int], eta: float, cap: int = 100000) -> tuple[int, bool]:
    """Largest positive-margin cycle count; `censored=True` if still safe at cap."""
    if robust_phase_margin(n, frequencies, eta, 1) <= 0.0:
        return 0, False
    lo, hi = 1, 2
    while hi <= cap and robust_phase_margin(n, frequencies, eta, hi) > 0.0:
        lo = hi
        hi *= 2
    if hi > cap:
        if robust_phase_margin(n, frequencies, eta, cap) > 0.0:
            return int(cap), True
        hi = cap
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if robust_phase_margin(n, frequencies, eta, mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return int(lo), False


@dataclass
class CertificateRow:
    relative_ratio_tolerance: float
    unconstrained_max_step_phase_error: float
    conditioned_max_step_phase_error: float
    unconstrained_certified_cycles: int
    unconstrained_censored: bool
    conditioned_certified_cycles: int
    conditioned_censored: bool
    certified_cycle_ratio: float
    unconstrained_margin_at_16: float
    conditioned_margin_at_16: float
    unconstrained_margin_at_64: float
    conditioned_margin_at_64: float


def main():
    p = argparse.ArgumentParser(description="Certify C_n phase-code tolerance under bounded static resonator-ratio mismatch")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--condition-cap", type=float, default=2.0)
    p.add_argument("--tolerances", nargs="+", type=float, default=[2e-6,5e-6,1e-5,2e-5,5e-5,1e-4])
    p.add_argument("--cycle-cap", type=int, default=10000)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    unconstrained = design.greedy_design(args.n, args.modes, None)
    conditioned = design.greedy_design(args.n, args.modes, args.condition_cap)
    rows = []
    for eta in args.tolerances:
        u_step = max(worst_mode_phase_errors(args.n, unconstrained.frequencies, eta))
        c_step = max(worst_mode_phase_errors(args.n, conditioned.frequencies, eta))
        u_cycles, u_censored = max_certified_cycles(args.n, unconstrained.frequencies, eta, args.cycle_cap)
        c_cycles, c_censored = max_certified_cycles(args.n, conditioned.frequencies, eta, args.cycle_cap)
        ratio = float(c_cycles / u_cycles) if u_cycles > 0 else float("inf")
        rows.append(
            CertificateRow(
                relative_ratio_tolerance=float(eta),
                unconstrained_max_step_phase_error=float(u_step),
                conditioned_max_step_phase_error=float(c_step),
                unconstrained_certified_cycles=u_cycles,
                unconstrained_censored=u_censored,
                conditioned_certified_cycles=c_cycles,
                conditioned_censored=c_censored,
                certified_cycle_ratio=ratio,
                unconstrained_margin_at_16=robust_phase_margin(args.n, unconstrained.frequencies, eta, 16),
                conditioned_margin_at_16=robust_phase_margin(args.n, conditioned.frequencies, eta, 16),
                unconstrained_margin_at_64=robust_phase_margin(args.n, unconstrained.frequencies, eta, 64),
                conditioned_margin_at_64=robust_phase_margin(args.n, conditioned.frequencies, eta, 64),
            )
        )

    payload = {
        "config": vars(args),
        "unconstrained_frequencies": unconstrained.frequencies,
        "conditioned_frequencies": conditioned.frequencies,
        "rows": [asdict(x) for x in rows],
        "interpretation": {
            "static_mismatch": "bounded static ratio bias gives phase drift proportional to time, so the certified free-run horizon scales approximately as 1/tolerance in the small-error regime",
            "dynamic_noise_contrast": "the earlier additive per-step noise frontier was diffusion-like, with free-run time roughly scaling as 1/sigma^2",
            "scope": "certificate is exact for the normalized independent phase-mode uncertainty model, not a full circuit interval proof",
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("eta u_step c_step u_cycles c_cycles ratio u_m64 c_m64")
        for r in rows:
            print(f"{r.relative_ratio_tolerance:.1e} {r.unconstrained_max_step_phase_error:.3e} "
                  f"{r.conditioned_max_step_phase_error:.3e} {r.unconstrained_certified_cycles:7d} "
                  f"{r.conditioned_certified_cycles:7d} {r.certified_cycle_ratio:6.2f} "
                  f"{r.unconstrained_margin_at_64:+.3f} {r.conditioned_margin_at_64:+.3f}")


if __name__ == "__main__":
    main()
