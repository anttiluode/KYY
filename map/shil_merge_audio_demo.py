from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

from map.shil_c4_quotient_backend import decode_c4, relax_phase

TAU = 2.0 * math.pi


def simulate_source_trials(
    source_q: int,
    alpha: float,
    count: int,
    seed: int,
    diffusion: float = 0.02,
    kappa: float = 4.0,
    dt: float = 0.002,
    merge_time: float = 0.5,
    relock_time: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decoded states and final phases for repeated trials from one C4 source state."""
    rng = np.random.default_rng(seed)
    phi = np.full(int(count), (int(source_q) % 4) * math.pi / 2.0, dtype=np.float64)
    merge_steps = max(1, int(round(merge_time / dt)))
    relock_steps = max(1, int(round(relock_time / dt)))
    phi = relax_phase(phi, 2, alpha, kappa, diffusion, dt, merge_steps, rng)
    phi = relax_phase(phi, 4, 0.0, kappa, diffusion, dt, relock_steps, rng)
    return decode_c4(phi), phi


def raised_cosine_envelope(n: int, fade_samples: int) -> np.ndarray:
    env = np.ones(int(n), dtype=np.float64)
    f = min(int(fade_samples), int(n) // 2)
    if f <= 0:
        return env
    x = np.linspace(0.0, math.pi / 2.0, f, endpoint=False)
    ramp = np.sin(x) ** 2
    env[:f] = ramp
    env[-f:] = ramp[::-1]
    return env


def render_phase_with_iq_reference(
    phase: float,
    sample_rate: int,
    duration: float,
    tone_hz: float,
    gain: float = 0.44,
) -> np.ndarray:
    """Stereo phase sonification using two physical same-frequency references.

    Absolute phase of an isolated steady sine is not audible.  The left channel
    interferes the oscillator with a 0-degree reference; the right channel uses
    a +90-degree reference.  The four C4 phases therefore produce four distinct
    stereo/amplitude patterns without changing the oscillator frequency.
    """
    n = max(1, int(round(sample_rate * duration)))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    w = TAU * float(tone_hz)
    osc = np.sin(w * t + float(phase))
    ref_i = np.sin(w * t)
    ref_q = np.cos(w * t)
    left = 0.5 * (osc + ref_i)
    right = 0.5 * (osc + ref_q)
    env = raised_cosine_envelope(n, int(round(0.012 * sample_rate)))
    stereo = np.column_stack([left * env, right * env]) * float(gain)
    return stereo


def render_marker(sample_rate: int, duration: float, hz: float, gain: float = 0.28) -> np.ndarray:
    n = max(1, int(round(sample_rate * duration)))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    sig = np.sin(TAU * float(hz) * t)
    env = raised_cosine_envelope(n, int(round(0.015 * sample_rate)))
    y = gain * sig * env
    return np.column_stack([y, y])


def silence(sample_rate: int, duration: float) -> np.ndarray:
    return np.zeros((max(1, int(round(sample_rate * duration))), 2), dtype=np.float64)


def section_audio(
    decoded: np.ndarray,
    sample_rate: int,
    tone_hz: float,
    trial_duration: float,
    gap_duration: float,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for q in np.asarray(decoded, dtype=np.int64):
        phase = (int(q) % 4) * math.pi / 2.0
        chunks.append(render_phase_with_iq_reference(phase, sample_rate, trial_duration, tone_hz))
        chunks.append(silence(sample_rate, gap_duration))
    return np.vstack(chunks) if chunks else np.zeros((0, 2), dtype=np.float64)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(audio, dtype=np.float64), -0.999, 0.999)
    pcm = (y * 32767.0).astype('<i2')
    with wave.open(str(p), 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def build_demo(
    output_wav: str | Path,
    output_json: str | Path | None = None,
    source_q: int = 1,
    count: int = 12,
    seed: int = 1234,
    diffusion: float = 0.02,
    sample_rate: int = 48000,
    tone_hz: float = 220.0,
) -> dict:
    bad_alpha = math.pi / 4.0
    good_alpha = math.pi / 8.0

    bad_decoded, bad_phi = simulate_source_trials(source_q, bad_alpha, count, seed, diffusion)
    good_decoded, good_phi = simulate_source_trials(source_q, good_alpha, count, seed, diffusion)

    # Low marker = locally symmetric but compositionally unsafe midpoint.
    # High marker = compiler-selected positive re-entry margin.
    chunks = [
        silence(sample_rate, 0.25),
        render_marker(sample_rate, 0.45, 110.0),
        silence(sample_rate, 0.18),
        section_audio(bad_decoded, sample_rate, tone_hz, 0.28, 0.07),
        silence(sample_rate, 0.65),
        render_marker(sample_rate, 0.45, 660.0),
        silence(sample_rate, 0.18),
        section_audio(good_decoded, sample_rate, tone_hz, 0.28, 0.07),
        silence(sample_rate, 0.25),
    ]
    audio = np.vstack(chunks)
    write_wav(output_wav, audio, sample_rate)

    target = 0 if source_q in (0, 1) else 2
    payload = {
        "source_q": int(source_q),
        "symbolic_target_after_merge": int(target),
        "trial_count_per_section": int(count),
        "seed": int(seed),
        "diffusion": float(diffusion),
        "audio": {
            "sample_rate": int(sample_rate),
            "tone_hz": float(tone_hz),
            "readout": (
                "stereo same-frequency I/Q interference reference; absolute phase is not claimed to be audible without a reference"
            ),
            "section_markers": "110 Hz mono marker = alpha=pi/4 bad midpoint; 660 Hz marker = alpha=pi/8 compiled",
        },
        "bad_midpoint": {
            "alpha": bad_alpha,
            "decoded_states": [int(x) for x in bad_decoded],
            "target_hits": int(np.sum(bad_decoded == target)),
            "accuracy": float(np.mean(bad_decoded == target)),
            "final_phases": [float(x) for x in bad_phi],
        },
        "compiled": {
            "alpha": good_alpha,
            "decoded_states": [int(x) for x in good_decoded],
            "target_hits": int(np.sum(good_decoded == target)),
            "accuracy": float(np.mean(good_decoded == target)),
            "final_phases": [float(x) for x in good_phi],
        },
        "listening_note": (
            "After the low marker the same source state resolves between two audible C4 reference patterns. "
            "After the high marker the repeated trials should stay on the target pattern."
        ),
        "scope": "Sonification of the reduced SHIL phase model, not a recording of physical oscillator hardware.",
    }
    if output_json is not None:
        jp = Path(output_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Generate an audible I/Q-reference demo of the C4 SHIL merge composition margin")
    p.add_argument('--wav', default='results/shil_merge_audio_demo.wav')
    p.add_argument('--json-out', default='results/shil_merge_audio_demo.json')
    p.add_argument('--source-q', type=int, default=1)
    p.add_argument('--count', type=int, default=12)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--diffusion', type=float, default=0.02)
    args = p.parse_args()
    payload = build_demo(args.wav, args.json_out, args.source_q, args.count, args.seed, args.diffusion)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
