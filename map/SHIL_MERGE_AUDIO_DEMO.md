# Audible SHIL merge demo

Date: 2026-08-11

The C4 SHIL composition-margin result now has an audio artifact.

This is a **sonification of the reduced phase model**, not a recording of oscillator hardware.

## Why a reference is necessary

An isolated steady sine wave does not make its absolute phase perceptually meaningful: changing only the global phase does not create a different steady-state pitch or timbre.

Therefore the demo does not pretend the four phase basins are directly audible by themselves.

Instead it uses a physical-style same-frequency I/Q reference:

```text
left  = oscillator + 0-degree reference
right = oscillator + 90-degree reference.
```

The oscillator remains at one frequency (`220 Hz`).

Different C4 phases interfere differently with the two references, producing distinct stereo/amplitude patterns.

## Demo structure

```text
110 Hz mono marker
    -> bad symmetric midpoint alpha=pi/4
    -> 12 repeated trials from source q=1

long gap

660 Hz mono marker
    -> compiler-selected alpha=pi/8
    -> 12 repeated trials from the same source q=1.
```

Noise level:

```text
phase diffusion D=.02.
```

Fixed seed:

```text
1234.
```

The symbolic merge target for source `q=1` is `q=0`.

## Generated result

Bad midpoint:

```text
decoded = [0,0,1,0,1,0,1,0,0,1,1,1]
correct = 6/12
accuracy = .5.
```

Compiled `pi/8`:

```text
decoded = [0,0,0,0,0,0,0,0,0,0,0,0]
correct = 12/12
accuracy = 1.0.
```

The audible point is therefore not a pitch comparison.

It is repeated phase-state resolution against a fixed reference:

> the locally symmetric `pi/4` merge hands C4 an ambiguous separatrix state; the `pi/8` merge hands C4 a state with positive re-entry margin.

## Files

- `map/shil_merge_audio_demo.py`
- `tests/test_shil_merge_audio_demo.py`
- `.github/workflows/shil-merge-audio-demo.yml`
- Actions artifact `shil-merge-audio-demo` containing:
  - `shil_merge_audio_demo.wav`
  - `shil_merge_audio_demo.json`

Focused CI is green.
