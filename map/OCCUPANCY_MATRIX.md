# Occupancy matrix

This is the most compact map in the folder.

The point is to stop asking **"has anyone done something like this?"** and instead ask which coordinates of the proposed object are already occupied.

Legend:

- `X` = clearly present / central to the cited work.
- `~` = partial, related, or present in a materially different setting.
- blank = not a central feature of that work as mapped here.

A blank is **not evidence of novelty**. It only means the current map does not use that paper to occupy the cell.

## Axes

| code | mechanism |
|---|---|
| `GEO` | fixed geometry/topology materially determines transition structure |
| `LOWCTRL` | a small/shared control parameterization generates a larger operator family |
| `XCOND` | transition/operator changes with current input/token |
| `LOCAL` | primitive communication is local/small-block rather than global |
| `OSC` | oscillatory / second-order / wave state is central |
| `REC` | genuine temporal recurrence/state update |
| `TRACK` | modern finite-state/state-tracking expressivity is directly evaluated/theorized |
| `PHYS` | literal wave/physical substrate is part of the computation |
| `RESOURCE` | communication/fan-in/depth/resource trade-off is directly part of the question |
| `COMPILE` | target operator is mapped/inverse-designed into a constrained physical realization |

## Matrix

| work / family | GEO | LOWCTRL | XCOND | LOCAL | OSC | REC | TRACK | PHYS | RESOURCE | COMPILE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| classical realization / control theory |  |  |  |  | ~ | X | ~ |  | X | X |
| uRNN |  |  |  |  | X | X |  |  | ~ |  |
| EUNN / Givens mesh |  |  |  | X | X | X |  | ~ | X | ~ |
| oRNN / Householder |  |  |  |  | ~ | X |  |  | X |  |
| ISAN |  |  | X |  |  | X | ~ |  |  |  |
| HyperNetworks |  | X | X/~ |  |  | X/~ |  |  | ~ |  |
| HyperNEAT | X | X |  | ~ |  | ~ |  |  | X |  |
| coRNN |  |  | X/~ | ~ | X | X |  |  | ~ |  |
| GraphCON | X |  | X/~ | X | X | ~ |  |  | ~ |  |
| delay reservoir computing | ~ | X/~ |  | X/~ | X | X | ~ | X/~ | X |  |
| Mamba / Mamba-2 |  | X/~ | X | ~ | ~ | X | X/~ |  | X |  |
| negative-eigenvalue LRNNs |  |  | X/~ |  | X | X | X |  | X |  |
| DeltaProduct |  |  | X |  | ~ | X | X |  | X |  |
| PD-SSM / Flash PD-SSM |  |  | X/~ | X/~ | X | X | X |  | X |  |
| H-LRU / BD-LRU |  |  | X/~ | X/~ | X/~ | X | X |  | X |  |
| SLiCEs structured linear CDEs |  |  | X | X/~ | ~ | X | X |  | X |  |
| bilinear state-transition RNNs |  |  | X |  |  | X | X |  | ~ |  |
| TCP-SSM token-conditioned poles |  | X | X | ~ | X | X | ~ |  | X |  |
| **Wave Physics as an Analog RNN** | X/~ |  |  | X | X | X |  | X | X | X |
| photonic unitary meshes / Clements | X/~ |  |  | X | X | ~ |  | X | X | X |
| OREO / photonic recurrent hardware | X/~ |  |  | X/~ | X | X |  | X | X | X/~ |
| Wanjura–Marquardt wave scattering | X/~ | ~ | X | X/~ | X | ~ |  | X | X | X/~ |
| programmable-metasurface structural encoding | X | ~ | X | X/~ | X | ~ |  | X | X | X/~ |
| Horne–Hush FSM fan-in bounds |  |  |  | X |  | X | X |  | X |  |
| TWC / TW-1A project semantics | X | ~ | ~ | X | X | X | ~ | X/~ | X | X |

## What the matrix immediately kills

These are not KYY openings by themselves:

```text
"use waves as an RNN"                 -> Hughes et al. 2019
"use local 2-port rotations"          -> EUNN / optical meshes
"make a graph define oscillator flow" -> GraphCON
"make Q depend on input"              -> ISAN / selective SSMs
"use richer mixing for state tracking"-> DeltaProduct / PD / BD-LRU / SLiCE
"generate weights compactly"          -> HyperNetworks / HyperNEAT
"shared base dynamics + token control"-> TCP-SSM and selective SSM family
"local fan-in changes FSM cost"        -> Horne & Hush 1993
"physical scattering computes"         -> large wave-computing literature
```

That is a lot of red ink. Good.

## The visible hole

No single mapped row currently has all of:

```text
GEO + LOWCTRL + XCOND + LOCAL + REC + TRACK
```

and certainly not all of:

```text
GEO + LOWCTRL + XCOND + LOCAL + OSC + REC + TRACK + PHYS + RESOURCE + COMPILE
```

But **that does not make the conjunction new**. Conjunction novelty is the easiest kind of novelty to hallucinate: the exact combination may exist under a different vocabulary, or each extra ingredient may add no useful capability.

The map therefore turns the hole into a question rather than a claim.

## Candidate KYY residual, version 2026-08-10

A stricter form than the first KYY model is:

```text
             fixed geometry G
          / lengths / topology \
         /  static local physics \
                    |
                    v
input x_t ---> tiny control a_t          dim(a_t)=r
                    |                       r << E
                    v
             Q_t = F(G, a_t)
                    |
                    v
          local recurrent propagation
                    |
                    v
                 state z_t
```

The key feature is **parameter tying imposed by geometry**:

```text
not:   one independent theta[token, edge]

but:   theta[token, edge]
       = sum_k alpha_k(token) * phi_k(edge geometry)
```

or another physically/structurally motivated generator.

The current map has strong neighbours on every side:

- HyperNEAT: geometry -> many weights.
- HyperNetworks: small generator -> many weights.
- TCP-SSM: shared base dynamics -> token-conditioned stable poles.
- ISAN/Mamba: input -> changing transition.
- GraphCON: geometry -> local oscillator coupling.
- EUNN: local mesh -> global recurrent unitary.
- Hughes et al.: wave geometry -> analog RNN.
- modern LRNNs: transition structure -> state-tracking ability.

So the **only defensible residual** is the exact structural tying and its measured consequence.

## Best no-new-architecture experiment suggested by the map

Before training `GeometryGeneratedSSM`, take already-trained KYY `geom_scatter` transitions and ask whether their free edge controls actually lie near a low-dimensional family.

For the learned angle tensor

```text
theta[token, sweep, edge]
```

measure:

1. SVD rank / effective rank over edges;
2. reconstruction from a Fourier basis over ring position;
3. reconstruction from graph-Laplacian eigenmodes;
4. reconstruction from small smooth coordinate MLP/CPPN-like generators;
5. behavior after replacing the free angles with each compressed reconstruction.

Sweep retained control dimension `r = 1, 2, 3, 4, ...`.

The decisive plot is:

```text
state-tracking accuracy
        ^
        |
        |               free-edge model
        |--------------------*
        |                 .
        |              .
        |           .
        |        .
        |_____.________________________> control dimension r
```

If behavior collapses until `r ≈ E`, the strongest remaining Geometric-Neuron story has no evidence on this task.

If a tiny `r` retains the behavior, **then** we have earned a reason to search the exact low-control geometry family harder and perhaps train it directly.

That is where the map currently points.