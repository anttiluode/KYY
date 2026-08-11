# Pass 39 — cyclic port symmetry compression

Date: 2026-08-10

Passes 31–38 made the recurrent operator exact and made port transport increasingly structured, but the final behavioral check still inspected every legal state.

Claude suggested replacing that enumeration by a Fejer–Riesz / trigonometric-polynomial positivity certificate. That suggestion exposed a useful mistake before it exposed a certificate.

## The important correction

For the C_n experiments the trained output layer is an arbitrary linear classifier

```text
W in R^{n x 2m},   b in R^n.
```

For C101 with m=8 this is

```text
101 * (16 + 1) = 1717
```

free scalars.

Before imposing symmetry, correctness is therefore not one fixed low-degree trigonometric polynomial. The winning row changes with the symbolic state, and the decoder itself contains O(nm) independent information.

There is a second correction. With m harmonic modes the Fourier *support cardinality* is at most `2m+1`, but the ordinary trigonometric degree is set by the actual character frequencies. In the ten C101 models below, m=8 while the largest centered character frequency ranges from 32 to 49.

So the statement

```text
8 modes => degree 8 => n-independent 9x9 SDP
```

is false for the machines actually trained here.

The right first move is to legalize/compress the **port symmetry**.

---

## Exact C_n-equivariant projection

Let the legalized hidden representation be rho(j), a block-diagonal bank of exact character rotations. Exact cyclic equivariance of the output classes requires

```text
w_j = rho(j) w_0
b_j = constant.
```

Given an arbitrary learned decoder, its least-squares projection onto this subspace is

```text
w_0 = (1/n) sum_j rho(j)^T w_j
w_j^eq = rho(j) w_0
b_j^eq = mean(b).
```

This is a standard Reynolds/group-average projection onto an equivariant linear-map space. Group-equivariant weight sharing is established mathematics and ML prior art; no novelty claim is made for the projection itself.

The free parameter count becomes

```text
raw decoder          n(2m+1)
equivariant decoder  2m+1
```

and for C101,m=8:

```text
1717 -> 17 parameters
101x compression.
```

---

## Ten-seed C101 result

Setup matches the recent legalization stress set:

```text
n                 101
complex modes       8
train length        16
train steps       2200
seeds             0..9
random start        yes
```

After exact character snapping and the existing zero-label midpoint recentering, the learned ports are already 101/101 correct on all ten seeds.

Projecting those decoders into exact C101 equivariance gives:

```text
10 / 10 exact
10 / 10 minimum margins improved
```

Selected rows:

| seed | inherited min margin | equivariant min margin | relative decoder move |
|---:|---:|---:|---:|
| 0 | +2.539 | +4.232 | 0.287 |
| 1 | +1.130 | +3.178 | 0.307 |
| 6 | +2.880 | +5.361 | 0.263 |
| 7 | +0.363 | +2.245 | 0.365 |
| 9 | +1.930 | +3.756 | 0.291 |

The projection moves the explicit learned decoder by roughly 26–36% in relative Frobenius norm, yet removes no required behavior and increases the legal-orbit classification margin on every seed.

The resulting logit table is circulant to numerical precision (roughly 1e-13 defect), whereas the inherited learned decoders have large symmetry defects.

This says the trained classifier contains substantial symmetry-breaking slack that the compiled machine does not need.

Full result:

- `results/cyclic_c101_equivariant_port.json`

---

## Does this subsume the timing / midpoint pass?

Yes for correctness in this controlled cyclic family.

A separate ablation removed midpoint recentering entirely and asked whether full port-symmetry projection alone could repair the raw snapped operator.

Before symmetry projection, four of ten inherited ports fail:

| seed | raw inherited accuracy | raw inherited min margin |
|---:|---:|---:|
| 1 | 96/101 | -0.684 |
| 5 | 97/101 | -0.531 |
| 7 | 56/101 | -2.887 |
| 9 | 88/101 | -1.401 |

After projecting the output layer into exact cyclic equivariance:

```text
10 / 10 become 101/101 exact
minimum raw-equivariant margin range: +0.636 .. +4.147
```

Adding midpoint timing/phase recentering afterward still raises the margins substantially:

```text
minimum midpoint+equivariant margin range: +2.216 .. +5.348
```

For seed 7:

```text
raw inherited snapped port      56/101,  margin -2.887
symmetry-projected port        101/101,  margin +0.636
+ midpoint phase conditioning  101/101,  margin +2.216
```

So the clean compiler interpretation is now:

```text
port symmetry projection  -> correctness repair
phase/timing recentering   -> optional conditioning / robustness knob
```

The biological timing detour therefore did useful work without becoming biology theory: it identified a one-dimensional slice of the interface symmetry. The later group projection subsumes that slice as a correctness mechanism while retaining it as a useful margin-control parameter.

Full ablation:

- `results/cyclic_c101_equivariant_port_ablation.json`

---

## Why this matters for certification

Once the decoder is equivariant,

```text
score(state k, class j) = q(j-k)
```

for one relative-score function q on C_n.

All n state-specific classification problems collapse to one displacement problem:

```text
q(0) > q(d)  for every d != 0.
```

The Fourier support of q is contained in

```text
{0, +/-f_1, ..., +/-f_m},
```

so its support cardinality is at most `2m+1`, independent of the explicit n-class decoder size.

This is finally the correct place to invoke finite-group Fourier positivity / SOS machinery if the learned interface must be preserved.

However, sparse Fourier SOS on finite abelian groups is established work, and its certificate size is not automatically independent of n. In particular, existing bounds depend on the support geometry / group order in various ways. We therefore do **not** claim that equivariance alone solves generic verification.

Pass 40 asks whether an even smaller decoder family makes the cyclic certificate elementary instead of semidefinite.

---

## Prior-art boundary

Established ingredients include:

- group-equivariant linear maps and parameter sharing;
- Reynolds/group averaging onto invariant/equivariant subspaces;
- Fourier analysis on finite cyclic/abelian groups;
- nonnegative trigonometric-polynomial / Fejer–Riesz machinery;
- sparse Fourier sum-of-squares certificates on finite abelian groups.

Relevant sources include:

- Cohen & Welling, *Group Equivariant Convolutional Networks*, 2016.
- Fawzi, Saunderson & Parrilo, *Sparse sum-of-squares certificates on finite abelian groups*, 2015.
- Yang, Ye & Zhi, *Computing sparse Fourier sum of squares on finite abelian groups in quasi-linear time*, 2022.
- Yang, Ye & Zhi, *Fourier sum of squares certificates*, 2022.

The KYY object under test remains the **compiler composition**, not these ingredients:

```text
learn approximate dynamics
    -> legalize task algebra
    -> project/transport the observable interface using that algebra
    -> certify the resulting behavioral machine.
```

## Files

- `map/cyclic_equivariant_port_probe.py`
- `map/cyclic_equivariant_port_ablation.py`
- `tests/test_cyclic_equivariant_port_probe.py`
- `.github/workflows/cyclic-equivariant-port.yml`
- `.github/workflows/cyclic-equivariant-port-ablation.yml`
- `results/cyclic_c101_equivariant_port.json`
- `results/cyclic_c101_equivariant_port_ablation.json`
