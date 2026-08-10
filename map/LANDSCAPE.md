# Landscape

This file maps the nearest known regions around KYY. It is organized by **mechanism**, not chronology.

Status labels:

- **OCCUPIED** — exact or very close mechanism already exists.
- **NEIGHBOUR** — important ingredient exists, but one major axis differs.
- **BRIDGE** — ingredients exist on both sides; the conjunction was not located in the current search.
- **UNMAPPED** — targeted search did not locate the exact object. Not a novelty claim.

---

## Region 1 — classical state-space / realization theory

**Status: OCCUPIED**

Core object:

```text
x[t+1] = A x[t] + B u[t]
y[t]   = C x[t] + D u[t]
```

Key inherited facts:

- state-space realizations have basis freedom under similarity transforms;
- controllability and observability determine minimal realizations;
- balanced truncation and Hankel singular values provide principled model reduction;
- input/output equivalence is not the same thing as equality of internal coordinates.

Modern SSM work is explicitly rediscovering/exploiting these tools. Balanced truncation has already been applied directly to S4/DSS compression, and later work regularizes Hankel singular values to create compressible SSMs.

**KYY implication:** neither gauge freedom nor balanced truncation is a new KYY hook. They are foundations and controls.

Primary anchors:

- Ezoe & Sato, *Model Compression Method for S4 with Diagonal State Space Layers using Balanced Truncation* (2024), https://arxiv.org/abs/2402.15993
- Schwerdtner et al., *Hankel Singular Value Regularization for Highly Compressible State Space Models* (2025), https://arxiv.org/abs/2510.22951
- Defourneau & Petreczky, *Realization theory of recurrent neural networks and rational systems* (2019), https://arxiv.org/abs/1903.05609

---

## Region 2 — unitary / orthogonal recurrent matrices

**Status: OCCUPIED**

### 2A. uRNN

Arjovsky, Shah & Bengio (2015/2016) use complex unitary recurrence to preserve norm and long-term dependencies.

- https://arxiv.org/abs/1511.06464

### 2B. EUNN / Givens mesh

Jing et al. (ICML 2017) parameterize a recurrent unitary transformation using alternating banks of local 2x2 transformations. The mesh depth is a capacity dial; full capacity can span the unitary group.

This is a direct hit on KYY's first `geom_scatter` mechanism.

- https://proceedings.mlr.press/v70/jing17a.html

The same 2-port mesh mathematics is standard in universal interferometers. Clements et al. provide the well-known rectangular optical mesh decomposition.

- Clements et al., *Optimal design for universal multiport interferometers* (Optica 2016), https://opg.optica.org/optica/article.cfm?uri=optica-3-12-1460

### 2C. Householder oRNN

Mhammedi et al. (ICML 2017) parameterize orthogonal RNN recurrence using Householder reflections.

- https://proceedings.mlr.press/v70/mhammedi17a.html

**KYY implication:** "local 2-port products vs global Householder products" is old architecture space. A modern state-tracking/resource comparison can still be useful, but it must be framed as a new benchmark question rather than a new recurrent primitive.

---

## Region 3 — input-switched / token-conditioned operators

**Status: OCCUPIED**

Input Switched Affine Networks (ISAN) already use input-dependent recurrent matrices:

```text
h[t] = W_{x[t]} h[t-1] + b_{x[t]}
```

The model was explicitly designed so that the input-dependent linear dynamics can be reverse engineered.

- Foerster et al., *Input Switched Affine Networks* (2016/2017), https://arxiv.org/abs/1611.09434

Modern selective SSMs make input dependence central at scale.

- Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* / Mamba-2 (2024), https://arxiv.org/abs/2405.21060
- Lahoti et al., *Mamba-3* (2026), https://arxiv.org/abs/2603.15569

**KYY implication:** `Q_t` being input-dependent is not a differentiator. The differentiator must be *how the family of possible Q_t is generated or constrained*.

---

## Region 4 — transition algebra and modern state tracking

**Status: OCCUPIED and extremely active**

This is the region KYY must benchmark against rather than narrate around.

### 4A. Negative and complex eigenvalues

Grazzi et al. show that restricting transition eigenvalues to positive values blocks parity and that complex eigenvalues are needed for modulo-3 counting in the analyzed LRNN setting. Their results connect algebraic structure directly to finite-state tracking.

- https://arxiv.org/abs/2411.12537

### 4B. Householder products / DeltaProduct

DeltaProduct forms transition matrices from multiple generalized Householder transformations per token and reports stronger state tracking and length extrapolation.

- https://arxiv.org/abs/2502.10297

### 4C. Structured sparse state tracking / PD-SSM

PD-SSM uses a product of a column one-hot matrix and complex diagonal matrix. It targets finite-state expressivity with recurrence cost comparable to diagonal SSMs and proves strong FSA realization guarantees.

- https://arxiv.org/abs/2509.22284

Flash PD-SSM continues this line with a memory-optimized implementation and modern mechanistic/state-tracking evaluation.

- https://arxiv.org/abs/2605.19150

### 4D. Mamba-3

Mamba-3 explicitly adds richer recurrence, complex state, and MIMO structure to improve state tracking and the performance-efficiency frontier.

- https://arxiv.org/abs/2603.15569

**KYY implication:** the live question is no longer "can non-diagonal/complex transitions track state?" It is whether **local geometric support or geometry-generated parameter tying** buys something beyond the already known expressive transition families.

---

## Region 5 — second-order oscillator RNNs

**Status: OCCUPIED**

### 5A. coRNN

Rusch & Mishra (2020) construct an RNN by discretizing a system of controlled nonlinear second-order oscillators. The biological/coupled-oscillator motivation, stable gradients, and long-time dependency framing are explicit.

- https://arxiv.org/abs/2010.00951

This is a close prior-art hit on KYY's first `geom_wave` intuition.

### 5B. neural oscillator universality

Lanthaler, Rusch & Mishra prove a broad universality result for neural oscillators, encompassing oscillator architectures used in sequence modeling, graph learning, and physical neural networks.

- https://arxiv.org/abs/2305.08753

**KYY implication:** "oscillators are a useful recurrent primitive" is occupied. KYY needs a narrower property: topology, physical locality, low-dimensional geometric control, reciprocal realization, or compileability.

---

## Region 6 — graph-coupled oscillator computation

**Status: OCCUPIED/NEIGHBOUR**

GraphCON uses second-order controlled/damped oscillator dynamics with coupling inherited from the graph adjacency structure.

- Rusch et al., *Graph-Coupled Oscillator Networks* (2022), https://arxiv.org/abs/2202.02296

GraphSSM later brings SSM theory directly to temporal graphs using structural information and Laplacian regularization.

- Li et al., *State Space Models on Temporal Graphs: A First-Principles Study* (2024), https://arxiv.org/abs/2406.00943

Reservoir literature also studies how graph topology changes computation, including ring reservoirs and connectome-inspired reservoirs.

- Gallicchio & Micheli, *Ring Reservoir Neural Networks for Graphs* (2020), https://arxiv.org/abs/2005.05294
- d'Andrea et al., *Complex topological features of reservoirs shape learning performances...* (2022), https://arxiv.org/abs/2211.00161

**KYY implication:** "topology matters" and "graph structure defines dynamics" are not new. A KYY result must identify a resource/capability trade-off not already explained by graph RNN / oscillator / reservoir work.

---

## Region 7 — delay-based and physical reservoir computing

**Status: OCCUPIED**

Delay systems have long been used as compact high-dimensional reservoirs. Appeltant et al. demonstrate computation using one nonlinear dynamical node with delayed feedback and time-multiplexed virtual nodes.

- Appeltant et al., *Information processing using a single dynamical node as complex system* (Nature Communications 2011), https://www.nature.com/articles/ncomms1476

More recent work explicitly links delayed embedding and reservoir dimension, including reductions to very small physical reservoirs.

- Duan et al., *Embedding Theory of Reservoir Computing and Reducing Reservoir Network Using Time Delays* (2023), https://arxiv.org/abs/2303.09042

**KYY implication:** delay geometry and physical memory are established computational resources. The old Geometric Neuron delay-manifold idea sits near this region.

---

## Region 8 — physical wave scattering as neural computation

**Status: OCCUPIED, but important bridge to TWC**

Wanjura & Marquardt (Nature Physics 2024) show that a system with **linear wave propagation** can implement nonlinear input-output computation by encoding the input into parameters of the scattering system. Other physical parameters are trainable. Their coupled-mode formulation covers optical resonators and general linear systems.

- https://www.nature.com/articles/s41567-024-02534-9

This is particularly close to the phrase:

```text
input -> changes physical operator -> probe -> scattering response -> output
```

It is therefore a major landmark for any KYY/TWC proposal involving a token-conditioned physical operator.

Related 2026 work on programmable-metasurface physical neural networks separately analyzes structural input encoding, mutual coupling, and depth.

- Hammami et al., *Expressivity of Programmable-Metasurface-Based Physical Neural Networks* (2026), https://arxiv.org/abs/2603.13602

**KYY implication:** "input encoded into a wave medium's operator" is not blank territory. The potential residual is the **recurrent/state-tracking** use of such an operator, or a compiler from learned recurrent operators into a constrained reciprocal physical medium.

---

## Region 9 — photonic meshes and local physical matrix realization

**Status: OCCUPIED**

Universal unitary transformations can be implemented by cascades of local 2x2 optical mixing units. This is not merely theory; programmable photonic meshes and recurrent loops have been proposed and demonstrated for neural computation.

Representative anchors:

- Clements et al. (2016), rectangular multiport interferometer mesh: https://opg.optica.org/optica/article.cfm?uri=optica-3-12-1460
- Bogaerts et al. / PyTorch photonic-circuit simulation work includes a unitary RNN formed by looping a unitary mesh onto itself: https://www.nature.com/articles/s41598-019-42408-2
- Radford et al., coupled-waveguide arrays inverse-designed for unitary matrices (2024): https://arxiv.org/abs/2409.18284
- OREO, an optoacoustic recurrent operator with acoustic-wave memory (Nature Communications 2024): https://www.nature.com/articles/s41467-024-47053-6

**KYY implication:** a local physical mesh realizing global linear transforms is established. TWC/KYY must contribute either a different constrained operator family, a compiler/diagnostic, or a measured resource advantage under its actual hardware constraints.

---

## Region 10 — oscillator-based physical alternatives to attention

**Status: NEIGHBOUR and very current**

A 2026 line from Pasqualetti & Guo uses coupled oscillators as computation for attention and sequential memory.

- *Attention by Synchronization in Coupled Oscillator Networks* (2026), https://arxiv.org/abs/2606.12059
- *Learnable Sequential Memory in Coupled Oscillator Networks* (2026), https://arxiv.org/abs/2607.18439

These papers are especially important because they are not old reservoir work; they explicitly ask how oscillator dynamics can implement modern AI operations on energy-constrained physical substrates.

**KYY implication:** the "maybe physical oscillators can replace expensive global AI operations" direction is actively occupied right now. Any KYY claim here must be much more specific than oscillator computation.

---

## Region 11 — symmetry, gauge freedom, and neural identifiability

**Status: OCCUPIED and rapidly developing**

Several modern lines treat function-preserving parameter transformations as symmetries/gauge freedoms.

- Hashimoto et al., *Unification of Symmetries Inside Neural Networks* (2024), https://arxiv.org/abs/2402.02362
- Lim et al., *The Empirical Impact of Neural Parameter Symmetries, or Lack Thereof* (2024), https://arxiv.org/abs/2405.20231
- Zhao, Walters & Yu, *Symmetry in Neural Network Parameter Spaces* (survey, 2025/2026), https://arxiv.org/abs/2506.13018
- Zhao et al., *Finding Symmetry in Neural Network Parameter Spaces* develops automatic symmetry discovery; 2026 OpenReview version: https://openreview.net/forum?id=0XhWusHpLq
- GENNI visualizes parameter equivalence classes: https://arxiv.org/abs/2011.07407

Sparse-autoencoder work now explicitly reports that individual learned features can vary across seeds while lower-dimensional **subspaces** are reproducible, a direct warning against interpreting arbitrary basis vectors as unique features.

- Gerasimov et al., *Unstable Features, Reproducible Subspaces* (2026), https://arxiv.org/abs/2606.12138

**KYY implication:** generic "build an identifiability audit for AI" is too broad. A viable residual would have to exploit a special exact realization group of a particular SSM/LRNN parameterization and produce a capability statement not already delivered by generic symmetry-discovery tools.

---

## Region 12 — controllability / observability applied to neural networks

**Status: OCCUPIED/NEIGHBOUR**

There is explicit modern work applying control-theoretic notions to neural networks and Mamba-like models.

- *Sparse Mamba: Introducing Controllability, Observability, And Stability To Structural State Space Models* (2024), https://arxiv.org/abs/2409.00563
- Moon, *From Black-Box to White-Box: Control-Theoretic Neural Network Interpretability* (2025), https://arxiv.org/abs/2511.12852

**KYY implication:** controllability/observability are tools we should use, not novelty claims.

---

# Where KYY actually sits after subtraction

After removing the occupied regions, the original cloudy proposition

> "geometric neuron as a better recurrent AI primitive"

shrinks into several much sharper questions:

```text
                    known EUNN / Givens
                         |
                         |
      coRNN ---------- KYY? ---------- modern state tracking
        |                |                   |
     GraphCON            |               DeltaProduct / PD
                         |
              low-dimensional geometry
                 generates Q_t
                         |
                         |
            physical reciprocal compiler
                         |
                         |
                       TWC
```

The remaining candidate is **not** "local rotations", "oscillators", "graphs", "input-dependent transitions", "complex state", or "wave hardware" separately.

The most distinctive residual currently visible is:

> **A small context/control signal acts on a fixed geometry whose local physical/delay/coupling structure generates a much larger recurrent operator family; the resulting computation is evaluated under modern state-tracking and communication/hardware resource constraints, and can in principle be lowered into the same reciprocal-wave semantics used by TWC.**

That sentence contains several known ingredients. The exact conjunction is only a **BRIDGE** until a deeper search establishes otherwise.

See [VALLEYS.md](VALLEYS.md) for how to test the residual without turning the benchmark into a self-fulfilling geometry task.