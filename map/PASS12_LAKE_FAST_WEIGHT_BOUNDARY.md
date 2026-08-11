# Pass 12 — the lake is a fast-weight programmer: read-before-write prior-art boundary

Date: 2026-08-10

This pass was triggered by a physical picture:

> A wave arrives carrying an address/value. The medium must first respond with what it already says at that address; only the discrepancy should bend the medium. Later a related wave probes the changed medium and retrieves the stored association.

That picture is useful.

It is also extremely close to established fast-weight / delta-rule memory.

---

# 1. The exact algebra already exists

Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* (ICML 2021):

https://proceedings.mlr.press/v139/schlag21a.html

interpret a dynamically changing matrix `W_t` as associative memory.

The key operations are, schematically,

```text
read current value:
    v_bar = W_(t-1) phi(k)

compute surprise:
    delta_v = v - v_bar

write correction:
    W_t = W_(t-1) + beta * delta_v outer phi(k)

query:
    y = W_t phi(q)
```

So the physical sentence

> read what the medium already says, then bend it only by the error

is not merely analogous to the delta-rule fast-weight update. It is the same abstract computation.

### Subtraction

The following are occupied:

```text
key/value/query associative memory
fast weights as dynamically changing short-term memory
read-before-write correction
delta-rule correction of an existing key-value mapping
learned write strength
```

---

# 2. This is now a major efficient-sequence-model line

Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (NeurIPS 2024):

https://arxiv.org/abs/2406.06484

scaled DeltaNet to standard language-model settings with hardware-efficient chunkwise training.

Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule* (ICLR 2025):

https://arxiv.org/abs/2412.06464

explicitly separate two complementary operations:

```text
delta rule -> targeted corrective write
gating     -> rapid memory erasure
```

That is strikingly close to Pass 11's algebraic split between reversible/state-preserving work and explicit forgetting/reset work, although the mathematical objects are not identical.

Recent 2026 extensions include preconditioned delta rules and independently addressed erase/write operations.

Examples:

- OSDN, https://arxiv.org/abs/2605.13473
- Preconditioned DeltaNet, https://arxiv.org/abs/2604.21100
- Erase-then-Delta Attention, https://arxiv.org/abs/2606.26560

So "give the memory a separate erase path" is also active frontier work.

---

# 3. Sparse delta memory closes another apparent hole

Cabannes et al., *Sparse Delta Memory: Scaling the State of Linear RNNs through Sparsity* (July 2026):

https://arxiv.org/abs/2607.07386

replace the dense DeltaNet memory with a much larger explicit sparse memory bank. Each token sparsely selects locations to read/write while using gated delta-style updates.

This kills the broad claim:

```text
"make delta-rule memory sparse/local rather than dense"
```

unless `local` is defined physically/geometrically rather than merely as sparse logical addressing.

The KYY distinction, if any, must therefore be something like:

```text
logical sparse address:
    token can select distant slots directly

versus

physical local geometry:
    influence must propagate through declared neighbouring couplings
```

That difference is a resource/cost statement, not a memory-rule novelty.

---

# 4. The optical / writable-medium version is old too

The lake/hologram picture also has direct physical ancestors.

Yoshinaga, Kitayama & Hori, *Experimental learning in an optical perceptronlike neural network* (Optics Letters 1989):

https://doi.org/10.1364/OL.14.000716

used photorefractive crystals as holographic interconnection media and experimentally implemented a delta learning rule.

Yoshinaga, Kitayama & Hara, *All-optical error-signal generation for backpropagation learning in optical multilayer neural networks* (Optics Letters 1989):

https://doi.org/10.1364/OL.14.000202

experimentally generated optical error signals including subtraction and derivative operations for adaptive photorefractive networks.

Suh & Lee, *Holographic associative memory based on adaptive learning including outer-product learning* (Applied Optics 1992):

https://doi.org/10.1364/AO.31.000199

implemented adaptive holographic associative memory.

And physical self-learning is a current research field, e.g.:

- Stern et al., *Physical Networks Become What They Learn* (PRL 2025), https://doi.org/10.1103/PhysRevLett.134.147402
- Ezraty, Stern & Rubinstein, *Harnessing intuitive local evolution rules for physical learning* (PRE 2026), https://doi.org/10.1103/51dl-czj3
- adaptive/self-learning physical reservoirs and photonic plastic networks.

### Subtraction

Do not claim:

```text
"waves in a medium implement associative memory"
"the medium itself changes as it learns"
"local physical adaptation can train a network"
"holographic interference can write weights"
"optics can implement a delta error signal"
```

All occupied.

---

# 5. What the physical lens still changes

The useful distinction is not the update equation.

DeltaNet-style memory is usually described as an explicit matrix/bank with an address primitive. Even Sparse Delta Memory allows a token to select sparse locations algorithmically.

The Geometric-Neuron/KYY constraint is harsher:

> **A memory location is not reachable because an index says so; it is reachable only through propagation supported by the declared geometry.**

That introduces costs absent from the abstract fast-weight equation:

```text
propagation distance
mixing time / wave travel time
wire span
number and location of writable couplings
local read/write bandwidth
interference/crosstalk
precision and dissipation
```

This is where the lake/bicycle picture is useful: it forces `addressing` to be a physical process instead of a free array lookup.

---

# 6. A possible bridge to Pass 11

Pass 11 found the automata-theoretic split:

```text
permutation/group behavior  +  reset/aperiodic behavior
```

Fast-weight memory gives a different split:

```text
retrieve current association
    +
correct/write association
    +
optional erase/forget gate
```

These are not the same decomposition.

But together they suggest a precise diagnostic question:

> When a finite task's transition monoid says that a step is behaviorally reversible, can a writable geometric memory execute that part using only conservative/local propagation, reserving actual contraction/erase/write events for the algebraically irreversible steps?

If yes, the task decomposition could tell the physical substrate **where it is worth paying the plasticity/erase cost**.

That is not established here and may have prior art in reversible computing, physical learning, and automata synthesis.

---

# 7. Current residual after this pass

Not:

> QKV but waves.

Not:

> read before write.

Not:

> delta-rule physical memory.

Not:

> sparse delta memory.

The surviving KYY question is narrower:

> **Can an associative fast-weight update be constrained to a declared local propagating geometry, while using the task's behavioral algebra to separate reversible transport from genuinely information-destroying write/erase operations, and does that reduce physical communication/energy/storage cost at fixed behavior?**

Status:

**BRIDGE / UNMAPPED IN THIS SEARCH / HIGH PRIOR-ART RISK.**

The next step remains an oracle/resource experiment, not a new neural layer.