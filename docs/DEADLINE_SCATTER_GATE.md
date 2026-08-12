# Deadline scatter gate — availability is free; legibility is not

Date: 2026-08-12

This gate came from the `PresentMoment` branch, but it belongs in KYY because the
cleanest test uses KYY's existing local scatter operator and its existing discipline:

> **change one thing at a time and keep the strong controls.**

The motivating question was:

> If a local physical machine does not impose global layer barriers, can a downstream
> readout use partially completed computation at arbitrary physical deadlines?

The first version of that intuition was too strong.

The current result is sharper:

> **Removing barriers makes partial computation available automatically. It does not
> make that partial state semantically legible automatically.**

That distinction is now executable in
`experiments/deadline_scatter_gate.py`.

---

## 1. Why `geom_scatter` is a clean object for this

KYY's `geom_scatter` transition is already a circuit of local reciprocal 2-port
scatterers.

For each token it applies two checkerboard phases per sweep:

```text
sweep 0: phase 0 -> phase 1
sweep 1: phase 0 -> phase 1
```

Within a phase the edges are disjoint, so the local gates commute.  Across phases,
gates that share a node do not in general commute.

The ordinary implementation therefore gives us a canonical dependency graph without
inventing one for this experiment.

### Synchronized execution

The existing implementation behaves like:

```text
all phase-0 gates finish
        |
        barrier
        |
all phase-1 gates finish
        |
        barrier
        |
...
```

### Asynchronous execution

Give every physical edge a positive heterogeneous gate duration.

A local gate may start when the previous canonical gates touching its two endpoints
have completed:

```text
start(op_ij) = max(last_finish(i), last_finish(j))
```

Disjoint gates can therefore be in different stages at the same wall-clock time.

Crucially, **no overlapping pair is reordered**.

When all work finishes, the asynchronous state is exactly the same as the ordinary
KYY transition.  The script asserts this numerically.

So the A/B comparison is unusually clean:

```text
same learned angles
same local operations
same dependencies
same shared readout
same final state

only synchronization differs
```

---

## 2. Prior-art guardrail

Nothing broad here is a novelty claim.

Continuous/cascade processing in cognition is old.
Anytime prediction and early-exit networks are established.
Adaptive computation and learned halting are established.
Asynchronous/event-driven neural systems are established.
Delay reservoirs are established.

So this gate cannot earn:

> "we invented partial computation"

or:

> "we invented asynchronous neural networks."

The narrower KYY question is whether an **existing local operator** gets useful
physical-time deadline behavior from its local dependency structure under strong
controls.

---

## 3. The first surprise: ordinary KYY partial states can be terrible answers

A scratch reconstruction used the exact `geom_scatter` operator and ordinary KYY
`perm3` training: cross-entropy only after a complete token transition.

The model learned the final task perfectly.

Then the same transition was exposed at intermediate asynchronous deadlines.

The final shared readout often interpreted those states catastrophically badly.
Across the small reconstruction sweep:

```text
ordinary final-only training
final accuracy                     ~1.000
mean async deadline AUC            ~0.56
mean minimum partial accuracy      ~0.005   [6-class chance = .167]
```

Some partially completed states were therefore **worse than random guessing by a very
large margin**, despite eventually becoming a perfect final state.

This kills the naive claim:

> local asynchronous computation should degrade gracefully for free.

It does not.

The medium can make unfinished computation physically available while the decoder
still reads that unfinished coordinate frame incorrectly.

---

## 4. Availability versus legibility

This suggests two separate properties.

### Availability

Has some useful local computation physically completed by the deadline?

A heterogeneous local dataflow machine gets this largely from scheduling:

```text
fast routes can finish useful work
without waiting for unrelated slow routes
```

### Legibility

Does the same downstream readout attach a useful meaning to the resulting partial
state?

That is **not** automatic.

In symbols, if complete processing produces

```text
h_final -> readout -> correct meaning
```

there is no reason that

```text
h_partial -> same readout
```

must preserve that meaning.

The internal coordinate system can rotate, reflect, permute or otherwise pass through
states whose final decoder semantics are wrong.

That is exactly what ordinary KYY training showed in the reconstruction.

---

## 5. A minimal fix: one readout, supervised across homogeneous maturities

The second condition changes no architecture.

There are **no extra exit heads**.
There is **no halting controller**.
There is **no deadline input**.
There is **no asynchronous state during training**.

Instead, during the ordinary synchronized forward pass, the same shared linear readout
is supervised after every checkerboard phase:

```text
phase 0 state --\
phase 1 state ---\
phase 2 state ----> SAME readout -> current target
phase 3 state ---/
final state -----/
```

Call this **phase supervision** for now.

It asks the recurrent operator to keep task meaning readable in a shared coordinate
system while computation progresses.

This is closely related to deep supervision / anytime ideas and is not itself a
novelty claim.

---

## 6. The interesting OOD test

Training sees only homogeneous phase states:

```text
all coordinates after phase 0
all coordinates after phase 1
all coordinates after phase 2
all coordinates after phase 3
```

Evaluation then removes the global barriers.

At a physical deadline the state can be a mixture:

```text
some local coordinates: later maturity
some local coordinates: earlier maturity
```

Those mixed-maturity states were **never training states**.

So the question becomes:

> Does semantic alignment learned across synchronized phase boundaries interpolate to
> asynchronous mixtures of those maturities?

In the scratch reconstruction, surprisingly, yes.

Across three model seeds x three unseen delay fabrics:

```text
phase-supervised shared readout
final accuracy                     ~1.000
async deadline AUC                 ~0.911
sync + phase-exit deadline AUC     ~0.816

accuracy at 10% sync latency
    async                          ~0.683
    sync                           ~0.331

accuracy at 20% sync latency
    async                          ~0.878
    sync                           ~0.331

async full latency / sync latency  ~0.855
```

The synchronized value around `.33` at very early barriers is specific to `perm3`:
the untouched previous state remains correct when the incoming token is the identity
operation, roughly one third of the time.

These scratch numbers motivated the committed experiment; they should be treated as
**exploratory until reproduced by the repo-native script**.

---

## 7. The current hypothesis is no longer "graceful degradation"

The useful phrase for the phenomenon is descriptive, not a novelty claim:

> **phase-consistent legibility**

or more generally:

> **maturity-consistent semantics**

The candidate mechanism is:

```text
local asynchronous availability
            +
semantic alignment across computation stages
            =
useful mixed-maturity readout
```

The first term can come from the physical/local schedule.
The second has to be learned or otherwise constrained.

That is much less magical than the original thought and much more testable.

---

## 8. Why this is relevant to the brain analogy

Do not claim that brains literally deep-supervise checkerboard phases.

The conceptual translation is only this:

> If downstream neural populations can respond while upstream processing continues,
> then useful behavior requires response-relevant variables to remain sufficiently
> stable/readable across those changing upstream states.

A biological system might accomplish that through recurrent attractors, population
codes, convergent pathways, learned downstream invariances, neuromodulatory gating,
or mechanisms not represented here.

The key correction is:

```text
partial activity exists
```

is not equivalent to:

```text
partial activity already means the right thing downstream
```

This is exactly the distinction the raw KYY experiment exposed.

---

## 9. Run the repo-native gate

Quick smoke run:

```bash
python experiments/deadline_scatter_gate.py --quick
```

Fuller exploratory run:

```bash
python experiments/deadline_scatter_gate.py \
    --training-modes final phase \
    --task perm3 \
    --state-dim 32 \
    --train-length 32 \
    --steps 300 \
    --model-seeds 0 1 2 \
    --delay-seeds 0 1 2
```

The script writes a timestamped JSON file to `results/`.

It also asserts:

```text
completed async state
== completed sync state
== canonical KYY state
```

within floating-point tolerance.

---

## 10. What would actually count next

The present gate is still structurally favorable to local asynchronous execution.
Before making any architectural claim, attack it with:

1. **unseen deadline distributions** — train with phase supervision but never with
   deadline input; test a denser and shifted deadline grid;
2. **delay warp OOD** — change the edge-duration distribution after training;
3. **straggler / failed-edge perturbations** — does local progress buy graceful
   robustness or merely different failure modes?;
4. **final-task cost** — does phase supervision damage KYY's long-length
   extrapolation or final state-tracking performance?;
5. **strong anytime baseline** — early-exit / monotonic-anytime model with comparable
   readout/training machinery;
6. **event-driven/asynchronous baseline** — because removing synchronization is prior
   art outside KYY;
7. **dense Householder control** — it has no obvious local edge schedule, which is
   precisely the hardware distinction, but comparisons must count actual operations
   and routing rather than giving locality free credit;
8. **simple-cycle / delay-reservoir control** — local physical-time recurrence itself
   is not unique to KYY.

A particularly strong result would be:

> train for final correctness plus homogeneous phase legibility, then show robust
> accuracy/calibration under **unseen asynchronous delay patterns and interruptions**
> without deadline-specific training.

If ordinary anytime/asynchronous controls match that under the same compute and
training budget, the KYY-specific claim dies.

---

## 11. Current ledger

### Survives

- heterogeneous local execution can expose useful computation before a global barrier;
- completed asynchronous and synchronized KYY transitions can be exactly identical;
- a final-only decoder need not understand intermediate states at all;
- shared phase supervision can, in the scratch experiment, make unseen mixed-maturity
  states surprisingly readable.

### Killed

- "partial results are automatically useful";
- "a delay mesh must degrade gracefully for free";
- "asynchronous neural computation is an unoccupied idea."

### Open

- whether phase-consistent legibility survives the repo-native multi-seed gate;
- whether it survives delay/straggler OOD;
- whether it buys anything over strong anytime and asynchronous controls;
- whether local physical support gives a real hardware/compute advantage rather than
  merely a pleasing interpretation.

The useful correction from this branch is therefore:

> **A physical medium can give you unfinished computation for free. Meaning is not
> free.**
