# Mixture of Experts (MoE)

*Source: `transcripts/_12_MoE.txt`*

Mixture of Experts extends transformers by introducing **dynamically chosen experts**, replacing part of the decoder block. It has two main components: **experts** and a **router**.

## Recap: the decoder block MoE modifies

A standard decoder block:
1. Input token vectors are **layer normalized**.
2. **Masked self-attention** weighs tokens by relative importance (see [06-self-attention-deep-dive.md](06-self-attention-deep-dive.md)).
3. Attention output is added back to the unprocessed input (residual/direct path + processed/indirect path).
4. **Layer normalized** again.
5. Passed through a **feedforward neural network (FFN)** — typically one of the largest components of an LLM, since it's where complex relationships in the attended information get worked out.

Normally this FFN is a single **dense network**: every parameter is activated (used, at least partially) for every input.

## What MoE changes

MoE replaces that single dense FFN with **several networks** to choose from, each called an **expert**. Note: an expert is *not* a domain specialist (not "the biology expert" or "the psychology expert") — at most, experts specialize on syntactic/token-level patterns like punctuation, verbs, or conjunctions.

- The FFN becomes an **MoE layer** made of, say, four experts.
- For each input, one or more experts are **selected** to process it; the rest sit unused for that input.
- Because only a subset of experts activate per input, this is called a **sparse model**.

## The router

The router decides which inputs go to which experts:
- It's itself a small feedforward neural network (much smaller than an expert — its only job is routing).
- For each expert, it produces a **probability score** indicating how well-suited that expert is for the current input.
- One strategy: route to the single highest-probability expert. Other strategies exist that trade determinism for more creative/varied outputs.
- If **multiple** experts are selected, their outputs are combined via a **weighted mean** — experts with higher router probability get more say in the final output.

**Worked example (illustrative, not from the transcript).** 4 experts, each producing a toy 2-dimensional output vector for the current input:

| Expert | Output |
|---|---|
| E1 | `[1.0, 0.2]` |
| E2 | `[0.3, 0.9]` |
| E3 | `[-0.5, 0.4]` |
| E4 | `[0.6, -0.3]` |

The router produces logits `[2.0, 0.5, -1.0, 0.2]` for E1–E4. Softmax turns these into routing probabilities: `[0.6953, 0.1551, 0.0346, 0.1149]` — E1 is the clear favorite at ~70%.

- **Top-1 routing**: only E1 fires. Its output is scaled by its own gate probability: $0.6953 \times [1.0, 0.2] = [\mathbf{0.6953,\ 0.1391}]$.
- **Top-2 routing**: E1 and E2 fire. Their probabilities are renormalized to sum to 1 first — $0.6953/(0.6953{+}0.1551) = 0.8177$ for E1, $0.1823$ for E2 — then combined as a weighted mean:

$$0.8177[1.0,0.2] + 0.1823[0.3,0.9] = [\mathbf{0.8723,\ 0.3277}]$$

Notice the top-2 result is measurably different from top-1 — E2's `[0.3, 0.9]` pulls the second dimension up from 0.14 to 0.33 — which is exactly why "how many experts to route to" (top-1 vs. top-2 vs. more) is itself a real design choice, not just an efficiency knob: it changes what the MoE layer actually outputs, not just how much compute it costs.

Together, router + experts make up the **MoE layer**, replacing the plain FFN inside the decoder block. (Since MoE only touches the FFN, not attention, non-transformer architectures like state-space models — Mamba, Jamba — can also have MoE variants.)

## Why MoE is efficient: sparse vs. active parameters

A model's parameters live in roughly five places: input embeddings, masked self-attention, the router, the experts, and output embeddings.

- **Sparse parameters** — *all* parameters that must be **loaded** into memory, including every expert, even though most won't be used for any given input.
- **Active parameters** — the subset actually **used during inference** for a given input (attention + router + only the selected expert(s), not all of them).

Net effect: MoE models need a lot of memory to *load* (all experts present), but comparatively little compute to *run* (only a few experts active at a time) — a favorable trade for production inference.

## Worked example: Mixtral 8x7B

Mixtral is a MoE model with 8 experts, each nominally "7B parameters."

- **Shared parameters** (always used, whether loading or running inference) live mostly in the attention mechanism — **1B+ parameters**.
- The **router** is tiny: only ~32,000 parameters.
- Each expert actually has **5.6B** parameters, not 7B — the "7B" figure appears to fold in the shared parameters, which is a bit misleading since those aren't expert-specific.
- Together, the 8 experts total **~45B parameters** — the majority of the model.
- **Total parameters to load: ~46B.**
- Mixtral only activates **2 experts** at inference time, so the active parameter count (and VRAM needed to *run* it) is much smaller than the 46B you need to *load* it.

**The arithmetic behind those totals:**

$$\underbrace{8 \times 5.6\text{B}}_{\text{8 experts}} = 44.8\text{B} \qquad 44.8\text{B} + \underbrace{1\text{B}}_{\text{shared/attention}} + \underbrace{0.032\text{B}}_{\text{router}} \approx \mathbf{45.83\text{B}} \approx \mathbf{46\text{B}}\ \text{total (to load)}$$

At inference, only 2 of the 8 experts activate per token, so the *active* parameter count is roughly $1\text{B (shared)} + 2 \times 5.6\text{B (experts)} + 0.032\text{B (router)} \approx \mathbf{12.23\text{B}}$ — close to the "~13B active" figure quoted elsewhere for Mixtral, and a fraction of the ~46B that had to be loaded into memory in the first place. That gap (46B to load vs. ~12B active) is the entire efficiency argument for MoE in one comparison.

## Pros and cons

| Pros | Cons |
|---|---|
| Higher performance than a same-size dense model (experts reduce redundant computation) | Requires a lot of memory to load (all experts) |
| Much lower VRAM/compute needed at inference vs. loading | Risk of overfitting on a single expert — needs careful load balancing |
| Flexible — which experts get used can vary per input | More complex architecture, requires more careful training |
| Not limited to transformers — applicable to other architectures too (Mamba, Jamba) | |
