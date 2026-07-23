# Inside a Transformer Block

*Source: `transcripts/_08_tranformer-block.txt`*

## The flow through the model

Take the prompt `"the Shawshank"`. The tokenizer breaks it into two tokens — `"the"` and `"Shawshank"` — each replaced by its embedding vector. From here on, everything is numeric.

Picture each token as its own "track" flowing through the stack of transformer blocks:

- The two token vectors flow into transformer block 1, which outputs a vector of the *same size* for each track (something happens to the values in between — that's the two sub-layers, below).
- Block 1's output flows into block 2, which processes it the same way, in parallel across tracks.
- This repeats down the entire stack of blocks.
- The **final block's output vector for the last token position** is handed to the language modeling head, which predicts the next token.

Flow is strictly one-directional: tokenizer → block 1 → block 2 → ... → LM head.

## The two sub-layers of a block

Each transformer block is made of two major components:

### 1. Feed-forward neural network (FFN)

A standard dense network: a layer that expands to a wider hidden layer, then shrinks back down. This is where most of a model's *learned facts/statistics* are thought to live.

**Intuition example:** if a block had *only* the FFN (no attention), it could still often complete `"the Shawshank"` → `"redemption"` — because in training data (internet, Wikipedia, etc.) "redemption" is statistically the word that follows "Shawshank" most often. The FFN acts like a storage of next-word statistics learned from training data. (Real behavior is more nuanced than this, but it's a reasonable first mental model.)

**Worked example (illustrative, not from the transcript).** A single hidden unit inside an FFN is just `ReLU(w1·x + b1)`, then a second weight turns that hidden value into a contribution to an output logit. Say `x = 0.8` is a toy scalar summarizing "how strongly does this input look like the phrase 'the Shawshank'," with a hidden unit tuned (during training, from seeing "Shawshank"/"Redemption" co-occur often) to `w1 = 1.5`, `b1 = -0.5`:

$$\text{hidden} = \text{ReLU}(1.5 \times 0.8 - 0.5) = \text{ReLU}(0.7) = 0.7$$

That hidden value then feeds two different output weights — one pushing toward the `"redemption"` logit, one toward an unrelated word like `"banana"`:

$$\text{logit}_{redemption} = 2.0 \times 0.7 + 0.1 = \mathbf{1.5} \qquad \text{logit}_{banana} = 0.05 \times 0.7 - 0.2 = \mathbf{-0.165}$$

`"redemption"`'s logit (1.5) is far higher than `"banana"`'s (-0.165) — a toy stand-in for "the FFN acts like a storage of next-word statistics." A real FFN has thousands of hidden units doing this in parallel across `d_ff` dimensions (8 of them in the fully worked example in [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md), §7), and the specific weights come from training, not hand-picking — but the mechanism (weighted sum → ReLU → weighted sum) is exactly this, repeated at scale.

### 2. Self-attention

Where the FFN is roughly static/statistical, self-attention lets the model **incorporate context from other tokens** into the token it's currently processing.

**Intuition example:** `"the dog chased the llama because it"` — to process the token `"it"`, the model needs to figure out what "it" refers to (the dog or the llama). This is the NLP task of **coreference resolution**. Self-attention is how the model bakes information from the correct referent (say, "llama", if prior context points that way) into the representation of "it".

**High-level formulation:**
- You have the vector representing the *current* position (the embedding, if this is block 1; the previous block's output, otherwise).
- You have vectors for *other* positions in the sequence.
- The goal: produce an updated vector for the current position that pulls in the relevant information from those other positions.

Self-attention does this in two steps:
1. **Relevance scoring** — assign a score to how relevant every other input token is to the token currently being processed.
2. **Combining information** — use those relevance scores to combine information from the relevant tokens into the current token's representation.

**Worked example (illustrative, not from the transcript; simplified — skips the Q/K/V projection step covered in full in [06-self-attention-deep-dive.md](06-self-attention-deep-dive.md)).** Take `"the dog barks"` with toy 2-dimensional token vectors, and treat each token's own vector directly as its query/key/value (a shortcut for intuition only):

| Token | Vector |
|---|---|
| the | `[0.1, 0.2]` |
| dog | `[0.9, 0.1]` |
| barks | `[0.3, 0.8]` |

To compute the updated representation for `"barks"`, score it against every token (including itself) via dot product:

$$\text{barks}\cdot\text{the} = 0.3(0.1)+0.8(0.2) = 0.19 \quad \text{barks}\cdot\text{dog} = 0.3(0.9)+0.8(0.1) = 0.35 \quad \text{barks}\cdot\text{barks} = 0.3(0.3)+0.8(0.8) = 0.73$$

Softmax those three scores → weights `[0.2571, 0.3017, 0.4412]` (the — dog — barks). Combine: weight each token's vector by its score and sum:

$$0.2571[0.1,0.2] + 0.3017[0.9,0.1] + 0.4412[0.3,0.8] = [0.4296,\ 0.4345]$$

`"barks"` ends up attending most to itself (0.44) and next-most to `"dog"` (0.30, its subject) — a small numeric echo of the "it → dog/llama" coreference idea: the token's new representation is a *blend*, dominated by whichever tokens are most relevant, not a hard pick of just one. The full version of this — with real learned projection matrices, causal masking, and multiple heads — is worked out end-to-end in [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md), §4.

**Worked example: the residual connection.** After self-attention produces its output, it's added back to the block's *original* input, not used on its own:

$$x = [0.6, 0.8, 0.2, 1.2] \qquad \text{attn\_out} = [0.1, -0.05, 0.3, -0.2]$$
$$x + \text{attn\_out} = [0.7,\ 0.75,\ 0.5,\ 1.0]$$

This elementwise add is the "residual/direct path" — it guarantees the original signal always makes it through the block even if the attention sub-layer's output were all zeros, which is what makes very deep stacks of blocks trainable. The full-precision version of this step (with the actual attention output from the running numeric example) is in [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md), §5.

The mechanics of how relevance scoring and combining are actually computed (queries/keys/values, multi-head attention, and efficient variants) are covered in [06-self-attention-deep-dive.md](06-self-attention-deep-dive.md).
