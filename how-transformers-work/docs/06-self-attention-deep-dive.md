# Self-Attention Mechanics: Q/K/V, Multi-Head, and Efficient Variants

*Source: `transcripts/_09_self-attention.txt`*

Self-attention breaks down into two high-level steps: **relevance scoring**, then **combining information**. This note covers how each is actually calculated, inside one **attention head**, and how attention has evolved for efficiency.

## Queries, keys, and values

Self-attention uses three learned weight matrices: the **query projection matrix**, the **key projection matrix**, and the **value projection matrix**. Each row of the resulting Q/K/V matrices is the query/key/value vector for one token position in the sequence.

- **Query** — represents the current position (the token you're computing an updated representation for).
- **Key** — represents each other position, for the purpose of being *matched against* by a query.
- **Value** — represents each other position, for the purpose of being *combined* once relevance is known.

## Step 1: relevance scoring

Goal: assign every token a score for how relevant it is to the token currently being represented, such that all scores add up to 100% (e.g. in `"the dog ..."`, `"dog"` might score highest).

Mechanically: multiply the **query vector of the current token** by the **key vectors of all tokens** (matrix multiplication) → raw relevance/attention scores.

> For the full mathematical derivation and implementation details, the transcript points to DeepLearning.AI's short course dedicated entirely to attention (by Josh Starmer / StatQuest) as a deeper resource.

## Step 2: combining information

Once you have relevance scores, multiply each token's score by that token's **value vector**, producing weighted value vectors (the most relevant token's values dominate; irrelevant tokens shrink toward zero). Summing the weighted values gives the output of this step — the enriched representation for the current token.

## Worked example: Q/K/V end to end (illustrative, not from the transcript)

A minimal, self-contained example with real projection matrices — small enough to trace by hand, `d_model = 2`, 2 tokens. (A larger, multi-token, multi-head, causally-masked version of this exact same math is worked out in full in [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md).)

Token embeddings: `x1 ("the") = [1.0, 0.0]`, `x2 ("cat") = [0.0, 1.0]`. Toy projection matrices:

$$W_Q = \begin{bmatrix}0.5 & -0.5 \\ 0.2 & 0.8\end{bmatrix} \qquad W_K = \begin{bmatrix}0.9 & 0.1 \\ 0.3 & 0.7\end{bmatrix} \qquad W_V = \begin{bmatrix}1.0 & 0.0 \\ 0.0 & 1.0\end{bmatrix}\ (\text{identity, for clean arithmetic})$$

Each token's query/key/value is its embedding row-multiplied by the corresponding matrix ($q = x \cdot W_Q$, etc.):

$$q_1 = x_1 W_Q = [0.5,\ -0.5] \qquad q_2 = x_2 W_Q = [0.2,\ 0.8]$$
$$k_1 = x_1 W_K = [0.9,\ 0.1] \qquad k_2 = x_2 W_K = [0.3,\ 0.7]$$
$$v_1 = x_1 W_V = [1.0,\ 0.0] \qquad v_2 = x_2 W_V = [0.0,\ 1.0]$$

**Relevance scoring** for token 2 (`"cat"`) — its query dotted with every key:

$$q_2 \cdot k_1 = 0.2(0.9) + 0.8(0.1) = 0.26 \qquad q_2 \cdot k_2 = 0.2(0.3) + 0.8(0.7) = 0.62$$

Softmax($[0.26, 0.62]$) → weights $[0.4110, 0.5890]$.

**Combining information** — weight each value vector and sum:

$$0.4110\,v_1 + 0.5890\,v_2 = 0.4110[1.0,0.0] + 0.5890[0.0,1.0] = [0.4110,\ 0.5890]$$

That's `"cat"`'s new, contextualized representation: it pulls 41% from `"the"`'s value and 59% from its own — exactly the "weighted value vectors... summing... gives the enriched representation" described above, with the actual matrix multiplications shown.

**Multi-head split/concat, numerically.** Say a self-attention layer's raw output (before splitting into heads) for one token is `z = [0.4, 0.7, 0.2, 0.9]` (`d_model = 4`, 2 heads → `d_head = 2`). **Split**: head 1 gets `[0.4, 0.7]`, head 2 gets `[0.2, 0.9]` — each head runs its *own* full Q/K/V/relevance/combine pipeline (as above) independently on its slice. Suppose that per-head processing produces `head1_out = [0.5, 0.6]` and `head2_out = [0.1, 0.3]`. **Combine**: concatenate back to `d_model` width: `[0.5, 0.6, 0.1, 0.3]`. That concatenated vector is then passed through one more learned projection ($W_O$) to mix information across heads before leaving the self-attention layer — see §4h of [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md) for this step with real numbers.

## Multi-head attention

The Q/K/V calculation above happens inside a single **attention head**. In practice, self-attention runs this **same operation in parallel across multiple attention heads**, each with its *own independent* set of Q/K/V projection matrices — so different heads can end up attending to different things.

Two extra steps wrap the per-head computation:
1. **Split** — the input is split so each head gets its own slice to work on.
2. **Combine** — each head's output is concatenated back together to form the self-attention layer's final output.

## Efficient attention variants

Self-attention is typically the most computationally expensive part of a transformer, so several variants trade off quality for speed/memory:

| Variant | Idea | Trade-off |
|---|---|---|
| **Multi-query attention (MQA)** | All attention heads share a *single* keys matrix and a *single* values matrix (only queries stay per-head) | Big parameter/compute reduction ("compression"); faster, but can lose some quality |
| **Grouped-query attention (GQA)** | A middle ground: multiple heads share keys/values, but in *groups* (a smaller number of K/V sets than the number of query heads, but more than just one) | Better results than MQA, especially for larger models; this is why model cards report both "number of attention heads" *and* "number of key/value groups" |
| **Sparse attention** | Not every layer computes full attention across the entire history. Interleaved layers only attend to a limited recent window (e.g. last 4/16/32 tokens) instead of the full sequence | Much cheaper for long sequences; used selectively (e.g. every other layer) rather than everywhere |

**Sparse attention patterns** (from *"Generating Long Sequences with Sparse Transformers"*):
- **Full attention** — every token attends to every previous token (the baseline described above).
- **Strided** — each position attends to the last few tokens *plus* certain fixed-stride positions further back.
- **Fixed** — fixed reference positions in the sequence; e.g. after token 4, you're only allowed to look back to position 4 onward.

For even longer contexts (100K–1M+ tokens), techniques like **ring attention** are used (see the "Ring Attention Explained" blog post referenced in the course for a visual walkthrough).

**Worked example: quantifying MQA/GQA's parameter savings.** Using Llama 3.1 8B's own dimensions (32 attention heads, model dimension 4096 — see the table below), each head's dimension is $d_{head} = 4096/32 = 128$. In standard multi-head attention, the key and value projection matrices are each `d_model × d_model` (they produce one 128-dim key/value slice per head, for all 32 heads at once):

| Variant | K + V projection size | K + V params/layer | Reduction vs. MHA |
|---|---|---|---|
| **MHA** (32 separate K/V heads) | $2 \times (4096\times4096)$ | $33{,}554{,}432 \approx 33.55\text{M}$ | 1× (baseline) |
| **MQA** (1 shared K/V head) | $2 \times (4096\times128)$ | $1{,}048{,}576 \approx 1.05\text{M}$ | **32×** fewer |
| **GQA** (8 shared K/V groups) | $2 \times (4096\times1024)$ | $8{,}388{,}608 \approx 8.39\text{M}$ | **4×** fewer |

The reduction factor is exactly "number of query heads ÷ number of K/V heads/groups" — 32÷1=32 for MQA, 32÷8=4 for GQA — which is why the table below reports "GQA, 8 KV heads" rather than a vague "reduced," and why GQA is described as "a middle ground": it keeps 8 of the 32 possible K/V variants instead of collapsing all the way down to 1.

## KV caching, revisited

Because keys and values for already-processed tokens don't change across generation steps, they can be computed once and reused — this is the **KV cache** introduced in [04-llm-architecture-pipeline.md](04-llm-architecture-pipeline.md). It's what makes token-by-token generation fast after the first pass.

## Reading a real architecture table: Llama 3.1 8B

Once you know this vocabulary, published model architecture tables become readable. For Llama 3.1 8B:

| Parameter | Value | What it means |
|---|---|---|
| Layers | 32 | Number of stacked transformer blocks |
| Model dimension | 4096 | Length of the vector flowing through the model per token |
| FFN dimension | 14,336 | Hidden-layer width inside each block's feedforward network |
| Attention heads | 32 | Number of parallel self-attention heads |
| Attention type | GQA, 8 KV heads | Grouped-query attention with 8 shared key/value groups (not 32 — MQA/GQA in action) |
| Vocabulary | 128,000 | Tokenizer vocabulary size (see [03-tokenization.md](03-tokenization.md)) |
| Positional embeddings | RoPE (rotary embeddings) | See note below |

**RoPE (Rotary Position Embeddings)** is the positional-embedding method Llama uses instead of the sinusoidal absolute positional encoding taught in the introductory lessons (used in [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md) for simplicity). RoPE encodes position by *rotating* the query/key vectors as a function of position, rather than adding a separate positional vector to the embedding.
