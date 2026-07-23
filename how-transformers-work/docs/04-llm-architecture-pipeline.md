# The LLM Pipeline: Tokenizer → Transformer Blocks → LM Head

*Source: `transcripts/_07_architecture.txt`*

## The three major components

A transformer LLM generates text token by token. Under the hood it's three components:

1. **Tokenizer** — breaks input text into tokens (see [03-tokenization.md](03-tokenization.md)).
2. **Stack of transformer blocks** — where the vast majority of computation happens; this is where the "magic" (really just well-understood neural network math) takes place. Covered in depth in [05-transformer-block.md](05-transformer-block.md) and [06-self-attention-deep-dive.md](06-self-attention-deep-dive.md).
3. **Language modeling (LM) head** — a neural network that turns the final transformer block's output into a probability distribution over the entire vocabulary.

## Token embeddings

If the tokenizer's vocabulary has, say, 50,000 tokens, the model has 50,000 corresponding embedding vectors — one per token. At the start of processing, each input token ID is substituted with its embedding vector.

## The LM head and decoding strategies

The LM head scores every token in the vocabulary with a probability (all probabilities sum to 100%). E.g. if `"dear"` scores 40% — the highest — it's a candidate for the next output token, but **not necessarily the one chosen**. How you pick from these probabilities is called a **decoding strategy**:

**Logits vs. probabilities.** The LM head doesn't produce probabilities directly — its raw output (`vector @ W_lm`, a plain matrix multiply) is a set of **logits**: one unbounded real number per vocabulary word, which can be negative or greater than 1, and which don't sum to anything meaningful on their own. A **softmax** is applied on top to turn logits into actual probabilities — bounded in `[0,1]`, summing to 1, interpretable as "the model's confidence this is the next token." Softmax is monotonic (it never changes *which* token has the highest score), but it reshapes the *gaps* between them: large logits get pulled further apart, small/negative ones get squeezed together, because of the exponential in its formula. So "the LM head scores every token with a probability" above is really two steps — logits, then softmax — not one; the worked example below shows both.

- **Greedy decoding** — always pick the highest-probability token. This is what happens when `temperature = 0`. Deterministic: same prompt → same output every time.
- **Top-p (nucleus) sampling** — consider multiple high-probability tokens and sample among them, rather than always taking the top one. This is why re-running the same prompt at `temperature > 0` can give different answers each time — it introduces useful variety/naturalness at the cost of determinism.

**Worked example.** Toy 5-word vocabulary, with the LM head producing these logits (raw, pre-softmax scores) for the next token:

| Word | `dear` | `hello` | `hi` | `dog` | `car` |
|---|---|---|---|---|---|
| Logit | 1.6 | 1.2 | 0.9 | 0.1 | -0.6 |

Softmax ($\text{softmax}(x_i) = e^{x_i}/\sum_j e^{x_j}$) turns these into a probability distribution that sums to 1:

| Word | `dear` | `hello` | `hi` | `dog` | `car` |
|---|---|---|---|---|---|
| Probability | **0.3999** | 0.2680 | 0.1986 | 0.0892 | 0.0443 |

- **Greedy decoding** just takes the argmax: `dear` (40%), every time, regardless of the other four probabilities.
- **Top-p sampling with $p=0.75$**: sort by probability descending and keep adding tokens until the cumulative probability reaches $p$. Cumulative sum: `dear` 0.3999 → `+hello` 0.6679 → `+hi` **0.8665** (≥0.75, stop). The **nucleus** is `{dear, hello, hi}` — `dog` and `car` (13.3% of the mass combined) are excluded from sampling entirely, then one token is drawn at random from the nucleus, weighted by their probabilities. So top-p can output `hello` or `hi` on a given run — a run greedy decoding could never produce for this exact distribution — while still never touching `dog`/`car`, which is the "useful variety... at the cost of determinism" trade-off described above, made concrete.

## Parallel processing and context size

Unlike RNNs, transformers process **all input tokens in parallel** — this is what makes them fast to train and run compared to the strictly sequential RNN approach.

Think of it as multiple "tracks" flowing through the stack of transformer blocks simultaneously, one per token position. The number of tracks a model can run at once is its **context size** — e.g. a 16,000-token context size means 16,000 tokens can be processed at the same time.

For a decoder-only LLM, the **generated token is the output of the model's final token position**.

## The generation loop and KV caching

Generating text is a loop:

1. **First pass**: process the entire prompt at once (all "tracks" are new — every arrow in the diagram is "red"/uncached).
2. Take the generated token, append it to the prompt, and feed the **entire** sequence back through the model.
3. Repeat.

On steps after the first, most of the computation for previously-seen tokens is *identical* to what was already computed — so it can be **cached** instead of recomputed. This is **KV caching** (K = keys, V = values; explained in depth in [06-self-attention-deep-dive.md](06-self-attention-deep-dive.md)). It's one of the major optimizations that makes transformer generation fast.

**Worked example: counting the savings.** Say a 5-token prompt ($P=5$) generates 5 more tokens one at a time ($G=5$), so the sequence grows from length 6 up to length 10 across the 5 generation steps. Self-attention needs one query·key dot product for every `(query position, key position)` pair it computes.

- **Without caching**, each generation step naively reruns the *entire* sequence through the model, recomputing the full attention matrix — that's $n^2$ dot products for a sequence of length $n$. Summed over the 5 steps ($n = 6, 7, 8, 9, 10$): $6^2+7^2+8^2+9^2+10^2 = 36+49+64+81+100 = \mathbf{330}$ dot products.
- **With caching**, keys/values for every already-seen token are stored once and reused — each step only computes the *new* token's query against all $n$ cached keys, i.e. $n$ dot products per step: $6+7+8+9+10 = \mathbf{40}$ dot products.

$330 / 40 = 8.25\times$ fewer dot products for this tiny 5-token example — and the gap only widens as the sequence gets longer (without caching it grows quadratically in the number of generation steps; with caching, linearly), which is why KV caching is treated as close to mandatory for any production decoder-only model.

Related efficiency metric: **time to first token (TTFT)** — how long the model takes to process the entire prompt before it produces the very first generated token. Generating each subsequent token is a comparatively cheaper, different process thanks to KV caching.
