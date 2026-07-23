# Transformers: Encoders, Decoders, and the Model Families

*Source: `transcripts/_05_transformers.txt`*

## "Attention Is All You Need"

The Transformer architecture (2017) is based **solely on attention**, with no recurrent neural network. Because there's no step-by-step recurrence, the model can be trained **in parallel**, which is a major speed-up over RNNs.

The original Transformer is an **encoder-decoder** architecture, built from stacked encoder and decoder blocks. Stacking blocks amplifies what a single encoder/decoder can represent.

## Inside an encoder block

Given input `"I love llamas"`:
1. Tokens are converted to embeddings — but unlike Word2Vec, these start as **random values** and are learned during training.
2. **Self-attention**: attention applied to a single sequence by comparing it *to itself* (as opposed to comparing two separate sequences, like the earlier RNN attention did between encoder and decoder). This updates each token's embedding with contextual information from the rest of the input.
3. The updated embeddings are passed through a **feedforward neural network** to produce the final contextualized token embeddings.

The encoder's job is representation — turning input tokens into rich, contextualized embeddings.

## Inside a decoder block

1. Previously generated tokens are passed through **masked self-attention** — like self-attention, but every token can only attend to tokens *before* it (all values above the diagonal are removed/masked). This prevents the model from "seeing the future" during generation.
2. The result is passed to another attention layer, this time combined with the **encoder's output embeddings** — so the decoder attends to both what it has generated so far *and* the full input representation.
3. This is passed through a feedforward network, which produces the next token.

**Worked example: masking a score matrix.** Take a 4-token sequence and a toy 4×4 matrix of raw attention scores (row = the token doing the attending, column = the token being attended to):

$$
S = \begin{bmatrix}
1.0 & 0.5 & -0.5 & 2.0 \\
0.2 & 1.2 & 0.1 & -0.3 \\
-0.4 & 0.6 & 1.5 & 0.2 \\
0.3 & -0.2 & 0.4 & 1.0
\end{bmatrix}
$$

Full (unmasked) self-attention runs softmax across every row as-is — every token sees every other token, including ones that come *after* it:

| Row (token) | softmax(row) |
|---|---|
| 1 | `[0.2199, 0.1334, 0.0491, 0.5977]` |
| 2 | `[0.1912, 0.5198, 0.1730, 0.1160]` |
| 3 | `[0.0818, 0.2223, 0.5468, 0.1490]` |
| 4 | `[0.2116, 0.1284, 0.2339, 0.4262]` |

Notice row 1 (token 1) still puts 60% of its weight on column 4 — a token three positions in its future. That's fine for an *encoder* (it's allowed to see the whole sequence), but wrong for a *decoder* generating token 1: it hasn't produced tokens 2–4 yet, so it can't be allowed to "look" at them.

**Masked** self-attention fixes this by setting every score above the diagonal (column index > row index) to $-\infty$ *before* the softmax — since $e^{-\infty}=0$, those positions get exactly zero weight:

$$
S_{masked} = \begin{bmatrix}
1.0 & -\infty & -\infty & -\infty \\
0.2 & 1.2 & -\infty & -\infty \\
-0.4 & 0.6 & 1.5 & -\infty \\
0.3 & -0.2 & 0.4 & 1.0
\end{bmatrix}
$$

| Row (token) | softmax(masked row) |
|---|---|
| 1 | `[1.0000, 0, 0, 0]` — can only attend to itself |
| 2 | `[0.2689, 0.7311, 0, 0]` — only tokens 1–2 |
| 3 | `[0.0961, 0.2613, 0.6426, 0]` — only tokens 1–3 |
| 4 | `[0.2116, 0.1284, 0.2339, 0.4262]` — unchanged (it's the last token, so it's already allowed to see everything before it) |

Row 4 is identical in both tables — the causal mask never removes anything a token is *already* entitled to see, it only ever removes future positions. This is exactly the "upper triangle → masked" rule described above, with actual numbers behind it. A full end-to-end version of this step (with real Q/K/V projections, not raw toy scores) is worked out in section 4c–4e of [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md).

## Two model families that came from this

The full encoder-decoder Transformer is well suited to translation-style tasks but isn't a natural fit for everything (e.g. plain text classification). Two specialized families emerged:

### BERT (2018) — encoder-only

**Bidirectional Encoder Representations from Transformers.** Stack of encoder blocks only (self-attention + feedforward), focused on producing rich contextual *representations* rather than generating text.

- Adds a special **`[CLS]`** (classification) token to the input, whose output embedding is used to represent the *entire* input — commonly used as the input to a downstream classifier.
- Trained with **masked language modeling (MLM)**: randomly mask words in the input and have the model predict them. This forces the model to learn to represent language deeply.
- Training is two-phase: **pre-training** (MLM on large unlabeled data) then **fine-tuning** (on a specific downstream task, e.g. classification).

### GPT — decoder-only, generative

Stack of decoder blocks only. **GPT-1** ("Generative Pre-Trained transformer") uses masked self-attention → feedforward network, with **no encoder** at all, and generates the next token directly.

- GPT-1: ~100M parameters
- GPT-2: 1B+ parameters
- GPT-3: 175B parameters

Bigger models → bigger capabilities, which is why the field kept scaling up.

## Context length

The **context length** is the number of tokens a model can process at once — the original query plus every token generated so far. E.g. a model with a max context length of 512 can only process 512 tokens total (inputs + generated tokens) at any given time.

**Worked example.** `microsoft/Phi-3-mini-4k-instruct` (loaded and inspected in [08-model-walkthrough-phi3.md](08-model-walkthrough-phi3.md)) has a max context length of 4,096 tokens — that's the "4k" in its name. Suppose a conversation with this model has used:

| Consumed so far | Tokens |
|---|---|
| System prompt | 180 |
| User messages (so far) | 432 |
| Model replies generated (so far) | 150 |
| **Total consumed** | **762** |

Remaining budget: $4096 - 762 = \mathbf{3334}$ tokens. Every further user message, model reply, or newly generated token draws down that same 3,334-token pool — once it's exhausted, the oldest tokens have to be dropped (or the conversation summarized/truncated) for the model to keep generating, because 4,096 is a hard ceiling, not a soft guideline.

## The "year of generative AI"

ChatGPT (GPT-3.5) kicked off a wave of proprietary models, quickly followed by open-source models — models whose weights are publicly available, some usable commercially.

## Summary of model flavors

| Family | Blocks used | Good at |
|---|---|---|
| Encoder-decoder (original Transformer) | Encoder + decoder | Translation-style seq2seq tasks |
| Encoder-only (BERT) | Encoder only | Representing/understanding language (classification, embeddings) |
| Decoder-only (GPT) | Decoder only | Generating text |
