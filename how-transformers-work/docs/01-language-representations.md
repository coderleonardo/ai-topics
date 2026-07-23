# From Bag-of-Words to Attention

*Source: `transcripts/_02_bag-of-words.txt`, `_03_word-embeddings.txt`, `_04_encoding-decoding.txt`*

This note covers the pre-Transformer history of representing language numerically: bag-of-words → Word2Vec → RNN encoder-decoder with attention. Each step fixes a limitation of the previous one, and the last one hands off directly to the Transformer.

## 1. Bag-of-words

Language is unstructured — text loses meaning once reduced to raw bytes or characters — so language AI has always focused on representing text in a more structured, numerical way.

**Pipeline:**
1. **Tokenization** — split text into pieces (tokens) by whitespace. `"That is a cute dog"` → `["That", "is", "a", "cute", "dog"]`.
2. **Vocabulary** — the set of *unique* tokens across all documents you're representing.
3. **Bag-of-words vector** — for a given input, count how often each vocabulary word appears (including the zeros for words that don't appear — absence is meaningful too).

Example: vocabulary built from `"That is a cute dog"` and `"My cat is cute"`. Representing `"My cat is cute"` against that vocabulary gives a vector like `0101011` — a **sparse** vector whose values are literal, interpretable word counts.

**Worked example.** Build the vocabulary in order of first appearance across both sentences:

| Index | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Word | That | is | a | cute | dog | My | cat |

Now count each vocabulary word's occurrences in `"My cat is cute"`:

| Word | That | is | a | cute | dog | My | cat |
|---|---|---|---|---|---|---|---|
| Count in sentence | 0 | 1 | 0 | 1 | 0 | 1 | 1 |

Reading the counts left to right gives the vector `[0, 1, 0, 1, 0, 1, 1]` — exactly the `0101011` from above. Every position is a literal count against a fixed vocabulary slot, which is why bag-of-words vectors are easy to interpret but grow as large as the vocabulary (mostly zeros, for any one sentence).

**Limitation:** bag-of-words treats text as a literal bag — it ignores word order beyond presence/absence and captures no semantic meaning. "elegant but flawed."

## 2. Word2Vec (2013): static embeddings

Word2Vec was one of the first successful attempts at capturing *meaning* in embeddings, using neural networks trained on large corpora (e.g. all of Wikipedia).

**How it's trained:**
- Every vocabulary word starts with a vector of random values (e.g. 5 dimensions).
- Training takes pairs of words from the data and asks the network to predict whether they're likely to be neighbors in a sentence.
- Words that tend to share neighbors end up with embeddings that are close together; unrelated words end up far apart.

**What the dimensions mean:** each dimension is loosely a learned "property." E.g. the embedding for *cats* might score low on *newborn*/*human*/*fruit* and high on *animal*/*plural*; *puppy* scores high on *animal*/*newborn* but low on the rest. In practice these properties aren't human-interpretable — they emerge from the training math — but the resulting vectors are still useful for comparing word similarity. Real embeddings commonly have 1,000+ dimensions.

**Worked example (illustrative, not from the transcript): measuring similarity with cosine similarity.** Suppose training produced these toy 3-dimensional embeddings (loosely: [animal-ness, pet-ness, vehicle-ness]):

| Word | Embedding |
|---|---|
| `cat` | `[0.90, 0.80, 0.10]` |
| `dog` | `[0.85, 0.75, 0.15]` |
| `car` | `[0.10, 0.05, 0.90]` |

Word similarity is measured with **cosine similarity**: $\cos(a,b) = \dfrac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$ — the dot product of the two vectors, divided by the product of their lengths (magnitudes). It ranges from −1 (opposite) to 1 (identical direction), and ignores vector *length* so it isolates *direction* (meaning) rather than magnitude.

`cat` vs. `dog`:
- Dot product: $0.90(0.85) + 0.80(0.75) + 0.10(0.15) = 0.765 + 0.600 + 0.015 = 1.380$
- $\lVert cat \rVert = \sqrt{0.90^2+0.80^2+0.10^2} = \sqrt{1.46} \approx 1.2083$
- $\lVert dog \rVert = \sqrt{0.85^2+0.75^2+0.15^2} = \sqrt{1.3075} \approx 1.1435$
- $\cos(cat, dog) = 1.380 / (1.2083 \times 1.1435) \approx 1.380/1.3818 \approx \mathbf{0.9988}$

`cat` vs. `car`:
- Dot product: $0.90(0.10) + 0.80(0.05) + 0.10(0.90) = 0.09+0.04+0.09 = 0.220$
- $\lVert car \rVert = \sqrt{0.10^2+0.05^2+0.90^2} = \sqrt{0.8225} \approx 0.9069$
- $\cos(cat, car) = 0.220 / (1.2083 \times 0.9069) \approx 0.220/1.0958 \approx \mathbf{0.2008}$

`cat`/`dog` score ≈0.999 (nearly identical direction — both score high on "animal"/"pet", low on "vehicle"), while `cat`/`car` scores only ≈0.201 (barely related). This is exactly the mechanism behind "words that tend to share neighbors end up with embeddings that are close together" above — closeness is literally this cosine computation.

**Beyond words:** a *representation model* like Word2Vec converts text into embeddings. Since real tokenizers split rare words into subword pieces (e.g. `"vocalization"` → `["vocal", "ization"]`), you can average a word's subtoken embeddings to get a word embedding, and average further to get sentence or document embeddings.

**Limitation:** embeddings are **static** — the word *bank* gets the exact same vector whether it means a financial bank or a riverbank. There's no context.

## 3. Encoding, decoding, and attention (RNNs)

**RNNs (recurrent neural networks)** model sequences and were used in two roles:
- **Encoder** — represents the entire input sentence as a single embedding (a "context vector").
- **Decoder** — generates the output sentence, one token at a time, from that context vector.

This is **autoregressive**: each newly generated token is appended to the input before generating the next one. E.g. translating "I love llamas" → "Ik hou van lama's":
- Step 1: input `"I love llamas"` → generate `"Ik"`
- Step 2: input `"I love llamas ik"` → generate `"hou"`
- ...continue until the full output is generated.

**Limitation:** the single context embedding is a bottleneck — it has to represent an entire (possibly long, complex) sentence in one fixed-size vector, which degrades on longer sequences.

**Attention (2014)** fixed this. Instead of passing only the final context embedding to the decoder, the encoder passes the **hidden state of every input word**. The decoder then learns to *attend* to the specific input words most relevant to whatever it's currently generating — e.g. English "I" and Dutch "Ik" get high attention weight (synonyms), while "I" and "llamas" get low attention weight (unrelated).

Attention does two things:
1. Determines which words matter most for the current generation step.
2. Lets the decoder pull information from across the *whole* input sequence rather than a single squashed vector.

**Worked example (illustrative, not from the transcript).** Toy 2-dimensional encoder hidden states for `"I love llamas"`:

| Position | Word | Hidden state |
|---|---|---|
| 1 | I | `[1.0, 0.0]` |
| 2 | love | `[0.0, 1.0]` |
| 3 | llamas | `[-1.0, 0.5]` |

Say the decoder's current query vector — its internal state while about to generate `"Ik"` (Dutch for "I") — is `q = [0.9, 0.1]`, i.e. it "looks like" the hidden state for `"I"`. Attention scores are the dot product of the query against each hidden state:

- $q \cdot h_{I} = 0.9(1.0) + 0.1(0.0) = 0.90$
- $q \cdot h_{love} = 0.9(0.0) + 0.1(1.0) = 0.10$
- $q \cdot h_{llamas} = 0.9(-1.0) + 0.1(0.5) = -0.85$

These raw scores are turned into weights that sum to 1 via **softmax**: $\text{softmax}(x_i) = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$.

$e^{0.90}\approx2.4596,\ e^{0.10}\approx1.1052,\ e^{-0.85}\approx0.4274$, sum $\approx3.9922$, giving weights:

| Word | Score | Weight |
|---|---|---|
| I | 0.90 | **0.6161** |
| love | 0.10 | 0.2768 |
| llamas | -0.85 | 0.1071 |

The context vector fed to the decoder is the weights times the hidden states, summed:

$$0.6161[1.0,0.0] + 0.2768[0.0,1.0] + 0.1071[-1.0,0.5] = [0.5090,\ 0.3304]$$

The result sits mostly in the direction of $h_I$ (weight 0.616, by far the largest) with a small pull from `love` and an even smaller, partially-canceling pull from `llamas` — numerically reproducing "English 'I' and Dutch 'Ik' get high attention weight... while 'I' and 'llamas' get low attention weight" from above.

**Remaining limitation:** RNNs are inherently sequential — each step depends on the previous one — so **training cannot be parallelized**. This is the problem the Transformer (next note) solves by dropping recurrence entirely and using attention alone.
