# Tokens and Tokenizers

*Source: `transcripts/_06_tokenizers.txt` + `tokenizers.ipynb`*

## Why tokenization

For a language model to process text, it first breaks it into smaller pieces called **tokens** (this splitting process is **tokenization**). Each token is then turned into a numerical embedding.

- These initial embeddings are **static** — each is created independently of the others.
- The model processes them and turns them into **contextualized embeddings** — still one per input token, but now shaped by every other token in the sequence.
- For generative models, the output is itself another token, which gets decoded back into text.

Each token has a fixed integer **ID**, used to look up/produce embeddings and to encode/decode between text and the model's internal representation.

## Tokenization levels

For the input `"Have the bards who precede..."`, the same text can be tokenized at different granularities:

| Level | Example | Notes |
|---|---|---|
| Word | `["Have", "the", "bards", "who", ...]` | One token per whole word |
| Subword | `["Have", "the", "b", "ards", "who", ...]` | Whole words *or* pieces; pieces recombine into the original word |
| Character | `["H","a","v","e", ...]` | One token per character |
| Byte | raw bytes | Smallest possible unit; some characters (e.g. emoji) need multiple bytes |

**Why subword tokenization dominates in practice:** a tokenizer has a fixed, finite vocabulary. It can't have an entry for every possible word, so unknown/rare words (e.g. *"bards"*) are represented as combinations of known subword pieces (*"b"* + *"ards"*). This keeps the vocabulary flexible without exploding its size.

### Worked example: how a subword vocabulary gets built (illustrative BPE walkthrough)

The transcript doesn't cover the *algorithm* that produces a subword vocabulary, only its effect. The dominant one is **Byte-Pair Encoding (BPE)**: start from individual characters, and repeatedly merge the *most frequent adjacent pair* into a new single symbol, until the vocabulary reaches a target size. Standard textbook math, not transcript content — but it grounds "pieces recombine into the original word" in an actual, reproducible process.

Take a tiny 3-word corpus with per-word frequencies, each split into characters plus an end-of-word marker `_`:

| Word | Frequency | Character sequence |
|---|---|---|
| low | 5 | `l o w _` |
| lower | 2 | `l o w e r _` |
| lowest | 2 | `l o w e s t _` |

**Step 1 — count every adjacent symbol pair, weighted by word frequency:**

| Pair | Count |
|---|---|
| (l, o) | 5+2+2 = **9** |
| (o, w) | 5+2+2 = **9** |
| (w, e) | 2+2 = 4 |
| (w, _) | 5 |
| (e, r) | 2 |
| (r, _) | 2 |
| (e, s) | 2 |
| (s, t) | 2 |
| (t, _) | 2 |

`(l, o)` and `(o, w)` are tied at 9 — pick `(l, o)` (tie-break arbitrary) and merge it into a new symbol `lo`. Corpus becomes: `lo w _` (×5), `lo w e r _` (×2), `lo w e s t _` (×2).

**Step 2 — recount pairs on the updated corpus:** `(lo, w)` is now the clear winner at 5+2+2=**9** (the old `(o,w)` count moved onto it). Merge `lo`+`w` → `low`. Corpus becomes: `low _` (×5), `low e r _` (×2), `low e s t _` (×2).

**Step 3 — recount again:** `(low, _)` = 5 is now the most frequent pair. Merge → `low_`.

After 3 merges, the vocabulary has grown from 8 characters (`l,o,w,e,r,s,t,_`) to include the subword pieces `lo`, `low`, `low_` — and critically, `lower` and `lowest` still share the `low` piece even though they diverge afterward, exactly like the transcript's `"bards"` → `"b"` + `"ards"` example. In a real tokenizer this repeats for thousands of merge steps until the vocabulary hits its target size (28,996 for `bert-base-cased`, 100,263 for GPT-4's tokenizer, etc. — see the table below).

## Notebook walkthrough (`tokenizers.ipynb`)

### Basic tokenization

```python
from transformers import AutoTokenizer

sentence = "Hello world!"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

token_ids = tokenizer(sentence).input_ids
print(token_ids)
# [101, 8667, 1362, 106, 102]

for id_ in token_ids:
    print(tokenizer.decode(id_))
# [CLS]
# Hello
# world
# !
# [SEP]
```

`bert-base-cased` wraps the sentence with two special tokens: **`[CLS]`** (classification token, represents the whole input — used for fine-tuning on tasks like classification) and **`[SEP]`** (separator, marks the end of a segment/sentence).

### Visualizing tokenization across tokenizers

```python
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    print(f"Vocab length: {len(tokenizer)}")
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) + '\x1b[0m', end=' '
        )
```

Run against a stress-test string containing capitalization, an emoji, a Chinese character, code-like syntax, tabs, and arithmetic:

```python
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""
```

### Observed results (actual notebook output)

| Tokenizer | Vocab length | Notable behavior |
|---|---|---|
| `bert-base-cased` | 28,996 | Breaks `CAPITALIZATION` into many pieces (`CA`, `##PI`, `##TA`, `##L`, `##I`, `##Z`, `##AT`, `##ION`); emoji/Chinese become `[UNK]` (unknown token); `##` prefix marks "continues the previous token" |
| `bert-base-uncased` | 30,522 | Lowercases everything first (`english`, `capital`, `##ization`); still `[UNK]` for emoji/Chinese |
| `Xenova/gpt-4` (tiktoken) | 100,263 | 3x+ BERT's vocab; `CAPITALIZATION` → only 2 tokens (`CAPITAL`, `IZATION`); no `[CLS]`/`[SEP]` (it's built for generation, not classification); represents tabs and whitespace faithfully |
| `gpt2` | 50,257 | Splits `CAPITALIZATION` further (`CAP`, `ITAL`, `IZ`, `ATION`); splits `tokens` into `t`, `ok`, `ens` |
| `google/flan-t5-small` | 32,100 | Splits `CAPITALIZATION` into individual-ish pieces; every space becomes `<unk>` |
| `bigcode/starcoder2-15b` | 49,152 | Code-oriented; represents each digit of `12.0*50=600` as a *separate* token (`1`,`2`,`.`,`0`, ...) — useful for arithmetic/code tasks |
| `microsoft/Phi-3-mini-4k-instruct` | 32,011 | Splits `CAPITALIZATION` character-by-character-ish (`C`,`AP`,`IT`,`AL`,`IZ`,`ATION`) |
| `Qwen/Qwen2-VL-7B-Instruct` | 151,657 | Largest vocab shown; represents the emoji `🎵` directly instead of falling back to `[UNK]`/replacement bytes |

### The vocabulary-size trade-off

A larger vocabulary means:
- ✅ Fewer tokens needed to represent the same text (more of each word is a single token)
- ✅ Better coverage of uncommon words/characters (less need for `[UNK]`)
- ❌ More embeddings the model has to learn and store — one per vocabulary entry

So tokenizer choice is a real design trade-off, not just an implementation detail — and it also differs a lot by target language (Western vs. Eastern scripts, code vs. natural language).

**Worked example: quantifying the compression.** The table above shows `bert-base-cased` breaking `"CAPITALIZATION"` (14 characters) into 8 pieces (`CA`, `##PI`, `##TA`, `##L`, `##I`, `##Z`, `##AT`, `##ION`), while GPT-4's tokenizer breaks the same word into only 2 (`CAPITAL`, `IZATION`). Turning that into a **characters-per-token** ratio:

- `bert-base-cased`: $14 \text{ chars} / 8 \text{ tokens} = 1.75$ chars/token
- GPT-4 tokenizer: $14 \text{ chars} / 2 \text{ tokens} = 7.0$ chars/token

$7.0 / 1.75 = 4\times$ — for this word, GPT-4's larger vocabulary (100,263 vs. 28,996 entries) represents the same text in a quarter as many tokens. That directly matters downstream: fewer tokens per input means more of the context window (see [02-transformer-models-overview.md](02-transformer-models-overview.md)) is left for actual content, and — since LLM APIs typically bill per token — a lower cost for the same text.
