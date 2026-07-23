# Real Model Walkthrough: Phi-3-mini-4k-instruct

*Source: `decoder-only-example.ipynb`*

This note grounds the abstract pipeline from [04-llm-architecture-pipeline.md](04-llm-architecture-pipeline.md) in one real, loaded model — `microsoft/Phi-3-mini-4k-instruct`, a decoder-only (causal) LLM.

## Loading the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
```

`AutoModelForCausalLM` is for **decoder-only / causal** models used for text generation — "causal" because, to predict the next token, the model can only attend to preceding tokens (see masked self-attention, [02](02-transformer-models-overview.md)/[05](05-transformer-block.md)). This contrasts with `AutoModelForMaskedLM`, for **encoder-only** models like BERT that predict a masked/hidden token using context from *both* directions.

## Generating full text via a pipeline

```python
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,   # do_sample=False == temperature=0 == greedy decoding
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. "
output = generator(prompt)
print(output[0]['generated_text'])
```

The pipeline hides the token-by-token loop described in [04-llm-architecture-pipeline.md](04-llm-architecture-pipeline.md) — internally it's still generating one token at a time and feeding the sequence back in.

## Inspecting the architecture directly

```python
model
```

```
Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
          (rotary_emb): Phi3RotaryEmbedding()
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Phi3RMSNorm()
        (post_attention_layernorm): Phi3RMSNorm()
      )
    )
    (norm): Phi3RMSNorm()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)
```

Mapping this back to the concepts in the other notes:

| In the printout | Concept | Note |
|---|---|---|
| `Embedding(32064, 3072, ...)` | vocab size = 32,064, embedding dim (`d_model`) = 3,072 | [03](03-tokenization.md), [04](04-llm-architecture-pipeline.md) |
| `32 x Phi3DecoderLayer` | 32 stacked transformer (decoder) blocks | [05](05-transformer-block.md) |
| `qkv_proj`, `o_proj` | fused query/key/value projection + output projection of self-attention (here `qkv_proj` computes Q, K, V in one matrix multiply instead of three separate ones — a common real-world optimization) | [06](06-self-attention-deep-dive.md) |
| `Phi3RotaryEmbedding` | RoPE positional embeddings, instead of the sinusoidal absolute positions taught in the intro lessons | [06](06-self-attention-deep-dive.md), [09](09-transformer-block-by-hand.md) |
| `mlp` (`gate_up_proj` + `down_proj` + `SiLU`) | the feedforward network, using a **SwiGLU-style gated MLP** with SiLU activation, rather than the plain `Linear → ReLU → Linear` used for hand-calculation simplicity | [05](05-transformer-block.md), [09](09-transformer-block-by-hand.md) |
| `Phi3RMSNorm` | **RMSNorm** instead of standard LayerNorm — normalizes by root-mean-square only (no mean-centering, no learned bias) — a common efficiency swap in modern LLMs | [09](09-transformer-block-by-hand.md) |
| `lm_head`: `Linear(3072 → 32064)` | the LM head, projecting the final hidden state to vocabulary logits | [04](04-llm-architecture-pipeline.md) |

This is a good reminder that the "textbook" transformer block (sinusoidal position encodings, separate Q/K/V matrices, plain LayerNorm, `Linear → ReLU → Linear` FFN) is a simplified teaching model — real production models like Phi-3 swap in more efficient variants of nearly every piece, while keeping the same overall structure (attention sub-layer + FFN sub-layer + residuals + normalization).

## Worked example: counting Phi-3-mini's parameters from the printout above

Every `Linear(in_features=A, out_features=B, ...)` in the printout is a dense `A × B` weight matrix (no bias, per `bias=False`), so its parameter count is just $A \times B$. Reading the numbers straight out of the architecture dump:

| Layer | Shape | Params |
|---|---|---|
| `qkv_proj` | 3072 × 9216 | 28,311,552 |
| `o_proj` | 3072 × 3072 | 9,437,184 |
| `gate_up_proj` | 3072 × 16384 | 50,331,648 |
| `down_proj` | 8192 × 3072 | 25,165,824 |

Per-layer subtotals: self-attention ($qkv\_proj + o\_proj$) = $28{,}311{,}552 + 9{,}437{,}184 = 37{,}748{,}736$. MLP ($gate\_up\_proj + down\_proj$) = $50{,}331{,}648 + 25{,}165{,}824 = 75{,}497{,}472$. One full `Phi3DecoderLayer` = $37{,}748{,}736 + 75{,}497{,}472 = \mathbf{113{,}246{,}208}$ params (RMSNorm's two learned scale vectors add only ~6K more — negligible next to the linear layers).

There are 32 identical layers (`(0-31): 32 x Phi3DecoderLayer`):

$$113{,}246{,}208 \times 32 = 3{,}623{,}878{,}656 \approx 3.624\text{B}$$

Add the two vocabulary-sized matrices — `embed_tokens` ($32064 \times 3072 = 98{,}500{,}608$) and `lm_head` ($3072 \times 32064 = 98{,}500{,}608$, same shape transposed):

$$3{,}623{,}878{,}656 + 98{,}500{,}608 + 98{,}500{,}608 = \mathbf{3{,}820{,}879{,}872} \approx \mathbf{3.82\text{B}}$$

That lands almost exactly on Phi-3-mini's published parameter count of **~3.8B** — the entire model's size, derived from nothing but the shapes already visible in the printout above and simple multiplication. It's also a concrete illustration of where an LLM's parameters actually live: here, the 32 stacked decoder layers alone account for ~95% of the total (3.624B of 3.82B), with the two vocabulary-sized projections making up the rest.

## Manually generating a single token

Instead of using the pipeline, you can call the model's two pieces (transformer stack, then LM head) directly:

```python
prompt = "The capital of France is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# tensor([[ 450, 7483,  310, 3444,  338]])   <- 5 tokens

# 1. Run just the stack of transformer blocks (no LM head)
model_output = model.model(input_ids)
model_output[0].shape
# torch.Size([1, 5, 3072])
#             ^  ^  ^
#             |  |  embedding size (d_model)
#             |  number of tokens
#             batch size

# 2. Run the LM head on that output
lm_head_output = model.lm_head(model_output[0])
lm_head_output.shape
# torch.Size([1, 5, 32064])   <- one 32,064-way probability distribution per token

# 3. Take the last token's logits (what comes after "is") and pick the argmax
token_id = lm_head_output[0, -1].argmax(-1)
token_id
# tensor(3681)

tokenizer.decode(token_id)
# 'Paris'
```

This is exactly the shape story from [04-llm-architecture-pipeline.md](04-llm-architecture-pipeline.md): `[batch, seq_len, d_model]` out of the transformer stack, `[batch, seq_len, vocab_size]` out of the LM head, and greedy decoding (`argmax`) picks `"Paris"` as the single most probable next token for `"The capital of France is"`.

The toy version of exactly this argmax step — with every number shown — is worked out in [09-transformer-block-by-hand.md](09-transformer-block-by-hand.md).
