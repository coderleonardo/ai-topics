# How Transformers Work — Notes

Notes distilled from the `docs/transcripts/` course transcripts and the companion notebooks (`tokenizers.ipynb`, `decoder-only-example.ipynb`), in reading order.

1. [Language Representations: Bag-of-Words → Word2Vec → RNN Attention](01-language-representations.md)
2. [Transformer Models Overview: Encoder-Decoder, BERT, GPT](02-transformer-models-overview.md)
3. [Tokens and Tokenizers](03-tokenization.md)
4. [The LLM Pipeline: Tokenizer → Transformer Blocks → LM Head](04-llm-architecture-pipeline.md)
5. [Inside a Transformer Block](05-transformer-block.md)
6. [Self-Attention Mechanics: Q/K/V, Multi-Head, and Efficient Variants](06-self-attention-deep-dive.md)
7. [Mixture of Experts (MoE)](07-mixture-of-experts.md)
8. [Real Model Walkthrough: Phi-3-mini-4k-instruct](08-model-walkthrough-phi3.md)
9. [A Transformer Block, Computed By Hand](09-transformer-block-by-hand.md) — the capstone: a full numeric forward pass through one decoder block, every matrix multiply and softmax spelled out

Source transcripts live in [`transcripts/`](transcripts/); the two Jupyter notebooks referenced above live in the project root.
