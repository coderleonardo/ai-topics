# AI Topics

A personal, growing collection of hands-on notes, workshops, and small tools for learning AI/ML end to end — from classical machine learning and transformer internals, to building agents (from scratch and with frameworks like LangGraph), to working with managed platforms like Amazon Bedrock, and packaging reusable Claude Code Skills.

Each top-level folder is a self-contained topic with its own notebooks/docs/code and (where relevant) a `README.md`/`hands-on.md` pair: the README explains what the folder is and how to run it, the hands-on doc walks through the concepts in depth.

> This repo evolves as topics are finished, so it isn't a fixed curriculum — folders are added, expanded, or reorganized over time. Only folders currently committed to this repository are listed below.

## Folders

- **[`agentic-patterns/`](./agentic-patterns)** — Core agent design patterns (Reflection, Tool use, ReAct/Planning) implemented from scratch in plain Python, run against a local model via Ollama. Good for understanding what frameworks like LangGraph automate under the hood.

- **[`aws-genAI-with-bedrock/`](./aws-genAI-with-bedrock)** — Notes on building generative AI applications with Amazon Bedrock: foundation models, the Bedrock Runtime API, Knowledge Bases (RAG), Prompt Management, the Converse API (incl. tool use), and Bedrock Flows.

- **[`how-transformers-work/`](./how-transformers-work)** — From-scratch notes and notebooks on transformer internals: language representations, tokenization, self-attention (Q/K/V, multi-head), Mixture of Experts, a real model walkthrough (Phi-3), and a full transformer block computed by hand.

- **[`langgraph-workshop/`](./langgraph-workshop)** — A multimodal agent workshop built with LangGraph: graphs/state/tools/ReAct, short- and long-term memory (checkpointers, RAG), speech-to-text, text-to-speech, vision, image generation, and a full Telegram bot tying it all together.

- **[`machine-learning/`](./machine-learning)** — Classical ML docs, notebooks, and from-scratch implementations: clustering, dimensionality reduction, statistical distributions, KNN, linear models, evaluation metrics, preprocessing, sampling, and SVMs.

- **[`skills-utils/`](./skills-utils)** — Reusable [Claude Code Skills](https://docs.claude.com/en/docs/claude-code/skills), packaged so anyone can download and install them. Currently checked in: `skills-installer-and-checker/` (the `skill-manager` skill), which bulk-installs a configurable bundle of agent skills from a GitHub repo and runs a local, offline security audit on whatever's already installed.
