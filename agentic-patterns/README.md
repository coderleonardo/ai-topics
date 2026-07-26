# Agentic Patterns — Minimalistic Multiagent Framework

This is part of the **Agentic Patterns Series** from [*The Neural Maze*](https://theneuralmaze.substack.com), a series of notes about understanding agents from scratch:

0. [What is an Agent?](https://theneuralmaze.substack.com/p/what-is-an-agent)
1. [Reflection Pattern: When Agents think twice](https://theneuralmaze.substack.com/p/reflection-pattern-agents-that-think) ([Andrew Ng notes](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?ref=dl-staging-website.ghost.io))
2. [Agent Tools: The bridge to the outside world](https://theneuralmaze.substack.com/p/agent-tools-the-bridge-to-the-outside)
3. [Reason + Act (ReAct): Thought -> Action -> Observation](https://theneuralmaze.substack.com/p/building-a-react-agent-from-scratch)

The framework is inspired by two fundamental concepts from CrewAI: **Crew** (a group of agents collaborating on a task) and **Agent** (an autonomous unit executing one specific task). Only individual agents are implemented here so far — this repo builds the *patterns* underneath an agent framework from scratch, in plain Python, rather than relying on a library like LangGraph or CrewAI to provide them.

> **This run uses [Ollama](https://ollama.com) with a local model (`llama3.2:3b`)**, not a hosted API. Every notebook and every agent class in `src/` talks to a local Ollama server. See `hands-on.md` for what that choice reveals about small local models' limits.

## Folder contents

| Path | Description |
|---|---|
| `notebooks/reflection_pattern.ipynb` | From-scratch build of the Reflection pattern (generate → critique → revise) using raw `ollama.chat()` calls. |
| `notebooks/reflection_agent_test.ipynb` | **Test run** — exercises the packaged `ReflectionAgent` class end-to-end against Ollama. |
| `notebooks/tool_pattern.ipynb` | From-scratch build of function/tool calling: prompt protocol, XML tag parsing, manual tool execution. |
| `notebooks/tool_agent_test.ipynb` | **Test run** — exercises the packaged `ToolAgent` class with a real tool (the Hacker News API) against Ollama. |
| `notebooks/planning_pattern.ipynb` | From-scratch build of the full ReAct loop (Thought → Action → Observation) with multi-step tool chaining. No packaged equivalent exists for this one. |
| `src/agentic_patterns/reflection_pattern/reflection_agent.py` | `ReflectionAgent` class. |
| `src/agentic_patterns/tool_pattern/tool.py` | `Tool` class, `@tool` decorator, function-signature introspection, argument validation. |
| `src/agentic_patterns/tool_pattern/tool_agent.py` | `ToolAgent` class (single-shot tool calling). |
| `src/agentic_patterns/utils/completions.py` | Shared chat-history helpers (`ChatHistory`, `FixedFirstChatHistory`, `completions_create`, `build_prompt_structure`, `update_chat_history`). |
| `src/agentic_patterns/utils/extraction.py` | `extract_tag_content` — regex-based extraction of `<tag>...</tag>` content from raw LLM text. |
| `src/agentic_patterns/utils/logging.py` | Colorized console helpers (`fancy_print`, `fancy_step_tracker`) for tracing an agent's internal loop. |
| `hands-on.md` | **Start here for concepts.** Storytelling walkthrough of every pattern, with the actual code, the recorded Ollama transcripts, and the specific ways the small local model deviated from the expected protocol. |
| `images/` | Diagrams referenced by the notebooks/README. |
| `pyproject.toml` | Package metadata (`agentic-patterns`, Python ≥3.10, installable via `pip install -e .`). |

## Setup

```bash
# 1. Install and start Ollama (https://ollama.com/download), then pull the model used throughout:
ollama pull llama3.2:3b

# 2. Install this package in editable mode (from the repo root)
pip install -e .

# 3. Install the runtime dependencies the notebooks/agents actually import
#    (pyproject.toml's dependency list is currently empty — these are installed ad hoc):
pip install ollama colorama python-dotenv requests jupyter
```

No API keys or `.env` file are required for the model calls — everything runs against your local Ollama server. (You may notice commented-out `# from groq import Groq` lines in a couple of files: this course was originally written against Groq's hosted API and was re-pointed at local Ollama models for these runs. `python-dotenv` is only a leftover import from that version, not actually required for the Ollama path.)

## How to use this folder

1. **Read `hands-on.md` first.** It walks through the Reflection pattern, the Tool pattern, and the Planning/ReAct pattern in that order, with real code and real (recorded) model output — including the points where `llama3.2:3b` didn't follow the protocol correctly. That's the most useful part of this repo: seeing *exactly* how a small local model can deviate from a well-designed agentic prompt.
2. **Run the notebooks in this order:** `reflection_pattern.ipynb` → `reflection_agent_test.ipynb` → `tool_pattern.ipynb` → `tool_agent_test.ipynb` → `planning_pattern.ipynb`. Each "from scratch" notebook explains the mechanics; each paired "test" notebook shows the same pattern via the packaged, reusable agent class.
3. Treat the `*_test.ipynb` notebooks as the reference for **how to actually invoke** the library from your own code (`from agentic_patterns import ReflectionAgent`, `from agentic_patterns.tool_pattern.tool_agent import ToolAgent`).
4. Use this repo alongside `../langgraph-workshop/` if you have it: this folder shows what tool calling and ReAct looping look like *without* a framework; the LangGraph workshop shows the same ideas using LangGraph's prebuilt `ToolNode`/`tools_condition`/checkpointer abstractions. `hands-on.md`'s closing section maps the two directly against each other.
