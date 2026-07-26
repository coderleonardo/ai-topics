# LangGraph Workshop — Building Multimodal Agents

Hands-on notebooks for learning [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) by incrementally building **Karan**, a multimodal Telegram agent that can chat, remember, listen, speak, see, and draw.

## Folder contents

| File | Description |
|---|---|
| `1_langgraph_crash_course.ipynb` | Core LangGraph primitives: state, nodes, edges, chains, the Router pattern, and the ReAct agent loop. |
| `2_1_short_term_memory.ipynb` | Conversation memory via checkpointers (`MemorySaver`, `SqliteSaver`) and thread-scoped state, plus conversation summarization. |
| `2_2_long_term_memory.ipynb` | Retrieval-Augmented Generation (RAG): indexing documents into ChromaDB and exposing retrieval as a tool. |
| `3_speech_to_text_systems.ipynb` | Transcribing voice notes to text (Whisper / Gemini), for feeding Telegram voice messages into the graph. |
| `4_text_to_speech_systems.ipynb` | Generating spoken responses with ElevenLabs. |
| `5_vision_language_models.ipynb` | Captioning/describing images so the agent can react to photos. |
| `6_image_generation.ipynb` | Generating on-brand character images (Gemini / `gpt-image-1`) with a structured, repeatable prompt template. |
| `7_telegram_agent.ipynb` | Capstone: combines every notebook above into one LangGraph app wired into a live Telegram bot. |
| `hands-on.md` | **Start here for concepts.** A storytelling walkthrough of every idea across all 7 notebooks, plus a cheat-sheet table (concept → API → when to use it). Written as reference material for building your own LangGraph agents. |
| `Langgraph_Workshop_Building_Multimodal_Agents.pdf` | Original workshop slide deck. |
| `gemini_api_test.py` | Minimal standalone script to sanity-check your `GEMINI_API_KEY` works, outside of any notebook. |
| `pyproject.toml` / `uv.lock` | Project dependencies, managed with [`uv`](https://docs.astral.sh/uv/). |
| `.python-version` | Pins Python `3.11`. |
| `.env` | API keys (gitignored) — see setup below. |

## Setup

This project uses `uv` for dependency management (Python 3.11 pinned via `.python-version`).

```bash
uv sync
```

Each notebook also installs its own extra dependencies in its first cell (`langchain_openai`, `langgraph`, `langchain-chroma`, `elevenlabs`, `python-telegram-bot`, etc.), since notebooks were originally authored for Google Colab.

### Environment variables

Create a `.env` file (already gitignored) with the keys the notebooks read via `python-dotenv` / `getpass`:

```
GEMINI_API_KEY=...
ELEVENLABS_API_KEY=...
```

`7_telegram_agent.ipynb` additionally needs:

```
OPENAI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
```

Verify your Gemini key works with:

```bash
uv run python gemini_api_test.py
```

## How to use this folder

1. **Read `hands-on.md` first** — it explains every concept (state/nodes/edges, memory, RAG, STT/TTS/VLM, image generation) in the order the notebooks introduce them, with the actual code and a cheat-sheet for quick lookup later.
2. **Work through the notebooks in numeric order** (`1` → `7`). Each one builds on APIs introduced in the previous one; `7` assembles all of them into the final agent.
3. Notebooks write local artifacts as you run them (SQLite DBs like `example.db` / `short_term_memory.db`, Chroma persist directories like `longterm_memory_db` / `long_term_memory`, downloaded media). These are gitignored — delete them if you want a clean re-run.
4. Use `hands-on.md`'s cheat-sheet table as a reference when designing your **own** LangGraph agents, not just for following along with Karan's.
