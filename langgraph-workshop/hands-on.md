# Hands-On LangGraph: The Making of Karan

> A storytelling walkthrough of every concept in this workshop's notebooks, told as one continuous build:
> we are assembling **Karan**, a multimodal Telegram agent, one capability at a time.
> Use this document as a reference whenever you need to remember *why* a LangGraph concept exists and *how* it's wired together.

**Cast of notebooks:**

| # | Notebook | What Karan learns in this chapter |
|---|----------|-----------------------------------|
| 1 | `1_langgraph_crash_course.ipynb` | How to think, decide, and act (graphs, chains, router, ReAct) |
| 2.1 | `2_1_short_term_memory.ipynb` | How to remember *this* conversation |
| 2.2 | `2_2_long_term_memory.ipynb` | How to remember facts from documents (RAG) |
| 3 | `3_speech_to_text_systems.ipynb` | How to listen (voice → text) |
| 4 | `4_text_to_speech_systems.ipynb` | How to speak (text → voice) |
| 5 | `5_vision_language_models.ipynb` | How to see (image → text) |
| 6 | `6_image_generation.ipynb` | How to draw (text → image) |
| 7 | `7_telegram_agent.ipynb` | How to become a whole person, wired into Telegram |

---

## Chapter 0 — Why a graph at all?

Before Karan can do anything, we need a way to describe *how he thinks*. A regular Python script executes top to bottom. An LLM chain (prompt → model → parse) is only slightly smarter — it's still a straight line.

But a real assistant needs to **loop** ("call a tool, look at the result, decide again"), **branch** ("should I speak, or draw, or just reply?"), and **remember**. That's exactly what LangGraph is for:

> LangGraph is a framework for building **stateful, graph-based LLM applications** — as opposed to linear chains — giving you conditional logic, looping, tool orchestration and shared memory as first-class citizens.

Every LangGraph application, no matter how complex Karan eventually gets, is built from exactly **three primitives**:

1. **State** — a shared data structure (a `TypedDict`) that flows through the graph.
2. **Nodes** — plain Python functions that receive the state and return a partial update to it.
3. **Edges** — the wiring that decides which node runs next (fixed, or conditional/dynamic).

Everything in this document is a variation on those three ideas.

---

## Chapter 1 — LangGraph Crash Course

### 1.1 State: the shared notebook everyone writes on

```python
from typing_extensions import TypedDict

class State(TypedDict):
    graph_state: str
```

The state schema is the contract: every node knows what shape of data it will receive and must return. In the toy example, `graph_state` is just a string that nodes append to.

### 1.2 Nodes: functions that read and rewrite the state

```python
def node_1(state):
    return {"graph_state": state['graph_state'] + " Welcome"}
```

Two rules that hold for the rest of this document:
- A node always takes `state` as its first argument.
- A node returns a **dict of updates**, not the full state — by default, each key overwrites the previous value (more on that in "reducers" below).

### 1.3 Edges: normal vs. conditional

- **Normal edge** — always go from A to B (`builder.add_edge("node_1", "node_2")`).
- **Conditional edge** — a router function inspects the state and returns the *name* of the next node:

```python
def decide_node(state) -> Literal["node_2", "node_3"]:
    return "node_2" if random.random() < 0.5 else "node_3"

builder.add_conditional_edges("node_1", decide_node)
```

Conditional edges are the mechanism behind every "should I call a tool or just answer?" decision Karan will ever make.

### 1.4 Building & running the graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_node)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"graph_state": "Hi there, it's Leonardo!"})
```

`START` and `END` are sentinel nodes marking entry and exit. `.compile()` validates the graph structure (catches orphan nodes, dead ends) and returns a **Runnable** — anything compiled this way exposes `.invoke()`, `.stream()`, etc., the same interface used across the LangChain ecosystem. `graph.get_graph().draw_mermaid_png()` renders the graph as a Mermaid diagram, useful for sanity-checking the wiring visually as it grows.

### 1.5 Messages: the vocabulary of conversation

Once we move from a toy string state to actual conversations, the state needs to hold a **list of chat messages**. LangChain models four message roles:

| Message type | Who it represents |
|---|---|
| `SystemMessage` | Instructions that steer the model's behavior |
| `HumanMessage` | The user |
| `AIMessage` | The model's own reply (may contain tool calls) |
| `ToolMessage` | The result of executing a tool |

```python
from langchain_core.messages import AIMessage, HumanMessage

messages = [AIMessage(content="Hello, I'm your assistant...", name="Agent")]
messages.append(HumanMessage(content="Hi, first time visiting...", name="Leonardo"))
```

### 1.6 Chat models

A chat model (in this workshop, `ChatGoogleGenerativeAI` with `gemini-2.5-flash` — the notebooks also show the commented-out `ChatOpenAI` equivalent) takes the message list and returns a new `AIMessage`. Both providers implement the same LangChain interface, so swapping models is a one-line change.

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
result = llm.invoke(messages)   # -> AIMessage
```

### 1.7 Tools: letting the model touch the outside world

An LLM only produces text. To let it *act* — call an API, run a calculation, query a database — you bind **tools**: plain Python functions with type hints and a docstring, which LangChain turns into a structured schema the model can request.

```python
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])
```

The model never executes the function itself — it emits a **tool call** (a structured `{"name": ..., "args": ...}` object living on `AIMessage.tool_calls`). Your code (or a prebuilt node) is responsible for actually running it and feeding the result back as a `ToolMessage`.

```python
tool_call = llm_with_tools.invoke([HumanMessage(content="What is 2 multiplied by 3")])
tool_call.tool_calls
# [{'name': 'multiply', 'args': {'a': 2, 'b': 3}, 'id': '...', 'type': 'tool_call'}]
```

### 1.8 MessagesState: the built-in conversational state

Rather than hand-rolling a `TypedDict` with a `messages: list[AnyMessage]` field, LangGraph ships `MessagesState`. Its superpower is a built-in **reducer**: instead of overwriting the message list on every node update (the default behavior), it *appends* new messages to the existing list.

```python
from langgraph.graph import MessagesState

class MessagesState(MessagesState):
    pass  # extend it with your own fields when you need to
```

> **Reducers, in one sentence:** a reducer is a function attached to a state field that controls *how* an update is merged into the existing value — append, overwrite, sum, deduplicate, etc. `MessagesState.messages` uses an append-with-dedup-by-id reducer under the hood, which is also what makes `RemoveMessage` (Chapter 2.1) work.

### 1.9 The Chain pattern

The simplest possible agent: one node, invoke the tool-bound model, done.

```python
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)
graph = builder.compile()
```

This graph will happily *decide* to call a tool (you'll see `tool_calls` on the returned `AIMessage`), but nothing in the graph actually executes it yet. That's the next pattern.

### 1.10 The Router pattern

Now we close the loop: add a node that executes the tool call, and a conditional edge that decides whether to go there at all.

```python
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)
```

Two prebuilt pieces do the heavy lifting:
- **`ToolNode`** — given a list of tools, automatically inspects the last `AIMessage`'s `tool_calls`, executes the matching Python function(s), and appends the results as `ToolMessage`s.
- **`tools_condition`** — a ready-made conditional-edge function: if the last message contains tool calls, route to `"tools"`; otherwise, route to `END`.

This is already a rudimentary **agent**: the LLM is in control of the flow (respond directly, or delegate to a tool), not your code.

### 1.11 The ReAct pattern (Reason + Act)

The Router pattern stops after *one* tool call. Real tasks often need several steps ("add these numbers, then multiply the result, then divide it"). ReAct is the Router pattern with one crucial change: **the `tools` node loops back into the assistant node instead of ending**.

```python
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")   # <-- the loop that makes this ReAct
react_graph = builder.compile()
```

Now the model can: call a tool → see the result → decide to call another tool → ... → eventually answer directly, at which point `tools_condition` routes to `END` because the final `AIMessage` has no tool calls left. This "reason, act, observe, repeat" cycle is the backbone of nearly every LangGraph agent, including Karan's final form in Chapter 7.

> **Modern shortcut:** everything in 1.9–1.11 can also be built in one line with `langgraph.prebuilt.create_react_agent(model, tools)` (or the newer `langchain.agents.create_agent`). The workshop builds it by hand because understanding the node/edge wiring is exactly what lets you customize the pattern later (as Karan's graph does in Chapter 7).

---

## Chapter 2.1 — Short-Term Memory

### The problem: state is transient

`graph.invoke(...)` runs one pass through the graph and returns. Call it again with a new message, and there's no trace of the previous call — the LLM has amnesia by default:

```python
graph.invoke({"messages": [HumanMessage("Hello, I'm Leonardo!")]})
graph.invoke({"messages": [HumanMessage("Remember my name?")]})
# -> "I don't have memory of past conversations..."
```

### The fix: checkpointers + thread_id

A **checkpointer** persists the graph's state after every step (every "super-step"), keyed by a `thread_id`. Compile the graph with one, then always pass a `config` carrying that id:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph_memory = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
graph_memory.invoke({"messages": [HumanMessage("Hello, my name is Leonardo!")]}, config)
graph_memory.invoke({"messages": [HumanMessage("Do you remember it?")]}, config)
# -> "Yes, I remember your name! It's Leonardo."
```

Mental model: the checkpointer is a versioned key–value store; `thread_id` is the key that selects *which conversation's history* to load before running the graph and save after.

> **Naming note:** the notebooks use `MemorySaver`. Current LangGraph docs alias this as `InMemorySaver` (`from langgraph.checkpoint.memory import InMemorySaver`) — same class, clearer name. Either import works; prefer `InMemorySaver` in new code.

### Keeping conversations cheap: summarization

An ever-growing message list eventually blows past the model's context window (and your token budget). The fix is a **summarization node** that periodically compresses old messages into a running summary.

```python
class State(MessagesState):
    summary: str

def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        messages = [SystemMessage(content=f"Summary of conversation earlier: {summary}")] + state["messages"]
    else:
        messages = state["messages"]
    return {"messages": llm.invoke(messages)}

def summarize_conversation(state: State):
    summary = state.get("summary", "")
    prompt = (f"This is summary of the conversation to date: {summary}\n\nExtend the summary..."
              if summary else "Create a summary of the conversation above:")
    response = llm.invoke(state["messages"] + [HumanMessage(content=prompt)])

    # Keep only the 2 most recent raw messages; everything else collapses into `summary`
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def should_continue(state: State) -> Literal["summarize_conversation", END]:
    return "summarize_conversation" if len(state["messages"]) > 20 else END
```

Two ideas worth internalizing here:
- **`RemoveMessage(id=...)`** is a special update understood by `MessagesState`'s reducer: instead of appending, it deletes the message with that id from state. This is how you *shrink* an append-only list.
- The conditional edge (`should_continue`) is just the router pattern from Chapter 1, applied to a memory-management decision instead of a tool-calling decision — the same primitive, a different job.

### Making memory durable across restarts

`MemorySaver` lives in RAM — restart the Python process and it's gone. For memory that survives kernel restarts (or, in production, server restarts), swap in a database-backed checkpointer with the *same* `.compile(checkpointer=...)` API:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("example.db", check_same_thread=False)
memory = SqliteSaver(conn)

graph = workflow.compile(checkpointer=memory)
```

Nothing else about the graph changes — this is the whole point of the checkpointer abstraction: your nodes and edges don't know or care where the state is stored. Karan's Telegram bot (Chapter 7) uses exactly this pattern with `short_term_memory.db`.

---

## Chapter 2.2 — Long-Term Memory (Retrieval-Augmented Generation)

Short-term memory (Chapter 2.1) is about *this conversation*. Long-term memory, in this workshop, means: **give the agent access to a body of knowledge that lives outside the conversation** — articles, PDFs, biographies — via a vector database. This is the classic **RAG (Retrieval-Augmented Generation)** pattern, wired into LangGraph as just another tool.

### Indexing: turning documents into searchable chunks

1. **Load** the source document(s).
2. **Split** into overlapping chunks (so no chunk is too big for the embedding model, and context isn't lost at chunk boundaries).
3. **Embed** each chunk into a vector.
4. **Store** the vectors in a vector database for similarity search.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)
vector_store = Chroma(
    collection_name="example_collection",   # ~ a table
    embedding_function=embeddings,
    persist_directory="longterm_memory_db",
)

loader = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
vector_store.add_documents(all_splits)
```

### Retrieval: querying by meaning, not keywords

```python
vector_store.similarity_search("What is task decomposition?", k=1)
```

Under the hood this embeds the query and returns the `k` chunks whose vectors are closest — semantic search rather than keyword matching.

### Wiring retrieval into the graph as a tool

Exactly like `multiply` in Chapter 1, retrieval becomes something the LLM can *choose* to invoke:

```python
from langchain_core.tools.retriever import create_retriever_tool

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="retrieve_article_info_tool",
    description="Retrieve some information from Lilian Weng's article",
)

llm_with_tool = llm.bind_tools([retriever_tool])

def assistant(state: MessagesState):
    return {"messages": [llm_with_tool.invoke(state["messages"])]}

workflow = StateGraph(MessagesState)
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode([retriever_tool]))
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "assistant")
```

Notice this is *the exact same ReAct graph shape as Chapter 1* — the only thing that changed is which tool is bound. That reuse is the point: once you understand "assistant decides → tool executes → loop back," adding retrieval, arithmetic, image generation, or anything else is just "bind another tool."

> **Terminology check (per current LangGraph docs):** what this notebook calls "long-term memory" is really a **RAG knowledge base** — static documents made searchable via a vector store and exposed as a tool. LangGraph also has an official, distinct concept called **long-term memory via a `Store`** (`langgraph.store.memory.InMemoryStore` or a DB-backed store), used to persist arbitrary facts *across threads/users* (e.g., "this user prefers concise answers"), addressed by `(namespace, key)` rather than by semantic search over documents. In practice the two ideas complement each other: use a vector-store-backed retriever tool for "knowledge the agent can look up" (what this chapter builds, and what Karan's biography PDF becomes in Chapter 7), and use a `Store` for "facts the agent should remember about a user between separate conversations." Keep both tools in your mental toolbox when designing new agents.

---

## Chapter 3 — Speech-to-Text (STT)

Telegram users send voice notes, not `HumanMessage` objects. Before anything reaches the graph, audio has to become text.

```python
from google import genai

client = genai.Client()
audio_file = client.files.upload(file='audio.mp3')

transcription = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Generate a transcript of the speech.", audio_file]
)
print(transcription.text)
```

The workshop's slides reference **Whisper** (OpenAI's STT model, robust to accents/background noise/multiple languages) as the canonical choice — the notebook itself demonstrates the same idea using Gemini's multimodal `generate_content` with an uploaded audio file. Either way, the architectural point is the same:

> **STT sits *outside* the graph, upstream of it.** It's a preprocessing step that converts one modality (audio) into the modality LangGraph state actually deals with (text/messages). The Telegram integration in Chapter 7 calls Whisper directly inside `handle_voice`, then feeds the resulting text into `graph.invoke(...)` exactly like a typed message.

---

## Chapter 4 — Text-to-Speech (TTS)

The mirror image of Chapter 3: turning the agent's textual reply into an audio clip, using **ElevenLabs**.

```python
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

voice_id = "Xb7hH8MSUJpSbSDYk0k2"     # a specific persona's voice
model_id = "eleven_flash_v2_5"

audio = client.text_to_speech.convert(text=msg, voice_id=voice_id, model_id=model_id)
audio_bytes = b"".join(audio)          # convert stream into bytes

with open("audio.mp3", "wb") as f:
    f.write(audio_bytes)
```

Two config values matter every time you use ElevenLabs:
- **`voice_id`** — which persona's voice to use (Karan gets his own dedicated voice id in Chapter 7).
- **`model_id`** — which underlying TTS model/quality tier to use (`eleven_flash_v2_5` trades a little quality for low latency, good for chat-like responsiveness).

> Same architectural placement as STT: **TTS sits *outside* the graph, downstream of it.** A node inside the graph decides "the reply should be spoken," but the actual bytes get generated in the delivery layer (Chapter 7's `generate_final_response_node` and, at the very edge, the Telegram handler that calls `reply_voice`).
>
> The notebook also documents a real-world failure mode worth remembering: ElevenLabs' free tier can throw `detected_unusual_activity` / `401` errors if it suspects automated/proxy usage. Production systems should not treat third-party API calls as infallible — pair them with retries (see the `tenacity` pattern in Chapter 7).

---

## Chapter 5 — Vision Language Models (VLMs)

Now the reverse of Chapter 4's output modality problem, but for *input*: users send Karan photos, and he needs to "see" them before responding.

### Step 1 — Resize before you send

```python
from PIL import Image

img = Image.open("image.png").convert("RGB")
img_resized = img.resize((512, 512))
```

> This is called out explicitly in the notebook as important: sending full-resolution photos to a multimodal model burns far more input tokens (and money/latency) than necessary. Downscaling to a size the model can still describe accurately is a cheap, high-leverage optimization.

### Step 2 — Encode as base64

Multimodal chat APIs expect images either as a URL or as inline base64-encoded data:

```python
import base64
from io import BytesIO

def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

base64_image = encode_image(img_resized)
```

### Step 3 — Ask the model to describe it

```python
chat_completion = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what you see in the picture"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ],
    }],
    model="gpt-4o",
)
```

(The notebook also shows the equivalent call against Gemini's `generate_content` with an inline `image_data` blob — same idea, different SDK.)

> **Where this plugs into the graph:** exactly like STT, image captioning happens *before* the graph is invoked. The caption text (wrapped in a marker like `[IMAGE_ANALYSIS] ...`, as Chapter 7 shows) is what actually becomes the `HumanMessage` content the graph sees. The graph itself never touches raw pixels — it only ever reasons over text.

---

## Chapter 6 — Image Generation

Karan needs to be able to *show* himself, not just describe things. This uses autoregressive/diffusion image-generation models rather than a plain chat model.

```python
from google import genai
from google.genai.types import GenerateContentConfig, Modality

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=prompt,
    config=GenerateContentConfig(
        response_modalities=[Modality.TEXT, Modality.IMAGE],
        candidate_count=1,
    ),
)

for part in response.candidates[0].content.parts:
    if part.inline_data:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("output.png")
```

The OpenAI equivalent used elsewhere in the workshop (and in Karan's final image node):

```python
from openai import OpenAI

client = OpenAI()
result = client.images.generate(model="gpt-image-1", prompt=prompt, quality="high", size="1024x1024")
image_bytes = base64.b64decode(result.data[0].b64_json)
image = Image.open(BytesIO(image_bytes))
```

The other big lesson in this chapter is **prompt engineering for character consistency**: the prompt used to generate "Karan" is a structured spec (Appearance → Clothing → Background → Rules) rather than a loose sentence. Reusing this same structured template every time (as Chapter 7's `basic_prompt` does, appending situational context at the end) is what keeps Karan looking like the same person across many independent generations, since none of these models have "memory" of a previous image by default.

> **Where this plugs into the graph:** unlike STT/TTS/VLM, image *generation* in Karan's final design happens **inside a graph node** (`generate_final_response_node`), because *deciding whether to generate an image at all* is itself a decision the graph needs to route on (see the router node in Chapter 7).

---

## Chapter 7 — The Telegram Agent (Karan, assembled)

This is where every previous chapter's concept becomes one wire in a single graph. Read this chapter as "how do all these pieces compose," not as new primitives.

### The state: everything Karan needs to carry between nodes

```python
class KaranState(MessagesState):
    summary: str          # Chapter 2.1 — conversation summarization
    response_type: str    # NEW — "text" | "audio" | "image", decided by the router
    audio_buffer: bytes    # Chapter 4 — generated speech, ready to send
    image_path: str        # Chapter 6 — generated image, ready to send
```

Notice this is just `MessagesState` (Chapter 1.8) extended with the extra fields each downstream capability needs — the same "start from the built-in state, bolt on what you need" move as Chapter 2.1's summarizing `State(MessagesState)`.

### Node 1 — `router_node`: deciding the shape of the reply

This is the router pattern (Chapter 1.10) applied to a new kind of decision: not "call a tool or not," but "should the *final* reply be text, a voice note, or an image?" It uses **structured output** — the model is forced to return a value matching a Pydantic schema rather than free text — which is the standard way to make an LLM's decision machine-readable.

```python
class RouterResponse(BaseModel):
    response_type: str = Field(description="...must be one of: 'text', 'image' or 'audio'")

def router_node(state: KaranState):
    llm_structured = llm.with_structured_output(RouterResponse)
    response = llm_structured.invoke([SystemMessage(content=ROUTER_SYSTEM_PROMPT), state["messages"][-1]])

    if response.response_type == "text" and random.random() > 0.5:
        return {"response_type": "audio"}   # adds spontaneity/realism

    return {"response_type": response.response_type}
```

### Node 2 — `generate_text_response_node`: Karan's personality + RAG

This is the ReAct assistant node from Chapter 1.11, with the retriever tool from Chapter 2.2 bound to it, plus the running summary from Chapter 2.1 injected into the system prompt when one exists:

```python
llm_with_tools = llm.bind_tools([retriever_tool])

def generate_text_response_node(state: KaranState):
    summary = state.get("summary", "")
    system_message = f"{SYSTEM_PROMPT} \n Summary of conversation earlier: {summary}" if summary else SYSTEM_PROMPT
    messages = [SystemMessage(content=system_message)] + state["messages"]
    return {"messages": llm_with_tools.invoke(messages)}
```

The `SYSTEM_PROMPT` itself is worth studying as a template for any persona-driven agent: it defines *who* Karan is, *how* he talks, and explicit **rules** (never admit to being an AI, keep replies under 100 words, never refuse to "send" media, etc.) — a persona spec is just a very carefully engineered `SystemMessage`.

### Node 3 — `summarize_conversation_node`

Identical in shape to Chapter 2.1's `summarize_conversation`, just renamed and using `KaranState`.

### Node 4 — `tools` (RAG retrieval)

```python
tool_node = ToolNode([retriever_tool])
```

This is `ToolNode` from Chapter 1.10/2.2 executing the retriever tool — Karan's 10+ page biography PDF, chunked and embedded exactly as in Chapter 2.2, is what makes him able to answer deep questions about "himself" without hardcoding every fact into the system prompt:

```python
loader = PyPDFLoader("karan_biography.pdf")
docs = loader.load()
all_splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

vector_store = Chroma(collection_name="karan_biography_collection", embedding_function=embeddings, persist_directory="long_term_memory")
vector_store.add_documents(all_splits)

retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    name="retrieve_karan_information_tool",
    description="Retrieve information about Karan's background, ...",
)
```

### Node 5 — `generate_final_response_node`: dispatching to TTS/image generation

This node reads the `response_type` decided by the router and calls out to Chapter 4 (ElevenLabs) or Chapter 6 (image generation) accordingly — this is where those "outside the graph" capabilities get pulled *inside* the graph, because the decision of *which one to use* is graph-native:

```python
def generate_final_response_node(state: KaranState):
    if state["response_type"] == "audio":
        audio = elevenlabs_client.text_to_speech.convert(
            text=state["messages"][-1].content, voice_id=voice_id, model_id=model_id)
        return {"audio_buffer": b"".join(audio)}

    elif state["response_type"] == "image":
        result = openai_client.images.generate(
            model="gpt-image-1", prompt=basic_prompt + state["messages"][-1].content,
            quality="high", size="1024x1024")
        image = PIL.Image.open(BytesIO(base64.b64decode(result.data[0].b64_json)))
        image_path = f"{uuid4()}.png"
        image.save(image_path)
        return {"image_path": image_path}

    else:
        return state
```

### Wiring the whole graph

```python
workflow = StateGraph(KaranState)

workflow.add_node("router_node", router_node)
workflow.add_node("generate_text_response_node", generate_text_response_node)
workflow.add_node("summarize_conversation_node", summarize_conversation_node)
workflow.add_node("tools", tool_node)
workflow.add_node("generate_final_response_node", generate_final_response_node)

workflow.add_edge(START, "router_node")
workflow.add_edge("router_node", "generate_text_response_node")
workflow.add_conditional_edges(
    "generate_text_response_node",
    tools_condition,
    {"tools": "tools", END: "generate_final_response_node"},
)
workflow.add_edge("tools", "generate_text_response_node")   # ReAct loop (Ch.1.11)
workflow.add_conditional_edges("generate_final_response_node", should_summarize_conversation)
workflow.add_edge("summarize_conversation_node", END)

graph = workflow.compile(checkpointer=short_term_memory)   # SqliteSaver, Ch.2.1
```

Trace the shape and you'll recognize every earlier chapter:
`START → router (Ch.7) → assistant with RAG tool in a ReAct loop (Ch.1+2.2) → dispatch to TTS/image (Ch.4/6) → conditionally summarize (Ch.2.1) → END`. Nothing here is a new primitive — it's composition of the seven ideas that came before.

### From graph to Telegram: the delivery layer

The final piece is the adapter code that has *nothing to do with LangGraph* — it just translates between Telegram's I/O and `graph.invoke(...)`:

- **`handle_text`** — takes `update.message.text`, invokes the graph directly.
- **`handle_voice`** — downloads the voice file, runs it through Whisper (Ch.3) to get text, *then* invokes the graph with that text.
- **`handle_photo`** — downloads the photo, runs it through a vision model (Ch.5) to get a caption, prepends the user's own caption if any, tags it `[IMAGE_ANALYSIS]`, then invokes the graph with that combined string.
- **`send_response`** — reads `response_type` off the final graph state and calls the matching Telegram reply method (`reply_text`, `reply_voice` with `audio_buffer`, or `reply_photo` with `image_path`).

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(Exception), reraise=True)
def safe_graph_invoke(payload, config=None):
    config = {"configurable": {"thread_id": "miguel"}}
    return graph.invoke(payload, config)
```

`tenacity`'s `@retry` decorator wraps every call to the graph in **exponential backoff** — a production concern that doesn't show up in the earlier, exploratory notebooks: any node in the graph might call a flaky third-party API (recall Chapter 4's ElevenLabs 401), so the entry point to the whole system should assume failure is normal and retry gracefully rather than crash the bot.

Finally, the bot itself is just `python-telegram-bot` routing message types to the three handlers above:

```python
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.VOICE, handle_voice))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
app.run_polling()
```

---

## Cheat Sheet — Concept → API → When to reach for it

| Concept | Key API | Use it when... |
|---|---|---|
| Shared data across a run | `TypedDict` state / `MessagesState` | Any graph — this is non-negotiable, it's the contract nodes read/write. |
| A step of computation | plain function `def node(state): -> dict` | You need to call an LLM, run logic, or transform data mid-graph. |
| Fixed flow | `add_edge(a, b)` | Node B should always run right after node A. |
| Dynamic flow | `add_conditional_edges(a, router_fn)` | The next node depends on the state (tool call present? summarize needed? which media type?). |
| Tool use | `bind_tools([...])` + `ToolNode([...])` + `tools_condition` | The model should be able to call external functions/APIs/retrievers. |
| Multi-step tool use | loop `tools → assistant` (ReAct) | A task may need more than one tool call before answering. |
| Conversation memory (this thread) | `checkpointer=` (`InMemorySaver`/`MemorySaver`, `SqliteSaver`, ...) + `thread_id` | The agent must recall earlier turns in the same conversation, or survive process restarts. |
| Trimming a growing history | `RemoveMessage(id=...)` + a `summary` field | Long conversations threaten the context window / token budget. |
| Knowledge from documents (RAG) | `Chroma` + `RecursiveCharacterTextSplitter` + `create_retriever_tool` | The agent needs facts from PDFs/articles/wikis that don't fit in a prompt. |
| Facts about a user across *different* threads | `Store` (`InMemoryStore` / DB-backed) | You need "long-term memory" in LangGraph's own sense: user preferences, cross-session facts. |
| Voice input | Whisper / multimodal `generate_content` on audio | Users send voice notes; convert to text *before* `graph.invoke`. |
| Voice output | ElevenLabs `text_to_speech.convert` | The agent's reply should be spoken, not just typed. |
| Image input | resize → base64 → multimodal chat call | Users send photos; caption them to text *before* `graph.invoke`. |
| Image output | `gemini-2.5-flash-image` / `gpt-image-1` | The agent should draw/generate a picture as its reply. |
| Structured decisions | `llm.with_structured_output(PydanticModel)` | You need a machine-parseable decision (e.g. routing between text/audio/image), not free text. |
| Resilience to flaky APIs | `tenacity.retry` with exponential backoff | Wrapping any call to a third-party model/API at the system's entry point. |

---

## Closing note: the one habit worth keeping

Every chapter in this workshop reduces to the same move: **identify a decision, express it as a conditional edge; identify a capability, express it as a tool or a node.** Karan's final graph in Chapter 7 looks intimidating at first glance, but it's nothing more than the Chapter 1 ReAct loop with a persona, a memory system, and a media-dispatch decision bolted on. When you design your next agent, start from that same minimal loop — `assistant ⇄ tools`, checkpointed — and add exactly the nodes and conditional edges your use case actually needs, in the same incremental order this workshop did: think → remember → listen → speak → see → draw → ship.
