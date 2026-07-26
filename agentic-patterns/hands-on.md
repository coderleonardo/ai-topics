# Hands-On Agentic Patterns: Building Agents From Scratch (with Ollama)

> A storytelling walkthrough of every concept in this repo's notebooks and source code.
> Where the LangGraph workshop shows you the *prebuilt* abstractions (`ToolNode`, `tools_condition`, the ReAct loop as two edges), this repo builds the exact same ideas **by hand**, in plain Python, so you can see what's actually happening under the hood — and, crucially, what breaks when you swap a frontier cloud model for a small local one.

All the notebooks here were run against a **local model served by [Ollama](https://ollama.com)** (`llama3.2:3b`), not a hosted API. That choice is why several of the transcripts below show the agent going off the rails — it's not a bug in the code, it's the recurring lesson of this repo: **small local models are far less reliable at following strict agentic protocols than large hosted ones.** Keep that in mind as you read; it's called out explicitly at the end of every chapter where it bites.

## Cast of files

| File | Role |
|---|---|
| `notebooks/reflection_pattern.ipynb` | **From scratch.** Builds the Reflection pattern by hand with raw `ollama.chat()` calls. |
| `notebooks/reflection_agent_test.ipynb` | **The test run.** Exercises the packaged `ReflectionAgent` class — this is the file the user actually ran end-to-end against Ollama. |
| `notebooks/tool_pattern.ipynb` | **From scratch.** Builds function/tool calling by hand: prompt templates, XML tags, JSON parsing. |
| `notebooks/tool_agent_test.ipynb` | **The test run.** Exercises the packaged `ToolAgent` class with a real tool (Hacker News API) against Ollama. |
| `notebooks/planning_pattern.ipynb` | **From scratch, no package.** Implements the full ReAct (Thought → Action → Observation) loop manually. There is no packaged "PlanningAgent" — this pattern only exists as a notebook. |
| `src/agentic_patterns/reflection_pattern/reflection_agent.py` | The packaged `ReflectionAgent` class. |
| `src/agentic_patterns/tool_pattern/tool.py` | The `Tool` class, the `@tool` decorator, function-signature introspection, argument validation. |
| `src/agentic_patterns/tool_pattern/tool_agent.py` | The packaged `ToolAgent` class (single-shot tool calling, no reflection loop). |
| `src/agentic_patterns/utils/completions.py` | Shared chat-history helpers used by every agent (`ChatHistory`, `FixedFirstChatHistory`, `build_prompt_structure`, `completions_create`). |
| `src/agentic_patterns/utils/extraction.py` | `extract_tag_content` — the regex-based parser that pulls `<tool_call>...</tool_call>` (or any tag) out of raw LLM text. |
| `src/agentic_patterns/utils/logging.py` | Colorized, human-friendly console output for following an agent's internal loop. |

---

## Chapter 0 — What is an agent, and why build it by hand?

The README frames this repo around three notes from *The Neural Maze*'s "Agentic Patterns" series:

1. **What is an Agent?** — an LLM wrapped in a loop that can *observe*, *decide*, and *act*, instead of just answering once.
2. **Reflection** — an agent that critiques and improves its own output before handing it back.
3. **Tools** — the bridge that lets an LLM touch the outside world (APIs, functions, calculators).
4. **ReAct (Reason + Act)** — Thought → Action → Observation, repeated until the task is done.

The framework's own two building blocks (borrowed conceptually from CrewAI) are:

- **Agent** — an autonomous unit that executes a specific task (e.g. `ReflectionAgent`, `ToolAgent`).
- **Crew** — a group of agents collaborating on a larger task (mentioned as the framework's north star in the README; not implemented in this codebase yet — there's no `Crew` class here, only individual agents).

Every pattern below is really the same four-step recipe, applied to a different problem:

1. Build a **system prompt** that tells the model exactly what protocol to follow (what tags to emit, what format to use).
2. Keep a **chat history** (a list of `{"role", "content"}` dicts) that the model reads and appends to.
3. **Parse** the model's raw text output for a structured signal (a tag, a JSON blob, a stop-word).
4. **Loop**, feeding the result of step 3 back into the chat history, until some stopping condition fires.

Once you see that shape once, you'll recognize it in every file in this repo.

---

## Chapter 1 — The Reflection Pattern

### The idea: generate, then critique yourself, then revise

A single LLM call answers once and stops — even if the first answer is mediocre. Reflection adds a second "voice" (often the *same* model, prompted differently) whose only job is to criticize the first voice's output. The generator then gets a chance to improve, and the cycle repeats until the critic is satisfied.

### Building it from scratch (`reflection_pattern.ipynb`)

Two independent conversations are kept — one for generation, one for reflection — because they have different system prompts and different "points of view":

```python
generation_chat_history = [
    {"role": "system", "content": "You are a Python programmer tasked with generating high quality Python code..."},
]
generation_chat_history.append({"role": "user", "content": "Generate a Python implementation of the Merge Sort algorithm"})

response = chat(model="llama3.2:3b", messages=generation_chat_history)
merge_sort_code = response.message.content
```

The critique step reuses the *same model*, but under a different persona:

```python
reflection_chat_history = [
    {"role": "system", "content": "You are Andrej Karpathy, an experienced computer scientist. You are tasked with generating critique and recommendations for the user's code"},
]
reflection_chat_history.append({"role": "user", "content": merge_sort_code})

critique = chat(model="llama3.2:3b", messages=reflection_chat_history).message.content
```

The critique is then fed back as a `user` message in the *generation* history, prompting a revised attempt — and the loop can repeat as many times as needed.

> Notice the model is `llama3.2:3b` throughout — this notebook talks to Ollama directly (`from ollama import chat`), even though a commented-out `# client = Groq()` line is left over from the original course material. That leftover is a good marker: **this codebase was adapted from a cloud-model course to run entirely on a local model.**

### The packaged version (`ReflectionAgent`, exercised by `reflection_agent_test.ipynb`)

`src/agentic_patterns/reflection_pattern/reflection_agent.py` turns the notebook's manual loop into a reusable class:

```python
class ReflectionAgent:
    def __init__(self, model: str = "llama3.2:3b"):
        self.client = OllamaClient()
        self.model = model

    def generate(self, generation_history, verbose=0): ...
    def reflect(self, reflection_history, verbose=0): ...

    def run(self, user_msg, generation_system_prompt="", reflection_system_prompt="", n_steps=10, verbose=0) -> str:
        ...
```

Three implementation details matter more than they look:

1. **Two base system prompts are always appended**, regardless of what you pass in:
   ```python
   BASE_GENERATION_SYSTEM_PROMPT = "...You must always output the revised content."
   BASE_REFLECTION_SYSTEM_PROMPT = "...If the user content is ok and there's nothing to change, output this: <OK>"
   ```
   This guarantees the stopping condition (`<OK>`) is always part of the critic's instructions, even if the caller forgets to mention it.

2. **`FixedFirstChatHistory`** (from `utils/completions.py`) is a `list` subclass that behaves like a bounded queue — except it never evicts the *first* message:
   ```python
   class ChatHistory(list):
       def append(self, msg):
           if len(self) == self.total_length:
               self.pop(0)          # drop the oldest message
           super().append(msg)

   class FixedFirstChatHistory(ChatHistory):
       def append(self, msg):
           if len(self) == self.total_length:
               self.pop(1)          # drop the *second* message, keep index 0 pinned
           super().append(msg)
   ```
   This is the trick that keeps the system prompt alive forever in a long-running loop while still bounding token usage — a cheap, dependency-free alternative to LangGraph's summarization node (see the LangGraph workshop's `hands-on.md`, Chapter 2.1, for the heavier-weight version of solving the same problem).

3. **The stop condition is a literal substring check**: `if "<OK>" in critique`. This is a recurring pattern in this codebase — protocols are enforced by *string/tag conventions*, not structured output APIs. It's simple and model-agnostic, but fragile: if the critic model paraphrases instead of emitting the exact token, the loop never stops early and just runs `n_steps` times.

### Try it

```python
from agentic_patterns import ReflectionAgent

agent = ReflectionAgent()   # defaults to llama3.2:3b via Ollama
final_response = agent.run(
    user_msg="Generate a Python implementation of the Insertion Sort algorithm",
    generation_system_prompt="You are a Python programmer tasked with generating high quality Python code",
    reflection_system_prompt="You are Andrej Karpathy, an experienced computer scientist",
    n_steps=1,
    verbose=1,
)
```

With `verbose=1`, `fancy_step_tracker` (from `utils/logging.py`) prints a colorized `STEP 1/1` banner, and each phase is labeled and colored (`GENERATION` in blue, `REFLECTION` in green) — genuinely useful when debugging a multi-step agent loop from the terminal, since it's otherwise hard to tell which "voice" produced which block of text.

---

## Chapter 2 — The Tool Pattern

### The idea: teach the model to ask for help instead of guessing

LLMs can't fetch live weather, query a database, or hit an API — but they can be taught to emit a structured request ("please call `get_current_weather(location="Madrid")`") that *your code* executes on their behalf, feeding the result back in.

### Building it from scratch (`tool_pattern.ipynb`)

Everything hinges on a system prompt that defines a strict wire protocol:

```python
TOOL_SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.
...
For each function call return a json object with function name and arguments within <tool_call></tool_call>
XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>}
</tool_call>

Here are the available tools:
<tools> {...} </tools>
"""
```

The model is expected to reply with something like:

```
<tool_call>
{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}}
</tool_call>
```

...which your code parses with a plain regex, executes, and feeds back as an *observation*:

```python
def parse_tool_call_str(tool_call_str: str):
    clean_tags = re.sub(r'</?tool_call>', '', tool_call_str)
    return json.loads(clean_tags)

result = get_current_weather(**parsed_output["arguments"])

agent_chat_history.append({"role": "user", "content": f"Observation: {result}"})
output_msg = CLIENT.chat(messages=agent_chat_history, model=MODEL).message.content
# -> "The current temperature in Madrid is 25°C."
```

Notice there are **two separate chat histories** here (`tool_chat_history` and `agent_chat_history`) — the tool-calling protocol (with its verbose XML-tag instructions) is kept out of the "clean" conversation the user actually sees, so the final answer doesn't sound like a robot reciting JSON schemas.

### Turning any function into a Tool (`tool.py`)

Three small utilities do all the work:

```python
def get_fn_signature(fn: Callable) -> dict:
    """Introspects a function's __name__, __doc__, and __annotations__ into a JSON-able schema."""
    return {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {"properties": {
            k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"
        }},
    }

def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """Coerces string arguments (e.g. '5') into the annotated type (e.g. int) before calling the function."""
    ...

class Tool:
    def __init__(self, name, fn, fn_signature): ...
    def run(self, **kwargs):
        return self.fn(**kwargs)

def tool(fn: Callable) -> Tool:
    """Decorator: wraps any type-hinted, docstringed function into a Tool automatically."""
    ...
```

The elegance here: **your function's docstring and type hints *are* the schema.** There's no separate Pydantic model or JSON schema to hand-maintain — `get_fn_signature` derives everything by reflection. This is a deliberately minimal stand-in for what `bind_tools()` does automatically in LangChain/LangGraph (see the LangGraph workshop's Chapter 1.7) — same idea, no framework.

> **Caveat, called out directly in the source:** `validate_arguments`'s `type_mapping` only understands `int`, `str`, `bool`, `float` — "This is overly simplified but enough for simple Tools," per the docstring. Don't reach for this for tools with list/dict/Optional arguments without extending it first.

### The packaged version (`ToolAgent`, exercised by `tool_agent_test.ipynb`)

`tool_agent.py` wraps the same protocol into a reusable, multi-tool-capable class:

```python
class ToolAgent:
    def __init__(self, tools: Tool | list[Tool], model: str = "llama3.2:3b"):
        self.client = OllamaClient()
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool = self.tools_dict[tool_call["name"]]
            validated_tool_call = validate_arguments(tool_call, json.loads(tool.fn_signature))
            result = tool.run(**validated_tool_call["arguments"])
            observations[validated_tool_call["id"]] = result
        return observations

    def run(self, user_msg: str) -> str:
        ...
        tool_call_response = completions_create(self.client, tool_chat_history, self.model)
        tool_calls = extract_tag_content(str(tool_call_response), "tool_call")
        if tool_calls.found:
            observations = self.process_tool_calls(tool_calls.content)
            update_chat_history(agent_chat_history, f'f"Observation: {observations}"', "user")
        return completions_create(self.client, agent_chat_history, self.model)
```

`extract_tag_content` (from `utils/extraction.py`) is the general-purpose version of the regex you saw in the from-scratch notebook — it finds *every* occurrence of a tag (not just the first) and returns a `TagContentResult(content=[...], found=bool)`, which is how `ToolAgent` supports multiple tool calls in a single model turn.

### Try it — and where the small local model actually breaks

```python
from agentic_patterns.tool_pattern.tool import tool
from agentic_patterns.tool_pattern.tool_agent import ToolAgent

def fetch_top_hacker_news_stories(top_n: int):
    """Fetch the top stories from Hacker News. ... Args: top_n (int): ..."""
    ...

hn_tool = tool(fetch_top_hacker_news_stories)   # wraps the plain function into a Tool
tool_agent = ToolAgent(tools=[hn_tool])

output = tool_agent.run(user_msg="Tell me the top 5 Hacker News stories right now")
```

In the actual recorded run (`tool_agent_test.ipynb`), the tool call fires correctly and returns real, live Hacker News titles/URLs as the `Tool result` — but the **final answer still says** *"I don't have real-time access to the latest headlines"* and invents a list of hypothetical story ideas instead of using the observation it was just given.

> **Lesson worth internalizing:** the plumbing worked perfectly (prompt → tool call → execution → observation, all correct). The failure is the model itself — `llama3.2:3b` is small enough that it doesn't reliably *use* the observation it's handed; it falls back on its training-time instincts ("I don't have real-time access") even when the current turn's context clearly contradicts that. This is exactly why frontier hosted models (or LangGraph's `tools_condition`/`ToolNode`, paired with a stronger model) are usually paired with production agents — the *pattern* here is correct and portable, but pattern correctness doesn't guarantee a small model will follow it faithfully. Always inspect the final output, not just whether the tool executed.

---

## Chapter 3 — The Planning Pattern (ReAct: Thought → Action → Observation)

### The idea: don't stop after one tool call — loop until the task is actually done

The Tool pattern (Chapter 2) calls **one** tool and returns. Real tasks are often multi-step ("sum these, then multiply, then take the log of the result") — each step's output feeds the next step's input, and the *agent itself* (not your code) has to decide how many steps are needed. That's ReAct, and unlike the previous two patterns, this repo never wraps it in a class — `planning_pattern.ipynb` is the full implementation, hand-rolled and run cell by cell.

### The protocol

The system prompt is an evolution of the Tool pattern's prompt, extended with an explicit worked example of the full cycle:

```python
REACT_SYSTEM_PROMPT = """
You are a function calling AI model. You operate by running a loop with the following steps: Thought, Action, Observation.
...
Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>
"""
```

Giving the model a **worked example inline in the system prompt** (few-shot prompting) is doing a lot of the heavy lifting here — it's how a model with no native "ReAct mode" learns the exact tag vocabulary (`<thought>`, `<tool_call>`, `<observation>`, `<response>`) it's expected to use.

### The loop, by hand

```python
chat_history = [
    {"role": "system", "content": REACT_SYSTEM_PROMPT},
    {"role": "user", "content": f"<question>{USER_QUESTION}</question>"},
]

while True:
    output = CLIENT.chat(messages=chat_history, model=MODEL).message.content
    chat_history.append({"role": "assistant", "content": output})

    tool_call = extract_tag_content(output, tag="tool_call")
    if not tool_call.found:
        break   # model emitted <response>...</response> instead — we're done

    tool_call = json.loads(tool_call.content[0])
    tool_result = available_tools[tool_call["name"]].run(**tool_call["arguments"])

    chat_history.append({"role": "user", "content": f"<observation>{tool_result}</observation>"})
```

(The notebook itself runs this manually, cell by cell, rather than in an actual Python `while` loop — useful for teaching, since you can inspect `chat_history` after every single step — but the logic above is the loop it's manually unrolling.)

### Where it goes wrong — and why that's the whole point of running this locally

The recorded run asks: *"I want to calculate the sum of 1234 and 5678 and multiply the result by 5. Then, I want to take the logarithm of this result."* Watch what `llama3.2:3b` actually does:

1. **Step 1** — the model emits *three* tool calls at once, chaining its own guessed intermediate values instead of waiting for real ones:
   ```
   sum_two_elements(1234, 5678, id=1)
   multiply_two_elements(6272, 5, id=2)      # 6272 is NOT 1234+5678 (=6912) — the model guessed wrong
   compute_log(31360, id=3)                   # built on the wrong guess
   ```
2. The harness (correctly, per the ReAct protocol) only executes the **first** tool call and returns the real observation: `<observation>6912</observation>` (the correct sum).
3. **Step 2** — instead of now multiplying `6912 * 5`, the model calls `multiply_two_elements(6912, 5)` — good! — but *also* pre-emptively calls `compute_log(34800)`, again guessing a product (`34800`) that doesn't match what `6912 * 5` actually is (`34560`).
4. This mismatch-and-recover cycle repeats for several more turns, and eventually the notebook hits `IndexError: list index out of range` — the model emitted a `<response>` with no `<tool_call>` at a point in the manual loop where the next cell still assumed one would be present.

> **This is the single most important lesson in this repo:** the ReAct *protocol* is sound — it's the same protocol LangGraph's `tools_condition`/`ToolNode` loop automates for you (see the LangGraph workshop's Chapter 1.11). But a 3B-parameter local model is not reliably capable of the discipline the protocol demands: waiting for an observation before computing the next step, and never guessing an intermediate result. When you build agents against small/local models via Ollama, budget for this — either by (a) using a larger local model for anything requiring multi-step reasoning, (b) hardening your harness so it *only* ever executes the first tool call per turn and re-prompts rather than trusting a batch of chained calls, or (c) falling back to a hosted frontier model for the planning/reasoning step even if execution stays local. The failure mode you'll see is almost always this exact one: **the model narrates the next step correctly in words, but computes with a number it invented instead of the number the tool actually returned.**

---

## Chapter 4 — Running everything locally with Ollama

Every agent in this repo — from-scratch notebooks and packaged classes alike — talks to the same thing: a local [Ollama](https://ollama.com) server, via the `ollama` Python package, using the model `llama3.2:3b`.

```python
from ollama import Client as OllamaClient
# or, for one-off calls:
from ollama import chat, ChatResponse

client = OllamaClient()
response = client.chat(messages=[...], model="llama3.2:3b")
content = response.message.content   # or response["message"]["content"], both work
```

`utils/completions.py`'s `completions_create` is the one seam every agent class calls through:

```python
def completions_create(client, messages: list, model: str) -> str:
    response = client.chat(messages=messages, model=model)
    return str(response["message"]["content"])
```

Because every agent (`ReflectionAgent`, `ToolAgent`) calls through this single function, **swapping the backend is a one-line change** — replace `OllamaClient()` with a Groq/OpenAI/Anthropic client that exposes a compatible `.chat(messages=..., model=...)` interface, and every pattern in this repo keeps working unmodified. (You can see the fossil of this flexibility in the commented-out `# from groq import Groq` / `# client = Groq()` lines left in several files — this course was originally written against Groq's hosted API and was re-pointed at local Ollama models for these runs.)

**Prerequisites to actually run any notebook in this repo:**

```bash
# 1. Install Ollama itself (see https://ollama.com/download)
# 2. Pull the model these notebooks use
ollama pull llama3.2:3b
# 3. Make sure the Ollama server is running (it usually auto-starts)
ollama serve
```

No API key, `.env` file, or network access is required for the model calls themselves — this is the appeal of the local-model setup: fast iteration, no per-token cost, no rate limits. The trade-off is exactly what Chapters 2 and 3 demonstrated: a 3B model is small enough to break the very protocols it's being asked to follow.

---

## Cheat Sheet — Concept → Where it lives → When to reach for it

| Concept | Where it lives | Use it when... |
|---|---|---|
| Generate → critique → revise loop | `ReflectionAgent.run()` | Output quality matters more than latency (code generation, writing, anything with an objective "better version"). |
| Bounded chat history that never forgets the system prompt | `FixedFirstChatHistory` | Any long-running loop where you must cap tokens but can't lose the instructions. |
| String-based stop condition | `"<OK>" in critique` | Simple, model-agnostic looping — but verify the model reliably emits the literal token, especially on weak models. |
| Turning a function into a callable schema | `@tool` decorator / `get_fn_signature` | You want the model to be able to invoke a specific Python function; docstring + type hints become the schema automatically. |
| Coercing string args to typed args | `validate_arguments` | Model output is JSON-ish but not type-safe (e.g. `"5"` instead of `5`) — only handles `int/str/bool/float`. |
| One-shot tool calling (no loop) | `ToolAgent.run()` | The task needs at most one tool call before answering. |
| Multi-step tool calling with reasoning | Manual ReAct loop (`planning_pattern.ipynb`) | The task requires chaining multiple tool results together — and you're prepared to harden the loop against a model that "computes ahead" instead of waiting for real observations. |
| Extracting structured data from raw LLM text | `extract_tag_content(text, tag)` | Your protocol is XML-tag-based rather than a native structured-output API. |
| Colorized step-by-step tracing | `utils/logging.py` (`fancy_print`, `fancy_step_tracker`) | Debugging a multi-turn agent loop from the terminal. |
| Swappable LLM backend | `completions_create(client, messages, model)` | You want every agent class to work with Ollama today and a hosted API tomorrow, without touching agent logic. |

---

## Closing note: how this repo relates to the LangGraph workshop

If you've also worked through `../langgraph-workshop/hands-on.md`, you've now seen the same three ideas twice, at two different altitudes:

- **Tool calling** — here, a hand-rolled prompt + regex parser + `Tool` class; in LangGraph, `bind_tools()` + `ToolNode` + `tools_condition`.
- **ReAct looping** — here, a manual `while` loop with hand-parsed `<tool_call>`/`<observation>` tags; in LangGraph, the `assistant ⇄ tools` conditional-edge loop.
- **Trimming a growing conversation** — here, `FixedFirstChatHistory`; in LangGraph, a summarization node + `RemoveMessage`.

The LangGraph abstractions exist precisely to remove the hand-rolled plumbing you've now seen line-by-line in this repo — and to add the things a from-scratch implementation doesn't give you for free: checkpointed persistence, visual graph inspection, conditional routing across many nodes instead of one linear loop, and (crucially, per Chapters 2–3 above) far more robust handling of a model that doesn't perfectly follow protocol. When picking which approach to reach for: build it by hand (like this repo) when you want to deeply understand or tightly control the loop with minimal dependencies and a local model; reach for LangGraph when the agent needs to scale past one loop, or when you need the reliability guarantees a small local model can't provide on its own.
