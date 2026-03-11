# I Ran a 35B AI Coding Agent Locally on My Mac Mini — For Free, Forever

*No API bills. No data leaving your machine. Just a Mac Mini M4, 64 GB of memory, and a weekend of debugging.*

---

I have a Mac Mini M4 with 64 GB of unified memory sitting on my desk. For months I kept paying OpenAI and Anthropic to power my AI coding assistant in VS Code. Then I thought: this machine has more memory than most servers. Why am I sending my code to the cloud?

This is the story of how I got a state-of-the-art 35-billion-parameter AI model running locally, integrated it with Cline (my VS Code AI agent), and fixed a handful of tricky bugs along the way. By the end of this post you'll know exactly how to do the same thing.

Spoiler: it works remarkably well. The model generates tokens at **35 tokens per second** — fast enough for real coding sessions with zero lag between thoughts.

---

## The Model: Qwen3.5-35B-A3B

The model I chose is **Qwen3.5-35B-A3B-4bit** from Alibaba's Qwen team. Let me explain what that mouthful means, because it's actually the key to why this works on a Mac.

"35B" means 35 billion parameters — this is a big, capable model on par with what you'd pay for via API. "A3B" means it's a **Mixture of Experts** (MoE) architecture, where only about 3 billion parameters are actually activated for each token it generates. The rest are dormant. "4bit" means it's been quantized down to 4 bits per parameter, compressing the whole thing to about 20 GB.

The practical result: a model with 35B-class intelligence that uses compute closer to a 3B model. On Apple Silicon, where memory bandwidth is the bottleneck, this translates directly to speed.

For comparison, I also tested a dense **Qwen3-32B-4bit** — a model with 32 billion parameters that are all active all the time. Same memory footprint, totally different performance profile.

Here's what I measured on my M4 Mac Mini with 64 GB, running both models side by side:

**Qwen3.5-35B-A3B-4bit (MoE):**
- Short prompt: 33.7 tokens/second
- Coding task: 37.0 tokens/second
- Long generation: 32.9 tokens/second
- Average: **35.2 tokens/second**

**Qwen3-32B-4bit (Dense):**
- Short prompt: 10.1 tokens/second
- Coding task: 9.4 tokens/second
- Long generation: 9.1 tokens/second
- Average: **10.0 tokens/second**

The MoE model is **3.5× faster** while being smarter. It's not even close. For an AI coding agent where you're waiting on every response, that difference is everything.

---

## The Server: omlx

To serve the model, I used **omlx** — a local LLM inference server built specifically for Apple Silicon. It speaks the OpenAI API protocol, which means any tool that works with ChatGPT can point at it instead. It handles the MLX inference backend, token streaming, KV caching, and a nice web admin panel.

You can find my fork with all the fixes at **github.com/kenhuangus/omlx**.

---

## The Agent: Cline

**Cline** is a VS Code extension that gives you an AI agent inside your editor. Unlike GitHub Copilot which just autocompletes, Cline can read your files, run terminal commands, make multi-step plans, and execute them. It's what "AI-assisted development" actually looks like in practice.

Cline supports any OpenAI-compatible API, which is exactly what omlx provides. In theory, you just plug in the local URL and go. In practice, I hit four different bugs that took a full day of debugging to untangle.

---

## The Bugs (and How I Fixed Them)

The Qwen3.5 model has a "thinking" capability — before answering, it internally reasons inside `<think>...</think>` blocks, similar to how Claude thinks before responding. This is great for quality. It's also a landmine for streaming APIs that weren't designed for it.

Here's every bug I found and fixed.

### Bug 1: Cline Couldn't Even Find the Model

The very first thing Cline does before sending any message is call `GET /v1/models/Qwen3.5-35B-A3B-4bit` to verify the model exists. omlx only had a `GET /v1/models` endpoint (which lists all models) but not the single-model lookup. Every single Cline request failed with a 404 before the model was ever touched.

The fix was straightforward — add the missing endpoint that looks up a model by ID:

```python
@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, _: bool = Depends(verify_api_key)) -> ModelInfo:
    for m in status["models"]:
        if m["id"] == model_id or m.get("alias") == model_id:
            return ModelInfo(id=model_id, owned_by="omlx")
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
```

### Bug 2: Whitespace Leaking Into Tool Calls

Cline is a coding *agent*, which means it doesn't just chat — it uses tools. It asks the model to call functions like `read_file` or `execute_command`, and the model responds with XML-style tool call markup.

The problem: the model would output `\n\n<function=execute_command>...`, and omlx correctly stripped the `<function=...>` XML — but not the `\n\n` that came before it. Cline received a response chunk with `content="\n\n"` alongside the tool call, which broke its parser.

One line fixed it — skip content that's whitespace-only when tools are present:

```python
if content_delta and (not has_tools or content_delta.strip()):
    yield content chunk
```

### Bug 3: Content and Tool Calls Arriving at the Same Time

In streaming mode, tokens arrive one by one. The problem is that the model emits text tokens *before* it's done emitting the tool call XML. Cline was receiving partial content text in one chunk and the tool call in the next chunk — and its parser expected them to be cleanly separated.

The fix: when a request includes tools, stop streaming token by token. Instead, buffer everything, then emit the full assembled response at the end:

```python
stream_content = not has_tools   # live-stream only when no tools involved

if has_tools:
    # accumulate everything, emit once complete
    tool_filter = ToolCallStreamFilter(engine.tokenizer)
```

This adds a tiny bit of latency for tool calls but eliminates an entire class of parsing failures.

### Bug 4: The Thinker That Never Stopped Thinking

This was the hard one. The one that kept me debugging with a network proxy, capturing raw bytes, staring at SSE frames.

Here's what happens with a thinking model:

1. omlx's scheduler prepends `<think>\n` to the model's output
2. The model generates its reasoning inside the `<think>` block — sometimes hundreds of tokens
3. Sometimes the model never generates content *after* `</think>` — it just... stops. The entire response is inside the `<think>` block.

omlx's `ThinkingParser` correctly routes thinking content to `reasoning_content` and regular content to `content`. Cline only reads `content`. So when the model put everything in `<think>`, Cline received a blank message.

I set up a socket-level logging proxy between Cline and omlx to capture exactly what was going over the wire. One request looked like this:

```
content(0 chunks), reasoning_content(470 tokens), finish_reason=stop
```

Zero content. 470 tokens of reasoning. Cline had nothing to work with.

The fix adds a `content_emitted` tracking flag. After the stream ends, if nothing was emitted as regular content, we extract the thinking text and emit it as content — so the client always gets something:

```python
content_emitted = False

# ... (set to True whenever a content chunk is yielded)

if not content_emitted and not has_tools and accumulated_text:
    thinking_text, regular_text = extract_thinking(accumulated_text)
    raw_fallback = regular_text.strip() or thinking_text.strip()
    fallback = re.sub(r'</?think>\n?', '', raw_fallback).strip()
    if fallback:
        yield content chunk with fallback text
```

The `re.sub` at the end handles one more edge case: if the model never closes its `<think>` tag, `extract_thinking` returns the raw text including the opening tag. Stripping those tags before emitting prevents `<think>` from leaking into Cline's response window.

After this fix, I ran a stress test — 10 prompts specifically designed to trigger thinking-only responses. All 10 passed. No empty content, no tag leakage.

---

## How to Set This Up Yourself

You need a Mac with Apple Silicon and at least 32 GB of unified memory. 64 GB is ideal — it fits the model plus a full 128K context window with room to spare.

### Step 1 — Install omlx

Clone my fork, which contains all the fixes:

```bash
git clone https://github.com/kenhuangus/omlx.git
cd omlx
pip install -e ".[all]"
```

### Step 2 — Download the model

```bash
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-MLX-4bit \
  --local-dir ~/.omlx/models/Qwen3.5-35B-A3B-4bit
```

This downloads about 20 GB. Get a coffee.

### Step 3 — Configure omlx

Run `omlx serve` once to create the config file, then edit `~/.omlx/settings.json`:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "model": {
    "model_dirs": ["~/.omlx/models"]
  },
  "sampling": {
    "max_context_window": 131072,
    "max_tokens": 131072
  },
  "auth": {
    "api_key": "pick-a-strong-key"
  }
}
```

Setting `host` to `0.0.0.0` means the API and admin panel are reachable from other devices on your network — useful if your Mac Mini is a headless server. The `131072` context window is 128K tokens, which the model fully supports and your 64 GB has memory for.

### Step 4 — Start the server

```bash
omlx serve --model-dir ~/.omlx/models
```

Check it's up:

```bash
curl http://localhost:8000/health
```

You can also open `http://localhost:8000` in a browser for the admin panel — monitor memory usage, loaded models, and active requests.

### Step 5 — Connect Cline in VS Code

1. Install the **Cline** extension from the VS Code marketplace
2. Click the Cline icon in the sidebar, then the gear icon
3. Set **API Provider** to `OpenAI Compatible`
4. **Base URL**: `http://localhost:8000/v1` — type it carefully, no trailing space
5. **API Key**: whatever you put in `settings.json`
6. **Model ID**: `Qwen3.5-35B-A3B-4bit` — must match the folder name exactly
7. Save and you're done

Open a project and ask Cline to do something. It should respond within a second or two and be able to read files, run commands, and iterate on tasks just like it would with a cloud model.

---

## What It's Like to Use

Honest impressions after using this daily:

It's fast enough. 35 tokens per second means you're not watching a spinner — responses feel nearly instant for short exchanges and comfortable for longer ones. Context switches to a new file or reading a long document are handled by the 290 tokens/second prefill speed, so there's no noticeable delay even on large codebases.

The thinking capability is genuinely useful for complex tasks. When you ask it to refactor something tricky or debug a subtle issue, it works through the problem before answering rather than just pattern-matching. You can see the reasoning in Cline's sidebar.

The model is private. Every token stays on your machine. I work on proprietary code and sending it to cloud APIs has always made me vaguely uncomfortable. This removes that entirely.

The model is free to run. The upfront cost is the Mac Mini, which you already have. After that, it's electricity.

---

## Common Pitfalls

**"Invalid API Response" in Cline** — Use my fork, not the upstream omlx. The upstream doesn't have the thinking-only response fallback. Also double-check the model ID in Cline matches the folder name exactly, and make sure there's no trailing space in the Base URL.

**Model ID mismatch** — The folder name in `~/.omlx/models/` is the model ID. If the folder is `Qwen3.5-35B-A3B-4bit`, type exactly that in Cline. No autocomplete will save you here.

**Port saved wrong** — If you ever run `omlx serve --port 8001` (say, for testing), omlx saves that to `settings.json`. Next time you start it without a `--port` flag, it comes up on 8001. Fix by editing `settings.json` and setting `"port": 8000`.

**Memory pressure** — 64 GB handles the model plus 128K context comfortably. If you're on 32 GB, drop `max_context_window` to `65536` (64K) and close other heavy apps. The model itself needs about 20 GB, which leaves 12 GB for context on a 32 GB machine — enough for most coding sessions.

---

## The Code

Everything is on GitHub:

**github.com/kenhuangus/omlx**

The fork includes all four bug fixes, two benchmark scripts (single-model and side-by-side comparison), and the detailed technical write-up in `MoE_FIXES_REPORT.md` if you want to understand exactly what changed and why.

---

## Final Thoughts

A year ago, running a 35B model locally with good tool-calling support and 128K context would have required a serious GPU workstation. Today it runs on a $1,400 Mac Mini that fits behind a monitor, uses no fan noise worth mentioning, and generates tokens faster than I can read them.

The MoE architecture is what makes it practical. You get the knowledge of a 35B model at the inference cost of a 3B model. Apple Silicon's unified memory means the model weights sit in the same pool as everything else — no PCIe bandwidth tax, no separate VRAM. They were made for each other.

If you have an M4 Mac with 64 GB, this setup gives you a capable, private, free coding agent. The setup takes about an hour including the download. The bugs I fixed are already in the fork. You just have to follow the steps.

Go try it.

---

*The fork with all fixes: [github.com/kenhuangus/omlx](https://github.com/kenhuangus/omlx)*

*Technical deep-dive into the bugs: `MoE_FIXES_REPORT.md` in the repo*

*Step-by-step setup guide: `QUICKSTART_M4_CLINE.md` in the repo*
