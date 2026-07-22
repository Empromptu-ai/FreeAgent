# context_architect

A standalone Python library that reworks a coding agent's context at every turn
boundary. Instead of carrying the full conversation history forward, it rewrites
history into a compact, labeled summary — while stashing the full detail on disk
so **nothing is truly lost** and the agent can pull it back on demand.

It is **lossy in view, losslessly recoverable.**

---

# Setup — exact steps

The proxy is an **Ollama device**: it forwards OpenCode's requests to an Ollama
server and uses that same server for its own summary/label calls. Pick one of
the two flows below. Both end with `./install_and_run`, which is idempotent —
re-run it any time.

## A. Local Ollama (everything gets set up on this machine)

```sh
cp .env.example .env        # optional — install_and_run does this for you
./install_and_run
```

That single command will: create the venv, install the Python deps, install
Ollama (if missing) and start it, pull the model in `CA_MODEL`, install
OpenCode, write `~/.config/opencode/opencode.json`, place the `recall_turn`
tool, and launch the proxy on `127.0.0.1:49786`.

Then, in OpenCode, run `/models` and pick **"Ollama/ (via context_architect)" / your-model-name (via context_architect)**.

## B. Remote Ollama (model runs on another machine)

On the **remote** machine, run Ollama so it listens on the network and pull the
model:

```sh
OLLAMA_HOST=0.0.0.0:11434 ollama serve      # bind all interfaces
ollama pull qwen3.6:35b                      # the model you'll use
```

On **this** machine, edit `.env` before running the installer:

```sh
cp .env.example .env
# then set these in .env:
#   OLLAMA_BASE_URL=http://<remote-host>:11434
#   CA_SKIP_OLLAMA=1        # don't install/start a local Ollama
#   CA_MODEL=qwen3.6:35b    # must already be pulled on the remote
./install_and_run
```

`CA_SKIP_OLLAMA=1` skips the local Ollama install/serve/pull but still installs
the Python deps and everything else. The proxy still runs locally on
`127.0.0.1:49786`; only the model calls go to the remote.

> **Note:** the proxy sends no auth header, so the remote Ollama must be
> reachable without a token (LAN / VPN / SSH tunnel). Hosted APIs that require a
> key (OpenAI, Anthropic) are **not** reachable through the proxy today.

## Changing the port

Set `CA_PROXY_PORT` (and/or `CA_PROXY_HOST`) in `.env` and re-run
`./install_and_run`. The generated `opencode.json` and every existing
`recall_turn.ts` are re-synced to the new address automatically — only the
`/recall` URL is rewritten, the tool definition itself is left untouched.







# The Longer how-to versions: 
---

## Simple Quickstart:
./install_and_run

### Use the install script (Assumes you have an Ollama server locally):
./install_ca.sh

This will do the following:
1. venv — creates venv/ if absent (same as run_ca_proxy.sh)
2. deps — pip installs library + fastapi/uvicorn/httpx (skippable with CA_SKIP_INSTALL=1)
3. .env — copies .env.example → .env (skipped if .env already exists; use --force to overwrite)
4. Ollama — pulls the model from CA_MODEL (defaults qwen3.6:35b)
5. OpenCode — installs via curl if not present
6. opencode.json — injects the context-architect/qwen3.6:35b provider block into your existing config (doesn't clobber other providers)
7. recall_turn.ts — places it in whichever tools/ directory exists

### Then start up Ollama (only if you need to - just make sure it's running):
ollama serve  # only if you need to - just make sure it's running

### Start up the proxy server:
./run_ca_proxy.sh 

# And run Opencode
Start up your favorite opencode connection and choose "Qwen3.6 35B (reworked) Ollama" as the model (using the /models command in opencode).  





## What happens each turn

When your agent finishes a turn and hands the full context to the library, it:

1. **Classifies** the context into four buckets — pinned setup/tools, prior
   summaries, the file ledger, and this turn's new activity.
2. **Summarizes** the new activity into a `label` + 1–2 sentence note (one LLM
   call) and **archives** the full, uncompressed turn to disk under a stable key.
3. **Updates a file ledger** — detects files read/written this turn and refines a
   one-line description per file (understanding accumulates, it is not overwritten).
4. **Reassembles** a canonical compact history: `pinned → older summaries →
   recent full-text turns → file ledger`. The most recent
   `num_full_text_turns` completed turns (default **3**) are kept as their full
   text — the text messages only, with tool use and file reads stripped —
   instead of a summary; every older turn falls back to its summary. Each turn
   is still summarized and archived regardless, so a turn aging out of the
   window is demoted to its already-computed summary with no extra LLM call.
   Set `num_full_text_turns=0` (env: `CA_NUM_FULL_TEXT_TURNS=0`) for the classic
   summarize-every-turn behavior.
5. **Persists** everything keyed to the session id so resume/fork reproduce state.
6. **Audits** exactly what went in and came out to `audit.log`.

A **recall tool** (`recall_turn`) lets the agent fetch the full detail of any past
turn by its archive key when a summary isn't enough.

## Install

```bash
pip install -e .                    # core (Ollama via stdlib, no extra deps)
pip install -e ".[anthropic]"      # + Anthropic backend
pip install -e ".[openai]"         # + OpenAI backend
```

## Quick start

```python
from context_architect import ContextArchitect, Config, LLMConfig

ca = ContextArchitect(Config(
    storage_root="~/.context_architect",
    llm=LLMConfig(provider="ollama", base_url="http://localhost:11434", model="llama3.1"),
    # or LLMConfig(provider="anthropic", model="claude-sonnet-5", api_key=...)
    # or LLMConfig(provider="openai", model="gpt-4o", api_key=...)
))

session = ca.session("my-session-id")   # create or resume by id
```

At the end of each turn, hand the library the full context and install what it
returns as the new live context.

### Provider-native messages (easiest)

```python
# Anthropic shape -> returns (system, messages)
system, messages = session.rework_native(messages, fmt="anthropic", system=system)

# OpenAI shape -> returns messages
messages = session.rework_native(messages, fmt="openai")
```

### Normalized messages

```python
new_context = session.rework(messages, tools=tools)          # sync
new_context = await session.arework(messages, tools=tools)   # async
```

## Recall tool wiring

```python
schema = ca.recall_tool_schema(fmt="anthropic")   # register with your host

# when the agent calls recall_turn(key=...):
tool_result_text = session.recall(key)
```

Every summary in the history ends with `(recall: turn-0003)` so the model knows
the key to ask for.

## Resume & fork

```python
ca.session("my-session-id")     # resuming reconstructs the exact compact state
session.fork("branch-session")  # deep-copies archive + state for a new branch
```

## Storage layout

```
{storage_root}/{session_id}/
  archive/turn-0001.json   full raw turn payloads (recall targets)
  state.json               compact live history + file ledger + turn counter
  audit.log                append-only JSONL: inputs, prompts, outputs per turn
```

## LLM backends

| provider   | config                                  | notes                    |
|------------|-----------------------------------------|--------------------------|
| `ollama`   | `base_url`, `model`                      | stdlib HTTP, no API key  |
| `anthropic`| `model`, `api_key` / `ANTHROPIC_API_KEY`| extra `[anthropic]`      |
| `openai`   | `model`, `api_key` / `OPENAI_API_KEY`   | extra `[openai]`         |
| `fake`     | —                                       | deterministic, for tests |

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Tests run fully offline against the `fake` backend.

## How file detection works

Detection is heuristic. Built-in tool-name sets cover common hosts:

- `tool_use` blocks whose name matches known **write** tools (`apply_patch`,
  `str_replace_editor`, `edit`, `write`, `notebook_edit`, …) or **read** tools
  (`read_file`, `view`, `open`, …); the path is pulled from common input keys in
  both snake_case and camelCase (`path`, `file_path`, `filePath`, `target_file`,
  …).
- **shell** tool commands beginning with a read-only viewer (`cat`, `head`,
  `grep`, `sed -n`, …) count their file arguments as reads. Mutating shell
  commands are ignored (writes surface through edit tools).

If your host exposes edit/read/shell tools under names the defaults don't
recognize, register them on `Config` (matched case-insensitively, merged with
the built-ins):

```python
Config(
    extra_write_tools={"MyEditor"},
    extra_read_tools={"my_reader"},
    extra_shell_tools={"my_shell"},
    extra_path_keys={"srcPath"},   # unusual key your tool puts the path under
)
```

## Integrating with OpenCode + local Ollama

[OpenCode](https://opencode.ai) is a TypeScript/Bun terminal agent; this library
is Python. Rather than fork OpenCode, the clean bridge is a small
**OpenAI-compatible proxy** that sits *between* OpenCode and Ollama:

```
OpenCode ──/v1/chat/completions──▶  ca-proxy (context_architect)  ──▶  Ollama
   │                                      │  compacts completed history
   └──── recall_turn tool ──────────────▶ │  serves /recall from the archive
```

OpenCode points its provider at the proxy. On each request the proxy splits the
transcript at the **last user message**: the in-flight turn from there on is
forwarded to Ollama *verbatim* (so the model actually answers it), while every
completed turn before it is rewritten into a `context_architect` summary + file
ledger and archived to disk. A custom OpenCode tool (`recall_turn`) calls the
proxy's `/recall` endpoint to pull any turn's full detail back.

> **Why a proxy and not a plugin?** OpenCode owns its message store and resends
> the full transcript each request; its plugin hooks are mostly *events* and
> don't let you substitute the outbound transcript.
>
> **Why keep the in-flight turn verbatim?** A completion proxy fires *before*
> the model answers, so the current turn is the request being answered, not
> finished history — summarizing it would hand the model a description of the
> question instead of the question. So the current turn is sent verbatim and gets
> summarized once the next turn arrives. Prior turns are always summaries (from
> turn 1), matching the library's contract, with a one-turn lag inherent to
> intercepting before the response exists.

### 1. Install and prime Ollama

```bash
# https://ollama.com/download  (or: brew install ollama)
ollama serve                       # starts the server on :11434
ollama pull qwen3.6:35b            # 262k context window — no Modelfile needed
```

OpenCode wants a large context window and Ollama defaults to 4096 tokens, but
`qwen3.6:35b` carries a 262k window, so no `num_ctx` Modelfile variant is needed
here.

### 2. Install OpenCode

```bash
curl -fsSL https://opencode.ai/install | bash   # or: npm i -g opencode-ai
```

### 3. Run the context_architect proxy

A ready-to-run proxy lives at [`examples/ca_proxy.py`](examples/ca_proxy.py). It
keeps the in-flight turn verbatim, rewrites every completed turn behind it into a
summary + ledger, passes OpenCode's tool-less auxiliary calls (title / summary
generation) straight through, forwards to Ollama (streaming included), and
serves `/recall`.

```bash
pip install -e .                    # this library
pip install fastapi uvicorn httpx   # proxy deps

uvicorn examples.ca_proxy:app --port 49786
```

Configure it via environment variables (or just copy .env.example to .env and change whatever you want):

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export CA_MODEL=qwen3.6:35b          # single source of truth: main agent loop + summary/label + ledger
#export CA_MAIN_MODEL=qwen3.6:35b     # optional: run the agent on a different model than the summarizer
#export CA_REASONING=medium           # optional: agent-loop reasoning effort (off|low|medium|high); unset = model default
#export CA_SUMM_REASONING=off         # optional: reasoning for the internal summary/ledger calls; default OFF (see note)
export CA_STORAGE_ROOT=~/.context_architect
export CA_NUM_FULL_TEXT_TURNS=2      # keep the last N turns as full text (0 = summaries only)
export CA_TOOLS_DENY=glob            # drop these tools from the host's set (default: glob); empty = pass all
#export CA_TOOLS_ALLOW=read,edit,grep # keep ONLY these (allowlist); wins over CA_TOOLS_DENY
export CA_AUDIT_OUTBOUND=1           # dump the exact main-LLM input per turn
export CA_AUDIT_INBOUND=1            # dump the exact main-LLM responses per turn
export CA_AUDIT_FULL=1               # dump the complete interleaved turn transcript

export CA_SYSTEM_OVERRIDE=1                              # master switch: replace the host's system prompt
export CA_SYSTEM_PROMPT_FILE=./system_prompt/my_prompt.md  # your prompt (file wins over CA_SYSTEM_PROMPT)
export CA_SYSTEM_MODE=replace                           # replace | prefix | suffix
```

### Overriding the system prompt

OpenCode resends its own system prompt on every request. The proxy can swap it
for one of yours before forwarding to the model — the substitution touches only
the **main agent loop**; OpenCode's auxiliary title/summary prompts and the
**tool definitions are always forwarded untouched**, so tool calling keeps
working exactly as before.

The override is gated behind a master switch so you can keep a prompt configured
and flip it on/off:

- `CA_SYSTEM_OVERRIDE=1` — turns the override on (any other value = off).
- `CA_SYSTEM_PROMPT_FILE` — path to your prompt (takes precedence; best for
  multi-line prompts). Or `CA_SYSTEM_PROMPT` for a short inline string. A
  relative path is resolved against the repo root (not the launch cwd), so the
  bundled [`system_prompt/my_system_prompt.md`](system_prompt/my_system_prompt.md)
  works out of the box; `~` and absolute paths are honored too.
- `CA_SYSTEM_MODE` — how your text combines with the host's leading `system`
  run: `replace` (default, swap it out), `prefix` (yours, then OpenCode's), or
  `suffix` (OpenCode's, then yours). `prefix`/`suffix` are useful when you want
  to keep OpenCode's environment/tool notes while adding your own guidance.

The startup banner prints whether the override is on and where the prompt came
from. The whole leading run of `system` messages is collapsed into your single
override, which then flows through as the library's normal pinned setup.

### 4. Point OpenCode at the proxy

Edit `~/.config/opencode/opencode.json` (or `.jsonc`) so a provider's `baseURL`
points at the **proxy**, not Ollama directly. OpenCode sends its own session id
as the `x-session-id` header, so each OpenCode conversation gets its own archive
automatically; the static header below is just a fallback for other clients.

> **Do not name the provider `ollama`.** That id collides with OpenCode's
> built-in Ollama provider, which makes it ignore your `baseURL` and connect
> straight to `:11434` — bypassing the proxy while still showing your label.
> Use a distinct id like `context-architect`.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "context-architect": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (via context_architect)",
      "options": {
        "baseURL": "http://localhost:49786/v1",
        "headers": { "x-session-id": "opencode" }
      },
      "models": { "qwen3.6:35b": { "name": "Qwen3.6 35B (reworked)" } }
    }
  }
}
```

The model then appears in `/models` under provider id
`context-architect/qwen3.6:35b`.

> **Note:** the model id in this `models` block is just a cosmetic label. The
> proxy stamps `CA_MODEL` (or `CA_MAIN_MODEL`) onto every request, so the model
> Ollama actually runs is chosen by your `.env` — change the model there and
> restart the proxy; you never need to edit this block again. When
> `CA_REASONING` is set, the proxy likewise stamps the reasoning effort onto
> every main-agent request as `reasoning_effort`; leave it unset to inherit
> whatever the model does by default.
>
> **Reasoning applies to the agent only, not the summarizer.** The internal
> summary/label/file-ledger calls run *blocking, before the agent is called*,
> and their thinking is discarded — so they default to no extra reasoning even
> when `CA_REASONING` is on. Turning it up there (`CA_SUMM_REASONING`) only adds
> latency and timeout risk on the turn's critical path. A slow or failed
> summarization no longer fails the turn (it's caught and retried next request),
> but it will still stall it — so leave `CA_SUMM_REASONING` unset unless you have
> a specific reason.

### 5. Add the recall tool

OpenCode reads custom tools from `~/.config/opencode/tools/` (global) or
`.opencode/tools/` (per project). The **filename becomes the tool name**, so
save this as `recall_turn.ts`:

```typescript
import { tool } from "@opencode-ai/plugin"

export default tool({
  description:
    "Fetch the full, uncompressed detail of an earlier turn by its archive " +
    "key. Summaries in the history end with '(recall: turn-NNNN)' — pass that " +
    "key here when a summary isn't enough.",
  args: {
    key: tool.schema.string().describe("Archive key, e.g. 'turn-0003'"),
  },
  async execute(args, context) {
    // context.sessionID matches the x-session-id the proxy keyed the archive on.
    const res = await fetch("http://localhost:49786/recall", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session: context.sessionID, key: args.key }),
    })
    const data = await res.json()
    return data.text ?? "No archived turn found."
  },
})
```

### 6. Run it

```bash
ollama serve  # only if you need to - just make sure it's running

uvicorn examples.ca_proxy:app --port 49786  # The long/specific way
OR:
./run_ca_proxy.sh # And just copy .env.example to .env to load the env variables

Then start up your favorite opencode connection and choose "Qwen3.6 35B (reworked) Ollama" as the model (this is literally the /models command in opencode).  
```

the example-case command ./run_ca_proxy.sh is meant as the one-command "set up and run". It does the following:
 - creates venv/ if it doesn't exist,
 - installs the library + fastapi uvicorn httpx (skippable with CA_SKIP_INSTALL=1 for fast restarts),
 - loads .env with set -a so every variable is exported into the environment the proxy inherits,
 - execs uvicorn on the host/port from .env (defaults 127.0.0.1:49786).


As you work, `~/.context_architect/<session-id>/` fills with `archive/`,
`state.json`, and `audit.log` (the proxy logs the session id and target path on
each call). When a summary isn't enough, the model calls `recall_turn` with the
`turn-NNNN` key and gets the full turn back.

### Caveats & tuning

- **One-turn lag.** Because the proxy runs before the model answers, a turn is
  summarized when the *next* turn arrives — so the first `turn-0001.json` appears
  once the second exchange starts, not during the first.
- **Recency window (`CA_NUM_FULL_TEXT_TURNS`, default 2).** The last N completed
  turns are kept in the live context as their full text (text messages only —
  tool use and file reads are stripped, since those surface through the file
  ledger and are recoverable via `recall_turn`); older turns are summaries. Turns
  are still summarized/archived every turn, so this only changes what the model
  *sees*, not what's stored. Combined with the one-turn lag above, the in-flight
  turn is verbatim and the N turns behind it are full text. Set to `0` for the
  original summarize-every-turn behavior.
- **Tool filtering (`CA_TOOLS_DENY`, default `glob` / `CA_TOOLS_ALLOW`).** The
  proxy can trim the host's tool set before it reaches the model. `CA_TOOLS_DENY`
  is a comma-separated denylist (e.g. `glob,ls`) and drops those tools; it
  defaults to `glob`, and setting it empty (`CA_TOOLS_DENY=`) passes every tool
  through. `CA_TOOLS_ALLOW` is a comma-separated allowlist that keeps *only* the
  named tools and takes precedence when both are set — stricter, but riskier: if
  it removes a tool the host still references in a prior `tool_call`, some
  backends error on the dangling reference, so prefer the denylist unless you
  need a hard whitelist. When a filter drops anything the proxy logs a `[tools]`
  line, and a `tool_choice` that pinned a removed tool is reset to `auto`. Only
  the main agent loop is affected; aux title/summary calls carry no tools.
- **`CA_AUDIT_OUTBOUND=1`** writes the exact messages sent to the main model at
  the start of each turn to `{root}/{session}/turn-NNN-msgs_to_main_llm.json`
  (just the messages array, one file per turn, written on the turn's first call
  — tool-loop continuations don't overwrite it).
- **`CA_AUDIT_INBOUND=1`** writes the exact messages the main model returned
  *during* each turn to `{root}/{session}/turn-NNN-msgs_from_main_llm.json` — one
  entry per completion call in the turn's tool loop, in order, reconstructed from
  the (possibly streamed) response without altering what the host receives.
- **`CA_AUDIT_FULL=1`** writes the complete interleaved turn to
  `{root}/{session}/turn-NNN-full_transcript.json` — the in-flight messages, tool
  calls, tool results, and final answer in order. (Tool results are produced by
  the host's tool executor, not the model, so they appear here but not in the
  `msgs_from_main_llm` file.)
- **Turn boundary is heuristic.** The proxy treats everything from the last
  `user` message onward as the in-flight turn and keeps it verbatim. That's the
  right cut for a normal request/tool-loop; unusual transcript shapes may need a
  smarter boundary.
- **Session mapping.** OpenCode sends its session id as `x-session-id`, so each
  conversation is isolated automatically; the recall tool passes the matching
  `context.sessionID`. Non-OpenCode clients fall back to the header default.
- **Fold tracking is in-memory.** `_folded` resets if the proxy restarts, which
  re-folds completed history once (harmless). Persist it if that matters.
- **Extra local calls.** Each fold makes one summary call plus (when files are
  touched) one ledger-refine call against Ollama, on top of the real completion.
- **Streaming** is proxied verbatim in every branch.
- **Register OpenCode's tool names** for accurate file detection — the proxy
  already sets `extra_read_tools={"read"}` / `extra_write_tools={"edit", "write",
  "patch"}`; see *How file detection works* above.

