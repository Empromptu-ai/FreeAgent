"""OpenAI-compatible proxy that inserts context_architect between a host
(e.g. OpenCode) and a local Ollama server.

    host ──/v1/chat/completions──▶  ca_proxy  ──▶  Ollama
                                       │  rewrites prior turns into summaries
                                       └──▶  /recall serves the archive

Faithful to the library's contract: at every turn boundary the prior turns are
rewritten into compact labeled summaries and each turn's full detail is archived
to disk (recoverable via ``recall_turn``). There is no size threshold — turn 1
is summarized as soon as turn 2 arrives.

The only wrinkle unique to a *proxy*: it is called BEFORE the model answers, so
the current (in-flight) turn cannot be summarized yet — summarizing the pending
request would send the model a description of the question instead of the
question. So the proxy splits each transcript at the last user message:

  * the in-flight turn (from the last user message on) is sent VERBATIM;
  * everything before it is completed history, which is folded into
    context_architect summaries the moment it becomes visible (i.e. on the next
    request). This gives the spec's behavior with a one-turn lag inherent to
    intercepting before the response exists.

Auxiliary host calls (title / summary generation) arrive without a ``tools``
array; those are passed through untouched so they don't become bogus turns.

Run:

    pip install -e ".[openai]"          # or just: pip install -e .
    pip install fastapi uvicorn httpx
    uvicorn examples.ca_proxy:app --port 49786

Environment:
    OLLAMA_BASE_URL   default http://localhost:11434
    CA_MODEL          default qwen3.6:35b   (used for summary/label + ledger,
                      and for the main agent loop unless CA_MAIN_MODEL is set)
    CA_MAIN_MODEL     default = CA_MODEL. The model the main agent loop runs on;
                      the proxy stamps it onto every request so the host's own
                      model id becomes a placeholder. Set this only to run the
                      summarizer on a different model than the agent.
    CA_REASONING      reasoning/thinking effort for the MAIN AGENT loop:
                      off | low | medium | high. Unset -> the model's own
                      default (nothing injected).
    CA_MAIN_REASONING kept-for-compat alias for CA_REASONING (agent loop).
    CA_SUMM_REASONING reasoning effort for the internal summary/label/ledger
                      calls. Defaults to OFF even when the agent uses reasoning:
                      these run blocking before the agent and discard their
                      thinking, so turning it up only adds latency and timeout
                      risk. Set it only if you specifically want it.
    CA_STORAGE_ROOT   default ~/.context_architect
    CA_TOOLS_DENY     comma-separated tool names to drop from the host's tool set
                      before it reaches the model. Defaults to "glob"; set it
                      empty (CA_TOOLS_DENY=) to pass every tool through.
    CA_TOOLS_ALLOW    comma-separated tool names to keep (allowlist). When set it
                      wins over CA_TOOLS_DENY. Stricter but riskier: removing a
                      tool the host still references in a prior tool_call can make
                      some backends error — prefer CA_TOOLS_DENY unless you need a
                      hard whitelist.
    CA_AUDIT_OUTBOUND set to 1 to dump the exact messages sent to the main model
                      at the start of each turn to
                      {root}/{session}/turn-NNN-msgs_to_main_llm.json
    CA_AUDIT_INBOUND  set to 1 to dump the exact messages the main model returned
                      during each turn (all tool-loop responses, in order) to
                      {root}/{session}/turn-NNN-msgs_from_main_llm.json
    CA_AUDIT_FULL     set to 1 to dump the complete interleaved turn (in-flight
                      messages + tool calls + tool results + final answer) to
                      {root}/{session}/turn-NNN-full_transcript.json
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from context_architect import Config, ContextArchitect, LLMConfig
from context_architect.adapters import openai as oai
from context_architect.llm.reasoning import normalize as _norm_reasoning
from context_architect.llm.reasoning import params_for as _reasoning_params

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("CA_MODEL", "qwen3.6:35b")
# The model the *main agent loop* runs on. Defaults to CA_MODEL so a single
# CA_MODEL env var drives everything; set CA_MAIN_MODEL only if you want the
# summary/label calls (CA_MODEL) to use a different model than the main agent.
# The proxy stamps this onto every main-agent request, so whatever model id the
# host (OpenCode) has configured becomes a cosmetic placeholder — change the
# model here, in one place, and restart the proxy.
MAIN_MODEL = os.environ.get("CA_MAIN_MODEL", MODEL)

# Reasoning/thinking effort for the MAIN AGENT loop (the model that answers the
# user). This is what CA_REASONING controls; CA_MAIN_REASONING is a kept-for-
# compat alias. Unset -> nothing injected -> the model's own default.
AGENT_REASONING = _norm_reasoning(
    os.environ.get("CA_MAIN_REASONING") or os.environ.get("CA_REASONING")
)
# Reasoning effort for the internal summary/label/file-ledger calls. These are
# mechanical JSON extractions that run *blocking, on the turn's critical path*
# (before the agent is even called) and whose thinking tokens are discarded — so
# turning reasoning up here is nearly all cost (latency, and timeouts that would
# surface as errors) for no benefit. It therefore defaults to OFF even when the
# agent runs with reasoning on; opt in explicitly with CA_SUMM_REASONING only if
# you have a specific reason to.
SUMM_REASONING = _norm_reasoning(os.environ.get("CA_SUMM_REASONING", "off"))
STORAGE_ROOT = os.environ.get("CA_STORAGE_ROOT", "~/.context_architect")
AUDIT_OUTBOUND = os.environ.get("CA_AUDIT_OUTBOUND") == "1"
AUDIT_INBOUND = os.environ.get("CA_AUDIT_INBOUND") == "1"
AUDIT_FULL = os.environ.get("CA_AUDIT_FULL") == "1"
NUM_FULL_TEXT_TURNS = int(os.environ.get("CA_NUM_FULL_TEXT_TURNS", "2"))

# --- System-prompt override -------------------------------------------------
# Master switch: the override only applies when CA_SYSTEM_OVERRIDE=1, so you can
# keep a prompt configured and toggle it on/off without deleting it.
SYSTEM_OVERRIDE = os.environ.get("CA_SYSTEM_OVERRIDE") == "1"
# The replacement text. CA_SYSTEM_PROMPT_FILE (a path) takes precedence over the
# inline CA_SYSTEM_PROMPT. A relative path (e.g. ./system_prompt/foo.md) is
# resolved against the repo root, not the cwd the proxy was launched from, so
# in-repo prompt files work no matter where run_ca_proxy.sh is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sp_file = os.environ.get("CA_SYSTEM_PROMPT_FILE")


def _resolve_prompt_path(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (_REPO_ROOT / q)


SYSTEM_PROMPT = (
    _resolve_prompt_path(_sp_file).read_text()
    if _sp_file
    else os.environ.get("CA_SYSTEM_PROMPT")
)
# How the override combines with the host's own system prompt:
#   replace : swap the whole leading system run for yours (default)
#   prefix  : your text, then the host's system prompt
#   suffix  : the host's system prompt, then your text
SYSTEM_MODE = os.environ.get("CA_SYSTEM_MODE", "replace")


def _apply_system_override(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite the leading contiguous run of ``system`` messages using the
    configured override. No-op unless CA_SYSTEM_OVERRIDE=1 and a prompt is set.

    Tool definitions live in ``body["tools"]`` and are never touched here — only
    the system messages inside ``body["messages"]`` are rewritten."""
    if not (SYSTEM_OVERRIDE and SYSTEM_PROMPT):
        return messages
    n = 0
    while n < len(messages) and messages[n].get("role") == "system":
        n += 1
    original = "\n\n".join(str(m.get("content", "") or "") for m in messages[:n])
    if SYSTEM_MODE == "prefix":
        text = f"{SYSTEM_PROMPT}\n\n{original}" if original else SYSTEM_PROMPT
    elif SYSTEM_MODE == "suffix":
        text = f"{original}\n\n{SYSTEM_PROMPT}" if original else SYSTEM_PROMPT
    else:  # replace
        text = SYSTEM_PROMPT
    return [{"role": "system", "content": text}] + messages[n:]


# --- Tool filtering ---------------------------------------------------------
# Drop/keep tools from the host's tool set before it reaches the model. Both
# vars are comma-separated tool names:
#   CA_TOOLS_ALLOW : keep ONLY these (allowlist). Wins if both are set.
#   CA_TOOLS_DENY  : drop these (denylist). Defaults to "glob".
# An empty CA_TOOLS_DENY (CA_TOOLS_DENY=) disables the default and passes every
# tool through. Allowlist is stricter but riskier: if it removes a tool the host
# still references in a prior tool_call/tool message, some backends error on the
# orphaned reference — prefer the denylist unless you need a hard whitelist.
def _csv_set(name: str, default: str = "") -> set:
    return {t.strip() for t in os.environ.get(name, default).split(",") if t.strip()}


TOOLS_ALLOW = _csv_set("CA_TOOLS_ALLOW")
TOOLS_DENY = _csv_set("CA_TOOLS_DENY", "glob")


def _tool_name(t: Dict[str, Any]) -> Optional[str]:
    """Tool name for an OpenAI-style tool def ({"function": {"name": ...}}),
    falling back to a top-level ``name`` for other shapes."""
    return (t.get("function") or {}).get("name") or t.get("name")


def _filter_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Apply the configured allow/deny filter. No-op unless a filter is set."""
    if not tools or (not TOOLS_ALLOW and not TOOLS_DENY):
        return tools
    if TOOLS_ALLOW:
        return [t for t in tools if _tool_name(t) in TOOLS_ALLOW]
    return [t for t in tools if _tool_name(t) not in TOOLS_DENY]


CONFIG = Config(
    storage_root=STORAGE_ROOT,
    # Backend used for the summary/label + file-ledger calls. Register the
    # host's edit/read tool names so file detection recognizes them.
    llm=LLMConfig(provider="ollama", base_url=OLLAMA, model=MODEL, reasoning=SUMM_REASONING),
    extra_read_tools={"read"},
    extra_write_tools={"edit", "write", "patch"},
    # Keep the most recent N completed turns as full text; older turns as
    # summaries (0 = every completed turn is summarized immediately).
    num_full_text_turns=NUM_FULL_TEXT_TURNS,
)
ca = ContextArchitect(CONFIG)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # ── startup ──
    print("── context_architect proxy ─────────────────────────────", flush=True)
    print(f"   ollama    : {OLLAMA}", flush=True)
    print(f"   main model: {MAIN_MODEL}  (agent loop)", flush=True)
    print(f"   summ model: {MODEL}  (summary/label + ledger)", flush=True)
    print(
        f"   reasoning : {AGENT_REASONING or 'model default'} (agent) / "
        f"{SUMM_REASONING or 'model default'} (summ)",
        flush=True,
    )
    print(f"   full-text : last {NUM_FULL_TEXT_TURNS} turns kept verbatim", flush=True)
    if SYSTEM_OVERRIDE and SYSTEM_PROMPT:
        src = str(_resolve_prompt_path(_sp_file)) if _sp_file else "CA_SYSTEM_PROMPT"
        print(f"   sys-prompt: OVERRIDE on ({SYSTEM_MODE}) ← {src}", flush=True)
    else:
        print("   sys-prompt: override off (host prompt passes through)", flush=True)
    print(f"   archive → : {CONFIG.resolved_root()}/<session-id>/", flush=True)
    print("   waiting for POST /v1/chat/completions from the host…", flush=True)
    print("─────────────────────────────────────────────────────────", flush=True)
    yield
    # ── shutdown ── (nothing to tear down)


app = FastAPI(lifespan=_lifespan)

# Per session: how many completed-history messages we've already folded into
# context_architect. In-memory; a restart re-folds once (harmless).
_folded: Dict[str, int] = {}

# Per session: number of user messages seen. When it grows, a new turn has
# started — used to dump the main-LLM input once per turn (CA_AUDIT_OUTBOUND).
_turns_seen: Dict[str, int] = {}


def _dump_outbound(session_id: str, turn_no: int, messages: List[Dict[str, Any]]):
    """Write EXACTLY the messages sent to the main model at the start of a turn."""
    d = CONFIG.resolved_root() / session_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"turn-{turn_no:03d}-msgs_to_main_llm.json"
    path.write_text(json.dumps(messages, indent=2))
    print(f"[outbound] turn {turn_no} → {path}", flush=True)


# Per (session, turn): the assistant messages the main model returned this turn
# (one per completion call in the tool loop). In-memory; resets on restart.
_responses: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}


def _capture_response(session_id: str, turn_no: int, msg: Dict[str, Any]):
    """Append one main-model response for a turn and rewrite the turn's file."""
    per_turn = _responses.setdefault(session_id, {}).setdefault(turn_no, [])
    per_turn.append(msg)
    d = CONFIG.resolved_root() / session_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"turn-{turn_no:03d}-msgs_from_main_llm.json"
    path.write_text(json.dumps(per_turn, indent=2))
    print(f"[inbound] turn {turn_no} response #{len(per_turn)} → {path}", flush=True)


class _StreamAcc:
    """Reconstruct an assistant message from an OpenAI-style SSE stream."""

    def __init__(self):
        self._buf = b""
        self.role = "assistant"
        self.content: List[str] = []
        self.tool_calls: Dict[int, Dict[str, Any]] = {}

    def feed(self, chunk: bytes):
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload in (b"", b"[DONE]"):
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("role"):
                    self.role = delta["role"]
                if delta.get("content"):
                    self.content.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = self.tool_calls.setdefault(
                        tc.get("index", 0),
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

    def message(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": self.role, "content": "".join(self.content)}
        if self.tool_calls:
            msg["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return msg


def _dump_full_transcript(
    session_id: str,
    turn_no: int,
    live: List[Dict[str, Any]],
    response: Dict[str, Any],
):
    """Write the complete interleaved turn: the in-flight turn's messages (user →
    assistant tool_calls → tool results → …) plus the model's latest response.

    Each tool-loop call's ``live`` already contains all prior tool calls and
    results, so overwriting on every call means the turn's *final* call writes
    the complete transcript."""
    d = CONFIG.resolved_root() / session_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"turn-{turn_no:03d}-full_transcript.json"
    path.write_text(json.dumps(list(live) + [response], indent=2))
    print(f"[full] turn {turn_no} → {path} ({len(live) + 1} msgs)", flush=True)


def _strip_meta(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop the ``metadata`` field the library tags onto its injected messages;
    Ollama's OpenAI-compatible endpoint doesn't expect it."""
    return [{k: v for k, v in m.items() if k != "metadata"} for m in messages]


def _last_user_index(messages: List[Dict[str, Any]]) -> int:
    """Index where the in-flight turn begins (the last user message). Everything
    from here on is sent verbatim; everything before it is completed history."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return 0


async def _forward(
    body: Dict[str, Any],
    capture: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Forward a (possibly rewritten) request to Ollama, streaming-aware.

    If ``capture`` is given, the reconstructed assistant message the model
    returns is handed to it (after streaming completes, without altering the
    bytes forwarded to the client)."""
    url = f"{OLLAMA}/v1/chat/completions"
    if body.get("stream"):

        async def gen():
            acc = _StreamAcc() if capture else None
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=body) as r:
                    async for chunk in r.aiter_raw():
                        if acc is not None:
                            acc.feed(chunk)
                        yield chunk
            if acc is not None:
                capture(acc.message())

        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, json=body)
    data = r.json()
    if capture is not None:
        try:
            capture(data["choices"][0]["message"])
        except Exception:
            pass
    return JSONResponse(data, status_code=r.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, x_session_id: str = Header("opencode")):
    body = await request.json()
    incoming = body.get("messages", []) or []

    # Auxiliary calls (title/summary generation) arrive without tools — pass
    # them through untouched so they never become bogus turns.
    if not body.get("tools"):
        print(f"[pass] session={x_session_id!r} aux/no-tools, {len(incoming)} msgs", flush=True)
        return await _forward(body)

    # This is the main agent loop (it carries tools). Force the model to
    # MAIN_MODEL so the model is chosen in one place (CA_MAIN_MODEL / CA_MODEL)
    # rather than in the host's config; the host's model id is just a label.
    body["model"] = MAIN_MODEL

    # Drop/keep tools per CA_TOOLS_ALLOW / CA_TOOLS_DENY (default: deny "glob").
    tools_in = body.get("tools") or []
    kept = _filter_tools(tools_in) or []
    body["tools"] = kept
    if len(kept) != len(tools_in):
        removed = sorted({_tool_name(t) for t in tools_in} - {_tool_name(t) for t in kept})
        print(
            f"[tools] session={x_session_id!r} {len(tools_in)}→{len(kept)} tools "
            f"(dropped: {', '.join(r for r in removed if r)})",
            flush=True,
        )
    # If tool_choice pins a tool we just removed, drop it back to auto so the
    # backend doesn't error on a reference to a missing tool.
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        chosen = (tc.get("function") or {}).get("name") or tc.get("name")
        if chosen and chosen not in {_tool_name(t) for t in kept}:
            body["tool_choice"] = "auto"

    # Force the reasoning level too (if configured), so it's chosen here in one
    # place rather than by the host. Ollama's OpenAI-compatible /v1 endpoint
    # takes it as ``reasoning_effort``. Unset -> nothing added -> model default.
    body.update(_reasoning_params("ollama-openai", AGENT_REASONING))

    # Substitute our own system prompt for the host's (main agent loop only, so
    # aux title/summary prompts above are left intact). Tool definitions in
    # body["tools"] are forwarded untouched.
    incoming = _apply_system_override(incoming)

    # Split at the last user message: the in-flight turn is sent verbatim so the
    # model can answer it; everything before it is completed history.
    boundary = _last_user_index(incoming)
    history, live = incoming[:boundary], incoming[boundary:]

    # Fold every newly-completed turn into context_architect (no threshold).
    session = ca.session(x_session_id)
    folded = _folded.get(x_session_id, 0)
    fresh = history[folded:]
    if fresh:
        try:
            session.rework(session.live_history + oai.to_internal(fresh))
            _folded[x_session_id] = len(history)
        except Exception as e:
            # Summarization runs blocking, before the agent is called. A failure
            # here (LLM timeout, unparseable output, upstream hiccup) must not
            # take down the whole agent turn with a 500. Skip folding this turn —
            # the agent still gets the prior compact history plus the live turn —
            # and leave _folded unadvanced so the fold is retried next request.
            print(
                f"[rework] session={x_session_id!r} FOLD FAILED "
                f"({type(e).__name__}: {e}); serving prior history, retrying next turn",
                flush=True,
            )

    # Send: prior turns as compact summaries + the in-flight turn verbatim.
    compact = _strip_meta(oai.from_internal(session.live_history))
    body["messages"] = compact + live
    print(
        f"[rework] session={x_session_id!r} history {len(history)}→{len(compact)} summary msgs "
        f"+ {len(live)} live → {CONFIG.resolved_root()}/{x_session_id}/",
        flush=True,
    )

    # A new turn = a new user message; the count is this turn's number.
    turn_no = sum(1 for m in incoming if m.get("role") == "user")

    # Dump the exact main-LLM input, once per turn (at the turn's first call,
    # before the tool loop appends anything).
    if AUDIT_OUTBOUND and turn_no > _turns_seen.get(x_session_id, 0):
        _turns_seen[x_session_id] = turn_no
        _dump_outbound(x_session_id, turn_no, body["messages"])

    # Capture the main model's response(s) for this turn (streamed or not) — for
    # the from-model file and/or the complete interleaved transcript.
    capture = None
    if AUDIT_INBOUND or AUDIT_FULL:
        def capture(msg, _sid=x_session_id, _tn=turn_no, _live=live):
            if AUDIT_INBOUND:
                _capture_response(_sid, _tn, msg)
            if AUDIT_FULL:
                _dump_full_transcript(_sid, _tn, _live, msg)

    return await _forward(body, capture=capture)


@app.post("/recall")
async def recall(request: Request):
    """Called by the host's ``recall_turn`` tool: return one archived turn."""
    data = await request.json()
    print(f"[recall] session={data.get('session', 'opencode')!r} key={data.get('key')!r}", flush=True)
    session = ca.session(data.get("session", "opencode"))
    return {"text": session.recall(data["key"])}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, request: Request):
    """Everything else (e.g. GET /v1/models) goes straight to Ollama."""
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.request(
            request.method, f"{OLLAMA}/v1/{path}", content=await request.body()
        )
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=49786)
