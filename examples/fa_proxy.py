# Empromptu FreeAgent - The free, local, entirely private agent coding system, by Empromptu!
# Copyright (C) 2025  Empromptu, Sean Robinson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of version 3 of the GNU General Public License as published by
# the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""OpenAI-compatible proxy that inserts free_agent between a host
(e.g. OpenCode) and a local Ollama server.

    host ──/v1/chat/completions──▶  fa_proxy  ──▶  Ollama
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
    free_agent summaries the moment it becomes visible (i.e. on the next
    request). This gives the spec's behavior with a one-turn lag inherent to
    intercepting before the response exists.

Auxiliary host calls (title / summary generation) arrive without a ``tools``
array; those are passed through untouched so they don't become bogus turns.

Run:

    pip install -e ".[openai]"          # or just: pip install -e .
    pip install fastapi uvicorn httpx
    uvicorn examples.fa_proxy:app --port 49786

Environment:
    FA_PROVIDER       which upstream to forward to: ollama (default) or openai.
                      Both speak the OpenAI chat-completions wire format, so the
                      only differences are the base URL, the auth header and the
                      reasoning-param flavor.
    OLLAMA_BASE_URL   default http://localhost:11434 (used when FA_PROVIDER=ollama)
    OPENAI_API_KEY    OpenAI bearer token (used when FA_PROVIDER=openai). Optional
                      here but required by OpenAI itself; also honored by the
                      summarizer backend.
    OPENAI_BASE_URL   default https://api.openai.com/v1. Point this at any
                      OpenAI-compatible endpoint (incl. Anthropic's /v1 compat
                      endpoint) to reuse the openai provider path.
    FA_MODEL          default qwen3.6:35b   (used for summary/label + ledger,
                      and for the main agent loop unless FA_MAIN_MODEL is set)
    FA_MAIN_MODEL     default = FA_MODEL. The model the main agent loop runs on;
                      the proxy stamps it onto every request so the host's own
                      model id becomes a placeholder. Set this only to run the
                      summarizer on a different model than the agent.
    FA_REASONING      reasoning/thinking effort for the MAIN AGENT loop:
                      off | low | medium | high. Unset -> the model's own
                      default (nothing injected).
    FA_MAIN_REASONING kept-for-compat alias for FA_REASONING (agent loop).
    FA_SUMM_REASONING reasoning effort for the internal summary/label/ledger
                      calls. Defaults to OFF even when the agent uses reasoning:
                      these run blocking before the agent and discard their
                      thinking, so turning it up only adds latency and timeout
                      risk. Set it only if you specifically want it.
    FA_STORAGE_ROOT   default ~/.free_agent
    FA_TOOLS_DENY     comma-separated tool names to drop from the host's tool set
                      before it reaches the model. Defaults to "glob"; set it
                      empty (FA_TOOLS_DENY=) to pass every tool through.
    FA_TOOLS_ALLOW    comma-separated tool names to keep (allowlist). When set it
                      wins over FA_TOOLS_DENY. Stricter but riskier: removing a
                      tool the host still references in a prior tool_call can make
                      some backends error — prefer FA_TOOLS_DENY unless you need a
                      hard whitelist.
    FA_AUDIT_OUTBOUND set to 1 to dump the exact messages sent to the main model
                      at the start of each turn to
                      {root}/{session}/turn-NNN-msgs_to_main_llm.json
    FA_AUDIT_INBOUND  set to 1 to dump the exact messages the main model returned
                      during each turn (all tool-loop responses, in order) to
                      {root}/{session}/turn-NNN-msgs_from_main_llm.json
    FA_AUDIT_FULL     set to 1 to dump the complete interleaved turn (in-flight
                      messages + tool calls + tool results + final answer) to
                      {root}/{session}/turn-NNN-full_transcript.json
    FA_CONTINUE_ON_EMPTY
                      set to 1 to transparently re-ask the model when it returns
                      an empty final message (no content, no tool calls). The
                      proxy appends a "please continue." nudge and returns the
                      non-empty result; the nudge never reaches the host's
                      transcript or free_agent's summaries. Off by default.
    FA_CONTINUE_MAX   max retries before giving up and forwarding the empty
                      response as-is (default 2).
    FA_CONTINUE_MSG   the nudge text sent on an empty reply (default
                      "please continue.").
"""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from free_agent import Config, FreeAgent, LLMConfig
from free_agent.adapters import openai as oai
from free_agent.llm.reasoning import normalize as _norm_reasoning
from free_agent.llm.reasoning import params_for as _reasoning_params

# --- Provider selection -----------------------------------------------------
# Which upstream the proxy forwards to. Both providers speak the OpenAI Chat
# Completions wire format the proxy already builds, so switching is just a
# matter of base URL + auth header + reasoning-param flavor.
#   FA_PROVIDER=ollama  (default) : local Ollama, no API key
#   FA_PROVIDER=openai            : OpenAI (or any OpenAI-compatible endpoint via
#                                   OPENAI_BASE_URL), authenticated with
#                                   OPENAI_API_KEY.
PROVIDER = os.environ.get("FA_PROVIDER", "ollama").strip().lower()

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if PROVIDER not in ("ollama", "openai"):
    raise ValueError(
        f"FA_PROVIDER must be 'ollama' or 'openai', got {PROVIDER!r}"
    )

# Resolve the single upstream used by BOTH the main agent loop and the /v1/*
# passthrough. ``UPSTREAM_V1`` already includes the trailing ``/v1`` so callers
# just append ``/chat/completions`` etc. ``UPSTREAM_HEADERS`` carries auth (empty
# for Ollama). ``REASONING_FLAVOR`` selects how FA_REASONING is translated into
# the request body (see free_agent.llm.reasoning.params_for).
if PROVIDER == "openai":
    UPSTREAM_V1 = OPENAI_BASE_URL
    UPSTREAM_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}
    REASONING_FLAVOR = "openai"
else:  # ollama
    UPSTREAM_V1 = f"{OLLAMA}/v1"
    UPSTREAM_HEADERS = {}
    REASONING_FLAVOR = "ollama-openai"

MODEL = os.environ.get("FA_MODEL", "qwen3.6:35b")
# The model the *main agent loop* runs on. Defaults to FA_MODEL so a single
# FA_MODEL env var drives everything; set FA_MAIN_MODEL only if you want the
# summary/label calls (FA_MODEL) to use a different model than the main agent.
# The proxy stamps this onto every main-agent request, so whatever model id the
# host (OpenCode) has configured becomes a cosmetic placeholder — change the
# model here, in one place, and restart the proxy.
MAIN_MODEL = os.environ.get("FA_MAIN_MODEL", MODEL)

# Reasoning/thinking effort for the MAIN AGENT loop (the model that answers the
# user). This is what FA_REASONING controls; FA_MAIN_REASONING is a kept-for-
# compat alias. Unset -> nothing injected -> the model's own default.
AGENT_REASONING = _norm_reasoning(
    os.environ.get("FA_MAIN_REASONING") or os.environ.get("FA_REASONING")
)
# Reasoning effort for the internal summary/label/file-ledger calls. These are
# mechanical JSON extractions that run *blocking, on the turn's critical path*
# (before the agent is even called) and whose thinking tokens are discarded — so
# turning reasoning up here is nearly all cost (latency, and timeouts that would
# surface as errors) for no benefit. It therefore defaults to OFF even when the
# agent runs with reasoning on; opt in explicitly with FA_SUMM_REASONING only if
# you have a specific reason to.
SUMM_REASONING = _norm_reasoning(os.environ.get("FA_SUMM_REASONING", "off"))
STORAGE_ROOT = os.environ.get("FA_STORAGE_ROOT", "~/.free_agent")
AUDIT_OUTBOUND = os.environ.get("FA_AUDIT_OUTBOUND") == "1"
AUDIT_INBOUND = os.environ.get("FA_AUDIT_INBOUND") == "1"
AUDIT_FULL = os.environ.get("FA_AUDIT_FULL") == "1"
NUM_FULL_TEXT_TURNS = int(os.environ.get("FA_NUM_FULL_TEXT_TURNS", "1"))

# --- Continue-on-empty ------------------------------------------------------
# Some models occasionally return an empty final message (no content, no tool
# calls) — opencode then shows a blank assistant turn. When enabled, the proxy
# transparently re-asks the model with a "please continue." nudge and returns
# the non-empty result instead. The nudge lives only inside the proxy's retry
# request; it never reaches opencode's transcript or free_agent's summaries, so
# from the host's seat the model simply answered normally.
#   FA_CONTINUE_ON_EMPTY  set to 1 to enable (default off).
#   FA_CONTINUE_MAX       max retries before giving up and returning the empty
#                         response as-is (default 2).
#   FA_CONTINUE_MSG       the nudge text (default "please continue.").
CONTINUE_ON_EMPTY = os.environ.get("FA_CONTINUE_ON_EMPTY") == "1"
CONTINUE_MAX = int(os.environ.get("FA_CONTINUE_MAX", "2"))
CONTINUE_MSG = os.environ.get("FA_CONTINUE_MSG", "please continue.")

# --- System-prompt override -------------------------------------------------
# Master switch: the override only applies when FA_SYSTEM_OVERRIDE=1, so you can
# keep a prompt configured and toggle it on/off without deleting it.
SYSTEM_OVERRIDE = os.environ.get("FA_SYSTEM_OVERRIDE") == "1"
# The replacement text. FA_SYSTEM_PROMPT_FILE (a path) takes precedence over the
# inline FA_SYSTEM_PROMPT. A relative path (e.g. ./system_prompt/foo.md) is
# resolved against the repo root, not the cwd the proxy was launched from, so
# in-repo prompt files work no matter where run_fa_proxy.sh is invoked.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_sp_file = os.environ.get("FA_SYSTEM_PROMPT_FILE")


def _resolve_prompt_path(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (_REPO_ROOT / q)


SYSTEM_PROMPT = (
    _resolve_prompt_path(_sp_file).read_text()
    if _sp_file
    else os.environ.get("FA_SYSTEM_PROMPT")
)
# How the override combines with the host's own system prompt:
#   replace : swap the whole leading system run for yours (default)
#   prefix  : your text, then the host's system prompt
#   suffix  : the host's system prompt, then your text
SYSTEM_MODE = os.environ.get("FA_SYSTEM_MODE", "replace")


def _apply_system_override(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite the leading contiguous run of ``system`` messages using the
    configured override. No-op unless FA_SYSTEM_OVERRIDE=1 and a prompt is set.

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
    elif SYSTEM_MODE == "prefix_env":# Custom prompt first, then ONLY the <env>...</env> block from the host.
        env_match = re.search(r"<env>(.*?)</env>", original, re.DOTALL)
        env_block = f"\n\n{env_match.group(0)}" if env_match else ""
        text = SYSTEM_PROMPT + env_block if env_block else SYSTEM_PROMPT
    else:  # replace
        text = SYSTEM_PROMPT
    return [{"role": "system", "content": text}] + messages[n:]


# --- Tool filtering ---------------------------------------------------------
# Drop/keep tools from the host's tool set before it reaches the model. Both
# vars are comma-separated tool names:
#   FA_TOOLS_ALLOW : keep ONLY these (allowlist). Wins if both are set.
#   FA_TOOLS_DENY  : drop these (denylist). Defaults to "glob".
# An empty FA_TOOLS_DENY (FA_TOOLS_DENY=) disables the default and passes every
# tool through. Allowlist is stricter but riskier: if it removes a tool the host
# still references in a prior tool_call/tool message, some backends error on the
# orphaned reference — prefer the denylist unless you need a hard whitelist.
def _csv_set(name: str, default: str = "") -> set:
    return {t.strip() for t in os.environ.get(name, default).split(",") if t.strip()}


TOOLS_ALLOW = _csv_set("FA_TOOLS_ALLOW")
TOOLS_DENY = _csv_set("FA_TOOLS_DENY", "glob")


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


# Backend used for the summary/label + file-ledger calls. Follows FA_PROVIDER
# so the summarizer runs on the same provider as the agent loop. For openai this
# uses free_agent's OpenAIBackend, which needs the openai package
# (pip install "free_agent[openai]").
if PROVIDER == "openai":
    _SUMM_LLM = LLMConfig(
        provider="openai",
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model=MODEL,
        reasoning=SUMM_REASONING,
    )
else:
    _SUMM_LLM = LLMConfig(
        provider="ollama", base_url=OLLAMA, model=MODEL, reasoning=SUMM_REASONING
    )

CONFIG = Config(
    storage_root=STORAGE_ROOT,
    # Register the host's edit/read tool names so file detection recognizes them.
    llm=_SUMM_LLM,
    extra_read_tools={"read"},
    extra_write_tools={"edit", "write", "patch"},
    # Keep the most recent N completed turns as full text; older turns as
    # summaries (0 = every completed turn is summarized immediately).
    num_full_text_turns=NUM_FULL_TEXT_TURNS,
)
ca = FreeAgent(CONFIG)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # ── startup ──
    print("── free_agent proxy ─────────────────────────────", flush=True)
    print(f"   provider  : {PROVIDER}", flush=True)
    print(f"   upstream  : {UPSTREAM_V1}", flush=True)
    if PROVIDER == "openai" and not OPENAI_API_KEY:
        print(
            "   WARNING   : FA_PROVIDER=openai but OPENAI_API_KEY is unset — "
            "requests will be sent unauthenticated and likely 401.",
            flush=True,
        )
    print(f"   main model: {MAIN_MODEL}  (agent loop)", flush=True)
    print(f"   summ model: {MODEL}  (summary/label + ledger)", flush=True)
    print(
        f"   reasoning : {AGENT_REASONING or 'model default'} (agent) / "
        f"{SUMM_REASONING or 'model default'} (summ)",
        flush=True,
    )
    print(f"   full-text : last {NUM_FULL_TEXT_TURNS} turns kept verbatim", flush=True)
    if CONTINUE_ON_EMPTY:
        print(
            f"   continue  : on empty reply, nudge {CONTINUE_MSG!r} "
            f"(max {CONTINUE_MAX} retries)",
            flush=True,
        )
    else:
        print("   continue  : off (empty replies pass through)", flush=True)
    if SYSTEM_OVERRIDE and SYSTEM_PROMPT:
        src = str(_resolve_prompt_path(_sp_file)) if _sp_file else "FA_SYSTEM_PROMPT"
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
# free_agent. In-memory; a restart re-folds once (harmless).
_folded: Dict[str, int] = {}

# Per session: number of user messages seen. When it grows, a new turn has
# started — used to dump the main-LLM input once per turn (FA_AUDIT_OUTBOUND).
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

    def has_content(self) -> bool:
        """True once a real content token or any tool call has arrived. Used to
        decide, mid-stream, whether the response is going to be non-empty."""
        return bool("".join(self.content).strip()) or bool(self.tool_calls)

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


def _msg_is_empty(msg: Dict[str, Any]) -> bool:
    """An assistant message counts as empty when it has no textual content AND
    no tool calls. A tool call is never empty — the host will act on it."""
    content = msg.get("content") or ""
    if isinstance(content, list):  # anthropic-style block list
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return not str(content).strip() and not msg.get("tool_calls")


def _continue_body(body: Dict[str, Any], prev_msg: Dict[str, Any]) -> Dict[str, Any]:
    """A shallow copy of ``body`` with the model's (empty) reply and a
    "please continue." nudge appended. This lives only in the retry request; it
    is never persisted to the host or to free_agent."""
    nb = dict(body)
    assistant: Dict[str, Any] = {"role": "assistant", "content": prev_msg.get("content") or ""}
    if prev_msg.get("tool_calls"):
        assistant["tool_calls"] = prev_msg["tool_calls"]
    nb["messages"] = list(body.get("messages") or []) + [
        assistant,
        {"role": "user", "content": CONTINUE_MSG},
    ]
    return nb


async def _forward(
    body: Dict[str, Any],
    capture: Optional[Callable[[Dict[str, Any]], None]] = None,
    allow_continue: bool = False,
):
    """Forward a (possibly rewritten) request to Ollama, streaming-aware.

    If ``capture`` is given, the reconstructed assistant message the model
    returns is handed to it (after streaming completes, without altering the
    bytes forwarded to the client).

    If ``allow_continue`` is set and FA_CONTINUE_ON_EMPTY is on, an empty
    response (no content, no tool calls) triggers a transparent retry with a
    "please continue." nudge, up to FA_CONTINUE_MAX times. For streaming we hold
    chunks back only until the first real content token arrives, so a normal
    response streams with negligible added latency; an empty one is discarded
    and the retry is streamed in its place."""
    url = f"{UPSTREAM_V1}/chat/completions"
    do_continue = allow_continue and CONTINUE_ON_EMPTY

    if body.get("stream"):

        async def gen():
            attempt_body = body
            for attempt in range(CONTINUE_MAX + 1):
                acc = _StreamAcc()
                buffer: List[bytes] = []
                flushed = False
                async with httpx.AsyncClient(timeout=None, headers=UPSTREAM_HEADERS) as client:
                    async with client.stream("POST", url, json=attempt_body) as r:
                        # An error status is NOT a stream — the body is a JSON
                        # error, not SSE. Feeding it to the accumulator finds no
                        # content, which would look like an "empty" reply and
                        # trigger pointless retries, ultimately surfacing to the
                        # host as a blank turn. Instead surface the upstream error
                        # verbatim (as a content chunk so it's visible in the host
                        # UI) and stop.
                        if r.status_code >= 400:
                            err = (await r.aread()).decode("utf-8", "replace")
                            print(f"[upstream error] {r.status_code}: {err[:800]}", flush=True)
                            payload = {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": f"[proxy] upstream error {r.status_code}: {err}",
                                        },
                                        "finish_reason": "stop",
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(payload)}\n\n".encode()
                            yield b"data: [DONE]\n\n"
                            return
                        async for chunk in r.aiter_raw():
                            acc.feed(chunk)
                            if flushed:
                                yield chunk
                            else:
                                buffer.append(chunk)
                                if acc.has_content():
                                    for b in buffer:
                                        yield b
                                    buffer = []
                                    flushed = True
                # Stream finished. If real content ever arrived it's already sent.
                if flushed:
                    if capture is not None:
                        capture(acc.message())
                    return
                # Empty response: retry with a nudge, or give up and forward it.
                if do_continue and attempt < CONTINUE_MAX:
                    print(
                        f"[continue] empty stream response, retrying "
                        f"({attempt + 1}/{CONTINUE_MAX})",
                        flush=True,
                    )
                    attempt_body = _continue_body(attempt_body, acc.message())
                    continue
                for b in buffer:
                    yield b
                if capture is not None:
                    capture(acc.message())
                return

        return StreamingResponse(gen(), media_type="text/event-stream")

    attempt_body = body
    for attempt in range(CONTINUE_MAX + 1):
        async with httpx.AsyncClient(timeout=None, headers=UPSTREAM_HEADERS) as client:
            r = await client.post(url, json=attempt_body)
        if r.status_code >= 400:
            print(f"[upstream error] {r.status_code}: {r.text[:800]}", flush=True)
            return JSONResponse(r.json(), status_code=r.status_code)
        data = r.json()
        try:
            msg = data["choices"][0]["message"]
        except Exception:
            msg = None
        if do_continue and msg is not None and _msg_is_empty(msg) and attempt < CONTINUE_MAX:
            print(
                f"[continue] empty response, retrying ({attempt + 1}/{CONTINUE_MAX})",
                flush=True,
            )
            attempt_body = _continue_body(attempt_body, msg)
            continue
        if capture is not None and msg is not None:
            try:
                capture(msg)
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
    # MAIN_MODEL so the model is chosen in one place (FA_MAIN_MODEL / FA_MODEL)
    # rather than in the host's config; the host's model id is just a label.
    body["model"] = MAIN_MODEL

    # Drop/keep tools per FA_TOOLS_ALLOW / FA_TOOLS_DENY (default: deny "glob").
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
    body.update(_reasoning_params(REASONING_FLAVOR, AGENT_REASONING))

    # Substitute our own system prompt for the host's (main agent loop only, so
    # aux title/summary prompts above are left intact). Tool definitions in
    # body["tools"] are forwarded untouched.
    incoming = _apply_system_override(incoming)

    # Split at the last user message: the in-flight turn is sent verbatim so the
    # model can answer it; everything before it is completed history.
    boundary = _last_user_index(incoming)
    history, live = incoming[:boundary], incoming[boundary:]

    # Fold every newly-completed turn into free_agent (no threshold).
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

    return await _forward(body, capture=capture, allow_continue=True)


@app.post("/recall")
async def recall(request: Request):
    """Called by the host's ``recall_turn`` tool: return one archived turn."""
    data = await request.json()
    print(f"[recall] session={data.get('session', 'opencode')!r} key={data.get('key')!r}", flush=True)
    session = ca.session(data.get("session", "opencode"))
    return {"text": session.recall(data["key"])}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, request: Request):
    """Everything else (e.g. GET /v1/models) goes straight to the upstream."""
    async with httpx.AsyncClient(timeout=None, headers=UPSTREAM_HEADERS) as client:
        r = await client.request(
            request.method, f"{UPSTREAM_V1}/{path}", content=await request.body()
        )
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=49786)
