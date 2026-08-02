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
    OLLAMA_BASE_URL   default http://localhost:11434
    FA_MAIN_PROVIDER  backend for the agent-facing model (ollama|openai, default
                      ollama). "openai" forwards the main loop to OPENAI_BASE_URL
                      with a bearer token instead of the local Ollama server.
    FA_SUMM_PROVIDER  backend for the internal summary/label/history-folding
                      calls (ollama|openai, default ollama).
    FA_EMBED_PROVIDER codegraph embedder (default ollama). Only "ollama" is
                      implemented; any other value disables codegraph entirely
                      (there is no OpenAI embedding path).
    OPENAI_API_KEY    required when any surface is set to "openai".
    FA_OPENAI_BASE_URL OpenAI-compatible base URL (default
                      https://api.openai.com/v1; include the /v1 suffix). Point
                      it at Azure or a gateway if needed.
    FA_OPENAI_API     which OpenAI wire API the main agent leg uses when
                      FA_MAIN_PROVIDER=openai: "responses" (default) or "chat".
                      The gpt-5 series reject tools+reasoning on chat-completions
                      and require the Responses API; "chat" forces the legacy
                      endpoint (fine for models without that restriction). Only
                      the outbound LLM call changes — OpenCode still talks
                      chat-completions to the proxy and context handling is
                      identical.
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
    FA_CODEGRAPH_TOOL enable the Graph-RAG code-concept system (default 1).
                      Requires the optional extra: pip install -e ".[codegraph]".
                      Missing extra -> feature disabled, proxy unaffected.
    FA_CODEGRAPH_LIVE set to 1 to rebuild the index incrementally on file
                      changes (watchdog). Off by default; the initial build runs
                      once the launcher POSTs the project root to /codegraph/init.
    FA_CODEGRAPH_MODEL       digest LLM model (default = FA_MODEL). A smaller,
                             faster model here cuts build time substantially.
    FA_CODEGRAPH_EMBED_MODEL embedding model via /api/embed (default
                             nomic-embed-text).
    FA_CODEGRAPH_RESOLUTION  Leiden resolution / concept granularity (default 1.0).
    FA_CODEGRAPH_WORKERS     concurrent digest calls to Ollama (default 6).
    FA_CODEGRAPH_REASONING   thinking effort for digest calls (default "off";
                             thinking models default it ON and run 10-20x slower).
    FA_CODEGRAPH_TIMEOUT     per digest call timeout, seconds (default 300).
    FA_CODEGRAPH_IDLE        1 (default) = pause the background build's GPU calls
                             while the agent is mid-turn; 0 = build flat-out.
    FA_CODEGRAPH_QUIET_SECONDS  agent-silence before the build resumes (default 5).
    FA_CODEGRAPH_SKIP_METHODS  1 = summarize only classes + top-level functions.
    FA_CODEGRAPH_EXCLUDE     extra comma-separated dir names to exclude.
    FA_CODEGRAPH_MAX_FILE_KB  skip source files larger than this (default 512).
"""

from __future__ import annotations

import asyncio
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

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
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

# --- Provider selection (per surface) ---------------------------------------
# Each model-call surface picks its backend independently; all default to
# "ollama" so an unconfigured proxy behaves exactly as before.
#   FA_MAIN_PROVIDER   agent-facing model the proxy forwards to (ollama|openai)
#   FA_SUMM_PROVIDER   internal summary/label/history-folding calls (ollama|openai)
#   FA_EMBED_PROVIDER  codegraph embedder — only "ollama" is implemented; any
#                      other value disables codegraph (see below).
# OpenAI surfaces read OPENAI_API_KEY and, optionally, FA_OPENAI_BASE_URL (for
# Azure / OpenAI-compatible gateways; must include the /v1 suffix).
MAIN_PROVIDER = os.environ.get("FA_MAIN_PROVIDER", "ollama").lower()
SUMM_PROVIDER = os.environ.get("FA_SUMM_PROVIDER", "ollama").lower()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("FA_OPENAI_BASE_URL", "https://api.openai.com/v1")

# Upstream the main-agent path forwards to, plus optional bearer auth. Both
# branches normalize to a base URL ending in /v1 so URL construction below is
# uniform. Ollama needs no auth; OpenAI needs the bearer token.
if MAIN_PROVIDER == "openai":
    UPSTREAM_BASE = OPENAI_BASE_URL.rstrip("/")
    UPSTREAM_HEADERS = {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}
    MAIN_REASON_PROVIDER = "openai"
else:
    UPSTREAM_BASE = f"{OLLAMA}/v1"
    UPSTREAM_HEADERS = {}
    MAIN_REASON_PROVIDER = "ollama-openai"

# Which OpenAI wire API the main-agent leg uses. The gpt-5 series reject
# tools+reasoning on /v1/chat/completions and require /v1/responses (which
# preserves reasoning across tool calls). Default to "responses" for OpenAI so
# reasoning + tools work out of the box; set FA_OPENAI_API=chat to force the
# legacy chat-completions endpoint. Ignored unless FA_MAIN_PROVIDER=openai.
# NOTE: this only affects the outbound LLM call inside _forward — OpenCode still
# talks chat-completions to the proxy, and all context management is unchanged.
OPENAI_API = os.environ.get(
    "FA_OPENAI_API", "responses" if MAIN_PROVIDER == "openai" else "chat"
).lower()
MAIN_USE_RESPONSES = MAIN_PROVIDER == "openai" and OPENAI_API == "responses"

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

# --- Graph-RAG code-concept system (optional) -------------------------------
# When FA_CODEGRAPH_TOOL=1 (default) AND the optional [codegraph] extra is
# installed, the proxy indexes the project into a concept graph, injects a
# concept index into the main-agent context each turn, and serves the
# recall/query code-concept tools. A missing extra degrades to "disabled" — the
# import is lazy and guarded so the proxy never crashes on a missing dep.
CODEGRAPH_TOOL = os.environ.get("FA_CODEGRAPH_TOOL", "1") == "1"
# Live mode: rebuild incrementally on file changes via watchdog (§4). Off by
# default; the launcher triggers an initial build regardless.
CODEGRAPH_LIVE = os.environ.get("FA_CODEGRAPH_LIVE") == "1"
# Codegraph embeds locally via Ollama's native /api/embed; there is no OpenAI
# embedding path. If the embedder is pointed anywhere but Ollama, disable the
# whole subsystem with an explicit reason rather than failing mid-build.
EMBED_PROVIDER = os.environ.get("FA_EMBED_PROVIDER", "ollama").lower()
_cg = None
CODEGRAPH_OK = False
_cg_disabled_reason = None
if CODEGRAPH_TOOL and EMBED_PROVIDER != "ollama":
    _cg_disabled_reason = (
        f"embedder provider is {EMBED_PROVIDER!r}; codegraph requires a local "
        f"Ollama embedder (set FA_EMBED_PROVIDER=ollama or FA_CODEGRAPH_TOOL=0)"
    )
    print(f"[codegraph] disabled: {_cg_disabled_reason}", flush=True)
elif CODEGRAPH_TOOL:
    try:
        from free_agent import codegraph as _cg  # lazy heavy deps inside
        CODEGRAPH_OK = _cg.available()
    except Exception as _e:  # pragma: no cover - import-time degradation
        print(f"[codegraph] import failed, disabling ({type(_e).__name__}: {_e})",
              flush=True)
        _cg, CODEGRAPH_OK = None, False

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


CONFIG = Config(
    storage_root=STORAGE_ROOT,
    # Backend used for the summary/label + file-ledger calls. Register the
    # host's edit/read tool names so file detection recognizes them.
    llm=LLMConfig(
        provider=SUMM_PROVIDER,
        base_url=(OPENAI_BASE_URL if SUMM_PROVIDER == "openai" else OLLAMA),
        api_key=(OPENAI_API_KEY if SUMM_PROVIDER == "openai" else None),
        model=MODEL,
        reasoning=SUMM_REASONING,
    ),
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
    _api = f", {OPENAI_API} API" if MAIN_PROVIDER == "openai" else ""
    print(f"   upstream  : {UPSTREAM_BASE}  (main={MAIN_PROVIDER}{_api})", flush=True)
    if MAIN_PROVIDER == "ollama" or SUMM_PROVIDER == "ollama" or EMBED_PROVIDER == "ollama":
        print(f"   ollama    : {OLLAMA}", flush=True)
    print(f"   main model: {MAIN_MODEL}  (agent loop, {MAIN_PROVIDER})", flush=True)
    print(f"   summ model: {MODEL}  (summary/label + ledger, {SUMM_PROVIDER})", flush=True)
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
    if not CODEGRAPH_TOOL:
        print("   codegraph : off (FA_CODEGRAPH_TOOL=0)", flush=True)
    elif _cg_disabled_reason:
        print(f"   codegraph : disabled — {_cg_disabled_reason}", flush=True)
    elif CODEGRAPH_OK:
        live = " + live watch" if CODEGRAPH_LIVE else ""
        idle = "idle-aware" if os.environ.get("FA_CODEGRAPH_IDLE", "1") == "1" else "flat-out"
        print(f"   codegraph : on{live}, {idle} build (awaiting POST /codegraph/init)",
              flush=True)
    else:
        miss = ", ".join(_cg.missing_deps()) if _cg is not None else "import failed"
        print(f"   codegraph : unavailable — install .[codegraph] ({miss})", flush=True)
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


# ── OpenAI Responses API translation (last-hop only) ───────────────────────
# The proxy speaks chat-completions everywhere; only the final outbound call to
# an OpenAI gpt-5-series model is translated to the Responses API, because that
# endpoint is the only one that accepts tools + reasoning together (and carries
# reasoning across tool calls). Nothing upstream of _forward changes: the same
# chat-completions ``body`` goes in, and chat-completions bytes come back out to
# OpenCode. These helpers are the shim for that one leg.


def _content_text(content: Any) -> str:
    """Flatten a chat message ``content`` (str or block list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("input_text") or b.get("output_text") or "")
        return "".join(parts)
    return str(content)


def _chat_messages_to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """chat ``messages`` -> Responses ``input`` items. Assistant tool_calls become
    ``function_call`` items and ``tool`` results become ``function_call_output``
    items, so the full tool-loop history round-trips."""
    items: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id") or m.get("call_id"),
                "output": _content_text(m.get("content")),
            })
            continue
        text = _content_text(m.get("content"))
        if role == "assistant":
            if text.strip():
                items.append({"role": "assistant",
                              "content": [{"type": "output_text", "text": text}]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id"),
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "") or "",
                })
        else:  # system / developer / user
            items.append({"role": role or "user",
                          "content": [{"type": "input_text", "text": text}]})
    return items


def _chat_tools_to_responses(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """chat tool defs ({"type":"function","function":{…}}) -> Responses flat form."""
    out = []
    for t in tools or []:
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            spec = {"type": "function", "name": fn.get("name"),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}
            if fn.get("description"):
                spec["description"] = fn["description"]
            out.append(spec)
        else:
            out.append(t)  # already flat / non-function tool
    return out


def _chat_tool_choice_to_responses(tc: Any) -> Any:
    if tc is None or isinstance(tc, str):
        return tc  # "auto" / "none" / "required" carry over unchanged
    if isinstance(tc, dict):
        name = (tc.get("function") or {}).get("name") or tc.get("name")
        if name:
            return {"type": "function", "name": name}
    return "auto"


def _chat_body_to_responses(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a chat-completions request body into a Responses request body."""
    r: Dict[str, Any] = {
        "model": body.get("model"),
        "input": _chat_messages_to_responses_input(body.get("messages") or []),
        "stream": bool(body.get("stream")),
    }
    tools = _chat_tools_to_responses(body.get("tools"))
    if tools:
        r["tools"] = tools
    tc = _chat_tool_choice_to_responses(body.get("tool_choice"))
    if tc is not None:
        r["tool_choice"] = tc
    eff = body.get("reasoning_effort")
    if eff:  # "none" / "low" / "high" are all valid Responses efforts for gpt-5
        r["reasoning"] = {"effort": eff}
    mt = body.get("max_tokens") or body.get("max_completion_tokens")
    if mt:
        r["max_output_tokens"] = mt
    # temperature is intentionally omitted: gpt-5 reasoning models reject a
    # non-default temperature, and the host's value is not meaningful here.
    return r


def _responses_usage_to_chat(u: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not u:
        return None
    return {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens", 0),
    }


def _responses_json_to_chat(resp: Dict[str, Any]) -> Dict[str, Any]:
    """A non-streamed Responses object -> a chat-completions completion object."""
    content_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for item in resp.get("output") or []:
        it = item.get("type")
        if it == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    content_parts.append(c.get("text", ""))
        elif it == "function_call":
            tool_calls.append({
                "id": item.get("call_id"),
                "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": item.get("arguments", "") or ""},
            })
    msg: Dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": resp.get("id"),
        "object": "chat.completion",
        "model": resp.get("model"),
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": _responses_usage_to_chat(resp.get("usage")),
    }


def _sse_chunk(obj: Dict[str, Any]) -> bytes:
    """Encode one chat-completions streaming chunk as an SSE ``data:`` line."""
    obj.setdefault("object", "chat.completion.chunk")
    return b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"


async def _translate_responses_stream(r):
    """Consume a Responses SSE stream and yield chat-completions SSE chunk bytes,
    so the existing _StreamAcc / buffering / OpenCode path is untouched.

    Reasoning events are intentionally dropped — they carry no assistant-visible
    content and OpenCode's chat-completions parser doesn't expect them."""
    yield _sse_chunk({"choices": [{"index": 0, "delta": {"role": "assistant"},
                                   "finish_reason": None}]})
    tc_index: Dict[str, int] = {}   # Responses item id (fc_…) -> tool_call index
    next_idx = 0
    finish = "stop"
    async for line in r.aiter_lines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        t = obj.get("type")
        if t == "response.output_text.delta":
            yield _sse_chunk({"choices": [{"index": 0,
                              "delta": {"content": obj.get("delta", "")},
                              "finish_reason": None}]})
        elif t == "response.output_item.added":
            item = obj.get("item") or {}
            if item.get("type") == "function_call":
                idx = next_idx
                next_idx += 1
                tc_index[item.get("id")] = idx
                finish = "tool_calls"
                yield _sse_chunk({"choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": idx, "id": item.get("call_id"), "type": "function",
                    "function": {"name": item.get("name", ""),
                                 "arguments": item.get("arguments", "") or ""},
                }]}, "finish_reason": None}]})
        elif t == "response.function_call_arguments.delta":
            idx = tc_index.get(obj.get("item_id"), 0)
            yield _sse_chunk({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": idx, "function": {"arguments": obj.get("delta", "")},
            }]}, "finish_reason": None}]})
    yield _sse_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    yield b"data: [DONE]\n\n"


def _sse_error_chunk(status: int, err_bytes: bytes) -> bytes:
    """Surface an upstream error to OpenCode as visible assistant text instead of
    a silent blank turn (the streaming path can't change the 200 it already
    committed to, so make the failure legible)."""
    detail = err_bytes.decode("utf-8", "replace")[:800]
    return _sse_chunk({"choices": [{"index": 0,
                       "delta": {"role": "assistant",
                                 "content": f"[proxy] upstream error {status}: {detail}"},
                       "finish_reason": "stop"}]})


async def _forward(
    body: Dict[str, Any],
    capture: Optional[Callable[[Dict[str, Any]], None]] = None,
    allow_continue: bool = False,
):
    """Forward a (possibly rewritten) request to the upstream, streaming-aware.

    If ``capture`` is given, the reconstructed assistant message the model
    returns is handed to it (after streaming completes, without altering the
    bytes forwarded to the client).

    If ``allow_continue`` is set and FA_CONTINUE_ON_EMPTY is on, an empty
    response (no content, no tool calls) triggers a transparent retry with a
    "please continue." nudge, up to FA_CONTINUE_MAX times. For streaming we hold
    chunks back only until the first real content token arrives, so a normal
    response streams with negligible added latency; an empty one is discarded
    and the retry is streamed in its place.

    When the request carries tools and the main leg is set to the OpenAI
    Responses API, the outbound call is translated to /v1/responses and its
    reply translated back to chat-completions — transparently to both OpenCode
    and the rest of this function. Tool-less calls (e.g. title/summary passes)
    stay on chat-completions since they have no tools+reasoning conflict."""
    use_responses = MAIN_USE_RESPONSES and bool(body.get("tools"))
    url = f"{UPSTREAM_BASE}/responses" if use_responses else f"{UPSTREAM_BASE}/chat/completions"
    do_continue = allow_continue and CONTINUE_ON_EMPTY

    def _payload(b: Dict[str, Any]) -> Dict[str, Any]:
        return _chat_body_to_responses(b) if use_responses else b

    if body.get("stream"):

        async def gen():
            attempt_body = body
            for attempt in range(CONTINUE_MAX + 1):
                acc = _StreamAcc()
                buffer: List[bytes] = []
                flushed = False
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=_payload(attempt_body),
                                             headers=UPSTREAM_HEADERS) as r:
                        # Surface upstream failures instead of letting an error
                        # body masquerade as an empty (and endlessly retried)
                        # response — the bug that hid the gpt-5 tools+reasoning
                        # 400 as a silent blank turn.
                        if r.status_code >= 400:
                            err = await r.aread()
                            print(f"[upstream] {r.status_code} error: "
                                  f"{err.decode('utf-8', 'replace')[:500]}", flush=True)
                            yield _sse_error_chunk(r.status_code, err)
                            yield b"data: [DONE]\n\n"
                            return
                        chunks = (_translate_responses_stream(r) if use_responses
                                  else r.aiter_raw())
                        async for chunk in chunks:
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
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=_payload(attempt_body), headers=UPSTREAM_HEADERS)
        # Surface upstream failures rather than forwarding an error body as if it
        # were a normal completion.
        if r.status_code >= 400:
            try:
                err_json = r.json()
            except Exception:
                err_json = {"error": r.text}
            print(f"[upstream] {r.status_code} error: {str(err_json)[:500]}", flush=True)
            return JSONResponse(err_json, status_code=r.status_code)
        data = _responses_json_to_chat(r.json()) if use_responses else r.json()
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
        return JSONResponse(data, status_code=200)


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

    # Tell the codegraph build the agent is active, so its background index build
    # pauses instead of fighting this request for the GPU. Marked again at the
    # response's completion (below) so a long streaming turn stays "active".
    if CODEGRAPH_OK and _cg is not None:
        _cg.note_activity()

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
    body.update(_reasoning_params(MAIN_REASON_PROVIDER, AGENT_REASONING))

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

    # Inject the code-concept index as an extra system message (§6/§8). This is
    # only reachable on the main-agent path (the aux/no-tools call returned early
    # above), so it never lands on title/summary passthroughs. Empty while the
    # index is still building or the extra is disabled → nothing injected.
    concept_msgs: List[Dict[str, Any]] = []
    if CODEGRAPH_OK and _cg is not None:
        try:
            idx = _cg.get_concept_index()
        except Exception:
            idx = ""
        if idx:
            concept_msgs = [{
                "role": "system",
                "content": (
                    "Code-concept index for this codebase (one line per "
                    "concept: '<tag>: <summary>'). Use the recall_codeconcept / "
                    "query_codeconcept tools to pull the actual code for any "
                    "concept:\n" + idx
                ),
            }]

    body["messages"] = compact + concept_msgs + live
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
    # the from-model file and/or the complete interleaved transcript, and to
    # re-mark agent activity at completion so a long streaming turn keeps the
    # codegraph build paused for its whole duration (not just its first token).
    capture = None
    if AUDIT_INBOUND or AUDIT_FULL or CODEGRAPH_OK:
        def capture(msg, _sid=x_session_id, _tn=turn_no, _live=live):
            if CODEGRAPH_OK and _cg is not None:
                _cg.note_activity()
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


@app.post("/codegraph/init")
async def codegraph_init(request: Request):
    """Launcher calls this once the proxy is up, POSTing the project root. The
    full build / incremental sync (§3/§4) runs OFF the request path in a
    background thread so a large repo never blocks startup or this response."""
    data = await request.json()
    input_dir = data.get("dir")
    if not (CODEGRAPH_OK and _cg is not None):
        reason = _cg_disabled_reason or (
            "extra not installed" if _cg is None else "unavailable"
        )
        return {"ok": False, "status": "disabled", "reason": reason}
    if not input_dir:
        return JSONResponse({"ok": False, "error": "missing 'dir'"}, status_code=400)

    def _run():
        try:
            _cg.init_or_sync(input_dir)
        except Exception as e:  # status endpoint surfaces this; don't crash
            print(f"[codegraph] build failed ({type(e).__name__}: {e})", flush=True)
        if CODEGRAPH_LIVE:
            try:
                _cg.start_watch(input_dir)
            except Exception:
                pass

    print(f"[codegraph] init requested for {input_dir!r} (building in background)",
          flush=True)
    asyncio.get_event_loop().run_in_executor(None, _run)
    return {"ok": True, "status": "building", "dir": input_dir}


@app.post("/codegraph/recall")
async def codegraph_recall(request: Request):
    """Called by the ``recall_codeconcept`` tool: concept tags → code digest."""
    data = await request.json()
    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    print(f"[codegraph] recall tags={tags!r}", flush=True)
    if not (CODEGRAPH_OK and _cg is not None):
        return {"text": "Code-concept index is not available."}
    return {"text": _cg.recall_codeconcept(tags)}


@app.post("/codegraph/query")
async def codegraph_query(request: Request):
    """Called by the ``query_codeconcept`` tool: free text → code digest."""
    data = await request.json()
    query = data.get("query") or ""
    print(f"[codegraph] query={query!r}", flush=True)
    if not (CODEGRAPH_OK and _cg is not None):
        return {"text": "Code-concept index is not available."}
    return {"text": _cg.query_codeconcept(query)}


@app.get("/codegraph/status")
async def codegraph_status():
    if not (CODEGRAPH_OK and _cg is not None):
        return {"status": "disabled",
                "reason": _cg_disabled_reason,
                "missing": (_cg.missing_deps() if _cg is not None else None)}
    return _cg.status()


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, request: Request):
    """Everything else (e.g. GET /v1/models) goes straight to the upstream."""
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.request(
            request.method, f"{UPSTREAM_BASE}/{path}",
            content=await request.body(), headers=UPSTREAM_HEADERS,
        )
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=49786)
