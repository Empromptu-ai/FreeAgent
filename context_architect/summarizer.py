"""Compress a turn's new activity into a labeled summary + an archived copy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .config import Config
from .llm.base import LLMBackend, complete_json
from .models import (
    CA_KIND,
    CA_SUMMARY,
    KIND_FULL_TURN,
    KIND_SUMMARY,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    messages_to_dicts,
)

_SYSTEM = (
    "You are the assistant. Summarize the new conversation and tool activity above into a concise, "
    "first-person account of what you are working on. "
    "Return ONLY a JSON object with keys 'label' and 'summary'. "
    "'label' is a 1-3 word kebab-case label of what this turn was about. 'summary' is at most {n} "
    "sentences, written in the FIRST PERSON and PRESENT PROGRESSIVE, as if you are mid-task and "
    "narrating your own ongoing work (e.g. \"Working on X: I've read Y and am now editing Z\"). "
    "Capture your intent, the actions taken, and where things stand so far. Be specific and "
    "factual; do not invent details."
)


@dataclass
class SummaryResult:
    label: str
    summary: str
    archive_key: str
    message: Message
    prompt: str
    raw_response: Dict[str, Any]


def render_activity(new_activity: List[Message]) -> str:
    """Flatten the turn's messages into a plain-text transcript for the LLM."""
    lines: List[str] = []
    for m in new_activity:
        text = m.text().strip()
        if text:
            lines.append(f"[{m.role.value}] {text}")
    return "\n\n".join(lines)


def _fallback_summary(transcript: str) -> Dict[str, str]:
    """Deterministic, LLM-free summary for when the model call fails.

    Summarization is a best-effort enrichment: a backend hiccup (unparseable
    output, empty reasoning-only completion, timeout) must never abort the turn's
    rework, because that would freeze the whole compacted history. We still have
    the full turn archived, so we degrade to a truncated snippet plus the recall
    pointer the caller appends, and let the pipeline proceed.
    """
    snippet = " ".join(transcript.split())[:280]
    return {"label": "turn", "summary": snippet or "(summary unavailable)"}


def summarize_turn(
    backend: LLMBackend,
    config: Config,
    new_activity: List[Message],
    turn_index: int,
) -> SummaryResult:
    transcript = render_activity(new_activity)
    system = _SYSTEM.format(n=config.summary_max_sentences)
    prompt = (
        "Here is the agent's activity for the turn to summarize:\n\n"
        f"{transcript if transcript else '(no textual activity)'}"
    )

    try:
        result = complete_json(backend, system, prompt)
    except Exception:
        # Best-effort: never let a summarization failure abort the turn's rework
        # (which would freeze the compacted history). Fall back to a snippet.
        result = _fallback_summary(transcript)
    label = str(result.get("label") or "turn").strip()
    summary = str(result.get("summary") or "").strip()

    archive_key = f"turn-{turn_index:04d}"
    rendered = f"[{label}] {summary} (recall: {archive_key})"

    message = Message(
        role=Role.ASSISTANT,
        blocks=[TextBlock(text=rendered)],
        metadata={
            CA_KIND: KIND_SUMMARY,
            "archive_key": archive_key,
            "label": label,
            "turn": turn_index,
        },
    )

    return SummaryResult(
        label=label,
        summary=summary,
        archive_key=archive_key,
        message=message,
        prompt=prompt,
        raw_response=result,
    )


def build_full_turn(
    new_activity: List[Message],
    turn_index: int,
    summary_message: Message,
    archive_key: str,
):
    """Build the full-text representation of a turn for the recency window.

    Keeps text, tool-use, and tool-result blocks (only images are dropped) and
    preserves each message's role, so a windowed turn retains its full tool
    chain. This matters for smaller models: seeing that they already ran e.g.
    ``todowrite`` or an edit — and what came back — is what keeps them executing
    the plan instead of re-announcing it and stopping. Once the turn ages out of
    the window it is demoted to its one-line summary, so the verbatim tool detail
    is bounded to the most recent ``num_full_text_turns`` turns. Messages left
    with no retained blocks are dropped so the model never sees empty turns.

    The first ("head") kept message carries the rendered summary text under
    ``CA_SUMMARY`` so the turn can be demoted to a plain summary once it ages out
    of the window, without a second summarization call.

    Returns ``(messages, had_content)``. When ``had_content`` is False the turn
    had nothing worth keeping and the caller should represent it by its summary
    directly.
    """
    summary_text = summary_message.text()
    kept: List[Message] = []
    for m in new_activity:
        blocks = [
            b
            for b in m.blocks
            if (isinstance(b, TextBlock) and b.text.strip())
            or isinstance(b, (ToolUseBlock, ToolResultBlock))
        ]
        if not blocks:
            continue
        metadata = {CA_KIND: KIND_FULL_TURN, "turn": turn_index}
        if not kept:
            metadata[CA_SUMMARY] = summary_text
            metadata["archive_key"] = archive_key
        kept.append(Message(role=m.role, blocks=blocks, metadata=metadata))
    return kept, bool(kept)


def build_archive_payload(
    new_activity: List[Message],
    turn_index: int,
    label: str,
    summary: str,
) -> Dict[str, Any]:
    return {
        "turn": turn_index,
        "label": label,
        "summary": summary,
        "messages": messages_to_dicts(new_activity),
    }
