"""Recency window: keep the most recent turns as full text, older as summaries.

The library keeps the last ``num_full_text_turns`` completed turns in the live
context as their full text (text blocks only); every older turn falls back to
its one-line summary. A turn that ages out of the window is *demoted* — its
summary was already produced when the turn was processed and stashed on the
turn's head message (``CA_SUMMARY``), so demotion is a metadata swap, not a
second summarization call.

The result is a single ordered ``body`` (everything between the pinned setup and
the file ledger), sorted by turn index, so summaries and full-text turns stay in
chronological order even in the edge case of a pure-tool turn (no text) sitting
next to full-text neighbors.
"""

from __future__ import annotations

from typing import List, Optional

from .models import (
    CA_KIND,
    CA_SUMMARY,
    KIND_SUMMARY,
    Message,
    Role,
    TextBlock,
)


def _turn_of(m: Message) -> int:
    return int(m.metadata.get("turn", 0))


def _demote(head: Message, turn: int) -> Message:
    """Rebuild the summary message for a full-text turn from its head metadata."""
    return Message(
        role=Role.ASSISTANT,
        blocks=[TextBlock(text=str(head.metadata.get(CA_SUMMARY, "")))],
        metadata={
            CA_KIND: KIND_SUMMARY,
            "turn": turn,
            "archive_key": head.metadata.get("archive_key", ""),
        },
    )


def apply_window(
    prior_summaries: List[Message],
    prior_full_turns: List[Message],
    new_full_turn: List[Message],
    new_summary: Optional[Message],
    current_turn: int,
    num_full_text_turns: int,
) -> List[Message]:
    """Return the ordered body: summaries + full-text turns, chronological.

    ``prior_summaries`` / ``prior_full_turns`` are this library's own previously
    injected messages (already tagged). ``new_full_turn`` is this turn's
    full-text messages (empty for a pure-tool turn); ``new_summary`` is this
    turn's summary message (None only when the turn had no activity at all).
    """
    # turn index -> ("summary", [msg]) | ("full", [msgs])
    reps: dict = {}

    for m in prior_summaries:
        reps[_turn_of(m)] = ("summary", [m])

    # Group prior full-turn messages by turn, preserving order.
    for m in prior_full_turns:
        t = _turn_of(m)
        kind, msgs = reps.get(t, ("full", []))
        if kind != "full":
            msgs = []
        msgs.append(m)
        reps[t] = ("full", msgs)

    # The new turn joins as full text if it has any, else as its summary.
    if new_full_turn:
        reps[current_turn] = ("full", list(new_full_turn))
    elif new_summary is not None:
        reps[current_turn] = ("summary", [new_summary])

    # Demote any full-text turn that has aged out of the window.
    cutoff = current_turn - num_full_text_turns
    for t, (kind, msgs) in list(reps.items()):
        if kind == "full" and t <= cutoff:
            reps[t] = ("summary", [_demote(msgs[0], t)])

    body: List[Message] = []
    for t in sorted(reps):
        body.extend(reps[t][1])
    return body
