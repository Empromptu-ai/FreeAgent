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

"""A failing summarizer/ledger backend must never abort a turn's rework.

Regression for the wedge where a backend that returns empty/unparseable output
(observed with reasoning models that spend their whole token budget thinking)
raised out of ``rework``, freezing the compacted history and — via the proxy's
unadvanced fold pointer — making every subsequent turn disappear.
"""

from free_agent.models import (
    FA_SUMMARY,
    KIND_FILE_LEDGER,
    KIND_SUMMARY,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from helpers import make_agent


class BrokenBackend:
    """Every JSON call comes back empty -> ``complete_json`` raises."""

    def complete(self, system: str, prompt: str) -> str:
        return ""


def _turn(text, path):
    return [
        Message(role=Role.USER, blocks=[TextBlock(text="question about " + path)]),
        Message(role=Role.ASSISTANT, blocks=[ToolUseBlock(id="1", name="read", input={"path": path})]),
        Message(role=Role.TOOL, blocks=[ToolResultBlock(tool_use_id="1", content="contents of " + path)]),
        Message(role=Role.ASSISTANT, blocks=[TextBlock(text=text)]),
    ]


def test_broken_backend_does_not_abort_rework(tmp_path):
    fa = make_agent(tmp_path, num_full_text_turns=1)
    s = fa.session("broken")
    s.backend = BrokenBackend()

    pinned = Message(role=Role.SYSTEM, blocks=[TextBlock(text="you are an agent")])

    # Three turns, each folded through a backend that fails every LLM call.
    h = s.rework([pinned] + _turn("answer one", "a.py"))
    h = s.rework(h + _turn("answer two", "b.py"))
    h = s.rework(h + _turn("answer three", "c.py"))

    # rework never threw: the turn counter advanced all the way through, so the
    # proxy's fold pointer would advance too (no freeze).
    assert s.turn_index == 3

    # Older turns still fall back to summaries (not dropped); the newest stays
    # full text. Degradation produced usable summary text, not an empty message.
    summaries = [m for m in h if m.fa_kind == KIND_SUMMARY]
    assert summaries, "older turns should demote to (fallback) summaries"
    assert all(m.text().strip() for m in summaries)

    # The full detail is still archived, so recall keeps working.
    assert "contents of a.py" in s.recall("turn-0001")

    # The ledger tracked every touched file even though description refinement
    # (also an LLM call) failed.
    ledger = [m for m in h if m.fa_kind == KIND_FILE_LEDGER][0]
    for path in ("a.py", "b.py", "c.py"):
        assert path in ledger.text()

    # State persisted normally despite the failures.
    reworks = [r for r in fa.store.read_audit("broken") if r["event"] == "rework"]
    assert len(reworks) == 3
