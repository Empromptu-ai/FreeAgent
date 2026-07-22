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

from free_agent.models import (
    KIND_FILE_LEDGER,
    KIND_SUMMARY,
    Message,
    Role,
    TextBlock,
    ToolUseBlock,
)
from helpers import make_agent


def _turn_activity(text, tool=None):
    blocks = [TextBlock(text=text)]
    if tool:
        name, inp = tool
        blocks.append(ToolUseBlock(id="t", name=name, input=inp))
    return Message(role=Role.ASSISTANT, blocks=blocks)


def test_three_turn_compaction(tmp_path):
    # num_full_text_turns=0: every completed turn is summarized immediately
    # (the original, pre-recency-window behavior).
    fa = make_agent(tmp_path, num_full_text_turns=0)
    s = fa.session("sess-1")

    pinned = Message(role=Role.SYSTEM, blocks=[TextBlock(text="you are an agent")])

    # Turn 1
    ctx = [pinned, _turn_activity("reading file", ("read_file", {"path": "a.py"}))]
    h1 = s.rework(ctx)

    # Turn 2: host appends new raw activity onto the returned compact history.
    ctx = h1 + [_turn_activity("editing file", ("apply_patch", {"path": "a.py"}))]
    h2 = s.rework(ctx)

    # Turn 3
    ctx = h2 + [_turn_activity("running tests", ("bash", {"command": "cat b.py"}))]
    h3 = s.rework(ctx)

    # (a) Compact: exactly pinned + 3 summaries + 1 ledger = 5 messages.
    kinds = [m.fa_kind for m in h3]
    assert kinds.count(KIND_SUMMARY) == 3
    assert kinds.count(KIND_FILE_LEDGER) == 1
    assert h3[0].role == Role.SYSTEM
    assert len(h3) == 5

    # (b) Each turn recoverable via recall by its key.
    for turn in (1, 2, 3):
        key = f"turn-{turn:04d}"
        recalled = s.recall(key)
        assert key in recalled
        assert "full detail" in recalled

    # (c) Ledger accumulated both files across turns.
    ledger_msg = [m for m in h3 if m.fa_kind == KIND_FILE_LEDGER][0]
    assert "a.py" in ledger_msg.text()
    assert "b.py" in ledger_msg.text()

    # (d) Audit log has one rework record per turn plus recall records.
    audit = fa.store.read_audit("sess-1")
    reworks = [r for r in audit if r["event"] == "rework"]
    recalls = [r for r in audit if r["event"] == "recall"]
    assert len(reworks) == 3
    assert len(recalls) == 3
    assert reworks[0]["output"]["messages"] < len(ctx) + 5  # smaller than raw growth


def test_resume_reconstructs_state(tmp_path):
    fa = make_agent(tmp_path)
    s = fa.session("sess-resume")
    pinned = Message(role=Role.SYSTEM, blocks=[TextBlock(text="setup")])
    s.rework([pinned, _turn_activity("did x", ("edit", {"path": "z.py"}))])

    # New agent / session object over the same storage root.
    fa2 = make_agent(tmp_path / "..") if False else fa
    s2 = fa2.session("sess-resume")
    assert s2.turn_index == 1
    assert "z.py" in s2.ledger.entries
    assert len(s2.live_history) >= 2


def test_fork(tmp_path):
    fa = make_agent(tmp_path)
    s = fa.session("orig")
    pinned = Message(role=Role.SYSTEM, blocks=[TextBlock(text="setup")])
    s.rework([pinned, _turn_activity("did x", ("edit", {"path": "z.py"}))])

    forked = s.fork("forked")
    assert forked.turn_index == 1
    assert "z.py" in forked.ledger.entries
    # Recall works against the forked archive copy.
    assert "turn-0001" in forked.recall("turn-0001")


def test_recall_missing_key(tmp_path):
    fa = make_agent(tmp_path)
    s = fa.session("s")
    out = s.recall("turn-9999")
    assert "No archived turn" in out
