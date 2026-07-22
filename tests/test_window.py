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

"""Recency-window behavior: recent turns kept as full text, older as summaries."""

from free_agent.models import (
    FA_KIND,
    KIND_FULL_TURN,
    KIND_SUMMARY,
    Message,
    Role,
    TextBlock,
    ToolUseBlock,
)
from free_agent.window import apply_window
from helpers import make_agent


def _activity(text, tool=None):
    blocks = [TextBlock(text=text)]
    if tool:
        name, inp = tool
        blocks.append(ToolUseBlock(id="t", name=name, input=inp))
    return Message(role=Role.ASSISTANT, blocks=blocks)


def _run_turns(s, texts):
    """Drive N turns the way a host does: append raw activity onto the compact
    history each turn and rework. Returns the final compact history."""
    history = [Message(role=Role.SYSTEM, blocks=[TextBlock(text="you are an agent")])]
    for i, text in enumerate(texts, start=1):
        ctx = history + [_activity(text, ("read_file", {"path": f"f{i}.py"}))]
        history = s.rework(ctx)
    return history


def test_window_keeps_recent_full_text_and_summarizes_older(tmp_path):
    fa = make_agent(tmp_path, num_full_text_turns=2)
    s = fa.session("w")
    texts = ["reading one", "editing two", "running three", "checking four"]
    h = _run_turns(s, texts)

    kinds = [m.fa_kind for m in h]
    # 4 turns, window 2 -> turns 1,2 summarized; turns 3,4 full text.
    assert kinds.count(KIND_SUMMARY) == 2
    assert kinds.count(KIND_FULL_TURN) == 2

    full = [m for m in h if m.fa_kind == KIND_FULL_TURN]
    full_text = "\n".join(m.text() for m in full)
    # Full-text turns carry the original message text ...
    assert "running three" in full_text
    assert "checking four" in full_text
    # ... AND the tool chain (so a small model sees it is mid-task, not done).
    assert "read_file" in full_text
    assert "f4.py" in full_text

    # Chronological order preserved: summaries (older) then full turns (newer).
    turns = [m.metadata.get("turn") for m in h if m.fa_kind in (KIND_SUMMARY, KIND_FULL_TURN)]
    assert turns == sorted(turns)


def test_aged_out_turn_is_recoverable_and_carries_recall_key(tmp_path):
    fa = make_agent(tmp_path, num_full_text_turns=2)
    s = fa.session("w2")
    h = _run_turns(s, ["a", "b", "c", "d"])

    # Turn 1 aged out into a summary that still points at its archive key.
    summaries = [m for m in h if m.fa_kind == KIND_SUMMARY]
    assert any("turn-0001" in m.text() for m in summaries)
    # And the full detail is still on disk.
    assert "turn-0001" in s.recall("turn-0001")


def test_num_full_text_turns_zero_matches_summary_only(tmp_path):
    fa = make_agent(tmp_path, num_full_text_turns=0)
    s = fa.session("w0")
    h = _run_turns(s, ["a", "b", "c"])
    kinds = [m.fa_kind for m in h]
    assert kinds.count(KIND_SUMMARY) == 3
    assert kinds.count(KIND_FULL_TURN) == 0


def test_window_survives_resume(tmp_path):
    fa = make_agent(tmp_path, num_full_text_turns=2)
    s = fa.session("w3")
    _run_turns(s, ["a", "b", "c", "d"])

    # Re-open the same session over the same storage; window is reconstructed.
    s2 = fa.session("w3")
    kinds = [m.fa_kind for m in s2.live_history]
    assert kinds.count(KIND_SUMMARY) == 2
    assert kinds.count(KIND_FULL_TURN) == 2


def test_apply_window_unit_demotes_by_cutoff():
    def full(turn, text, summary):
        return Message(
            role=Role.ASSISTANT,
            blocks=[TextBlock(text=text)],
            metadata={FA_KIND: KIND_FULL_TURN, "turn": turn, "fa_summary": summary,
                      "archive_key": f"turn-{turn:04d}"},
        )

    prior_full = [full(1, "one", "[t] s1 (recall: turn-0001)"),
                  full(2, "two", "[t] s2 (recall: turn-0002)")]
    new_full = [Message(role=Role.ASSISTANT, blocks=[TextBlock(text="three")],
                        metadata={FA_KIND: KIND_FULL_TURN, "turn": 3,
                                  "fa_summary": "[t] s3 (recall: turn-0003)",
                                  "archive_key": "turn-0003"})]
    body = apply_window([], prior_full, new_full, None, current_turn=3, num_full_text_turns=2)

    # cutoff = 3 - 2 = 1 -> turn 1 demoted, turns 2,3 stay full.
    by_turn = {m.metadata["turn"]: m.fa_kind for m in body}
    assert by_turn == {1: KIND_SUMMARY, 2: KIND_FULL_TURN, 3: KIND_FULL_TURN}
    assert [m.metadata["turn"] for m in body] == [1, 2, 3]
