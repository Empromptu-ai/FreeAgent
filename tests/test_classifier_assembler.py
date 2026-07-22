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

from free_agent import Config
from free_agent.assembler import assemble
from free_agent.classifier import classify
from free_agent.models import (
    FA_KIND,
    KIND_FILE_LEDGER,
    KIND_SUMMARY,
    Message,
    Role,
    TextBlock,
)


def _sys(text):
    return Message(role=Role.SYSTEM, blocks=[TextBlock(text=text)])


def _user(text, meta=None):
    return Message(role=Role.USER, blocks=[TextBlock(text=text)], metadata=meta or {})


def test_classify_buckets():
    cfg = Config()
    msgs = [
        _sys("setup instructions"),
        _user("prior summary", {FA_KIND: KIND_SUMMARY, "archive_key": "turn-0001"}),
        _user("file ledger", {FA_KIND: KIND_FILE_LEDGER}),
        _user("agent did stuff this turn"),
        Message(role=Role.ASSISTANT, blocks=[TextBlock(text="assistant reply")]),
    ]
    b = classify(msgs, cfg)
    assert len(b.pinned) == 1
    assert len(b.prior_summaries) == 1
    assert len(b.file_ledger) == 1
    assert len(b.new_activity) == 2


def test_assemble_order():
    pinned = [_sys("setup")]
    body = [
        _user("s1", {FA_KIND: KIND_SUMMARY}),
        _user("s2", {FA_KIND: KIND_SUMMARY}),
    ]
    ledger = _user("ledger", {FA_KIND: KIND_FILE_LEDGER})
    out = assemble(pinned, body, ledger)
    assert [m.text() for m in out] == ["setup", "s1", "s2", "ledger"]
    # ledger always last, pinned always first
    assert out[0].role == Role.SYSTEM
    assert out[-1].fa_kind == KIND_FILE_LEDGER
