from context_architect import Config
from context_architect.assembler import assemble
from context_architect.classifier import classify
from context_architect.models import (
    CA_KIND,
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
        _user("prior summary", {CA_KIND: KIND_SUMMARY, "archive_key": "turn-0001"}),
        _user("file ledger", {CA_KIND: KIND_FILE_LEDGER}),
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
        _user("s1", {CA_KIND: KIND_SUMMARY}),
        _user("s2", {CA_KIND: KIND_SUMMARY}),
    ]
    ledger = _user("ledger", {CA_KIND: KIND_FILE_LEDGER})
    out = assemble(pinned, body, ledger)
    assert [m.text() for m in out] == ["setup", "s1", "s2", "ledger"]
    # ledger always last, pinned always first
    assert out[0].role == Role.SYSTEM
    assert out[-1].ca_kind == KIND_FILE_LEDGER
