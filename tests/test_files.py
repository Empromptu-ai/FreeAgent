from context_architect.config import Config
from context_architect.files import FileLedger, detect_touches
from context_architect.llm.fake import FakeBackend
from context_architect.models import Message, Role, ToolUseBlock


def _tool_msg(name, inp):
    return Message(role=Role.ASSISTANT, blocks=[ToolUseBlock(id="t", name=name, input=inp)])


def test_detect_edit_and_read_tools():
    acts = [
        _tool_msg("apply_patch", {"path": "src/a.py"}),
        _tool_msg("read_file", {"file_path": "src/b.py"}),
    ]
    touches = detect_touches(acts)
    kinds = {(t.path, t.kind) for t in touches}
    assert ("src/a.py", "write") in kinds
    assert ("src/b.py", "read") in kinds


def test_detect_camelcase_path_key():
    # Hosts like OpenCode pass the path under camelCase `filePath`.
    touches = detect_touches([_tool_msg("read", {"filePath": "src/a.py"})])
    assert any(t.path == "src/a.py" and t.kind == "read" for t in touches)


def test_extra_path_keys():
    cfg = Config(extra_path_keys={"srcPath"})
    touches = detect_touches([_tool_msg("read", {"srcPath": "src/b.py"})], cfg)
    assert any(t.path == "src/b.py" and t.kind == "read" for t in touches)
    # Unregistered key is still ignored.
    assert detect_touches([_tool_msg("read", {"weirdKey": "src/c.py"})]) == []


def test_detect_readonly_shell():
    acts = [_tool_msg("bash", {"command": "cat src/c.py"})]
    touches = detect_touches(acts)
    assert any(t.path == "src/c.py" and t.kind == "read" for t in touches)


def test_custom_tool_names_detected():
    # A host that exposes edit/read/shell tools under custom names can register
    # them via Config; they merge with (don't replace) the built-in sets.
    cfg = Config(
        extra_write_tools={"MyEditor"},
        extra_read_tools={"my_reader"},
        extra_shell_tools={"my_shell"},
    )
    acts = [
        _tool_msg("MyEditor", {"path": "src/x.py"}),
        _tool_msg("my_reader", {"file_path": "src/y.py"}),
        _tool_msg("my_shell", {"command": "cat src/z.py"}),
    ]
    kinds = {(t.path, t.kind) for t in detect_touches(acts, cfg)}
    assert ("src/x.py", "write") in kinds
    assert ("src/y.py", "read") in kinds
    assert ("src/z.py", "read") in kinds

    # Built-in names still work alongside the custom ones.
    builtin = detect_touches([_tool_msg("apply_patch", {"path": "src/a.py"})], cfg)
    assert any(t.path == "src/a.py" and t.kind == "write" for t in builtin)


def test_shell_write_ignored():
    # A mutating shell command is not counted as a file touch by the ledger.
    acts = [_tool_msg("bash", {"command": "rm -rf build"})]
    assert detect_touches(acts) == []


def test_description_refined_not_overwritten():
    cfg = Config()
    backend = FakeBackend()
    ledger = FileLedger()

    # Seed an existing description.
    ledger.entries  # noqa
    from context_architect.files import FileEntry

    ledger.entries["src/a.py"] = FileEntry(path="src/a.py", description="original desc")

    touches = detect_touches([_tool_msg("edit", {"path": "src/a.py"})])
    touched = ledger.apply_touches(touches, turn_index=2)
    audit = ledger.refine_descriptions(backend, cfg, touched, touches)

    # The refine prompt must carry the prior description as evidence.
    assert "original desc" in audit["prompt"]
    assert ledger.entries["src/a.py"].writes == 1


def test_ledger_message_render_and_roundtrip():
    from context_architect.files import FileEntry

    ledger = FileLedger()
    ledger.entries["x.py"] = FileEntry(path="x.py", description="thing", reads=2, writes=1)
    msg = ledger.to_message()
    assert "x.py" in msg.text()
    restored = FileLedger.from_dict(ledger.to_dict())
    assert restored.entries["x.py"].reads == 2
