from context_architect import recall_tool_schema
from context_architect.adapters import anthropic as a_adapt
from context_architect.adapters import openai as o_adapt
from context_architect.models import Role, TextBlock, ToolResultBlock, ToolUseBlock


def test_anthropic_roundtrip():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "let me look"},
                {"type": "tool_use", "id": "1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "file contents"}
            ],
        },
    ]
    internal = a_adapt.to_internal(msgs, system="you are helpful")
    assert internal[0].role == Role.SYSTEM
    assert any(isinstance(b, ToolUseBlock) for b in internal[2].blocks)
    assert any(isinstance(b, ToolResultBlock) for b in internal[3].blocks)

    system, out = a_adapt.from_internal(internal)
    assert system == "you are helpful"
    assert out[0]["role"] == "user"


def test_openai_roundtrip():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "edit", "arguments": '{"path": "a.py"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]
    internal = o_adapt.to_internal(msgs)
    tool_uses = [b for m in internal for b in m.blocks if isinstance(b, ToolUseBlock)]
    assert tool_uses and tool_uses[0].input == {"path": "a.py"}

    out = o_adapt.from_internal(internal)
    roles = [m["role"] for m in out]
    assert "tool" in roles
    assert any(m.get("tool_calls") for m in out)


def test_recall_schema_formats():
    a = recall_tool_schema("anthropic")
    assert a["name"] == "recall_turn"
    assert "input_schema" in a

    o = recall_tool_schema("openai")
    assert o["type"] == "function"
    assert o["function"]["name"] == "recall_turn"
