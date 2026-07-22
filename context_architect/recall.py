"""The recall tool: schema (per provider format) + result rendering.

The agent calls ``recall_turn`` with an archive ``key`` (surfaced in each
summary as ``(recall: <key>)``) to pull back the full, uncompressed detail of a
past turn when the short summary is not enough.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

TOOL_NAME = "recall_turn"
_DESCRIPTION = (
    "Retrieve the full, uncompressed detail of a past turn by its archive key. "
    "Summaries in the history end with '(recall: <key>)'. Use this ONLY when a "
    "summary is insufficient and you need the exact original messages, tool "
    "calls, or results from that turn."
)


def recall_tool_schema(fmt: str = "anthropic") -> Dict[str, Any]:
    fmt = fmt.lower()
    params = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The archive key of the turn to recall, e.g. 'turn-0003'.",
            }
        },
        "required": ["key"],
    }
    if fmt == "anthropic":
        return {"name": TOOL_NAME, "description": _DESCRIPTION, "input_schema": params}
    if fmt == "openai":
        return {
            "type": "function",
            "function": {"name": TOOL_NAME, "description": _DESCRIPTION, "parameters": params},
        }
    raise ValueError(f"unknown tool schema format: {fmt!r}")


def render_recall_result(payload: Optional[Dict[str, Any]], key: str) -> str:
    """Render an archived turn payload into text for a tool_result."""
    if payload is None:
        return f"No archived turn found for key {key!r}."
    lines = [f"Recalled turn {payload.get('turn')} (key {key}) — {payload.get('label', '')}"]
    lines.append(f"Summary was: {payload.get('summary', '')}")
    lines.append("--- full detail ---")
    for msg in payload.get("messages", []):
        role = msg.get("role", "?")
        for block in msg.get("blocks", []):
            btype = block.get("type")
            if btype == "text":
                lines.append(f"[{role}] {block.get('text', '')}")
            elif btype == "thinking":
                lines.append(f"[{role}:thinking] {block.get('thinking', '')}")
            elif btype == "tool_use":
                lines.append(f"[{role}:tool_use {block.get('name')}] {block.get('input')}")
            elif btype == "tool_result":
                err = " (error)" if block.get("is_error") else ""
                lines.append(f"[{role}:tool_result{err}] {block.get('content')}")
            elif btype == "image":
                lines.append(f"[{role}:image] {block.get('source')}")
    return "\n".join(lines)
