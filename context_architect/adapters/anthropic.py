"""Convert between Anthropic Messages format and the normalized model.

Anthropic shape: a separate ``system`` string plus ``messages`` where each has
a ``role`` and ``content`` that is either a string or a list of content blocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    Block,
    ImageBlock,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _block_from_anthropic(b: Any) -> Optional[Block]:
    if isinstance(b, str):
        return TextBlock(text=b)
    t = b.get("type")
    if t == "text":
        return TextBlock(text=b.get("text", ""))
    if t == "thinking":
        return ThinkingBlock(thinking=b.get("thinking", ""))
    if t == "tool_use":
        return ToolUseBlock(id=b.get("id", ""), name=b.get("name", ""), input=b.get("input", {}) or {})
    if t == "tool_result":
        content = b.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        return ToolResultBlock(
            tool_use_id=b.get("tool_use_id", ""),
            content=content if isinstance(content, str) else str(content),
            is_error=bool(b.get("is_error", False)),
        )
    if t == "image":
        return ImageBlock(source=str(b.get("source", "")))
    return TextBlock(text=str(b))


def to_internal(
    messages: List[Dict[str, Any]],
    system: Optional[str] = None,
) -> List[Message]:
    out: List[Message] = []
    if system:
        out.append(Message(role=Role.SYSTEM, blocks=[TextBlock(text=system)]))
    for m in messages:
        role = Role(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, str):
            blocks: List[Block] = [TextBlock(text=content)]
        else:
            blocks = []
            for b in content:
                blk = _block_from_anthropic(b)
                if blk is not None:
                    blocks.append(blk)
        out.append(Message(role=role, blocks=blocks, metadata=dict(m.get("metadata", {}) or {})))
    return out


def _block_to_anthropic(b: Block) -> Any:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ThinkingBlock):
        return {"type": "thinking", "thinking": b.thinking}
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": b.content,
            "is_error": b.is_error,
        }
    if isinstance(b, ImageBlock):
        return {"type": "image", "source": b.source}
    return {"type": "text", "text": str(b)}


def from_internal(messages: List[Message]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Return ``(system, messages)`` in Anthropic shape.

    Leading system messages are concatenated into the ``system`` string.
    """
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM and not out:
            system_parts.append(m.text())
            continue
        content = [_block_to_anthropic(b) for b in m.blocks]
        entry: Dict[str, Any] = {"role": m.role.value, "content": content}
        if m.metadata:
            entry["metadata"] = m.metadata
        out.append(entry)
    system = "\n\n".join(p for p in system_parts if p) or None
    return system, out
