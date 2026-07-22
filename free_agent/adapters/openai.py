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

"""Convert between OpenAI Chat Completions format and the normalized model.

OpenAI shape: a flat ``messages`` list. Assistant tool calls live in a
``tool_calls`` array; tool outputs are separate messages with role ``tool`` and
a ``tool_call_id``. We map those onto our block types.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..models import (
    Block,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def to_internal(messages: List[Dict[str, Any]]) -> List[Message]:
    out: List[Message] = []
    for m in messages:
        role_raw = m.get("role", "user")
        if role_raw == "tool":
            out.append(
                Message(
                    role=Role.TOOL,
                    blocks=[
                        ToolResultBlock(
                            tool_use_id=m.get("tool_call_id", ""),
                            content=str(m.get("content", "") or ""),
                        )
                    ],
                )
            )
            continue

        role = Role(role_raw) if role_raw in Role._value2member_map_ else Role.USER
        blocks: List[Block] = []
        content = m.get("content")
        if isinstance(content, str) and content:
            blocks.append(TextBlock(text=content))
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") in ("text", "input_text", "output_text"):
                    blocks.append(TextBlock(text=c.get("text", "")))
        for tc in m.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            blocks.append(ToolUseBlock(id=tc.get("id", ""), name=fn.get("name", ""), input=args))
        out.append(Message(role=role, blocks=blocks, metadata=dict(m.get("metadata", {}) or {})))
    return out


def from_internal(messages: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[ToolResultBlock] = []
        for b in m.blocks:
            if isinstance(b, TextBlock):
                text_parts.append(b.text)
            elif isinstance(b, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {"name": b.name, "arguments": json.dumps(b.input)},
                    }
                )
            elif isinstance(b, ToolResultBlock):
                tool_results.append(b)
            else:
                text_parts.append(m.text())
                break

        # Tool-result blocks become their own role="tool" messages.
        if tool_results:
            for tr in tool_results:
                out.append(
                    {"role": "tool", "tool_call_id": tr.tool_use_id, "content": tr.content}
                )
            if not text_parts and not tool_calls:
                continue

        entry: Dict[str, Any] = {"role": m.role.value, "content": "\n".join(text_parts)}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if m.metadata:
            entry["metadata"] = m.metadata
        out.append(entry)
    return out
