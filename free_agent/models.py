"""Normalized message and content-block model.

This is the canonical internal representation the library operates on. Host
systems convert their provider-native messages into these types (see
``free_agent.adapters``) before handing them to a :class:`Session`.

Every type is JSON round-trippable via ``to_dict`` / ``from_dict`` so the full
context can be archived to disk and reloaded on resume/fork.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """Base class for content blocks. Subclasses set a distinct ``type``."""

    type: str = field(init=False, default="block")

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Block":
        t = data.get("type")
        cls = _BLOCK_TYPES.get(t)
        if cls is None:
            raise ValueError(f"unknown block type: {t!r}")
        return cls._from_dict(data)  # type: ignore[attr-defined]


@dataclass
class TextBlock(Block):
    text: str = ""

    def __post_init__(self) -> None:
        self.type = "text"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "text", "text": self.text}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "TextBlock":
        return cls(text=data.get("text", ""))


@dataclass
class ThinkingBlock(Block):
    thinking: str = ""

    def __post_init__(self) -> None:
        self.type = "thinking"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "thinking", "thinking": self.thinking}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ThinkingBlock":
        return cls(thinking=data.get("thinking", ""))


@dataclass
class ToolUseBlock(Block):
    id: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = "tool_use"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ToolUseBlock":
        return cls(id=data.get("id", ""), name=data.get("name", ""), input=data.get("input", {}) or {})


@dataclass
class ToolResultBlock(Block):
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False

    def __post_init__(self) -> None:
        self.type = "tool_result"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ToolResultBlock":
        return cls(
            tool_use_id=data.get("tool_use_id", ""),
            content=data.get("content", ""),
            is_error=bool(data.get("is_error", False)),
        )


@dataclass
class ImageBlock(Block):
    source: str = ""  # opaque reference / path / data-url descriptor

    def __post_init__(self) -> None:
        self.type = "image"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "image", "source": self.source}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ImageBlock":
        return cls(source=data.get("source", ""))


_BLOCK_TYPES = {
    "text": TextBlock,
    "thinking": ThinkingBlock,
    "tool_use": ToolUseBlock,
    "tool_result": ToolResultBlock,
    "image": ImageBlock,
}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

# Well-known metadata keys the library uses to re-identify its own injected
# messages when a later turn's context comes back in.
FA_KIND = "fa_kind"
KIND_PINNED = "pinned"
KIND_SUMMARY = "summary"
KIND_FILE_LEDGER = "file_ledger"
# A completed turn kept as its full text (text blocks only) inside the recency
# window. The turn's head message additionally carries FA_SUMMARY (the rendered
# summary text) so it can be demoted to a summary once it ages out — without a
# second summarization call.
KIND_FULL_TURN = "full_turn"
FA_SUMMARY = "fa_summary"


@dataclass
class Message:
    role: Role
    blocks: List[Block] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.role, str):
            self.role = Role(self.role)

    @property
    def fa_kind(self) -> Optional[str]:
        return self.metadata.get(FA_KIND)

    def text(self) -> str:
        """Concatenate all textual content for rendering / token estimation."""
        parts: List[str] = []
        for b in self.blocks:
            if isinstance(b, TextBlock):
                parts.append(b.text)
            elif isinstance(b, ThinkingBlock):
                parts.append(b.thinking)
            elif isinstance(b, ToolUseBlock):
                parts.append(f"{b.name}({b.input})")
            elif isinstance(b, ToolResultBlock):
                parts.append(b.content if isinstance(b.content, str) else str(b.content))
            elif isinstance(b, ImageBlock):
                parts.append(f"[image {b.source}]")
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "blocks": [b.to_dict() for b in self.blocks],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=Role(data["role"]),
            blocks=[Block.from_dict(b) for b in data.get("blocks", [])],
            metadata=data.get("metadata", {}) or {},
        )


def messages_to_dicts(messages: List[Message]) -> List[Dict[str, Any]]:
    return [m.to_dict() for m in messages]


def messages_from_dicts(data: List[Dict[str, Any]]) -> List[Message]:
    return [Message.from_dict(d) for d in data]
