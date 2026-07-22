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

"""free_agent — rework a coding agent's context at every turn boundary.

Compact summaries stay in the model's view; full detail is archived to disk and
recoverable on demand via the ``recall_turn`` tool.
"""

from __future__ import annotations

from .config import Config, LLMConfig
from .models import (
    Block,
    ImageBlock,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    messages_from_dicts,
    messages_to_dicts,
)
from .recall import TOOL_NAME as RECALL_TOOL_NAME
from .recall import recall_tool_schema
from .session import FreeAgent, Session
from .store import FilesystemStore, StorageBackend

__version__ = "0.1.0"

__all__ = [
    "FreeAgent",
    "Session",
    "Config",
    "LLMConfig",
    "Message",
    "Role",
    "Block",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ImageBlock",
    "messages_to_dicts",
    "messages_from_dicts",
    "recall_tool_schema",
    "RECALL_TOOL_NAME",
    "FilesystemStore",
    "StorageBackend",
]
