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

"""Configuration objects for the library."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set


@dataclass
class LLMConfig:
    """Selects and configures the backend used for summaries / labels.

    provider:
        - "ollama"    : local server, ``base_url`` + ``model`` (no api key)
        - "anthropic" : requires ``api_key`` (or ANTHROPIC_API_KEY) + ``model``
        - "openai"    : requires ``api_key`` (or OPENAI_API_KEY) + ``model``
        - "fake"      : deterministic, offline (for tests / dry runs)
    """

    provider: str = "ollama"
    model: str = "llama3.1"
    base_url: Optional[str] = None  # ollama, e.g. http://localhost:11434
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0
    # Reasoning/thinking effort for reasoning-capable models. None leaves the
    # model's own default; otherwise "off" / "low" / "medium" / "high" (see
    # free_agent.llm.reasoning). Translated per-provider at call time.
    reasoning: Optional[str] = None


@dataclass
class Config:
    """Top-level library configuration."""

    storage_root: str = "~/.free_agent"
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Summary shaping.
    summary_max_sentences: int = 3

    # Recency window: keep this many of the most recent turns as their full text
    # messages (text blocks only) in the live context instead of a summary.
    # Older turns fall back to the summarized form. 0 reproduces the original
    # behavior (every completed turn is a summary immediately).
    num_full_text_turns: int = 3

    # How many leading system messages to treat as pinned setup when none are
    # explicitly tagged. The contiguous leading system run is pinned regardless;
    # this only bounds it.
    max_pinned_system_messages: int = 4

    # Custom tool names for file-touch detection, merged with the built-in
    # heuristics. Names are matched case-insensitively. Use these when your host
    # exposes edit/read/shell tools under names the defaults don't recognize.
    extra_write_tools: Set[str] = field(default_factory=set)
    extra_read_tools: Set[str] = field(default_factory=set)
    extra_shell_tools: Set[str] = field(default_factory=set)

    # Extra input keys a tool_use block may carry a file path under, merged with
    # the built-ins (which already cover common snake_case and camelCase names).
    extra_path_keys: Set[str] = field(default_factory=set)

    def resolved_root(self) -> Path:
        return Path(os.path.expanduser(self.storage_root)).resolve()
