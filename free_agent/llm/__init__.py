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

"""LLM backend selection."""

from __future__ import annotations

from ..config import LLMConfig
from .base import LLMBackend, LLMError, complete_json, extract_json
from .fake import FakeBackend
from .ollama import OllamaBackend


def build_backend(cfg: LLMConfig) -> LLMBackend:
    provider = (cfg.provider or "").lower()
    if provider == "ollama":
        return OllamaBackend(
            base_url=cfg.base_url or "http://localhost:11434",
            model=cfg.model,
            temperature=cfg.temperature,
            timeout=cfg.timeout,
            reasoning=cfg.reasoning,
        )
    if provider == "fake":
        return FakeBackend()
    if provider == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            reasoning=cfg.reasoning,
        )
    if provider == "openai":
        from .openai import OpenAIBackend

        return OpenAIBackend(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            reasoning=cfg.reasoning,
        )
    raise ValueError(f"unknown llm provider: {cfg.provider!r}")


__all__ = [
    "LLMBackend",
    "LLMError",
    "build_backend",
    "complete_json",
    "extract_json",
    "FakeBackend",
    "OllamaBackend",
]
