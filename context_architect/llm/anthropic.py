"""Anthropic backend (optional extra: ``pip install context_architect[anthropic]``)."""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMError
from .reasoning import params_for


class AnthropicBackend:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        reasoning: Optional[str] = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on extra
            raise LLMError(
                "the anthropic package is required: pip install context_architect[anthropic]"
            ) from e
        import anthropic

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning

    def complete(self, system: str, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            **params_for("anthropic", self.reasoning),
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
