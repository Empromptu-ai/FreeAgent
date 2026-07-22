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

"""OpenAI backend (optional extra: ``pip install free_agent[openai]``)."""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMError
from .reasoning import params_for


class OpenAIBackend:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        reasoning: Optional[str] = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on extra
            raise LLMError(
                "the openai package is required: pip install free_agent[openai]"
            ) from e
        import openai

        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning

    def complete(self, system: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **params_for("openai", self.reasoning),
        )
        return resp.choices[0].message.content or ""
