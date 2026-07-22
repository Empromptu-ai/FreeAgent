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

"""Ollama backend using the local HTTP API (no API key).

Uses only the standard library (``urllib``) so the core install stays
dependency-free.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from .base import LLMError
from .reasoning import params_for


class OllamaBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        reasoning: Optional[str] = None,
    ):
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.reasoning = reasoning

    def complete(self, system: str, prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        # Native /api/chat carries reasoning as a top-level ``think`` field.
        body.update(params_for("ollama", self.reasoning))
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError(f"ollama request failed: {e}") from e
        try:
            return payload["message"]["content"]
        except (KeyError, TypeError) as e:  # pragma: no cover - defensive
            raise LLMError(f"unexpected ollama response shape: {payload!r}") from e
