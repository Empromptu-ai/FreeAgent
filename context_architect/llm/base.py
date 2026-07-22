"""LLM backend protocol and shared helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, system: str, prompt: str) -> str:
        """Return the model's completion text for a single-shot prompt."""
        ...


class LLMError(RuntimeError):
    pass


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort parse of a JSON object out of a model completion.

    Tolerates code fences and surrounding prose by grabbing the outermost
    ``{...}`` span. Raises :class:`LLMError` if nothing parses.
    """
    text = text.strip()
    # Strip common ```json fences.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception as e:  # pragma: no cover - defensive
            raise LLMError(f"could not parse JSON from model output: {e}\n---\n{text}")
    raise LLMError(f"no JSON object found in model output:\n{text}")


def complete_json(
    backend: LLMBackend,
    system: str,
    prompt: str,
    retries: int = 1,
) -> Dict[str, Any]:
    """Ask ``backend`` for a JSON object, retrying once on parse failure."""
    last_err: Optional[Exception] = None
    attempt_prompt = prompt
    for _ in range(retries + 1):
        raw = backend.complete(system, attempt_prompt)
        try:
            return extract_json(raw)
        except LLMError as e:
            last_err = e
            attempt_prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON. "
                "Respond with ONLY a single JSON object, no prose, no code fences."
            )
    raise last_err  # type: ignore[misc]
