"""Reasoning-level normalization and per-provider request translation.

Reasoning-capable models express "effort" differently per provider, so the
library carries a single normalized level (``off`` / ``low`` / ``medium`` /
``high`` / ``None``) and translates it into each provider's request shape at the
call site via :func:`params_for`.

``None`` means "emit nothing" — the model's own default applies (the historical
behavior before this knob existed).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Canonical "disabled" value; several human spellings map onto it.
OFF = "off"
_OFF_ALIASES = {"off", "none", "false", "no", "0", "disable", "disabled"}

# Anthropic takes a thinking token budget rather than a label, so discrete
# levels map onto budgets. Tune here if you want deeper/shallower thinking.
_ANTHROPIC_BUDGET = {"low": 1024, "medium": 4096, "high": 16384}


def normalize(raw: Optional[str]) -> Optional[str]:
    """Canonicalize a raw env value into a reasoning level.

    Blank/unset -> ``None`` (model default). Off-aliases -> ``"off"``. Anything
    else is lowercased and passed through (so ``low``/``medium``/``high`` work,
    and future levels do too without a code change).
    """
    if raw is None:
        return None
    v = raw.strip().lower()
    if not v:
        return None
    if v in _OFF_ALIASES:
        return OFF
    return v


def params_for(provider: str, level: Optional[str]) -> Dict[str, Any]:
    """Return the request-body fragment to merge for ``provider`` at ``level``.

    Returns ``{}`` when ``level`` is ``None`` so callers can unconditionally
    ``body.update(params_for(...))`` and get the model default when unset.

    Providers:
      - ``"ollama-openai"`` : Ollama's OpenAI-compatible ``/v1`` endpoint
      - ``"ollama"``        : Ollama's native ``/api/chat`` endpoint
      - ``"openai"``        : OpenAI chat completions
      - ``"anthropic"``     : Anthropic messages
    """
    if level is None:
        return {}
    p = (provider or "").lower()

    if p in ("ollama-openai", "openai"):
        # Both speak the OpenAI ``reasoning_effort`` field.
        return {"reasoning_effort": "none" if level == OFF else level}

    if p == "ollama":
        # Native endpoint uses ``think``: a bool, or a level string on models
        # that support graded thinking.
        return {"think": False if level == OFF else level}

    if p == "anthropic":
        if level == OFF:
            return {}  # omit ``thinking`` entirely -> disabled
        budget = _ANTHROPIC_BUDGET.get(level)
        if budget is None:
            return {}  # unknown level -> leave default rather than guess
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    return {}
