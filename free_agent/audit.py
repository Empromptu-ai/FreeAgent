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

"""Durable per-turn audit records.

One JSON record is appended to the session's ``audit.log`` for every ``rework``
and every ``recall``, capturing exactly what went in and what came out so a
user can audit the compression process.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .models import Message


def estimate_tokens(messages: List[Message]) -> int:
    """Cheap char/4 token estimate; good enough for audit sizing."""
    chars = sum(len(m.text()) for m in messages)
    return chars // 4


def rework_record(
    *,
    turn_index: int,
    input_messages: List[Message],
    buckets: Dict[str, int],
    summary_prompt: str,
    summary_response: Dict[str, Any],
    files_audit: Dict[str, Any],
    detected_touches: List[Dict[str, str]],
    output_messages: List[Message],
    archive_key: str,
) -> Dict[str, Any]:
    return {
        "event": "rework",
        "turn": turn_index,
        "input": {
            "messages": len(input_messages),
            "est_tokens": estimate_tokens(input_messages),
        },
        "buckets": buckets,
        "summary": {"prompt": summary_prompt, "response": summary_response},
        "files": {
            "detected": detected_touches,
            "llm": files_audit,
        },
        "archive_key": archive_key,
        "output": {
            "messages": len(output_messages),
            "est_tokens": estimate_tokens(output_messages),
        },
    }


def recall_record(*, key: str, found: bool) -> Dict[str, Any]:
    return {"event": "recall", "key": key, "found": found}
