"""Reassemble the compact canonical history.

Fixed order:  pinned setup  ->  body  ->  file ledger.

The ``body`` is produced by :func:`context_architect.window.apply_window` and is
already ordered chronologically — older turns as summaries, the most recent
``num_full_text_turns`` turns as their full text. Everything not in these buckets
is dropped from the model's view (it remains recoverable from the on-disk
archive).
"""

from __future__ import annotations

from typing import List, Optional

from .models import Message


def assemble(
    pinned: List[Message],
    body: List[Message],
    file_ledger: Optional[Message],
) -> List[Message]:
    out: List[Message] = []
    out.extend(pinned)
    out.extend(body)
    if file_ledger is not None:
        out.append(file_ledger)
    return out
