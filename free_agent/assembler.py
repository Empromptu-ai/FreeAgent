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

"""Reassemble the compact canonical history.

Fixed order:  pinned setup  ->  body  ->  file ledger.

The ``body`` is produced by :func:`free_agent.window.apply_window` and is
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
