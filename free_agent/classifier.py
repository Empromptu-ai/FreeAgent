"""Split an incoming live context into the four canonical buckets.

The library re-identifies its own previously-injected messages by their
``metadata[fa_kind]`` marker. Whatever is left over after removing pinned
setup, prior summaries, and the file ledger is, by definition, the new
activity produced this turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import Config
from .models import (
    FA_KIND,
    KIND_FILE_LEDGER,
    KIND_FULL_TURN,
    KIND_PINNED,
    KIND_SUMMARY,
    Message,
    Role,
)


@dataclass
class Buckets:
    pinned: List[Message] = field(default_factory=list)
    prior_summaries: List[Message] = field(default_factory=list)
    prior_full_turns: List[Message] = field(default_factory=list)
    file_ledger: List[Message] = field(default_factory=list)  # 0 or 1
    new_activity: List[Message] = field(default_factory=list)


def classify(messages: List[Message], config: Config) -> Buckets:
    b = Buckets()

    # 1) Pinned: the contiguous leading run of system messages (bounded), plus
    #    anything explicitly tagged pinned anywhere.
    n_leading = 0
    for i, m in enumerate(messages):
        if m.role == Role.SYSTEM and m.fa_kind in (None, KIND_PINNED):
            if i == n_leading and n_leading < config.max_pinned_system_messages:
                n_leading += 1
            else:
                break
        else:
            break

    pinned_idx = set(range(n_leading))
    for i, m in enumerate(messages):
        if m.fa_kind == KIND_PINNED:
            pinned_idx.add(i)

    for i, m in enumerate(messages):
        if i in pinned_idx:
            b.pinned.append(m)
        elif m.fa_kind == KIND_SUMMARY:
            b.prior_summaries.append(m)
        elif m.fa_kind == KIND_FULL_TURN:
            b.prior_full_turns.append(m)
        elif m.fa_kind == KIND_FILE_LEDGER:
            b.file_ledger.append(m)
        else:
            b.new_activity.append(m)

    return b
