"""Deterministic offline backend for tests and dry runs.

Produces plausible JSON for the two prompt shapes the library issues (a
turn summary and a batch of file descriptions) without any network access.
"""

from __future__ import annotations

import json
from typing import List, Tuple


class FakeBackend:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        low = (system + "\n" + prompt).lower()

        if "file" in low and "describe" in low:
            # File-ledger refinement prompt: echo back one line per listed path.
            paths = _paths_from_prompt(prompt)
            return json.dumps({p: f"file at {p} (auto-described)" for p in paths})

        # Default: turn summary prompt.
        first = prompt.strip().splitlines()[0][:60] if prompt.strip() else "activity"
        return json.dumps(
            {
                "label": "turn",
                "summary": f"Summarized turn activity starting with: {first}",
            }
        )


def _paths_from_prompt(prompt: str) -> List[str]:
    paths: List[str] = []
    for line in prompt.splitlines():
        line = line.strip()
        # The files prompt lists paths as "- <path>: <evidence>".
        if line.startswith("- "):
            body = line[2:]
            path = body.split(":", 1)[0].strip()
            if path:
                paths.append(path)
    return paths
