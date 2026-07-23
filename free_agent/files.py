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

"""File-touch detection and the refining per-file ledger.

Detection is heuristic:
  * tool_use blocks whose name matches known read/write tool sets;
  * shell tool_use blocks whose command begins with a read-only viewer.

The built-in tool-name sets cover common hosts; a caller can register
additional edit/read/shell tool names via ``Config.extra_write_tools`` /
``extra_read_tools`` / ``extra_shell_tools``, which are merged with the defaults.

Each detected path accumulates read/write counts and a one-line description
that is *refined* (not overwritten) across turns via a single batched LLM call.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .config import Config
from .llm.base import LLMBackend, complete_json
from .models import (
    FA_KIND,
    KIND_FILE_LEDGER,
    Message,
    Role,
    TextBlock,
    ToolUseBlock,
)

WRITE_TOOLS = {
    "str_replace_editor",
    "str_replace_based_edit_tool",
    "edit",
    "apply_patch",
    "patch",
    "write",
    "write_file",
    "create",
    "notebook_edit",
    "multiedit",
}

READ_TOOLS = {
    "read",
    "read_file",
    "open",
    "view",
    "cat_file",
    "view_image",
    "read_image",
}

SHELL_TOOLS = {"bash", "shell", "run", "run_command", "execute", "sh", "terminal"}

# Read-only leading commands whose file arguments we count as reads.
READONLY_CMDS = {
    "cat", "head", "tail", "less", "more", "nl", "grep", "rg", "sed",
    "awk", "od", "hexdump", "xxd", "wc", "diff", "file",
}

# Common keys a tool_use block puts a file path under. Both snake_case and
# camelCase are included, since hosts differ (e.g. OpenCode uses ``filePath``).
_PATH_INPUT_KEYS = (
    "path",
    "file_path",
    "filePath",
    "file",
    "filename",
    "fileName",
    "target_file",
    "targetFile",
    "notebook_path",
    "notebookPath",
)


@dataclass
class FileEntry:
    path: str
    description: str = ""
    first_seen_turn: int = 0
    last_touched_turn: int = 0
    reads: int = 0
    writes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileEntry":
        return cls(**d)


@dataclass
class Touch:
    path: str
    kind: str  # "read" | "write"
    evidence: str


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _extract_path_from_input(
    inp: Dict[str, Any], path_keys=_PATH_INPUT_KEYS
) -> Optional[str]:
    for key in path_keys:
        val = inp.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _looks_like_path(tok: str) -> bool:
    if not tok or tok.startswith("-"):
        return False
    return ("/" in tok) or ("." in tok and not tok.endswith("."))


def _paths_from_shell(command: str) -> List[str]:
    """Extract file args from a read-only shell command; [] otherwise."""
    command = command.strip()
    if not command:
        return []
    # Only consider the first simple command (before pipes/redirs/;&).
    head = re.split(r"[|&;><]", command, maxsplit=1)[0].strip()
    try:
        tokens = shlex.split(head)
    except ValueError:
        tokens = head.split()
    if not tokens:
        return []
    cmd = tokens[0].rsplit("/", 1)[-1]
    if cmd not in READONLY_CMDS:
        return []
    return [t for t in tokens[1:] if _looks_like_path(t)]


def detect_touches(
    new_activity: List[Message], config: Optional[Config] = None
) -> List[Touch]:
    write_tools = WRITE_TOOLS
    read_tools = READ_TOOLS
    shell_tools = SHELL_TOOLS
    path_keys = _PATH_INPUT_KEYS
    if config is not None:
        # Merge caller-registered names (matched case-insensitively) with the
        # built-in heuristics.
        write_tools = write_tools | {t.lower() for t in config.extra_write_tools}
        read_tools = read_tools | {t.lower() for t in config.extra_read_tools}
        shell_tools = shell_tools | {t.lower() for t in config.extra_shell_tools}
        path_keys = tuple(path_keys) + tuple(config.extra_path_keys)

    touches: List[Touch] = []
    for m in new_activity:
        for b in m.blocks:
            if not isinstance(b, ToolUseBlock):
                continue
            name = (b.name or "").lower()
            inp = b.input or {}

            if name in write_tools:
                p = _extract_path_from_input(inp, path_keys)
                if p:
                    touches.append(Touch(p, "write", f"{name} tool"))
            elif name in read_tools:
                p = _extract_path_from_input(inp, path_keys)
                if p:
                    touches.append(Touch(p, "read", f"{name} tool"))
            elif name in shell_tools:
                cmd = inp.get("command") or inp.get("cmd") or ""
                if isinstance(cmd, list):
                    cmd = " ".join(str(c) for c in cmd)
                for p in _paths_from_shell(str(cmd)):
                    touches.append(Touch(p, "read", f"shell: {str(cmd)[:80]}"))
    return touches


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class FileLedger:
    def __init__(self, entries: Optional[Dict[str, FileEntry]] = None):
        self.entries: Dict[str, FileEntry] = entries or {}

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {p: e.to_dict() for p, e in self.entries.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileLedger":
        return cls({p: FileEntry.from_dict(v) for p, v in (d or {}).items()})

    # -- update -------------------------------------------------------------
    def apply_touches(self, touches: List[Touch], turn_index: int) -> List[str]:
        """Record counts/timestamps. Returns the paths touched this turn."""
        touched_paths: List[str] = []
        for t in touches:
            e = self.entries.get(t.path)
            if e is None:
                e = FileEntry(path=t.path, first_seen_turn=turn_index)
                self.entries[t.path] = e
            e.last_touched_turn = turn_index
            if t.kind == "write":
                e.writes += 1
            else:
                e.reads += 1
            if t.path not in touched_paths:
                touched_paths.append(t.path)
        return touched_paths

    def refine_descriptions(
        self,
        backend: LLMBackend,
        config: Config,
        touched_paths: List[str],
        touches: List[Touch],
    ) -> Dict[str, Any]:
        """Refine one-line descriptions for the touched paths via one LLM call.

        Returns audit info: {"prompt": str, "response": dict} (empty if no work).
        """
        if not touched_paths:
            return {}

        # Gather this turn's evidence per path.
        ev: Dict[str, List[str]] = {}
        for t in touches:
            ev.setdefault(t.path, []).append(f"{t.kind} ({t.evidence})")

        lines = []
        for p in touched_paths:
            existing = self.entries[p].description
            evidence = "; ".join(ev.get(p, []))
            prior = f" | current description: {existing}" if existing else ""
            lines.append(f"- {p}: this turn -> {evidence}{prior}")

        system = (
            "You maintain a running one-line description per file for a coding "
            "agent. For each file, describe its role/purpose based on all evidence. "
            "REFINE the current description with the new evidence; keep prior "
            "understanding unless it is contradicted. Return ONLY a JSON object "
            "mapping each file path to its updated one-line description."
        )
        prompt = "Files to describe (refine current descriptions):\n" + "\n".join(lines)

        try:
            result = complete_json(backend, system, prompt)
        except Exception as e:
            # Best-effort refinement: a backend failure must not abort rework.
            # The touch counts are already applied; we just keep the prior
            # one-line descriptions and record the failure for the audit.
            return {"prompt": prompt, "response": {}, "error": f"{type(e).__name__}: {e}"}
        for p in touched_paths:
            desc = result.get(p)
            if isinstance(desc, str) and desc.strip():
                self.entries[p].description = desc.strip()

        return {"prompt": prompt, "response": result}

    # -- rendering ----------------------------------------------------------
    def to_message(self) -> Optional[Message]:
        if not self.entries:
            return None
        lines = ["File ledger (files the agent has touched):"]
        for p in sorted(self.entries):
            e = self.entries[p]
            desc = e.description or "(no description yet)"
            lines.append(f"- {p} — {desc}  [r{e.reads}/w{e.writes}]")
        return Message(
            role=Role.ASSISTANT, # USER
            blocks=[TextBlock(text="\n".join(lines))],
            metadata={FA_KIND: KIND_FILE_LEDGER},
        )
