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

"""On-disk storage backend (the default).

Layout under ``{root}/{session_id}/``::

    archive/turn-0001.json   full raw turn payloads (recall targets)
    state.json               compact live history + file ledger + counters
    audit.log                append-only JSONL, one record per turn/recall
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_id(session_id: str) -> str:
    # Keep session ids filesystem-safe without silently colliding.
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in session_id)


class FilesystemStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    # -- paths --------------------------------------------------------------
    def _session_dir(self, session_id: str) -> Path:
        return self.root / _safe_id(session_id)

    def _archive_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "archive"

    # -- archive ------------------------------------------------------------
    def write_archive(self, session_id: str, key: str, payload: Dict[str, Any]) -> None:
        d = self._archive_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{key}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def read_archive(self, session_id: str, key: str) -> Optional[Dict[str, Any]]:
        p = self._archive_dir(session_id) / f"{key}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def list_archive_keys(self, session_id: str) -> List[str]:
        d = self._archive_dir(session_id)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    # -- state --------------------------------------------------------------
    def write_state(self, session_id: str, state: Dict[str, Any]) -> None:
        d = self._session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(d / "state.json")

    def read_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        p = self._session_dir(session_id) / "state.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    # -- audit --------------------------------------------------------------
    def append_audit(self, session_id: str, record: Dict[str, Any]) -> None:
        d = self._session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "audit.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_audit(self, session_id: str) -> List[Dict[str, Any]]:
        p = self._session_dir(session_id) / "audit.log"
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    # -- fork ---------------------------------------------------------------
    def copy_session(self, src_session_id: str, dst_session_id: str) -> None:
        src = self._session_dir(src_session_id)
        dst = self._session_dir(dst_session_id)
        if not src.exists():
            raise FileNotFoundError(f"no such session to fork: {src_session_id}")
        if dst.exists():
            raise FileExistsError(f"fork target already exists: {dst_session_id}")
        shutil.copytree(src, dst)
