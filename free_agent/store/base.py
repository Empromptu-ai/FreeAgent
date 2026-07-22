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

"""Storage backend protocol.

A backend persists three kinds of data per session:
  * archive blobs  - full raw turn payloads, keyed (recall targets)
  * state          - the compact live history + ledger + turn counter
  * audit records  - append-only JSONL log of what went in/out each turn
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    def write_archive(self, session_id: str, key: str, payload: Dict[str, Any]) -> None: ...

    def read_archive(self, session_id: str, key: str) -> Optional[Dict[str, Any]]: ...

    def list_archive_keys(self, session_id: str) -> List[str]: ...

    def write_state(self, session_id: str, state: Dict[str, Any]) -> None: ...

    def read_state(self, session_id: str) -> Optional[Dict[str, Any]]: ...

    def append_audit(self, session_id: str, record: Dict[str, Any]) -> None: ...

    def read_audit(self, session_id: str) -> List[Dict[str, Any]]: ...

    def copy_session(self, src_session_id: str, dst_session_id: str) -> None: ...
