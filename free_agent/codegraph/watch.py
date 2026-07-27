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

"""watchdog live mode (§4 live mode): a thin, debounced wrapper that calls
engine.on_file_changed for each edited source file.

Imported lazily so a missing watchdog dep never breaks the rest of codegraph.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional

from . import engine as _engine

_SRC_EXTS = {".py", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs"}
_DEBOUNCE_S = 2.0

_observers: Dict[str, object] = {}


def start_watch(input_dir: str, debounce_s: float = _DEBOUNCE_S) -> bool:
    """Start (or reuse) an observer for ``input_dir``. Returns True if watching."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:
        return False

    key = str(Path(input_dir).expanduser().resolve())
    if key in _observers:
        return True

    class _Handler(FileSystemEventHandler):
        def __init__(self):
            self._timers: Dict[str, threading.Timer] = {}
            self._lock = threading.Lock()

        def _schedule(self, path: str):
            if os.path.splitext(path)[1] not in _SRC_EXTS:
                return
            with self._lock:
                t = self._timers.get(path)
                if t:
                    t.cancel()
                timer = threading.Timer(
                    debounce_s, _engine.on_file_changed, args=(path, key)
                )
                timer.daemon = True
                self._timers[path] = timer
                timer.start()

        def on_modified(self, event):
            if not event.is_directory:
                self._schedule(event.src_path)

        on_created = on_modified

        def on_moved(self, event):
            if not event.is_directory:
                self._schedule(event.dest_path)

        def on_deleted(self, event):
            if not event.is_directory:
                self._schedule(event.src_path)

    obs = Observer()
    obs.schedule(_Handler(), key, recursive=True)
    obs.daemon = True
    obs.start()
    _observers[key] = obs
    return True


def stop_watch(input_dir: Optional[str] = None) -> None:
    if input_dir is None:
        for obs in _observers.values():
            obs.stop()
        _observers.clear()
        return
    key = str(Path(input_dir).expanduser().resolve())
    obs = _observers.pop(key, None)
    if obs:
        obs.stop()
