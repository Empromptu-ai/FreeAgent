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

"""Graph-RAG code-concept system (optional extra ``free_agent[codegraph]``).

Everything heavy (tree-sitter, networkx, leidenalg, igraph, numpy, watchdog) is
imported lazily inside submodules. The proxy imports THIS module at top level;
if the extra isn't installed, ``available()`` returns False and every entry
point degrades to a no-op / "disabled" message rather than raising.

Public API (§6):
    init_or_sync, on_file_changed, get_concept_index,
    recall_codeconcept, query_codeconcept, status, start_watch
"""

from __future__ import annotations

from typing import List

# These are pure-stdlib and always import.
from .engine import (
    get_concept_index,
    init_or_sync,
    on_file_changed,
    query_codeconcept,
    recall_codeconcept,
    status,
)

_REQUIRED = ("tree_sitter", "networkx", "numpy", "igraph", "leidenalg")


def available() -> bool:
    """True iff the optional runtime deps (and at least one grammar) import."""
    import importlib.util

    for mod in _REQUIRED:
        if importlib.util.find_spec(mod) is None:
            return False
    # Need at least one tree-sitter grammar to parse anything.
    if (importlib.util.find_spec("tree_sitter_python") is None
            and importlib.util.find_spec("tree_sitter_typescript") is None):
        return False
    return True


def missing_deps() -> List[str]:
    """Names of the required modules that are not importable (for diagnostics)."""
    import importlib.util

    out = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
    if (importlib.util.find_spec("tree_sitter_python") is None
            and importlib.util.find_spec("tree_sitter_typescript") is None):
        out.append("tree_sitter_python|tree_sitter_typescript")
    return out


def start_watch(input_dir: str) -> bool:
    from .watch import start_watch as _sw
    return _sw(input_dir)


__all__ = [
    "available",
    "missing_deps",
    "init_or_sync",
    "on_file_changed",
    "get_concept_index",
    "recall_codeconcept",
    "query_codeconcept",
    "status",
    "start_watch",
]
