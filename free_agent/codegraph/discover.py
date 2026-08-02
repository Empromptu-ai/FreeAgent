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

"""Single source of truth for *what counts as code under investigation*.

Both the parser (full/incremental parse) and the engine (hash manifest, file
watching) discover files through here, so the "skip node_modules / caches /
vendored / generated files" rules can't drift between them.

Rules, in order:
  * only files whose extension is a known source language (``SRC_EXTS``);
  * never descend into a skipped directory (``SKIP_DIRS`` + any hidden dir +
    ``*.egg-info`` + user additions via ``FA_CODEGRAPH_EXCLUDE``);
  * skip obviously-generated files (``*.min.js`` etc.) and anything larger than
    ``FA_CODEGRAPH_MAX_FILE_KB`` (default 512 KB) — a minified bundle or a giant
    vendored blob is not worth an LLM summary and dwarfs everything else.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Set

# Source languages we can actually parse (tree-sitter grammars available).
SRC_EXTS: Set[str] = {
    ".py", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs",
}

# Directory names that are never "the code under investigation": VCS metadata,
# dependency trees, build output, caches, tooling, IDE state, virtualenvs.
SKIP_DIRS: Set[str] = {
    # VCS / tooling
    ".git", ".hg", ".svn",
    # Python
    "__pycache__", "venv", ".venv", "env", ".env", ".tox", ".nox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".hypothesis",
    "site-packages", ".eggs",
    # JS / TS
    "node_modules", "bower_components", ".pnpm-store", ".yarn",
    ".next", ".nuxt", ".svelte-kit", ".turbo", ".parcel-cache",
    # Generic build / output / deps
    "dist", "build", "out", "target", "vendor", "coverage", ".coverage",
    ".cache", ".gradle", ".idea", ".vscode", ".DS_Store",
}

# Filename suffixes that are generated/minified and not worth summarizing.
_SKIP_FILE_SUFFIXES = (
    ".min.js", ".min.ts", ".min.mjs", ".min.cjs",
    ".bundle.js", ".d.ts",
)

# Default max file size to consider (KB). Overridable via env.
_DEFAULT_MAX_KB = 512


def _extra_excludes() -> Set[str]:
    raw = os.environ.get("FA_CODEGRAPH_EXCLUDE", "")
    return {d.strip() for d in raw.split(",") if d.strip()}


def _max_file_bytes() -> int:
    try:
        kb = int(os.environ.get("FA_CODEGRAPH_MAX_FILE_KB", _DEFAULT_MAX_KB))
    except ValueError:
        kb = _DEFAULT_MAX_KB
    return max(1, kb) * 1024


def skip_dir(name: str, extra: Set[str] = frozenset()) -> bool:
    """True if a directory named ``name`` should not be descended into.

    Hidden directories (leading dot) are skipped wholesale — dotfolders are
    almost always tooling/cache state, not code — with the small allow-list of
    dotted names above already covered by equality.
    """
    if name in SKIP_DIRS or name in extra:
        return True
    if name.endswith(".egg-info"):
        return True
    if name.startswith(".") and name not in (".",):
        return True
    return False


def is_source_file(path: Path) -> bool:
    """True if ``path`` is a parseable, non-generated, reasonably-sized source
    file. Size/suffix checks keep minified bundles and vendored blobs out."""
    name = path.name
    if path.suffix not in SRC_EXTS:
        return False
    lower = name.lower()
    if any(lower.endswith(sfx) for sfx in _SKIP_FILE_SUFFIXES):
        return False
    try:
        if path.stat().st_size > _max_file_bytes():
            return False
    except OSError:
        return False
    return True


def is_tracked_file(path: Path, root: Path) -> bool:
    """True if ``path`` (under ``root``) is a source file we track AND none of
    its parent directories are skipped. Used by the file watcher, which is handed
    individual paths rather than a walk it can prune."""
    try:
        rel_parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        return False
    extra = _extra_excludes()
    for part in rel_parts[:-1]:
        if skip_dir(part, extra):
            return False
    return is_source_file(Path(path))


def iter_source_files(root: Path) -> Iterator[Path]:
    """Yield every source file under ``root``, pruning skipped directories in
    place so we never even walk into node_modules/caches/etc."""
    extra = _extra_excludes()
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place so os.walk does not descend into skipped trees.
        dirnames[:] = [d for d in dirnames if not skip_dir(d, extra)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_source_file(p):
                yield p
