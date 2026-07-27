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

"""Tree-sitter → networkx structural graph (§3.1, §5-structural).

Extracts function / class / method entities per file and the structural edges
between them (contains / calls / inherits / imports). Resolution of calls and
inheritance is name-based and best-effort — imprecise, but more than enough to
seed the community detection that concepts are built on.

Python + TypeScript/TSX out of the box. Grammars are loaded lazily and a
missing grammar for a language is skipped (not fatal), so the parser degrades
per-language rather than failing the whole build.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schemas import EntityRecord
from .store import sha1_text

# extension -> language key
_EXT_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".jsx": "tsx",
    ".mjs": "javascript",
}

# Node types that define an entity, per grammar. (kind is normalized later.)
_DEF_TYPES = {
    "python": {"function_definition", "class_definition"},
    "typescript": {
        "function_declaration", "method_definition", "class_declaration",
        "abstract_class_declaration", "generator_function_declaration",
    },
    "tsx": {
        "function_declaration", "method_definition", "class_declaration",
        "abstract_class_declaration", "generator_function_declaration",
    },
    "javascript": {
        "function_declaration", "method_definition", "class_declaration",
        "generator_function_declaration",
    },
}

_CLASS_TYPES = {
    "class_definition", "class_declaration", "abstract_class_declaration",
}
_CALL_TYPES = {"call", "call_expression"}

_SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".egg-info", "site-packages",
}


class _Grammars:
    """Lazily-built, cached tree-sitter parsers keyed by language."""

    def __init__(self, languages: List[str]):
        self._languages = languages
        self._parsers: Dict[str, object] = {}
        self._built = False

    def _language_obj(self, lang: str):
        from tree_sitter import Language

        if lang == "python":
            import tree_sitter_python as m
            ptr = m.language()
        elif lang in ("typescript",):
            import tree_sitter_typescript as m
            ptr = m.language_typescript()
        elif lang in ("tsx",):
            import tree_sitter_typescript as m
            ptr = m.language_tsx()
        elif lang == "javascript":
            try:
                import tree_sitter_javascript as m
                ptr = m.language()
            except Exception:
                # Fall back to the TS grammar, a superset for our purposes.
                import tree_sitter_typescript as m
                ptr = m.language_typescript()
        else:
            raise KeyError(lang)
        try:
            return Language(ptr)              # tree-sitter >= 0.22
        except TypeError:
            return Language(ptr, lang)        # tree-sitter 0.21

    def parser(self, lang: str):
        if lang in self._parsers:
            return self._parsers[lang]
        from tree_sitter import Parser

        language = self._language_obj(lang)
        try:
            p = Parser(language)              # newer API
        except TypeError:
            p = Parser()
            p.set_language(language)          # older API
        self._parsers[lang] = p
        return p


def _iter_source_files(input_dir: Path, exts) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.endswith(".egg-info")
        ]
        for fn in filenames:
            if Path(fn).suffix in exts:
                out.append(Path(dirpath) / fn)
    return out


def _node_name(node, src: bytes) -> Optional[str]:
    name = node.child_by_field_name("name")
    if name is not None:
        return src[name.start_byte:name.end_byte].decode("utf-8", "replace")
    return None


def _signature(node, src: bytes) -> str:
    """Header text: from the node start to the first line of its body."""
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    text = src[node.start_byte:end].decode("utf-8", "replace").strip()
    # Keep it to the first ~3 lines so embeddings/prompts stay compact.
    return "\n".join(text.splitlines()[:3])


def _entity_kind(node, enclosing_class: bool) -> str:
    if node.type in _CLASS_TYPES:
        return "class"
    if node.type == "method_definition" or enclosing_class:
        return "method"
    return "function"


def _base_names(node, src: bytes) -> List[str]:
    """Superclass / implemented-interface names for a class node (best-effort)."""
    names: List[str] = []
    for field in ("superclasses",):  # python: argument_list under 'superclasses'
        sc = node.child_by_field_name(field)
        if sc is not None:
            for ch in sc.named_children:
                names.append(src[ch.start_byte:ch.end_byte].decode("utf-8", "replace"))
    # typescript: class_heritage child holding extends/implements clauses
    for ch in node.named_children:
        if ch.type in ("class_heritage", "extends_clause", "implements_clause"):
            for ident in ch.named_children:
                txt = src[ident.start_byte:ident.end_byte].decode("utf-8", "replace")
                names.append(txt.split("<")[0].strip())
    return [n.split(".")[-1].strip() for n in names if n.strip()]


def _call_names(node, src: bytes) -> List[str]:
    """Callee names invoked anywhere inside ``node``'s subtree."""
    out: List[str] = []
    stack = list(node.named_children)
    while stack:
        n = stack.pop()
        if n.type in _CALL_TYPES:
            fn = n.child_by_field_name("function")
            if fn is not None:
                txt = src[fn.start_byte:fn.end_byte].decode("utf-8", "replace")
                out.append(txt.split(".")[-1].split("(")[0].strip())
        stack.extend(n.named_children)
    return out


def parse_dir(
    input_dir: str,
    languages: List[str],
    only_files: Optional[List[str]] = None,
) -> Tuple["object", Dict[str, List[EntityRecord]]]:
    """Parse ``input_dir`` (or just ``only_files``) into a structural graph.

    Returns (networkx.MultiDiGraph, {rel_file_path: [EntityRecord, ...]}). The
    graph carries the same nodes (node_id) plus edges; EntityRecords carry the
    span/signature/code_hash needed downstream. Node ids are
    ``<rel_path>::<Qualified.Name>``.
    """
    import networkx as nx

    root = Path(input_dir).expanduser().resolve()
    exts = {e for e, l in _EXT_LANG.items() if _EXT_LANG.get(e) in languages
            or l in languages}
    grammars = _Grammars(languages)

    graph = nx.MultiDiGraph()
    per_file: Dict[str, List[EntityRecord]] = {}

    # name -> node_ids, for best-effort call/inherit resolution.
    name_index: Dict[str, List[str]] = {}
    # module rel path (no ext) -> file rel path, for import resolution.
    module_index: Dict[str, str] = {}

    if only_files is not None:
        files = [Path(f) for f in only_files]
    else:
        files = _iter_source_files(root, exts)

    parsed: List[Tuple[str, List[EntityRecord], List[Tuple[str, List[str], List[str]]]]] = []

    for path in files:
        lang = _EXT_LANG.get(path.suffix)
        if lang not in languages and lang not in ("tsx",):
            # tsx handled under the typescript umbrella
            if not (lang == "tsx" and "typescript" in languages):
                continue
        try:
            src = path.read_bytes()
            parser = grammars.parser(lang)
            tree = parser.parse(src)
        except Exception:
            continue  # unparseable / missing grammar: skip this file

        rel = os.path.relpath(path, root)
        module_index[os.path.splitext(rel)[0]] = rel
        entities: List[EntityRecord] = []
        # (node_id, call_names, base_names) collected for edge wiring pass 2.
        edge_facts: List[Tuple[str, List[str], List[str]]] = []
        def_types = _DEF_TYPES.get(lang, _DEF_TYPES["typescript"])

        def visit(node, class_qual: Optional[str]):
            for child in node.named_children:
                if child.type in def_types:
                    name = _node_name(child, src)
                    if not name:
                        visit(child, class_qual)
                        continue
                    enclosing_class = class_qual is not None
                    qual = f"{class_qual}.{name}" if class_qual else name
                    node_id = f"{rel}::{qual}"
                    kind = _entity_kind(child, enclosing_class)
                    start = child.start_point[0] + 1
                    end = child.end_point[0] + 1
                    body_text = src[child.start_byte:child.end_byte].decode(
                        "utf-8", "replace"
                    )
                    rec = EntityRecord(
                        node_id=node_id,
                        kind=kind,
                        file=rel,
                        span=[start, end],
                        code_hash=sha1_text(body_text),
                        signature=_signature(child, src),
                    )
                    entities.append(rec)
                    name_index.setdefault(name, []).append(node_id)
                    edge_facts.append((
                        node_id,
                        _call_names(child, src),
                        _base_names(child, src) if child.type in _CLASS_TYPES else [],
                    ))
                    # Recurse INTO classes (to reach methods); a class becomes the
                    # new qualifier. Recurse into functions too (nested defs).
                    next_qual = qual if child.type in _CLASS_TYPES else class_qual
                    visit(child, next_qual)
                else:
                    visit(child, class_qual)

        visit(tree.root_node, None)
        per_file[rel] = entities
        parsed.append((rel, entities, edge_facts))

    # Pass 1: nodes + contains edges.
    for rel, entities, _ in parsed:
        prev_class: Dict[str, str] = {}
        for rec in entities:
            graph.add_node(
                rec.node_id, kind=rec.kind, file=rec.file, span=rec.span,
            )
            qual = rec.node_id.split("::", 1)[1]
            if "." in qual:  # method -> its class
                parent_qual = qual.rsplit(".", 1)[0]
                parent_id = f"{rel}::{parent_qual}"
                if graph.has_node(parent_id):
                    graph.add_edge(parent_id, rec.node_id, type="contains")

    # Pass 2: calls + inherits (name-based).
    for rel, entities, edge_facts in parsed:
        for node_id, calls, bases in edge_facts:
            for cn in calls:
                for target in name_index.get(cn, []):
                    if target != node_id:
                        graph.add_edge(node_id, target, type="calls")
            for bn in bases:
                for target in name_index.get(bn, []):
                    if graph.nodes.get(target, {}).get("kind") == "class":
                        graph.add_edge(node_id, target, type="inherits")

    return graph, per_file
