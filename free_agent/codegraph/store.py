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

"""On-disk layout for one indexed codebase (§1 of the spec).

    FA_STORAGE_ROOT/_codegraph/<sha1(abs_input_dir)>/
        config.json      embedder/llm model, ollama url, leiden resolution, sim threshold
        manifest.json    {file_path: {hash, mtime, node_ids: [...]}}
        graph.pkl        networkx.MultiDiGraph (pickled)
        entities.json    {node_id: EntityRecord}
        concepts.json    {concept_id: ConceptRecord}
        embeddings.npz   {id -> float32 vector}

networkx / numpy are imported lazily so importing this module never fails when
the optional extra is missing; only graph/embedding IO touches them.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import ConceptRecord, EntityRecord

CODEGRAPH_DIRNAME = "_codegraph"

DEFAULT_CONFIG = {
    "embedder_model": "nomic-embed-text",
    "llm_model": None,            # filled from FA_CODEGRAPH_MODEL / FA_MODEL at build
    "ollama_url": "http://localhost:11434",
    "resolution": 1.0,            # main Leiden granularity knob (§5)
    "sim_threshold": 0.75,        # semantic-edge cutoff (§5)
    "sim_top_k": 8,               # top-k semantic neighbors per node (§5)
    "w_struct": 1.0,
    "w_sim": 1.0,
    "query_top_k": 5,             # concepts returned by query_codeconcept (§6)
    "match_jaccard": 0.5,         # concept identity-matching threshold (§notes)
    "languages": ["python", "typescript"],
    "summarize_workers": 6,       # concurrent Ollama digest calls (FA_CODEGRAPH_WORKERS)
    "skip_methods": False,        # summarize only classes + top-level functions
}


def store_dir_for(input_dir: str, storage_root: Path) -> Path:
    """FA_STORAGE_ROOT/_codegraph/<sha1(abs_input_dir)>/ for ``input_dir``."""
    abs_dir = str(Path(input_dir).expanduser().resolve())
    sha = hashlib.sha1(abs_dir.encode("utf-8")).hexdigest()
    return Path(storage_root).expanduser().resolve() / CODEGRAPH_DIRNAME / sha


class Store:
    """Reads/writes the six on-disk artifacts for one indexed directory."""

    def __init__(self, input_dir: str, storage_root: Path):
        self.input_dir = str(Path(input_dir).expanduser().resolve())
        self.root = store_dir_for(input_dir, storage_root)

    # ---- paths -----------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def graph_path(self) -> Path:
        return self.root / "graph.pkl"

    @property
    def entities_path(self) -> Path:
        return self.root / "entities.json"

    @property
    def concepts_path(self) -> Path:
        return self.root / "concepts.json"

    @property
    def embeddings_path(self) -> Path:
        return self.root / "embeddings.npz"

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def ensure_dir(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- config ----------------------------------------------------------
    def load_config(self) -> Dict[str, Any]:
        cfg = dict(DEFAULT_CONFIG)
        if self.config_path.exists():
            cfg.update(json.loads(self.config_path.read_text()))
        return cfg

    def save_config(self, cfg: Dict[str, Any]) -> None:
        self.ensure_dir()
        self.config_path.write_text(json.dumps(cfg, indent=2))

    # ---- manifest --------------------------------------------------------
    def load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        self.ensure_dir()
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

    # ---- entities --------------------------------------------------------
    def load_entities(self) -> Dict[str, EntityRecord]:
        if not self.entities_path.exists():
            return {}
        raw = json.loads(self.entities_path.read_text())
        return {k: EntityRecord.from_dict(v) for k, v in raw.items()}

    def save_entities(self, entities: Dict[str, EntityRecord]) -> None:
        self.ensure_dir()
        raw = {k: v.to_dict() for k, v in entities.items()}
        self.entities_path.write_text(json.dumps(raw, indent=2))

    # ---- concepts --------------------------------------------------------
    def load_concepts(self) -> Dict[str, ConceptRecord]:
        if not self.concepts_path.exists():
            return {}
        raw = json.loads(self.concepts_path.read_text())
        return {k: ConceptRecord.from_dict(v) for k, v in raw.items()}

    def save_concepts(self, concepts: Dict[str, ConceptRecord]) -> None:
        self.ensure_dir()
        raw = {k: v.to_dict() for k, v in concepts.items()}
        self.concepts_path.write_text(json.dumps(raw, indent=2))

    # ---- graph -----------------------------------------------------------
    def load_graph(self):
        import networkx as nx  # lazy

        if self.graph_path.exists():
            with open(self.graph_path, "rb") as fh:
                return pickle.load(fh)
        return nx.MultiDiGraph()

    def save_graph(self, graph) -> None:
        self.ensure_dir()
        with open(self.graph_path, "wb") as fh:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- embeddings ------------------------------------------------------
    def load_embeddings(self) -> Dict[str, Any]:
        import numpy as np  # lazy

        if not self.embeddings_path.exists():
            return {}
        with np.load(self.embeddings_path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def save_embeddings(self, vectors: Dict[str, Any]) -> None:
        import numpy as np  # lazy

        self.ensure_dir()
        # npz keys can't contain some characters cleanly, but node_ids (paths +
        # "::") are fine for np.savez. Force float32 to keep the file small.
        clean = {k: np.asarray(v, dtype="float32") for k, v in vectors.items()}
        np.savez(self.embeddings_path, **clean)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
