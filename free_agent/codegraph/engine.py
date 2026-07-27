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

"""Orchestration + the public API (§6).

Full build (§3), incremental sync (§4), and the three read functions the tools
call. One Engine per indexed directory; a module-level registry keyed by the
resolved input dir lets the proxy endpoints look the right one up by path (or
just use the single active engine).

Status is exposed so query functions can degrade to "index still building"
(§8) rather than error while a background build runs.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .discover import iter_source_files
from .schemas import ConceptRecord, EntityRecord
from .store import Store, file_hash

# ── module-level engine registry ────────────────────────────────────────────
_ENGINES: Dict[str, "Engine"] = {}
_ACTIVE: Optional["Engine"] = None
_LOCK = threading.Lock()


def _resolve_env_config(store: Store) -> Dict[str, object]:
    cfg = store.load_config()
    cfg["ollama_url"] = os.environ.get("OLLAMA_BASE_URL", cfg["ollama_url"])
    cfg["llm_model"] = (
        os.environ.get("FA_CODEGRAPH_MODEL")
        or os.environ.get("FA_MODEL")
        or cfg.get("llm_model")
        or "qwen3.6:35b"
    )
    cfg["embedder_model"] = os.environ.get(
        "FA_CODEGRAPH_EMBED_MODEL", cfg["embedder_model"]
    )
    if os.environ.get("FA_CODEGRAPH_RESOLUTION"):
        cfg["resolution"] = float(os.environ["FA_CODEGRAPH_RESOLUTION"])
    if os.environ.get("FA_CODEGRAPH_WORKERS"):
        try:
            cfg["summarize_workers"] = max(1, int(os.environ["FA_CODEGRAPH_WORKERS"]))
        except ValueError:
            pass
    if os.environ.get("FA_CODEGRAPH_SKIP_METHODS"):
        cfg["skip_methods"] = os.environ["FA_CODEGRAPH_SKIP_METHODS"] == "1"
    return cfg


class Engine:
    def __init__(self, input_dir: str, storage_root: Path):
        self.input_dir = str(Path(input_dir).expanduser().resolve())
        self.store = Store(self.input_dir, storage_root)
        self.status = "idle"           # idle | building | ready | error
        self.error: Optional[str] = None
        self._build_lock = threading.Lock()

    # ── file discovery / hashing ────────────────────────────────────────
    def _current_hashes(self) -> Dict[str, Dict[str, object]]:
        root = Path(self.input_dir)
        out: Dict[str, Dict[str, object]] = {}
        for p in iter_source_files(root):
            rel = os.path.relpath(p, root)
            try:
                out[rel] = {"hash": file_hash(p), "mtime": p.stat().st_mtime}
            except OSError:
                continue
        return out

    # ── LLM/embedding helpers ───────────────────────────────────────────
    def _summarizer(self, cfg):
        from .summarize import Summarizer
        return Summarizer(model=cfg["llm_model"], base_url=cfg["ollama_url"],
                          max_workers=int(cfg.get("summarize_workers", 6)))

    def _embed(self, texts: List[str], cfg):
        from .embed import embed_texts
        return embed_texts(texts, model=cfg["embedder_model"],
                           base_url=cfg["ollama_url"])

    def _entity_code(self, rec: EntityRecord) -> str:
        return read_span(Path(self.input_dir) / rec.file, rec.span)

    def _embed_text_for(self, rec: EntityRecord) -> str:
        return f"{rec.summary}\n{rec.signature}".strip() or rec.node_id

    # ── full build (§3) ─────────────────────────────────────────────────
    def build(self) -> None:
        from .parse import parse_dir

        cfg = _resolve_env_config(self.store)
        self.store.save_config(cfg)

        graph, per_file = parse_dir(self.input_dir, cfg["languages"])
        entities: Dict[str, EntityRecord] = {}
        for recs in per_file.values():
            for rec in recs:
                entities[rec.node_id] = rec
        _log(f"parsed {len(per_file)} files → {len(entities)} entities, "
             f"{graph.number_of_edges()} edges")

        # Entity summaries + tags (expensive; concurrent).
        _log(f"summarizing {len(entities)} entities via {cfg['llm_model']} …")
        self._summarize_entities(list(entities.values()), entities, cfg)
        # Entity embeddings.
        _log(f"embedding {len(entities)} entities via {cfg['embedder_model']} …")
        embeddings = self._embed_entities(list(entities.values()), cfg)

        # Cluster + concept digests.
        concepts = self._recluster(
            graph, entities, embeddings, prev_concepts={}, cfg=cfg,
        )
        _log(f"clustered into {len(concepts)} concepts")
        concept_emb = self._embed_concepts(concepts, cfg)
        embeddings.update(concept_emb)

        manifest = self._build_manifest(per_file)
        self._persist(graph, entities, concepts, embeddings, manifest)
        _log(f"build complete — {len(concepts)} concepts, {len(entities)} entities")

    def _summarize_entities(self, recs: List[EntityRecord],
                            entities: Dict[str, EntityRecord], cfg) -> None:
        if cfg.get("skip_methods"):
            # Methods stay in the graph (structure + clustering) but skip the
            # per-method LLM call; their embedding falls back to the signature.
            skipped = sum(1 for r in recs if r.kind == "method")
            recs = [r for r in recs if r.kind != "method"]
            if skipped:
                _log(f"skip_methods on — not summarizing {skipped} methods")
        if not recs:
            return
        items = [(r.node_id, r.file, self._entity_code(r)) for r in recs]

        def _progress(done, total):
            # Log at every 10% (and the final one) so a big repo shows movement.
            step = max(1, total // 10)
            if done == total or done % step == 0:
                _log(f"  summarized {done}/{total} entities")

        out = self._summarizer(cfg).entities(items, on_progress=_progress)
        for node_id, res in out.items():
            if node_id in entities:
                entities[node_id].tag = res.get("tag", "") or entities[node_id].tag
                entities[node_id].summary = res.get("summary", "")

    def _embed_entities(self, recs: List[EntityRecord], cfg) -> Dict[str, object]:
        if not recs:
            return {}
        texts = [self._embed_text_for(r) for r in recs]
        mat = self._embed(texts, cfg)
        return {r.node_id: mat[i] for i, r in enumerate(recs)}

    def _embed_concepts(self, concepts: Dict[str, ConceptRecord], cfg
                        ) -> Dict[str, object]:
        recs = [c for c in concepts.values() if c.summary or c.tag]
        if not recs:
            return {}
        texts = [f"{c.tag}: {c.summary}".strip() for c in recs]
        mat = self._embed(texts, cfg)
        return {c.concept_id: mat[i] for i, c in enumerate(recs)}

    # ── (re)clustering + concept digests (§3.4, §4, §notes) ──────────────
    def _recluster(self, graph, entities: Dict[str, EntityRecord],
                   embeddings: Dict[str, object],
                   prev_concepts: Dict[str, ConceptRecord], cfg,
                   touched: Optional[set] = None) -> Dict[str, ConceptRecord]:
        from .cluster import build_communities, match_concepts

        node_ids = [n for n in graph.nodes()
                    if graph.nodes[n].get("kind") in ("function", "class", "method")]
        communities = build_communities(
            graph, embeddings, node_ids,
            resolution=cfg["resolution"], sim_threshold=cfg["sim_threshold"],
            sim_top_k=cfg["sim_top_k"], w_struct=cfg["w_struct"], w_sim=cfg["w_sim"],
        )

        prev_members = {cid: c.members for cid, c in prev_concepts.items()}
        seq_start = _next_seq(prev_concepts)
        assignments, _ = match_concepts(
            communities, prev_members,
            jaccard_threshold=cfg["match_jaccard"], id_seq_start=seq_start,
        )

        concepts: Dict[str, ConceptRecord] = {}
        digest_jobs: List[Tuple[str, List[str]]] = []
        for cid, members, lineage in assignments:
            prev = prev_concepts.get(cid)
            member_set = set(members)
            member_files = sorted({entities[m].file for m in members if m in entities})

            # Keep the old record verbatim when members are byte-identical AND
            # nothing in this cluster was touched this sync (§4).
            unchanged = (
                prev is not None
                and set(prev.members) == member_set
                and (touched is None or not (member_set & touched))
            )
            rec = ConceptRecord(
                concept_id=cid, members=members, member_files=member_files,
                split_from=lineage.get("split_from", []),
                merged_from=lineage.get("merged_from", []),
            )
            if unchanged:
                rec.tag, rec.summary = prev.tag, prev.summary
            else:
                lines = []
                for m in members[:40]:
                    e = entities.get(m)
                    if e:
                        lines.append(f"{e.tag}: {e.summary}")
                digest_jobs.append((cid, lines))
            concepts[cid] = rec

        if digest_jobs:
            out = self._summarizer(cfg).concepts(digest_jobs)
            for cid, res in out.items():
                concepts[cid].tag = res.get("tag", "") or concepts[cid].tag
                concepts[cid].summary = res.get("summary", "")

        _wire_neighbors(graph, concepts)
        return concepts

    # ── incremental sync (§4) ────────────────────────────────────────────
    def sync(self, changed_files: Optional[List[str]] = None) -> None:
        from .parse import parse_dir

        if not self.store.exists():
            return self.build()

        cfg = _resolve_env_config(self.store)
        manifest = self.store.load_manifest()
        graph = self.store.load_graph()
        entities = self.store.load_entities()
        prev_concepts = self.store.load_concepts()
        embeddings = self.store.load_embeddings()

        current = self._current_hashes()
        old_files = set(manifest.keys())
        new_files = set(current.keys())
        added = new_files - old_files
        removed = old_files - new_files
        changed = {
            f for f in (new_files & old_files)
            if current[f]["hash"] != manifest[f].get("hash")
        }
        if changed_files is not None:
            rels = {os.path.relpath(f, self.input_dir) if os.path.isabs(f) else f
                    for f in changed_files}
            added &= rels
            changed &= rels
            removed &= rels
        if not (added or removed or changed):
            return  # no-op resync

        # 1. Drop nodes/records/embeddings for removed+changed files.
        drop_files = removed | changed
        touched: set = set()
        for f in drop_files:
            for node_id in list(manifest.get(f, {}).get("node_ids", [])):
                touched |= set(graph.predecessors(node_id)) if graph.has_node(node_id) else set()
                touched |= set(graph.successors(node_id)) if graph.has_node(node_id) else set()
                if graph.has_node(node_id):
                    graph.remove_node(node_id)
                entities.pop(node_id, None)
                embeddings.pop(node_id, None)

        # 2. Re-parse added+changed files, merge nodes/edges in.
        reparse = sorted(added | changed)
        if reparse:
            abs_files = [str(Path(self.input_dir) / f) for f in reparse]
            sub_graph, per_file = parse_dir(self.input_dir, cfg["languages"],
                                            only_files=abs_files)
            graph.add_nodes_from(sub_graph.nodes(data=True))
            graph.add_edges_from(sub_graph.edges(data=True))
            new_recs: List[EntityRecord] = []
            for recs in per_file.values():
                for rec in recs:
                    entities[rec.node_id] = rec
                    new_recs.append(rec)
                    touched.add(rec.node_id)
            # 3. Re-summarize + re-embed only the new/changed entities (§4.4).
            self._summarize_entities(new_recs, entities, cfg)
            embeddings.update(self._embed_entities(new_recs, cfg))

        # 4. 2-hop neighborhood of changed nodes joins the touched set (§4).
        touched |= _bfs(graph, touched, hops=2)

        # 5. Global recluster; only touched/changed concepts get re-digested.
        concepts = self._recluster(graph, entities, embeddings, prev_concepts,
                                   cfg, touched=touched)
        # Concept embeddings for any concept whose digest changed / is new.
        need_emb = {cid for cid in concepts
                    if cid not in embeddings or cid not in prev_concepts}
        recs = [concepts[c] for c in need_emb if concepts[c].summary or concepts[c].tag]
        if recs:
            texts = [f"{c.tag}: {c.summary}".strip() for c in recs]
            mat = self._embed(texts, cfg)
            for i, c in enumerate(recs):
                embeddings[c.concept_id] = mat[i]
        # Drop embeddings for retired concepts.
        for cid in list(embeddings.keys()):
            if cid.startswith("concept-") and cid not in concepts:
                embeddings.pop(cid, None)

        new_manifest = self._build_manifest_from_entities(current, entities)
        self._persist(graph, entities, concepts, embeddings, new_manifest)

    # ── persistence helpers ──────────────────────────────────────────────
    def _build_manifest(self, per_file) -> Dict[str, Dict[str, object]]:
        current = self._current_hashes()
        node_ids_by_file: Dict[str, List[str]] = {}
        for rel, recs in per_file.items():
            node_ids_by_file[rel] = [r.node_id for r in recs]
        manifest = {}
        for rel, meta in current.items():
            manifest[rel] = {
                "hash": meta["hash"], "mtime": meta["mtime"],
                "node_ids": node_ids_by_file.get(rel, []),
            }
        return manifest

    def _build_manifest_from_entities(self, current, entities
                                     ) -> Dict[str, Dict[str, object]]:
        by_file: Dict[str, List[str]] = {}
        for node_id, rec in entities.items():
            by_file.setdefault(rec.file, []).append(node_id)
        manifest = {}
        for rel, meta in current.items():
            manifest[rel] = {
                "hash": meta["hash"], "mtime": meta["mtime"],
                "node_ids": by_file.get(rel, []),
            }
        return manifest

    def _persist(self, graph, entities, concepts, embeddings, manifest) -> None:
        self.store.save_graph(graph)
        self.store.save_entities(entities)
        self.store.save_concepts(concepts)
        self.store.save_embeddings(embeddings)
        self.store.save_manifest(manifest)

    # ── public read API (§6) ─────────────────────────────────────────────
    def concept_index(self) -> str:
        concepts = self.store.load_concepts()
        lines = []
        for c in sorted(concepts.values(), key=lambda x: x.concept_id):
            if c.tag or c.summary:
                lines.append(f"{c.tag or c.concept_id}: {c.summary}")
        return "\n".join(lines)

    def recall(self, tags: List[str]) -> str:
        concepts = self.store.load_concepts()
        seeds = _match_tags(concepts, tags)
        if not seeds:
            return "No matching code concepts found."
        return self._assemble_digest(seeds, concepts)

    def query(self, query_text: str) -> str:
        cfg = self.store.load_config()
        concepts = self.store.load_concepts()
        embeddings = self.store.load_embeddings()
        concept_ids = [c for c in concepts if c in embeddings]
        if not concept_ids:
            return "Concept index is empty."
        from .embed import cosine_matrix, embed_texts
        import numpy as np

        cfg = _resolve_env_config(self.store)
        qvec = embed_texts([query_text], model=cfg["embedder_model"],
                           base_url=cfg["ollama_url"])[0]
        mat = np.asarray([embeddings[c] for c in concept_ids], dtype="float32")
        sims = cosine_matrix(qvec, mat)
        top_k = min(int(cfg.get("query_top_k", 5)), len(concept_ids))
        order = np.argsort(-sims)[:top_k]
        seeds = [concept_ids[i] for i in order]
        return self._assemble_digest(seeds, concepts)

    def _assemble_digest(self, seed_ids: List[str],
                         concepts: Dict[str, ConceptRecord]) -> str:
        entities = self.store.load_entities()
        # seeds + their 1-hop neighbor concepts (§6).
        expanded: List[str] = []
        seen = set()
        for cid in seed_ids:
            for c in [cid] + concepts.get(cid, ConceptRecord(cid)).neighbor_concepts:
                if c not in seen and c in concepts:
                    seen.add(c)
                    expanded.append(c)

        out: List[str] = []
        for cid in expanded:
            c = concepts[cid]
            out.append(f"## {c.tag or cid}\n{c.summary}\n")
            for node_id in c.members:
                e = entities.get(node_id)
                if not e:
                    continue
                lang = _lang_fence(e.file)
                code = read_span(Path(self.input_dir) / e.file, e.span)
                out.append(
                    f"### {e.tag or node_id} ({e.file}:{e.span[0]}-{e.span[1]})\n"
                    f"{e.summary}\n```{lang}\n{code}\n```\n"
                )
        return "\n".join(out).strip()


# ── module functions matching §6 signatures ─────────────────────────────────
def _get_engine(input_dir: Optional[str] = None) -> Optional[Engine]:
    global _ACTIVE
    if input_dir is None:
        return _ACTIVE
    key = str(Path(input_dir).expanduser().resolve())
    return _ENGINES.get(key)


def init_or_sync(input_dir: str, storage_root: Optional[Path] = None) -> None:
    """Create the store if absent (full build), else diff+update (§4). Entry
    point for the launcher / CLI. Runs synchronously here — callers that need
    it off the request path (the proxy) run it in a background thread."""
    global _ACTIVE
    if storage_root is None:
        storage_root = _default_storage_root()
    key = str(Path(input_dir).expanduser().resolve())
    with _LOCK:
        engine = _ENGINES.get(key) or Engine(input_dir, storage_root)
        _ENGINES[key] = engine
        _ACTIVE = engine
    with engine._build_lock:
        engine.status = "building"
        engine.error = None
        try:
            if engine.store.exists():
                engine.sync()
            else:
                engine.build()
            engine.status = "ready"
        except Exception as e:  # keep the proxy alive; surface via status
            engine.status = "error"
            engine.error = f"{type(e).__name__}: {e}"
            raise


def on_file_changed(file_path: str, input_dir: Optional[str] = None) -> None:
    """Scope sync() to a single changed file plus its 2-hop neighborhood (§4)."""
    engine = _get_engine(input_dir)
    if engine is None:
        return
    with engine._build_lock:
        try:
            engine.status = "building"
            engine.sync(changed_files=[file_path])
            engine.status = "ready"
        except Exception as e:
            engine.status = "error"
            engine.error = f"{type(e).__name__}: {e}"


def get_concept_index(input_dir: Optional[str] = None) -> str:
    engine = _get_engine(input_dir)
    if engine is None:
        return ""
    if engine.status == "building" and not engine.store.exists():
        return ""
    return engine.concept_index()


def recall_codeconcept(tags: List[str], input_dir: Optional[str] = None) -> str:
    engine = _get_engine(input_dir)
    if engine is None:
        return "Code-concept index is not available."
    if engine.status == "building" and not engine.store.exists():
        return "Code-concept index is still building — try again shortly."
    return engine.recall(tags)


def query_codeconcept(query_text: str, input_dir: Optional[str] = None) -> str:
    engine = _get_engine(input_dir)
    if engine is None:
        return "Code-concept index is not available."
    if engine.status == "building" and not engine.store.exists():
        return "Code-concept index is still building — try again shortly."
    return engine.query(query_text)


def status(input_dir: Optional[str] = None) -> Dict[str, object]:
    engine = _get_engine(input_dir)
    if engine is None:
        return {"status": "disabled"}
    return {"status": engine.status, "error": engine.error, "dir": engine.input_dir}


# ── free helpers ─────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    """Progress line. Goes to stdout, which the proxy redirects to proxy.log."""
    print(f"[codegraph] {msg}", flush=True)


def _default_storage_root() -> Path:
    return Path(os.path.expanduser(
        os.environ.get("FA_STORAGE_ROOT", "~/.free_agent")
    )).resolve()


def read_span(path: Path, span: List[int]) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    start = max(0, span[0] - 1)
    end = min(len(lines), span[1])
    return "\n".join(lines[start:end])


def _lang_fence(file: str) -> str:
    ext = os.path.splitext(file)[1]
    return {
        ".py": "python", ".ts": "typescript", ".tsx": "tsx",
        ".js": "javascript", ".jsx": "jsx",
    }.get(ext, "")


def _next_seq(prev_concepts: Dict[str, ConceptRecord]) -> int:
    mx = -1
    for cid in prev_concepts:
        if cid.startswith("concept-"):
            try:
                mx = max(mx, int(cid.split("-")[1]))
            except (IndexError, ValueError):
                pass
    return mx + 1


def _match_tags(concepts: Dict[str, ConceptRecord], tags: List[str]) -> List[str]:
    wanted = [t.strip().lower() for t in tags if t.strip()]
    hits: List[str] = []
    for cid, c in concepts.items():
        ctag = c.tag.lower()
        for w in wanted:
            if w == ctag or w in ctag or ctag in w:  # exact + fuzzy substring
                hits.append(cid)
                break
    return hits


def _wire_neighbors(graph, concepts: Dict[str, ConceptRecord]) -> None:
    """Fill neighbor_concepts: concepts joined by a structural edge crossing the
    cluster boundary (§2)."""
    node_to_concept: Dict[str, str] = {}
    for cid, c in concepts.items():
        for m in c.members:
            node_to_concept[m] = cid
    neigh: Dict[str, set] = {cid: set() for cid in concepts}
    for a, b in graph.edges():
        ca, cb = node_to_concept.get(a), node_to_concept.get(b)
        if ca and cb and ca != cb:
            neigh[ca].add(cb)
            neigh[cb].add(ca)
    for cid, c in concepts.items():
        c.neighbor_concepts = sorted(neigh.get(cid, set()))


def _bfs(graph, seeds: set, hops: int) -> set:
    frontier = set(seeds)
    reached = set(seeds)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            if not graph.has_node(n):
                continue
            nxt |= set(graph.successors(n))
            nxt |= set(graph.predecessors(n))
        nxt -= reached
        reached |= nxt
        frontier = nxt
        if not frontier:
            break
    return reached
