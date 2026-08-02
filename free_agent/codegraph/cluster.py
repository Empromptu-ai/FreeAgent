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

"""Combined structural+semantic graph, Leiden partition, and concept identity
matching (§5 and the "Additional notes" stabilization layer).

Produces community member-sets and matches them to prior concept_ids by member
Jaccard overlap so ids stay stable across resyncs (with split/merge lineage).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple


def build_communities(
    graph,
    embeddings: Dict[str, "object"],
    node_ids: List[str],
    *,
    resolution: float,
    sim_threshold: float,
    sim_top_k: int,
    w_struct: float,
    w_sim: float,
) -> List[List[str]]:
    """Partition entity nodes into communities (lists of node_ids).

    Combines the structural graph with top-k semantic-similarity edges (§5),
    then runs Leiden (RBConfigurationVertexPartition, weighted).
    """
    import igraph as ig
    import leidenalg as la
    import numpy as np

    if not node_ids:
        return []

    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    # Accumulate undirected edge weights in a dict keyed by (min,max) index.
    weights: Dict[Tuple[int, int], float] = {}

    def add(u: int, v: int, w: float):
        if u == v:
            return
        key = (u, v) if u < v else (v, u)
        weights[key] = weights.get(key, 0.0) + w

    # Structural edges (collapse the MultiDiGraph, ignore direction).
    for a, b in graph.edges():
        if a in idx and b in idx:
            add(idx[a], idx[b], w_struct)

    # Semantic edges: top-k cosine neighbors above threshold, per node.
    have_emb = [nid for nid in node_ids if nid in embeddings]
    if have_emb and sim_top_k > 0:
        mat = np.asarray([embeddings[nid] for nid in have_emb], dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        unit = mat / norms
        sims = unit @ unit.T  # (m, m)
        m = len(have_emb)
        k = min(sim_top_k + 1, m)  # +1 because the top hit is the node itself
        for i in range(m):
            order = np.argpartition(-sims[i], k - 1)[:k]
            for j in order:
                if i == j:
                    continue
                s = float(sims[i, j])
                if s > sim_threshold:
                    add(idx[have_emb[i]], idx[have_emb[j]],
                        w_sim * (s - sim_threshold))

    g = ig.Graph(n=n)
    if weights:
        edges = list(weights.keys())
        g.add_edges(edges)
        g.es["weight"] = [weights[e] for e in edges]
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=resolution,
        )
    else:
        part = la.find_partition(g, la.RBConfigurationVertexPartition,
                                 resolution_parameter=resolution)

    communities: List[List[str]] = []
    for comm in part:
        communities.append([node_ids[i] for i in comm])
    return communities


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_concepts(
    communities: List[List[str]],
    prev_members: Dict[str, List[str]],
    *,
    jaccard_threshold: float,
    id_seq_start: int,
) -> Tuple[List[Tuple[str, List[str], Dict[str, List[str]]]], int]:
    """Assign stable concept_ids to new communities by member overlap.

    prev_members: {concept_id: [node_id, ...]} from the previous clustering.

    Returns (assignments, next_seq), where each assignment is
    (concept_id, members, lineage) and lineage may hold {"split_from": [...]}
    or {"merged_from": [...]}. Greedy 1:1 matching above ``jaccard_threshold``;
    detect splits (one old -> many new) and merges (many old -> one new).
    """
    prev_sets = {cid: set(members) for cid, members in prev_members.items()}
    new_sets = [set(c) for c in communities]

    # Score every (new, old) pair above threshold.
    scored: List[Tuple[float, int, str]] = []
    for i, ns in enumerate(new_sets):
        for cid, ps in prev_sets.items():
            j = _jaccard(ns, ps)
            if j >= jaccard_threshold:
                scored.append((j, i, cid))
    scored.sort(reverse=True)

    new_to_old: Dict[int, str] = {}
    old_to_new: Dict[str, int] = {}
    for _, i, cid in scored:
        if i in new_to_old or cid in old_to_new:
            continue
        new_to_old[i] = cid
        old_to_new[cid] = i

    # Count, per old concept, how many new communities overlap it at all (for
    # split detection); and per new community, how many old ones (merge).
    old_overlap_count: Dict[str, int] = {}
    new_overlap_count: Dict[int, int] = {}
    for _, i, cid in scored:
        old_overlap_count[cid] = old_overlap_count.get(cid, 0) + 1
        new_overlap_count[i] = new_overlap_count.get(i, 0) + 1

    seq = id_seq_start
    assignments: List[Tuple[str, List[str], Dict[str, List[str]]]] = []
    for i, community in enumerate(communities):
        lineage: Dict[str, List[str]] = {}
        matched = new_to_old.get(i)
        is_merge = new_overlap_count.get(i, 0) > 1
        is_split = matched is not None and old_overlap_count.get(matched, 0) > 1
        if matched is not None and not is_merge and not is_split:
            cid = matched                       # clean 1:1 -> inherit id
        else:
            cid = f"concept-{seq:04d}"
            seq += 1
            if is_split and matched is not None:
                lineage["split_from"] = [matched]
            if is_merge:
                lineage["merged_from"] = sorted(
                    {c for _, j, c in scored if j == i}
                )
        assignments.append((cid, community, lineage))
    return assignments, seq
