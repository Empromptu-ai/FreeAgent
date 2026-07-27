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

"""Record shapes for the code-graph store (§2 of the spec).

These are plain dataclasses with dict (de)serialization so the JSON files on
disk map 1:1 to the spec. No heavy deps imported here — safe to load anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class EntityRecord:
    """One function / class / method — a structural node (§2)."""

    node_id: str                       # "path/to/file.py::ClassName.method_name"
    kind: str                          # "function" | "class" | "method"
    file: str                          # repo-relative path
    span: List[int]                    # [start_line, end_line] (1-based, inclusive)
    code_hash: str                     # sha1 of the entity's source text
    tag: str = ""                      # kebab-case, 2-4 words (LLM)
    summary: str = ""                  # 1-2 sentences (LLM)
    signature: str = ""                # header line(s), used for embedding text

    @property
    def embedding_id(self) -> str:
        return self.node_id

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["embedding_id"] = self.embedding_id
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EntityRecord":
        return cls(
            node_id=d["node_id"],
            kind=d["kind"],
            file=d["file"],
            span=list(d.get("span", [0, 0])),
            code_hash=d.get("code_hash", ""),
            tag=d.get("tag", ""),
            summary=d.get("summary", ""),
            signature=d.get("signature", ""),
        )


@dataclass
class ConceptRecord:
    """One Leiden cluster — a concept node (§2)."""

    concept_id: str                    # "concept-0007"
    tag: str = ""
    summary: str = ""
    members: List[str] = field(default_factory=list)          # entity node_ids
    member_files: List[str] = field(default_factory=list)
    neighbor_concepts: List[str] = field(default_factory=list)
    # Optional lineage for churn explanation (§ "Additional notes").
    split_from: List[str] = field(default_factory=list)
    merged_from: List[str] = field(default_factory=list)

    @property
    def embedding_id(self) -> str:
        return self.concept_id

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["embedding_id"] = self.embedding_id
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConceptRecord":
        return cls(
            concept_id=d["concept_id"],
            tag=d.get("tag", ""),
            summary=d.get("summary", ""),
            members=list(d.get("members", [])),
            member_files=list(d.get("member_files", [])),
            neighbor_concepts=list(d.get("neighbor_concepts", [])),
            split_from=list(d.get("split_from", [])),
            merged_from=list(d.get("merged_from", [])),
        )
