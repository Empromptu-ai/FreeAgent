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

"""Local embeddings via Ollama's batch ``/api/embed`` endpoint (§0, §2).

Uses only urllib (like free_agent.llm.ollama) so no new runtime dep. Returns
float32 numpy arrays; numpy is imported lazily.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import List

_BATCH = 64  # inputs per request


class EmbedError(RuntimeError):
    pass


def embed_texts(texts: List[str], model: str, base_url: str, timeout: float = 120.0):
    """Embed ``texts`` -> np.ndarray of shape (len(texts), dim), float32.

    Empty strings are embedded as-is (Ollama returns a zero-ish vector); callers
    that care should pass non-empty text. Batches to keep requests reasonable.
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype="float32")

    url = f"{base_url.rstrip('/')}/api/embed"
    vectors: List[List[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        body = json.dumps({"model": model, "input": batch}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:  # pragma: no cover - network
            raise EmbedError(f"ollama embed request failed: {e}") from e
        embs = payload.get("embeddings")
        if not embs:
            raise EmbedError(f"unexpected ollama embed response: {payload!r}")
        vectors.extend(embs)

    return np.asarray(vectors, dtype="float32")


def cosine_matrix(query, matrix):
    """Cosine similarity of a (dim,) query against an (n, dim) matrix -> (n,)."""
    import numpy as np

    if matrix.size == 0:
        return np.zeros((0,), dtype="float32")
    q = query / (np.linalg.norm(query) + 1e-12)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return m @ q
