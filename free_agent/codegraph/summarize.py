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

"""LLM tag+summary extraction for entities and concepts (§7).

Reuses free_agent.llm.ollama.OllamaBackend (with the format="json" extension)
so all LLM traffic goes through one place. Prompts are JSON-only, low
temperature; a fenced/prefixed reply is tolerated by _parse_json.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from free_agent.llm.ollama import OllamaBackend

_ENTITY_SYS = (
    "You summarize code. Respond with ONLY JSON, no preamble, no code fences."
)
_ENTITY_PROMPT = (
    "Given this code (file, signature, body, and any docstring), respond with "
    "ONLY JSON:\n"
    '{{"tag": "<2-4 word kebab-case phrase>", '
    '"summary": "<1-2 sentences: what it does and why>"}}\n\n'
    "file: {file}\n"
    "```\n{code}\n```"
)

_CONCEPT_SYS = _ENTITY_SYS
_CONCEPT_PROMPT = (
    "Given these related code entities (tag + summary for each), respond with "
    "ONLY JSON:\n"
    '{{"tag": "<2-4 word kebab-case phrase for the overall concept/system>", '
    '"summary": "<2-3 sentences describing the system/concept these entities '
    'together implement>"}}\n\n'
    "{members}"
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> Dict[str, str]:
    try:
        obj = json.loads(text)
    except Exception:
        m = _JSON_RE.search(text or "")
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return {}
    tag = str(obj.get("tag", "")).strip()
    summary = str(obj.get("summary", "")).strip()
    return {"tag": tag, "summary": summary}


class Summarizer:
    def __init__(self, model: str, base_url: str, temperature: float = 0.15,
                 max_workers: int = 6, timeout: float = 120.0):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_workers = max_workers
        self.timeout = timeout

    def _backend(self) -> OllamaBackend:
        return OllamaBackend(
            base_url=self.base_url, model=self.model,
            temperature=self.temperature, timeout=self.timeout,
        )

    def _one(self, system: str, prompt: str) -> Dict[str, str]:
        try:
            raw = self._backend().complete(system, prompt, format="json")
        except Exception:
            return {}
        return _parse_json(raw)

    def entities(self, items: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, str]]:
        """items: [(node_id, file, code)]  ->  {node_id: {tag, summary}}.

        Runs the per-entity calls concurrently against Ollama (the expensive
        full-build step, §3.2).
        """
        results: Dict[str, Dict[str, str]] = {}

        def work(item):
            node_id, file, code = item
            prompt = _ENTITY_PROMPT.format(file=file, code=code[:6000])
            return node_id, self._one(_ENTITY_SYS, prompt)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for node_id, res in pool.map(work, items):
                results[node_id] = res
        return results

    def concept(self, member_lines: List[str]) -> Dict[str, str]:
        members = "\n".join(f"- {ln}" for ln in member_lines)
        prompt = _CONCEPT_PROMPT.format(members=members[:8000])
        return self._one(_CONCEPT_SYS, prompt)

    def concepts(self, jobs: List[Tuple[str, List[str]]]) -> Dict[str, Dict[str, str]]:
        """jobs: [(concept_id, member_lines)] -> {concept_id: {tag, summary}}."""
        results: Dict[str, Dict[str, str]] = {}

        def work(job):
            cid, member_lines = job
            return cid, self.concept(member_lines)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for cid, res in pool.map(work, jobs):
                results[cid] = res
        return results
