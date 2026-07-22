"""The public entry points: ``ContextArchitect`` and ``Session``.

A ``Session`` owns the per-turn pipeline and all persisted state for a single
session id. One turn boundary is one call to :meth:`Session.rework`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import audit
from .assembler import assemble
from .classifier import classify
from .config import Config
from .files import FileLedger, detect_touches
from .llm import build_backend
from .llm.base import LLMBackend
from .models import Message, messages_from_dicts, messages_to_dicts
from .recall import render_recall_result
from .store import FilesystemStore, StorageBackend
from .summarizer import build_archive_payload, build_full_turn, summarize_turn
from .window import apply_window


class Session:
    def __init__(
        self,
        session_id: str,
        config: Config,
        backend: LLMBackend,
        store: StorageBackend,
    ):
        self.session_id = session_id
        self.config = config
        self.backend = backend
        self.store = store

        state = store.read_state(session_id)
        if state is None:
            self.turn_index = 0
            self.ledger = FileLedger()
            self.live_history: List[Message] = []
        else:
            self.turn_index = state.get("turn_index", 0)
            self.ledger = FileLedger.from_dict(state.get("ledger", {}))
            self.live_history = messages_from_dicts(state.get("live_history", []))

    # -- persistence --------------------------------------------------------
    def _save(self) -> None:
        self.store.write_state(
            self.session_id,
            {
                "turn_index": self.turn_index,
                "ledger": self.ledger.to_dict(),
                "live_history": messages_to_dicts(self.live_history),
            },
        )

    # -- core turn pipeline -------------------------------------------------
    def rework(self, messages: List[Message]) -> List[Message]:
        """Rework the current live context into a compact history.

        ``messages`` is the full context the model can currently see. Returns
        the new live history to install as the model's context.

        Tool *definitions* are not passed here: they are a top-level request
        parameter the host owns and re-sends each turn, so the library never
        needs them. To teach file detection about custom edit/read/shell tool
        names, set ``Config.extra_write_tools`` / ``extra_read_tools`` /
        ``extra_shell_tools``.
        """
        self.turn_index += 1
        turn = self.turn_index

        buckets = classify(messages, self.config)

        # 1) Summarize new activity + archive the full detail. Also build the
        #    turn's full-text form for the recency window (kept regardless of
        #    the window size; apply_window decides what is actually shown).
        new_summary = None
        new_full_turn: List[Message] = []
        archive_key = ""
        summary_prompt = ""
        summary_response: Dict[str, Any] = {}
        if buckets.new_activity:
            sr = summarize_turn(self.backend, self.config, buckets.new_activity, turn)
            new_summary = sr.message
            archive_key = sr.archive_key
            summary_prompt = sr.prompt
            summary_response = sr.raw_response
            self.store.write_archive(
                self.session_id,
                sr.archive_key,
                build_archive_payload(buckets.new_activity, turn, sr.label, sr.summary),
            )
            new_full_turn, _ = build_full_turn(
                buckets.new_activity, turn, sr.message, sr.archive_key
            )

        # 2) Detect file touches and refine the ledger.
        touches = detect_touches(buckets.new_activity, self.config)
        touched_paths = self.ledger.apply_touches(touches, turn)
        files_audit = self.ledger.refine_descriptions(
            self.backend, self.config, touched_paths, touches
        )
        file_ledger_msg = self.ledger.to_message()

        # 3) Apply the recency window and reassemble the compact history:
        #    pinned -> (older summaries + recent full-text turns) -> file ledger.
        body = apply_window(
            prior_summaries=buckets.prior_summaries,
            prior_full_turns=buckets.prior_full_turns,
            new_full_turn=new_full_turn,
            new_summary=new_summary,
            current_turn=turn,
            num_full_text_turns=self.config.num_full_text_turns,
        )
        new_history = assemble(
            pinned=buckets.pinned,
            body=body,
            file_ledger=file_ledger_msg,
        )

        # 4) Persist + audit.
        self.live_history = new_history
        self._save()
        self.store.append_audit(
            self.session_id,
            audit.rework_record(
                turn_index=turn,
                input_messages=messages,
                buckets={
                    "pinned": len(buckets.pinned),
                    "prior_summaries": len(buckets.prior_summaries),
                    "prior_full_turns": len(buckets.prior_full_turns),
                    "file_ledger": len(buckets.file_ledger),
                    "new_activity": len(buckets.new_activity),
                },
                summary_prompt=summary_prompt,
                summary_response=summary_response,
                files_audit=files_audit,
                detected_touches=[{"path": t.path, "kind": t.kind} for t in touches],
                output_messages=new_history,
                archive_key=archive_key,
            ),
        )
        return new_history

    # -- recall -------------------------------------------------------------
    def recall(self, key: str) -> str:
        payload = self.store.read_archive(self.session_id, key)
        self.store.append_audit(self.session_id, audit.recall_record(key=key, found=payload is not None))
        return render_recall_result(payload, key)

    # -- fork ---------------------------------------------------------------
    def fork(self, new_session_id: str) -> "Session":
        self._save()
        self.store.copy_session(self.session_id, new_session_id)
        return Session(new_session_id, self.config, self.backend, self.store)


class ContextArchitect:
    def __init__(self, config: Optional[Config] = None, store: Optional[StorageBackend] = None):
        self.config = config or Config()
        self.backend = build_backend(self.config.llm)
        self.store: StorageBackend = store or FilesystemStore(self.config.resolved_root())

    def session(self, session_id: str) -> Session:
        """Create or resume a session by id."""
        return Session(session_id, self.config, self.backend, self.store)
