import threading
from typing import Dict, Optional, Any

from backend.ke.ke_models import KnowledgeEdit
from backend.ke.rome.engine import edit_fact
from backend.ke.rome_config import get_gpt2_rome_config


class KEManager:
    """
    Knowledge Editing Manager.

    Responsibilities:
    - Apply a ROME edit to an in-memory model
    - Track snapshots for rollback
    - Provide a 'busy' flag to prevent concurrent edits
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False

        # Store rollback snapshots per model_id
        # snapshot: Dict[param_name, original_tensor]
        self._snapshots: Dict[str, Dict[str, Any]] = {}

        # Store last applied edit per model_id (optional, useful for logs)
        self._last_edit: Dict[str, KnowledgeEdit] = {}

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def apply_edit(self, model_id: str, model, tokenizer, edit: KnowledgeEdit) -> None:
        """
        Apply a ROME edit (blocking).

        This should be executed in a background thread (e.g., asyncio.to_thread)
        to avoid blocking the async websocket loop.
        """
        edit.is_valid()

        with self._lock:
            if self._busy:
                raise RuntimeError("Knowledge editing is already in progress.")
            self._busy = True

        try:
            # Explicit ROME configuration (no auto-inference)
            rome_config = get_gpt2_rome_config()

            # Apply ROME edit
            _, snapshot = edit_fact(
                model=model,
                tok=tokenizer,
                subject=edit.subject,
                prompt=edit.prompt,
                new_object=edit.new_object,
                config=rome_config,
                copy_model=False,
                return_orig_weights=True,
            )

            with self._lock:
                self._snapshots[model_id] = snapshot
                self._last_edit[model_id] = edit

        finally:
            with self._lock:
                self._busy = False

    def rollback(self, model_id: str, model) -> bool:
        """
        Roll back the last edit for the given model_id.
        Returns True if rollback happened, False if there was nothing to roll back.
        """
        with self._lock:
            snapshot = self._snapshots.get(model_id)

        if not snapshot:
            return False

        from backend.ke.rome import nethook

        with self._lock:
            if self._busy:
                raise RuntimeError("Cannot rollback while an edit is in progress.")
            self._busy = True

        try:
            for name, weight in snapshot.items():
                param = nethook.get_parameter(model, name)
                param.data.copy_(weight)

            with self._lock:
                self._snapshots.pop(model_id, None)
                self._last_edit.pop(model_id, None)

            return True

        finally:
            with self._lock:
                self._busy = False
