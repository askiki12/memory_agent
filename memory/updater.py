"""MemoryUpdater — summary dedup + capacity management (v3).

Same as v2 but cleaner: dedup near-duplicate summaries, prune oldest
items when over capacity (summaries first, then raw turns).
"""

import numpy as np


class MemoryUpdater:
    """Dedup for session summaries + capacity guard for all items."""

    def __init__(
        self,
        embed_model,
        similarity_threshold: float = 0.92,
        max_memories: int = 3000,
    ):
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.max_memories = max_memories

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeds = self.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        if embeds.ndim == 1:
            embeds = embeds.reshape(1, -1)
        return embeds

    def merge(
        self, new_summaries: list[dict], existing_summaries: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """Merge new summaries, deduplicating near-identical ones.

        Only summaries are passed here. Raw turns are never deduped.
        Returns (to_add, to_delete).
        """
        if not new_summaries:
            return [], []

        if not existing_summaries:
            return new_summaries, []

        new_texts = [m["text"] for m in new_summaries]
        existing_texts = [m.get("text", "") for m in existing_summaries]

        new_embeds = self._encode(new_texts)
        existing_embeds = self._encode(existing_texts)

        if new_embeds.size == 0 or existing_embeds.size == 0:
            return new_summaries, []

        sim = new_embeds @ existing_embeds.T

        to_add = []
        to_delete = []

        for i, new_mem in enumerate(new_summaries):
            best_idx = int(np.argmax(sim[i]))
            best_score = float(sim[i][best_idx])

            if best_score >= self.similarity_threshold:
                existing = existing_summaries[best_idx]
                new_date = new_mem.get("date_time", "")
                old_date = existing.get("date_time", "")
                if new_date >= old_date:
                    to_add.append(new_mem)
                    old_id = existing.get("mem_id")
                    if old_id:
                        to_delete.append(old_id)
            else:
                to_add.append(new_mem)

        return to_add, to_delete

    def prune(self, items: list[dict]) -> list[str]:
        """Remove oldest items when over capacity.

        Summaries are pruned before raw turns (summaries can be regenerated).
        """
        if len(items) <= self.max_memories:
            return []

        summaries = [i for i in items if i.get("category") == "session_summary"]
        turns = [i for i in items if i.get("category") == "raw_turn"]

        excess = len(items) - self.max_memories
        to_prune = []

        # Remove oldest summaries first
        sorted_summaries = sorted(summaries, key=lambda m: m.get("date_time", ""))
        for m in sorted_summaries[:excess]:
            to_prune.append(m.get("mem_id", ""))

        # If still over capacity, remove oldest turns
        if len(to_prune) < excess:
            remaining = excess - len(to_prune)
            sorted_turns = sorted(turns, key=lambda m: m.get("date_time", ""))
            for m in sorted_turns[:remaining]:
                to_prune.append(m.get("mem_id", ""))

        return [mid for mid in to_prune if mid]
