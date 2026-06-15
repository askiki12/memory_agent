import time

import numpy as np


class MemoryUpdater:
    """Deduplicates, merges, and manages memory lifecycle.

    Improvements over baseline:
      - Ebbinghaus forgetting: R = exp(-t / S) controls retention
      - Importance-aware pruning: low-importance, forgotten memories deleted first
      - Strengthen-on-access: used memories get stronger and resist forgetting
      - Merge conflict resolution based on importance + recency, not just date
    """

    def __init__(
        self,
        embed_model,
        similarity_threshold: float = 0.90,
        max_memories: int = 500,
        forget_threshold: float = 0.05,
    ):
        """
        Args:
            embed_model: SentenceTransformer instance for encoding text
            similarity_threshold: cosine sim above which two facts are considered duplicates
            max_memories: hard cap on total stored memories
            forget_threshold: retention R below which memories are candidate for removal
        """
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.max_memories = max_memories
        self.forget_threshold = forget_threshold
        self._now = time.time

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts to normalized embeddings."""
        if not texts:
            return np.array([])
        embeds = self.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        if embeds.ndim == 1:
            embeds = embeds.reshape(1, -1)
        return embeds

    # ------------------------------------------------------------------
    # Ebbinghaus forgetting model
    # ------------------------------------------------------------------

    @staticmethod
    def retention(memory: dict, now: float | None = None) -> float:
        """Compute Ebbinghaus retention R = exp(-t / S).

        t = seconds since last_accessed (or creation time if never accessed)
        S = memory strength (higher = more resistant to forgetting)

        Returns value in [0, 1].  R ~ 1.0 → fresh/strong;  R ~ 0.0 → forgotten.
        """
        if now is None:
            now = time.time()
        last = memory.get("last_accessed", memory.get("created_at", now))
        t = max(0.0, now - last)
        s = max(memory.get("strength", 1.0), 0.1)  # floor at 0.1 to avoid div/zero
        return float(np.exp(-t / (s * 86400)))  # scale S to days for intuitive decay

    def strengthen(self, memory: dict) -> None:
        """Called when a memory is used in answering. Boosts strength (S+1)."""
        memory["strength"] = memory.get("strength", 1.0) + 1.0
        memory["last_accessed"] = self._now()

    def get_forgotten(self, memories: list[dict]) -> list[str]:
        """Return mem_ids whose retention has fallen below the forget threshold."""
        now = self._now()
        forgotten = []
        for m in memories:
            if self.retention(m, now) < self.forget_threshold:
                forgotten.append(m.get("mem_id", ""))
        return forgotten

    # ------------------------------------------------------------------
    # Merge (dedup)
    # ------------------------------------------------------------------

    def merge(
        self, new_memories: list[dict], existing_memories: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """Merge new memories into existing store.

        For each new memory:
          - If semantic similarity > threshold to an existing memory:
              keep the one with higher (importance * retention) score.
          - Otherwise: mark for addition (novel fact).

        Returns (to_add, to_delete) where to_delete is a list of mem_ids.
        """
        if not new_memories:
            return [], []

        if not existing_memories:
            return new_memories, []

        new_texts = [m["text"] for m in new_memories]
        existing_texts = [m.get("text", "") for m in existing_memories]

        new_embeds = self._encode(new_texts)
        existing_embeds = self._encode(existing_texts)

        if new_embeds.size == 0 or existing_embeds.size == 0:
            return new_memories, []

        sim = new_embeds @ existing_embeds.T

        to_add = []
        to_delete = []

        for i, new_mem in enumerate(new_memories):
            best_idx = int(np.argmax(sim[i]))
            best_score = float(sim[i][best_idx])

            if best_score >= self.similarity_threshold:
                existing = existing_memories[best_idx]
                # Score by importance × retention to decide which to keep
                new_quality = new_mem.get("importance", 5) * self.retention(new_mem)
                old_quality = existing.get("importance", 5) * self.retention(existing)
                if new_quality >= old_quality:
                    to_add.append(new_mem)
                    old_id = existing.get("mem_id")
                    if old_id:
                        to_delete.append(old_id)
                # else: existing has higher quality, discard new
            else:
                to_add.append(new_mem)

        return to_add, to_delete

    # ------------------------------------------------------------------
    # Pruning (capacity management)
    # ------------------------------------------------------------------

    def prune(self, memories: list[dict]) -> list[str]:
        """Remove memories when over capacity.

        Order of eviction (worst first):
          1. Forgotten memories (R < forget_threshold)
          2. Lowest (importance × retention) score

        Returns list of mem_ids to delete.
        """
        if len(memories) <= self.max_memories:
            return []

        now = self._now()
        # Score each memory: higher = more worth keeping
        scored = []
        for m in memories:
            imp = m.get("importance", 5)
            r = self.retention(m, now)
            score = imp * r
            scored.append((score, m))

        # Sort ascending: lowest quality first → will be pruned
        scored.sort(key=lambda x: x[0])

        excess = len(memories) - self.max_memories
        to_prune = [
            m.get("mem_id", "") for _, m in scored[:excess] if m.get("mem_id")
        ]
        return to_prune
