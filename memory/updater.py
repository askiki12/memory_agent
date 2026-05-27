import numpy as np


class MemoryUpdater:
    """Deduplicates and merges new memories against existing ones.

    For efficiency, dedup is done by comparing raw text via embedding similarity
    rather than calling the LLM per conflict. Only high-confidence conflicts
    (similarity > threshold) are merged; the rest pass through.
    """

    def __init__(self, embed_model, similarity_threshold: float = 0.90, max_memories: int = 500):
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.max_memories = max_memories

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts to normalized embeddings."""
        embeds = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if embeds.ndim == 1:
            embeds = embeds.reshape(1, -1)
        return embeds

    def merge(
        self, new_memories: list[dict], existing_memories: list[dict]
    ) -> tuple[list[dict], list[str]]:
        """Merge new memories into existing ones. Returns (memories_to_add, mem_ids_to_delete).

        For each new memory:
        - If similarity to an existing memory > threshold: keep the newer one
          (by date_time), mark the old one for deletion.
        - Otherwise: mark for addition.

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

        # Compute cosine similarity matrix: [new x existing]
        sim = new_embeds @ existing_embeds.T

        to_add = []
        to_delete = []

        for i, new_mem in enumerate(new_memories):
            best_idx = int(np.argmax(sim[i]))
            best_score = float(sim[i][best_idx])

            if best_score >= self.similarity_threshold:
                existing = existing_memories[best_idx]
                # Keep the newer one (compare date strings for simplicity)
                new_date = new_mem.get("date_time", "")
                old_date = existing.get("date_time", "")
                if new_date >= old_date:
                    to_add.append(new_mem)
                    old_id = existing.get("mem_id")
                    if old_id:
                        to_delete.append(old_id)
                # else: old one is newer, discard the new one
            else:
                to_add.append(new_mem)

        return to_add, to_delete

    def prune(self, memories: list[dict]) -> list[str]:
        """Prune oldest memories if count exceeds max_memories.

        Returns list of mem_ids to delete.
        """
        if len(memories) <= self.max_memories:
            return []

        # Sort by date_time (oldest first) — these get pruned
        sorted_mems = sorted(memories, key=lambda m: m.get("date_time", ""))
        excess = len(memories) - self.max_memories

        return [m.get("mem_id", "") for m in sorted_mems[:excess] if m.get("mem_id")]
