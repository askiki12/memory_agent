"""MemoryStore — FAISS-backed unified vector store (v3).

Single index for both summaries and raw turns, same as the proven v2
approach. Competition between items for top-k slots acts as a natural
relevance filter.

Ghost-vector mitigation: fetch extra candidates per search to compensate
for soft-deleted items. Rebuild thresholds track when a full re-index
would be beneficial (requires re-embedding at controller level).
"""

import uuid

import faiss
import numpy as np


class MemoryStore:
    """FAISS IndexFlatIP store. Summaries + raw turns share one index."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: dict[int, dict] = {}
        self._faiss_id_to_mem_id: dict[int, str] = {}
        self._deleted_count: int = 0

    @property
    def live_count(self) -> int:
        return len(self.metadata)

    @property
    def total_count(self) -> int:
        return self.index.ntotal

    @property
    def ghost_count(self) -> int:
        return self._deleted_count

    def add(self, embeddings: np.ndarray, metadatas: list[dict]) -> list[str]:
        """Add embeddings with metadata. Returns list of memory IDs."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        mem_ids = []
        start_idx = self.index.ntotal
        self.index.add(embeddings.astype(np.float32))

        for i, meta in enumerate(metadatas):
            faiss_id = start_idx + i
            mem_id = meta.get("mem_id") or str(uuid.uuid4())
            self.metadata[faiss_id] = {**meta, "mem_id": mem_id}
            self._faiss_id_to_mem_id[faiss_id] = mem_id
            mem_ids.append(mem_id)

        return mem_ids

    def search(self, query_emb: np.ndarray, k: int = 10) -> list[dict]:
        """Semantic search, compensating for ghost vectors.

        Fetches extra candidates to offset ghosts from soft-deleted items.
        """
        if self.index.ntotal == 0:
            return []

        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
        query_emb = query_emb.astype(np.float32)

        # Fetch extra to compensate for ghost vectors
        fetch_k = min(k + self._deleted_count + 3, self.index.ntotal)
        scores, indices = self.index.search(query_emb, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata.get(int(idx))
            if meta is None:  # ghost vector
                continue
            results.append({
                "mem_id": meta.get("mem_id", ""),
                "score": float(score),
                "text": meta.get("text", ""),
                "metadata": meta,
            })
            if len(results) >= k:
                break

        return results

    def delete(self, mem_id: str) -> bool:
        """Soft-delete: removes metadata, vector persists as ghost.

        Ghost vectors are compensated for by fetching extra candidates
        in search(). Full rebuild happens when ghost ratio exceeds
        threshold (triggered by controller via needs_rebuild).
        """
        for faiss_id, mid in list(self._faiss_id_to_mem_id.items()):
            if mid == mem_id:
                self.metadata.pop(faiss_id, None)
                self._faiss_id_to_mem_id.pop(faiss_id, None)
                self._deleted_count += 1
                return True
        return False

    def needs_rebuild(self, threshold: float = 0.15) -> bool:
        """Check if ghost ratio exceeds threshold, warranting a rebuild."""
        if self.index.ntotal == 0:
            return False
        return self._deleted_count / self.index.ntotal > threshold

    def get_all(self) -> list[dict]:
        return list(self.metadata.values())

    def __len__(self) -> int:
        return len(self.metadata)

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata.clear()
        self._faiss_id_to_mem_id.clear()
        self._deleted_count = 0
