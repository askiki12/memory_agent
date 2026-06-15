"""Memory storage with separate stores for summaries and raw dialogue turns.

v3 design:
  - SummaryStore: plain list with brute-force cosine search (~10 summaries)
  - TurnStore: FAISS-backed vector store for raw dialogue turns (hundreds)

Why separate?
  Summaries and raw turns serve different roles:
    Summaries → "router" — find which sessions are relevant to a query
    Raw turns → "evidence" — exact dialogue wording for generating answers

  Storing them in separate stores prevents summaries from competing with
  raw turns for the same FAISS top-k slots. Summaries have cleaner prose
  and tend to win semantic similarity races, crowding out the raw turns
  that actually contain the precise answer wording.
"""

import uuid

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# SummaryStore — plain list, brute-force search
# ---------------------------------------------------------------------------

class SummaryStore:
    """Lightweight summary store using plain Python lists + brute-force search.

    A conversation has ~10 summaries (one per session), so FAISS overhead
    is unnecessary. Brute-force cosine similarity over 10 vectors is
    faster than FAISS (no C++ call overhead for tiny N) and supports
    true deletion without ghost vectors.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._embeddings: list[np.ndarray] = []
        self._metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, metadatas: list[dict]) -> list[str]:
        """Add summaries with pre-computed embeddings. Returns list of IDs."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        ids = []
        for vec, meta in zip(embeddings, metadatas):
            mem_id = meta.get("mem_id") or str(uuid.uuid4())
            self._embeddings.append(vec.astype(np.float32))
            self._metadata.append({**meta, "mem_id": mem_id})
            ids.append(mem_id)
        return ids

    def search(self, query_emb: np.ndarray, k: int = 3) -> list[dict]:
        """Brute-force cosine similarity search.

        Returns list of {mem_id, score, session_id, metadata}.
        """
        if not self._embeddings:
            return []

        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
        query_emb = query_emb.astype(np.float32)

        stack = np.stack(self._embeddings)  # (N, dim)
        scores = (stack @ query_emb.T).flatten()  # (N,)

        # Take top-k indices
        k_eff = min(k, len(scores))
        if k_eff == 0:
            return []
        top_indices = np.argsort(-scores)[:k_eff]

        results = []
        for idx in top_indices:
            idx = int(idx)
            meta = self._metadata[idx]
            results.append({
                "mem_id": meta.get("mem_id", ""),
                "score": float(scores[idx]),
                "session_id": meta.get("session_id", -1),
                "date_time": meta.get("date_time", ""),
                "text": meta.get("text", ""),
                "metadata": meta,
            })
        return results

    def delete(self, mem_id: str) -> bool:
        """True deletion — removes both embedding and metadata."""
        for i, meta in enumerate(self._metadata):
            if meta.get("mem_id") == mem_id:
                self._embeddings.pop(i)
                self._metadata.pop(i)
                return True
        return False

    def clear(self) -> None:
        self._embeddings.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        return len(self._metadata)


# ---------------------------------------------------------------------------
# TurnStore — FAISS-backed vector store for raw dialogue turns
# ---------------------------------------------------------------------------

class TurnStore:
    """FAISS-backed store for raw dialogue turns.

    Uses IndexFlatIP (inner product = cosine similarity for normalized vectors).
    Unlike the v2 mixed store, this only holds raw turns — no summaries.
    Ghost vectors from soft-deletes are minimized because the only deletions
    come from capacity pruning (rare with 3000 item capacity).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: dict[int, dict] = {}
        self._faiss_id_to_mem_id: dict[int, str] = {}

    def add(self, embeddings: np.ndarray, metadatas: list[dict]) -> list[str]:
        """Add raw turn embeddings with metadata. Returns list of memory IDs."""
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
        """Semantic search over raw turns.

        Returns list of {mem_id, score, text, metadata}.
        """
        if self.index.ntotal == 0:
            return []

        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
        query_emb = query_emb.astype(np.float32)

        scores, indices = self.index.search(query_emb, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata.get(int(idx))
            if meta is None:
                continue  # ghost vector from soft-delete
            results.append({
                "mem_id": meta.get("mem_id", ""),
                "score": float(score),
                "text": meta.get("text", ""),
                "metadata": meta,
            })
        return results

    def delete(self, mem_id: str) -> bool:
        """Soft-delete: removes metadata, FAISS vector persists.

        Ghost vectors are rare because pruning only happens at capacity
        (3000 items) and most conversations have <700 turns.
        """
        for faiss_id, mid in list(self._faiss_id_to_mem_id.items()):
            if mid == mem_id:
                self.metadata.pop(faiss_id, None)
                self._faiss_id_to_mem_id.pop(faiss_id, None)
                return True
        return False

    def get_all(self) -> list[dict]:
        return list(self.metadata.values())

    def __len__(self) -> int:
        return len(self.metadata)

    def clear(self) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata.clear()
        self._faiss_id_to_mem_id.clear()
