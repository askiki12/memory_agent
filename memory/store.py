"""MemoryStore — FAISS-backed vector store for dialogue chunks and summaries.

Stores two types of content in a single index:
  - raw_turn: individual dialogue turns (zero information loss, like Vanilla RAG)
  - session_summary: natural-language summaries per session (dense retrieval targets)
"""

import uuid

import faiss
import numpy as np


class MemoryStore:
    """FAISS vector store with metadata. Simple, fast, reliable."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: dict[int, dict] = {}
        self._faiss_id_to_mem_id: dict[int, str] = {}

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
        """Semantic search. Returns list of {mem_id, score, text, metadata}."""
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
        query_emb = query_emb.astype(np.float32)

        scores, indices = self.index.search(query_emb, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata.get(int(idx), {})
            results.append({
                "mem_id": meta.get("mem_id", ""),
                "score": float(score),
                "text": meta.get("text", ""),
                "metadata": meta,
            })
        return results

    def delete(self, mem_id: str) -> bool:
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
