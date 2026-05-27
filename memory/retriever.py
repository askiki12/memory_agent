import numpy as np


class MemoryRetriever:
    """Retrieves relevant memories for a query using semantic search + optional recency boost."""

    def __init__(self, embed_model, store, top_k: int = 10, recency_weight: float = 0.2):
        self.embed_model = embed_model
        self.store = store
        self.top_k = top_k
        self.recency_weight = recency_weight

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve top-k memories for a query.

        Optionally applies a recency boost: later sessions score slightly higher.
        """
        if len(self.store) == 0:
            return []

        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        results = self.store.search(q_emb, k=self.top_k)

        if not results or self.recency_weight <= 0:
            return results

        # Apply recency boost based on session_id order
        all_mems = self.store.get_all()
        if not all_mems:
            return results

        max_session = max(
            (m.get("session_id", 0) for m in all_mems), default=0
        )
        if max_session <= 0:
            return results

        for r in results:
            sid = r["metadata"].get("session_id", 0)
            recency_score = (sid / max_session) * self.recency_weight
            r["score"] = r["score"] + recency_score

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def format_context(self, memories: list[dict]) -> str:
        """Format retrieved memories into a prompt-ready context string."""
        if not memories:
            return "No relevant memories found."

        lines = ["Relevant memories (most relevant first):"]
        for i, m in enumerate(memories, 1):
            date = m["metadata"].get("date_time", "unknown date")
            text = m.get("text", "")
            lines.append(f"{i}. [{date}] {text}")
        return "\n".join(lines)
