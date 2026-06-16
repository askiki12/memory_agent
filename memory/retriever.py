"""MemoryRetriever — semantic search with importance-based noise filtering (v4).

v4 Phase 1: instead of re-ranking by importance (which biases toward
summaries), we use importance as a noise FILTER. Low-importance
candidates (small talk, fillers) are removed AFTER FAISS retrieval,
but the remaining candidates keep their original relevance order.

This helps single_hop (less noise) without hurting temporal (no
re-ranking that favors summary prose).
"""


class MemoryRetriever:
    """Semantic retrieval with importance-based noise filtering."""

    def __init__(
        self,
        embed_model,
        store,
        top_k: int = 10,
        importance_weight: float = 0.15,
        importance_filter: float = 0.25,
        fetch_multiplier: int = 3,
    ):
        self.embed_model = embed_model
        self.store = store
        self.top_k = top_k
        self.importance_weight = importance_weight
        self.importance_filter = importance_filter  # min importance to keep
        self.fetch_multiplier = fetch_multiplier

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve with noise filtering.

        1. FAISS search for top_k * fetch_multiplier candidates
        2. Filter out candidates with importance < importance_filter
           (removes small talk: "Hi!", "Yeah, me too", etc.)
        3. Apply mild importance boost to remaining candidates
        4. Return top_k

        The filter threshold is low (0.25) — only pure noise is removed.
        The boost is small — just enough to break ties in favor of
        fact-dense turns.
        """
        if len(self.store) == 0:
            return []

        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )

        fetch_k = min(self.top_k * self.fetch_multiplier, self.store.total_count)
        candidates = self.store.search(q_emb, k=fetch_k)

        if not candidates:
            return []

        # Filter: remove pure noise (importance < threshold)
        # Keep at least top_k candidates even if some are below threshold
        filtered = [c for c in candidates
                    if c["metadata"].get("importance", 0.0) >= self.importance_filter]

        # If filtering removed too many, fall back to unfiltered
        if len(filtered) < self.top_k:
            filtered = candidates

        # Keep original relevance order (no re-ranking by importance).
        # The filter already removed noise; the remaining candidates
        # preserve their natural semantic similarity order.
        result = filtered[:self.top_k]

        return result

    def format_context(self, memories: list[dict]) -> str:
        """Format retrieved items — summaries first (temporal anchors),
        then raw dialogue turns grouped by session (exact evidence)."""
        if not memories:
            return "No relevant information found."

        summaries = [m for m in memories
                     if m["metadata"].get("category") == "session_summary"]
        turns = [m for m in memories
                 if m["metadata"].get("category") == "raw_turn"]

        lines = []

        if summaries:
            lines.append("=== Session Summaries ===")
            for m in summaries:
                sid = m["metadata"].get("session_id", "?")
                date = m["metadata"].get("date_time", "")
                text = m.get("text", "")
                lines.append(f"[Session {sid} @ {date}] {text}")

        if turns:
            groups: dict[int, list[dict]] = {}
            session_order: list[int] = []
            for m in turns:
                sid = m["metadata"].get("session_id", -1)
                if sid not in groups:
                    groups[sid] = []
                    session_order.append(sid)
                groups[sid].append(m)

            lines.append("\n=== Relevant Dialogue ===")
            for sid in session_order:
                grp = groups[sid]
                date = grp[0]["metadata"].get("date_time", "unknown")
                lines.append(f"--- Session {sid} ({date}) ---")
                for m in grp:
                    lines.append(m.get("text", ""))

        return "\n".join(lines) if lines else "No relevant information found."
