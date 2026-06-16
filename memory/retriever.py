"""MemoryRetriever — semantic search with metadata collection (v4 Phase 2).

Importance and recency metadata are collected at ingest time and stored
in FAISS metadata. Retrieval currently uses pure relevance (cosine
similarity) — same as V3. The metadata is available for future phases
(query-type-aware weighting, answer verification, etc.).

What we learned:
  - Any re-ranking by non-relevance signals hurts temporal questions
  - Pure noise filtering (importance < 0.25) helps single_hop but regresses
    temporal at full scale (160 QA)
  - The safest path: collect metadata, use relevance-only retrieval,
    leverage metadata in context formatting or query-type detection
"""


class MemoryRetriever:
    """Semantic retrieval with metadata collection (relevance-only for now)."""

    def __init__(
        self,
        embed_model,
        store,
        top_k: int = 10,
        importance_weight: float = 0.0,    # reserved for future use
        recency_weight: float = 0.0,        # reserved for future use
        importance_filter: float = 0.0,     # reserved for future use
        fetch_multiplier: int = 3,
    ):
        self.embed_model = embed_model
        self.store = store
        self.top_k = top_k
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.importance_filter = importance_filter
        self.fetch_multiplier = fetch_multiplier

    def retrieve(self, query: str) -> list[dict]:
        """Pure relevance retrieval (V3-compatible).

        Importance and recency metadata are collected at ingest time
        and available for future phases. Current retrieval uses only
        cosine similarity — proven optimal for this setup.
        """
        if len(self.store) == 0:
            return []

        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        return self.store.search(q_emb, k=self.top_k)

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
