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
        importance_weight: float = 0.0,
        recency_weight: float = 0.0,
        importance_filter: float = 0.25,
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
        """Relevance retrieval with optional importance noise filter.

        If importance_filter > 0: removes low-importance candidates
        (small talk, fillers) before taking top_k. Relevance order
        is preserved — no re-ranking.
        """
        if len(self.store) == 0:
            return []

        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )

        if self.importance_filter <= 0:
            return self.store.search(q_emb, k=self.top_k)

        # Fetch extra to compensate for filtered-out items
        fetch_k = min(self.top_k * self.fetch_multiplier, self.store.total_count)
        candidates = self.store.search(q_emb, k=fetch_k)

        # Filter out noise, keep relevance order
        filtered = [c for c in candidates
                    if c["metadata"].get("importance", 0.0) >= self.importance_filter]

        if len(filtered) < self.top_k:
            filtered = candidates

        return filtered[:self.top_k]

    def format_context(self, memories: list[dict], question: str = "") -> str:
        """Format retrieved items — summaries first, turns grouped by session.

        V5: for temporal ('when') questions, uses timeline-style headers
        to make dates more prominent for the model.
        """
        if not memories:
            return "No relevant information found."

        summaries = [m for m in memories
                     if m["metadata"].get("category") == "session_summary"]
        turns = [m for m in memories
                 if m["metadata"].get("category") == "raw_turn"]

        # V5: detect temporal questions
        is_temporal = question.strip().lower().startswith("when ")

        lines = []

        # --- Summaries: temporal anchors ---
        if summaries:
            if is_temporal:
                lines.append("=== Timeline (session summaries with dates) ===")
            else:
                lines.append("=== Session Summaries ===")
            for m in summaries:
                sid = m["metadata"].get("session_id", "?")
                date = m["metadata"].get("date_time", "")
                text = m.get("text", "")
                if is_temporal:
                    # Emphasize date for temporal questions
                    lines.append(f"📅 {date} — {text}")
                else:
                    lines.append(f"[Session {sid} @ {date}] {text}")

        # --- Raw turns: exact evidence ---
        if turns:
            groups: dict[int, list[dict]] = {}
            session_order: list[int] = []
            for m in turns:
                sid = m["metadata"].get("session_id", -1)
                if sid not in groups:
                    groups[sid] = []
                    session_order.append(sid)
                groups[sid].append(m)

            if is_temporal:
                lines.append("\n=== Dialogue (look for dates and times) ===")
            else:
                lines.append("\n=== Relevant Dialogue ===")
            for sid in session_order:
                grp = groups[sid]
                date = grp[0]["metadata"].get("date_time", "unknown")
                lines.append(f"--- Session {sid} ({date}) ---")
                for m in grp:
                    lines.append(m.get("text", ""))

        return "\n".join(lines) if lines else "No relevant information found."
