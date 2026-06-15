"""MemoryRetriever — two-stage retrieval with session routing.

v3 design:
  Stage 1: Query summaries → find relevant sessions (router layer)
  Stage 2: Retrieve raw turns, prioritizing turns from matched sessions

Summaries are NOT included in the generation context — they serve purely
as retrieval targets to route the query to the right sessions. The LLM
only sees raw dialogue turns with session headers for temporal context.

Why this works better:
  - Summaries have clean, query-like prose → good at finding relevant sessions
  - Raw turns have exact facts and wording → good for generating precise answers
  - Keeping them separate prevents summaries from crowding raw turns out of context
"""


class MemoryRetriever:
    """Two-stage retriever: summaries route to sessions, turns provide evidence."""

    def __init__(
        self,
        embed_model,
        summary_store,
        turn_store,
        top_k: int = 8,
        summary_k: int = 2,
    ):
        self.embed_model = embed_model
        self.summary_store = summary_store
        self.turn_store = turn_store
        self.top_k = top_k
        self.summary_k = summary_k

    def _find_relevant_sessions(self, query_emb) -> set[int]:
        """Stage 1: Use summaries to find which sessions are relevant."""
        if len(self.summary_store) == 0:
            return set()

        matches = self.summary_store.search(query_emb, k=self.summary_k)
        return {m["session_id"] for m in matches}

    def retrieve(self, query: str) -> list[dict]:
        """Two-stage retrieval: summaries → sessions → boosted turn search.

        1. Search summaries to identify relevant sessions
        2. Retrieve raw turns with a bonus for turns from matched sessions
        3. Return top-k turns (no summaries in results)
        """
        if len(self.turn_store) == 0:
            return []

        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )

        # Stage 1: find relevant sessions via summaries
        matched_session_ids = self._find_relevant_sessions(q_emb)

        # Stage 2: retrieve more candidates than needed, then re-rank
        fetch_k = max(self.top_k * 3, 15)
        candidates = self.turn_store.search(q_emb, k=fetch_k)

        if matched_session_ids:
            # Boost turns from matched sessions: they sort before non-matched
            # turns, regardless of raw similarity. Within each group, original
            # similarity order is preserved.
            candidates.sort(key=lambda m: (
                0 if m["metadata"].get("session_id") in matched_session_ids else 1,
                -m["score"],
            ))

        return candidates[:self.top_k]

    def format_context(self, memories: list[dict]) -> str:
        """Format retrieved raw turns grouped by session with date headers.

        Summaries are deliberately excluded — the LLM generates from exact
        dialogue wording, not from lossy summaries.
        """
        if not memories:
            return "No relevant information found."

        # Group turns by session_id, preserving retrieval order within each
        groups: dict[int, list[dict]] = {}
        session_order: list[int] = []
        for m in memories:
            sid = m["metadata"].get("session_id", -1)
            if sid not in groups:
                groups[sid] = []
                session_order.append(sid)
            groups[sid].append(m)

        lines = []
        for sid in session_order:
            turns = groups[sid]
            date = turns[0]["metadata"].get("date_time", "unknown")
            lines.append(f"--- Session {sid} ({date}) ---")
            for m in turns:
                lines.append(m.get("text", ""))

        return "\n".join(lines)
