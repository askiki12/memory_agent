"""MemoryRetriever — parallel search with session-grouped context.

v3 design:
  Summaries and raw turns are stored and searched independently (separate
  stores → no FAISS slot competition). At format time, items are grouped
  by session: the summary goes first (temporal/topical anchor), followed
  by raw turns from that session (exact evidence).

  This grouping ensures the generation model reads the summary's resolved
  dates and topics immediately before the corresponding dialogue turns,
  combining the strengths of both sources.
"""


class MemoryRetriever:
    """Parallel retriever with session-grouped context formatting."""

    def __init__(
        self,
        embed_model,
        summary_store,
        turn_store,
        top_k: int = 10,
        summary_k: int = 2,
    ):
        self.embed_model = embed_model
        self.summary_store = summary_store
        self.turn_store = turn_store
        self.top_k = top_k
        self.summary_k = summary_k

    def retrieve(self, query: str) -> dict:
        """Parallel retrieval from both stores.

        Returns dict with summaries list and turns list.
        """
        result: dict = {"summaries": [], "turns": []}

        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )

        if len(self.summary_store) > 0:
            result["summaries"] = self.summary_store.search(
                q_emb, k=self.summary_k
            )

        if len(self.turn_store) > 0:
            result["turns"] = self.turn_store.search(q_emb, k=self.top_k)

        return result

    def format_context(self, retrieved: dict) -> str:
        """Format retrieved items grouped by session.

        For each session that appears in either summaries or turns:
        - Session header with date
        - Summary (if available) as a context note
        - Raw dialogue turns from that session

        Sessions with both summary AND turns are shown first (highest
        relevance), followed by sessions with only turns or only summaries.
        """
        summaries = retrieved.get("summaries", [])
        turns = retrieved.get("turns", [])

        if not summaries and not turns:
            return "No relevant information found."

        # Build a lookup from session_id → summary text
        summary_by_session: dict[int, str] = {}
        for s in summaries:
            sid = s.get("session_id", -1)
            if sid not in summary_by_session:
                summary_by_session[sid] = s.get("text", "")

        # Group turns by session_id
        turn_groups: dict[int, list[dict]] = {}
        session_order: list[int] = []
        for t in turns:
            sid = t["metadata"].get("session_id", -1)
            if sid not in turn_groups:
                turn_groups[sid] = []
                session_order.append(sid)
            turn_groups[sid].append(t)

        # Add any summary-only sessions
        for sid in summary_by_session:
            if sid not in turn_groups:
                session_order.append(sid)
                turn_groups[sid] = []

        # Sort sessions: those with BOTH summary+turns first
        def session_priority(sid: int) -> tuple[int, int]:
            has_summary = 0 if sid in summary_by_session else 1
            has_turns = 0 if turn_groups.get(sid) else 1
            return (has_summary + has_turns, sid)

        session_order.sort(key=session_priority)

        lines = []
        for sid in session_order:
            turns_in_session = turn_groups.get(sid, [])
            date = (
                turns_in_session[0]["metadata"].get("date_time", "unknown")
                if turns_in_session
                else "unknown"
            )

            # Session header with date
            lines.append(f"--- Session {sid} ({date}) ---")

            # Summary first (temporal/topical anchor)
            if sid in summary_by_session:
                lines.append(f"[Context] {summary_by_session[sid]}")

            # Raw turns (exact evidence)
            if turns_in_session:
                lines.append("[Dialogue]")
                for t in turns_in_session:
                    lines.append(t.get("text", ""))

        return "\n".join(lines) if lines else "No relevant information found."
