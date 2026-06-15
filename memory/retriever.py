"""MemoryRetriever — simple semantic search with evidence-first formatting.

v3 improvements over v2:
  - Raw dialogue turns are presented FIRST (primary evidence)
  - Session summaries follow as supporting context notes
  - top_k increased to 10 (more raw turns in context)

The retrieval itself is unchanged from v2: embed query → cosine similarity
→ top-k. The value is in how results are formatted for the generation model.
"""


class MemoryRetriever:
    """Simple semantic retrieval with evidence-first context formatting."""

    def __init__(self, embed_model, store, top_k: int = 10):
        self.embed_model = embed_model
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve top-k items by cosine similarity."""
        if len(self.store) == 0:
            return []

        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        return self.store.search(q_emb, k=self.top_k)

    def format_context(self, memories: list[dict]) -> str:
        """Format retrieved items — raw dialogue first, summaries as notes.

        v3 key change: raw turns are the primary evidence (shown first),
        summaries provide temporal/topical context (shown after).
        This ensures the generation model anchors on exact dialogue
        wording before seeing the compressed summary view.
        """
        if not memories:
            return "No relevant information found."

        summaries = [m for m in memories
                     if m["metadata"].get("category") == "session_summary"]
        turns = [m for m in memories
                 if m["metadata"].get("category") == "raw_turn"]

        lines = []

        # --- Raw dialogue turns first: primary evidence ---
        if turns:
            # Group by session for readability
            groups: dict[int, list[dict]] = {}
            session_order: list[int] = []
            for m in turns:
                sid = m["metadata"].get("session_id", -1)
                if sid not in groups:
                    groups[sid] = []
                    session_order.append(sid)
                groups[sid].append(m)

            lines.append("=== Relevant Dialogue ===")
            for sid in session_order:
                grp = groups[sid]
                date = grp[0]["metadata"].get("date_time", "unknown")
                lines.append(f"--- Session {sid} ({date}) ---")
                for m in grp:
                    lines.append(m.get("text", ""))

        # --- Summaries second: context anchors ---
        if summaries:
            lines.append("\n=== Session Context ===")
            for m in summaries:
                sid = m["metadata"].get("session_id", "?")
                date = m["metadata"].get("date_time", "")
                text = m.get("text", "")
                lines.append(f"[Session {sid} @ {date}] {text}")

        return "\n".join(lines) if lines else "No relevant information found."
