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
        """Format retrieved items — summaries first (temporal anchors),
        then raw dialogue turns grouped by session (exact evidence).

        Summaries provide resolved dates and topic context that help the
        model interpret raw turns. Raw turns provide the exact wording.
        """
        if not memories:
            return "No relevant information found."

        summaries = [m for m in memories
                     if m["metadata"].get("category") == "session_summary"]
        turns = [m for m in memories
                 if m["metadata"].get("category") == "raw_turn"]

        lines = []

        # --- Summaries first: temporal/topical anchors ---
        if summaries:
            lines.append("=== Session Summaries ===")
            for m in summaries:
                sid = m["metadata"].get("session_id", "?")
                date = m["metadata"].get("date_time", "")
                text = m.get("text", "")
                lines.append(f"[Session {sid} @ {date}] {text}")

        # --- Raw dialogue turns: exact evidence, grouped by session ---
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
