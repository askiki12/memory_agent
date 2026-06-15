"""MemoryRetriever — simple semantic search with dual-source context formatting.

Keeps it simple like Vanilla RAG: embed query → cosine similarity → top-k.
The value-add is in context formatting: session summaries are presented first
as "memory", raw turns as supporting evidence.
"""


class MemoryRetriever:
    """Simple semantic retrieval over a combined index of chunks + summaries."""

    def __init__(self, embed_model, store, top_k: int = 8):
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
        """Format retrieved items into a prompt-ready context.

        Session summaries first (dense, high-signal), then raw turns.
        """
        if not memories:
            return "No relevant information found."

        summaries = [m for m in memories if m["metadata"].get("category") == "session_summary"]
        turns = [m for m in memories if m["metadata"].get("category") == "raw_turn"]

        lines = []

        if summaries:
            lines.append("=== Session Summaries ===")
            for i, m in enumerate(summaries, 1):
                date = m["metadata"].get("date_time", "")
                text = m.get("text", "")
                lines.append(f"[Session {m['metadata'].get('session_id', '?')} @ {date}] {text}")

        if turns:
            lines.append("\n=== Relevant Dialogue ===")
            for i, m in enumerate(turns, 1):
                text = m.get("text", "")
                lines.append(text)

        return "\n".join(lines) if lines else "No relevant information found."
