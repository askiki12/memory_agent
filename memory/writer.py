"""MemoryWriter — lightweight session summarization.

Key design choice (learned from evaluation results):
  A 3B model cannot reliably produce structured JSON. When it fails, facts are
  silently lost and the memory store is empty. Vanilla RAG beats us simply by
  keeping raw dialogue chunks with zero information loss.

Instead, we ask the 3B model to write a short natural-language summary per session.
This is a generation task the model can actually do well. The summary is stored
alongside raw chunks as a "high-quality retrieval target" — it captures the
essence of each session in dense form, making semantic search more effective.

No JSON parsing. No structured categories. Just natural language.
"""

EXTRACTION_SYSTEM = """You are a note-taker. Read a conversation and write down the key facts in 3-5 sentences. Focus on:
- Who did what, when, and where
- Preferences, opinions, and plans mentioned
- Important personal information shared
- Events and experiences discussed

Write in natural English. Be specific — include names, dates, places.
Do NOT include greetings, small talk, or filler."""

EXTRACTION_PROMPT = """Summarize the key facts from this conversation.

Date: {date_time}
Speakers: {speaker_a} and {speaker_b}

{dialogue}

Key facts (3-5 sentences):"""


class MemoryWriter:
    """Generates natural-language session summaries using an LLM.

    The output is a short paragraph (3-5 sentences) — no JSON, no structure.
    If the LLM call fails, returns an empty list (caller falls back to raw chunks).
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def _format_session(self, session: dict) -> str:
        lines = []
        for turn in session.get("turns", []):
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            lines.append(f"[{speaker}]: {text}")
        return "\n".join(lines)

    def summarize_session(
        self, session: dict, speaker_a: str = "A", speaker_b: str = "B"
    ) -> str | None:
        """Generate a short NL summary for a single session.

        Returns a string (3-5 sentences) or None on failure.
        """
        dialogue = self._format_session(session)
        date_time = session.get("date_time", "unknown")

        prompt = EXTRACTION_PROMPT.format(
            date_time=date_time,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            dialogue=dialogue,
        )

        try:
            response = self.llm.generate(
                prompt, max_tokens=256, temperature=0.0, system=EXTRACTION_SYSTEM
            )
        except Exception as e:
            print(f"[MemoryWriter] LLM call failed for session {session.get('session_id', -1)}: {e}")
            return None

        summary = response.strip()
        # Filter out clearly bad outputs
        if not summary or len(summary) < 20:
            return None
        return summary

    def extract_from_sessions(
        self, sessions: list[dict], speaker_a: str = "A", speaker_b: str = "B"
    ) -> list[dict]:
        """Generate summaries for all sessions.

        Returns list of summary memory dicts with keys:
          text, category, session_id, date_time
        """
        summaries = []
        for session in sessions:
            summary = self.summarize_session(session, speaker_a, speaker_b)
            if summary:
                summaries.append({
                    "text": summary,
                    "category": "session_summary",
                    "session_id": session.get("session_id", -1),
                    "date_time": session.get("date_time", "unknown"),
                })
        return summaries
