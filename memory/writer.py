import json
import re
from typing import Optional

EXTRACTION_SYSTEM = """You are a memory extraction system. Your job is to read conversation transcripts and extract atomic, self-contained facts about the speakers.

For each conversation session, identify factual statements that would be useful for answering future questions. Focus on:
- personal_info: name, age, occupation, background, family members
- event: things that happened, meetings, trips, activities
- preference: likes, dislikes, opinions, habits
- relationship: connections between people, how they know each other
- plan: future intentions, scheduled events, goals
- knowledge: facts shared, information exchanged, advice given

Rules:
1. Each fact must be a self-contained declarative sentence that is understandable without the original context.
2. Skip greetings, filler, small talk, and purely emotional expressions.
3. Include specific names, dates, times, and locations when mentioned.
4. Output ONLY a JSON array of objects with keys: "fact" (str), "category" (one of the 6 types above).
5. If no facts are found, output an empty array [].
6. Do NOT include any text outside the JSON array."""

EXTRACTION_PROMPT = """Extract all atomic facts from the following conversation session.

Session date/time: {date_time}

Dialogue:
{dialogue}

Output ONLY a JSON array. Each fact object: {{"fact": "...", "category": "..."}}"""


class MemoryWriter:
    """Extracts atomic memory facts from conversation sessions using an LLM."""

    def __init__(self, llm_client, max_turns_per_batch: int = 60):
        self.llm = llm_client
        self.max_turns_per_batch = max_turns_per_batch

    def _format_session(self, session: dict) -> str:
        """Format a single session's dialogue as text."""
        lines = []
        for turn in session.get("turns", []):
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            lines.append(f"[{speaker}]: {text}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> list[dict]:
        """Parse LLM JSON response into memory dicts. Handles code fences and malformed output."""
        # Strip code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
        cleaned = re.sub(r"```\s*$", "", cleaned)

        try:
            facts = json.loads(cleaned)
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            # Try to recover by extracting JSON array
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return []

    def extract_from_sessions(
        self, sessions: list[dict], speaker_a: str = "A", speaker_b: str = "B"
    ) -> list[dict]:
        """Extract memories from a list of sessions.

        Returns list of memory dicts with keys: text, category, session_id, date_time.
        """
        all_memories = []

        for session in sessions:
            dialogue = self._format_session(session)
            date_time = session.get("date_time", "unknown")
            session_id = session.get("session_id", -1)

            prompt = EXTRACTION_PROMPT.format(date_time=date_time, dialogue=dialogue)

            try:
                response = self.llm.generate(
                    prompt, max_tokens=1024, temperature=0.0, system=EXTRACTION_SYSTEM
                )
            except Exception as e:
                print(f"[MemoryWriter] LLM call failed for session {session_id}: {e}")
                continue

            facts = self._parse_response(response)

            for item in facts:
                all_memories.append({
                    "text": item.get("fact", "").strip(),
                    "category": item.get("category", "knowledge"),
                    "session_id": session_id,
                    "date_time": date_time,
                })

        return all_memories
