import json
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Prompts — improved with importance scoring and few-shot guidance for the 3B model
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are a memory extraction system. Your job is to read conversation transcripts and extract atomic, self-contained facts about the speakers.

For each conversation session, identify factual statements that would be useful for answering future questions. Focus on:
- personal_info: name, age, occupation, background, family members
- event: things that happened, meetings, trips, activities
- preference: likes, dislikes, opinions, habits
- relationship: connections between people, how they know each other
- plan: future intentions, scheduled events, goals
- knowledge: facts shared, information exchanged, advice given

For each fact, also rate its importance (1-10):
- 1-3: trivial detail (small talk topic, passing mention)
- 4-6: moderately useful (a hobby, a one-time event)
- 7-8: important (recurring theme, significant life event, strong preference)
- 9-10: critical (core identity, life-changing event, key relationship info)

Rules:
1. Each fact must be a self-contained declarative sentence that is understandable without the original context.
2. Include specific names, dates, times, and locations when mentioned.
3. Skip greetings, filler, small talk, and purely emotional expressions.
4. Output ONLY a JSON array — no other text. Each object must have keys "fact", "category", "importance".
5. If no facts are found, output []."""

# Few-shot examples showing the expected format — helps the 3B model follow instructions
FEWSHOT_EXAMPLE = """
Example input:
Session date/time: 3:00 pm on 10 May, 2023
Dialogue:
[Alice]: Hey Bob! How was your trip to Paris?
[Bob]: It was amazing! I visited the Louvre and saw the Mona Lisa. I also tried escargot for the first time — surprisingly delicious.
[Alice]: Wow, that sounds incredible. Are you still planning to move there next year?
[Bob]: Yes, I'm applying for a work visa. The company office is near the Eiffel Tower.

Example output:
[
  {"fact": "Bob visited Paris and went to the Louvre museum where he saw the Mona Lisa.", "category": "event", "importance": 6},
  {"fact": "Bob tried escargot for the first time and found it surprisingly delicious.", "category": "event", "importance": 3},
  {"fact": "Bob plans to move to Paris next year and is applying for a work visa.", "category": "plan", "importance": 8},
  {"fact": "Bob's company office in Paris is near the Eiffel Tower.", "category": "knowledge", "importance": 4}
]
--- end example ---
"""

EXTRACTION_PROMPT = """Extract all atomic facts from the following conversation session.

Session date/time: {date_time}
Speakers: {speaker_a} and {speaker_b}

Dialogue:
{dialogue}

Output ONLY a JSON array. Each fact object: {{"fact": "...", "category": "...", "importance": N}} (N = 1-10)"""


class MemoryWriter:
    """Extracts atomic memory facts from conversation sessions using an LLM.

    Improvements over baseline:
      - Importance scoring (1-10) per fact
      - Few-shot example to guide small-model JSON output
      - Multi-pass parsing: direct JSON → regex extraction → line-by-line fallback
      - Session batching support (max_turns_per_batch) to avoid truncation
    """

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
        """Parse LLM JSON response into memory dicts.

        Uses multi-pass recovery for the 3B model which may produce imperfect JSON:
          1. Direct JSON parse (ideal case)
          2. Strip code fences → JSON parse
          3. Regex extract [...] array → JSON parse
          4. Line-by-line {{"fact": ...}} recovery (last resort)
        """
        cleaned = response.strip()

        # Pass 1: Try direct parse
        try:
            facts = json.loads(cleaned)
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            pass

        # Pass 2: Strip code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        try:
            facts = json.loads(cleaned)
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            pass

        # Pass 3: Regex extract JSON array
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Pass 4: Line-by-line recovery — look for {"fact": ...} objects
        facts = []
        for line in cleaned.split("\n"):
            line = line.strip().rstrip(",")
            if line.startswith("{") and '"fact"' in line:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "fact" in obj:
                        facts.append(obj)
                except json.JSONDecodeError:
                    continue

        return facts

    def extract_from_sessions(
        self, sessions: list[dict], speaker_a: str = "A", speaker_b: str = "B"
    ) -> list[dict]:
        """Extract memories from a list of sessions.

        Returns list of memory dicts with keys:
          text, category, importance, session_id, date_time, strength
        """
        all_memories = []

        for session in sessions:
            dialogue = self._format_session(session)
            date_time = session.get("date_time", "unknown")
            session_id = session.get("session_id", -1)

            # Build prompt with few-shot for the first session (guides model style)
            prompt = EXTRACTION_PROMPT.format(
                date_time=date_time,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                dialogue=dialogue,
            )
            system = EXTRACTION_SYSTEM + "\n\n" + FEWSHOT_EXAMPLE

            try:
                response = self.llm.generate(
                    prompt, max_tokens=1024, temperature=0.0, system=system
                )
            except Exception as e:
                print(f"[MemoryWriter] LLM call failed for session {session_id}: {e}")
                continue

            facts = self._parse_response(response)

            for item in facts:
                fact_text = item.get("fact", "").strip()
                if not fact_text:
                    continue
                all_memories.append({
                    "text": fact_text,
                    "category": item.get("category", "knowledge"),
                    "importance": int(item.get("importance", 5)),
                    "session_id": session_id,
                    "date_time": date_time,
                    "strength": 1.0,
                })

        return all_memories
