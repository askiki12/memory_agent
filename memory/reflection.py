"""Reflection module — synthesizes higher-level insights from atomic memories.

Inspired by Generative Agents (Park et al., UIST 2023):
  - Clusters related memories by entity and topic
  - Generates entity profiles, relationship summaries, and behavioral patterns
  - Produces "reflection memories" that are stored alongside atomic facts
  - These higher-level memories enable multi-hop and inferential reasoning

Trigger condition (Generative Agents style):
  - After all sessions are ingested, if cumulative importance > threshold,
    run a reflection pass.
"""

import json
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# Reflection prompts
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM = """You are a memory synthesis system. Your job is to read a list of atomic facts about two speakers and generate higher-level insights.

Generate insights in these categories:
- entity_profile: summarize what is known about one specific person (traits, background, preferences, goals)
- relationship: describe the relationship between the two speakers (how they know each other, dynamics, shared history)
- behavioral_pattern: identify recurring patterns, habits, or tendencies
- timeline: summarize key events in chronological order for a specific topic or person
- inference: draw a moderate, well-supported conclusion that is not explicitly stated but follows from multiple facts

Rules:
1. Each insight must be a self-contained declarative sentence.
2. Each insight must reference the specific facts it's based on (by fact index number).
3. Do NOT hallucinate — only synthesize from the provided facts.
4. Output ONLY a JSON array. No other text.
5. Each object: {"insight": "...", "category": "entity_profile|relationship|behavioral_pattern|timeline|inference", "importance": 7-10, "based_on": [1, 3, 5]}"""

REFLECTION_PROMPT = """Synthesize higher-level insights from the following atomic facts about {speaker_a} and {speaker_b}.

Numbered facts:
{fact_list}

Based on these facts, generate insights. For each insight, list which fact numbers it's based on.
Output ONLY a JSON array. Each object: {{"insight": "...", "category": "...", "importance": N, "based_on": [N, ...]}}"""


class ReflectionModule:
    """Generates higher-level memory insights from clusters of atomic facts.

    Attributes:
        importance_threshold: cumulative importance needed to trigger reflection
        max_facts_per_batch: cap on facts sent to LLM per reflection call
    """

    def __init__(
        self,
        llm_client,
        importance_threshold: int = 150,
        max_facts_per_batch: int = 80,
    ):
        self.llm = llm_client
        self.importance_threshold = importance_threshold
        self.max_facts_per_batch = max_facts_per_batch
        self._total_importance_since_last_reflection = 0

    def should_reflect(self, new_memories: list[dict]) -> bool:
        """Check if cumulative importance exceeds threshold."""
        for m in new_memories:
            self._total_importance_since_last_reflection += m.get("importance", 5)
        trigger = self._total_importance_since_last_reflection >= self.importance_threshold
        if trigger:
            self._total_importance_since_last_reflection = 0
        return trigger

    # ------------------------------------------------------------------
    # Entity-based clustering (no LLM needed — cheap preprocessing)
    # ------------------------------------------------------------------

    @staticmethod
    def _cluster_by_entity(
        memories: list[dict], speaker_a: str, speaker_b: str
    ) -> dict[str, list[int]]:
        """Group fact indices by which speaker entity they mention."""
        clusters: dict[str, list[int]] = defaultdict(list)
        clusters["all"] = []
        a_lower = speaker_a.lower()
        b_lower = speaker_b.lower()

        for i, m in enumerate(memories):
            text = m.get("text", "").lower()
            clusters["all"].append(i)
            if a_lower in text:
                clusters[f"about_{speaker_a}"].append(i)
            if b_lower in text:
                clusters[f"about_{speaker_b}"].append(i)
            if a_lower in text and b_lower in text:
                clusters["joint"].append(i)

        return dict(clusters)

    # ------------------------------------------------------------------
    # Main reflection pass
    # ------------------------------------------------------------------

    def reflect(
        self,
        memories: list[dict],
        speaker_a: str,
        speaker_b: str,
        session_id: int = -1,
        date_time: str = "",
    ) -> list[dict]:
        """Generate higher-level insights from a set of memories.

        Args:
            memories: list of memory dicts from the store
            speaker_a, speaker_b: speaker names
            session_id: session ID for the generated reflection memories
            date_time: date string for the generated reflection memories

        Returns:
            list of reflection memory dicts with keys:
              text, category, importance, session_id, date_time, strength,
              based_on (list of mem_ids)
        """
        if len(memories) < 8:
            return []  # not enough material to reflect on

        # Take the most important memories (up to max_facts_per_batch)
        sorted_mems = sorted(
            memories, key=lambda m: m.get("importance", 5), reverse=True
        )[:self.max_facts_per_batch]

        # Build numbered fact list for the prompt
        fact_lines = []
        mem_id_map: dict[int, str] = {}  # index → mem_id
        for i, m in enumerate(sorted_mems):
            fact_lines.append(f"[{i + 1}] ({m.get('category', '?')}) {m.get('text', '')}")
            mem_id_map[i + 1] = m.get("mem_id", "")

        prompt = REFLECTION_PROMPT.format(
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            fact_list="\n".join(fact_lines),
        )

        try:
            response = self.llm.generate(
                prompt, max_tokens=1024, temperature=0.3, system=REFLECTION_SYSTEM
            )
        except Exception as e:
            print(f"[Reflection] LLM call failed: {e}")
            return []

        insights = self._parse_response(response)
        if not insights:
            return []

        # Convert to memory dicts
        reflection_memories = []
        for item in insights:
            insight_text = item.get("insight", "").strip()
            if not insight_text:
                continue

            # Map fact indices back to mem_ids
            based_on_indices = item.get("based_on", [])
            based_on_ids = [
                mem_id_map[idx] for idx in based_on_indices if idx in mem_id_map
            ]

            reflection_memories.append({
                "text": insight_text,
                "category": f"reflection_{item.get('category', 'inference')}",
                "importance": int(item.get("importance", 8)),
                "session_id": session_id,
                "date_time": date_time,
                "strength": 2.0,  # reflections start stronger than atomic facts
                "based_on": based_on_ids,
            })

        return reflection_memories

    # ------------------------------------------------------------------
    # JSON parsing (same multi-pass logic as MemoryWriter)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: str) -> list[dict]:
        cleaned = response.strip()

        for attempt in range(3):
            if attempt == 0:
                candidate = cleaned
            elif attempt == 1:
                candidate = re.sub(r"^```(?:json)?\s*", "", cleaned)
                candidate = re.sub(r"```\s*$", "", candidate)
            else:
                match = re.search(r"\[.*\]", cleaned, re.DOTALL)
                if match:
                    candidate = match.group()
                else:
                    return []

            try:
                result = json.loads(candidate)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue

        return []
