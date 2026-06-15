"""MemoryUpdater — lightweight capacity management for raw turns.

v3 simplification:
  Summaries live in a plain list (~10 items per conversation) and don't
  need dedup or pruning. The updater now only manages raw turn capacity.
  Pruning is chronological (oldest turns first) and rare — the default
  capacity of 3000 exceeds most conversations (~600 turns).
"""


class MemoryUpdater:
    """Capacity guard for raw turns. Summaries are managed separately."""

    def __init__(self, max_turns: int = 3000):
        self.max_turns = max_turns

    def prune(self, items: list[dict]) -> list[str]:
        """Remove oldest raw turns when over capacity.

        Only operates on items with category 'raw_turn'. Returns list of
        mem_ids to delete.
        """
        turns = [i for i in items if i.get("category") == "raw_turn"]
        if len(turns) <= self.max_turns:
            return []

        excess = len(turns) - self.max_turns
        sorted_turns = sorted(turns, key=lambda m: m.get("date_time", ""))
        to_prune = [m.get("mem_id", "") for m in sorted_turns[:excess]]
        return [mid for mid in to_prune if mid]
