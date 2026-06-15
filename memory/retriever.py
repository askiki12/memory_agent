import re
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Simple English stopwords for keyword extraction
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "this", "that", "these", "those", "in", "on", "at", "to", "for",
    "of", "with", "from", "by", "about", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "because",
    "now", "then", "here", "there", "when", "where", "why", "how",
    "what", "which", "who", "whom", "did", "does", "doing", "also",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop stopwords and short tokens."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class MemoryRetriever:
    """Multi-strategy memory retrieval.

    Improvements over baseline:
      - Weighted score: semantic (0.50) + keyword (0.20) + importance (0.15) + recency (0.15)
      - Keyword overlap as a cheap, robust fallback for named entities
      - Query entity extraction for targeted matching
      - Multi-hop expansion: follow entity links from top results
      - Structured context formatting grouped by category
    """

    def __init__(
        self,
        embed_model,
        store,
        top_k: int = 15,             # retrieve more initially, then re-rank
        final_k: int = 10,           # how many to return after re-ranking
        recency_weight: float = 0.15,
        semantic_weight: float = 0.50,
        keyword_weight: float = 0.20,
        importance_weight: float = 0.15,
    ):
        self.embed_model = embed_model
        self.store = store
        self.top_k = top_k
        self.final_k = final_k
        self.recency_weight = recency_weight
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.importance_weight = importance_weight

    # ------------------------------------------------------------------
    # Query analysis
    # ------------------------------------------------------------------

    @staticmethod
    def extract_query_entities(query: str) -> set[str]:
        """Extract potential entity names from a query using simple heuristics.

        Looks for:
          - Capitalized words / proper nouns
          - Quoted strings
          - Key question targets (after "did", "does", "is", "are", "was", "were")
        """
        entities = set()

        # Capitalized words (likely proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        entities.update(capitalized)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.update(quoted)

        # All-caps acronyms
        acronyms = re.findall(r"\b[A-Z]{2,}\b", query)
        entities.update(acronyms)

        return entities

    # ------------------------------------------------------------------
    # Keyword scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        """Compute weighted keyword overlap score (Jaccard-like).

        Gives extra weight to multi-word entity matches.
        """
        q_tokens = _tokenize(query)
        t_tokens = _tokenize(text)

        if not q_tokens or not t_tokens:
            return 0.0

        q_set = set(q_tokens)
        t_set = set(t_tokens)

        intersection = q_set & t_set
        if not intersection:
            return 0.0

        # Jaccard coefficient
        union = q_set | t_set
        jaccard = len(intersection) / len(union)

        # Bonus: check for multi-word substring matches (entity names)
        query_lower = query.lower()
        text_lower = text.lower()
        substring_bonus = 0.0
        for word in q_set:
            if len(word) >= 4 and word in text_lower:
                # Count occurrences to boost important terms
                count = text_lower.count(word)
                substring_bonus += min(count, 3) * 0.05

        return min(1.0, jaccard + substring_bonus)

    # ------------------------------------------------------------------
    # Multi-strategy retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve and re-rank memories using multi-strategy scoring.

        Scoring formula:
          final = semantic * w_sem + keyword * w_kw + importance_norm * w_imp + recency * w_rec
        """
        if len(self.store) == 0:
            return []

        # Step 1: Semantic search (get top_k * 2 for re-ranking pool)
        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        candidates = self.store.search(q_emb, k=max(self.top_k, self.final_k * 2))

        if not candidates:
            return []

        # Pre-compute normalization bounds
        all_mems = self.store.get_all()
        if not all_mems:
            return candidates[:self.final_k]

        max_importance = max((m.get("importance", 5) for m in all_mems), default=10)
        max_session = max((m.get("session_id", 0) for m in all_mems), default=0)

        # Step 2: Multi-factor re-ranking
        query_entities = self.extract_query_entities(query)

        for r in candidates:
            meta = r["metadata"]
            text = r.get("text", "")

            # --- Semantic score (already cosine similarity, in [-1, 1] → normalize to [0,1]) ---
            sem = (r["score"] + 1.0) / 2.0  # map [-1, 1] to [0, 1]

            # --- Keyword score ---
            kw = self._keyword_score(query, text)
            # Entity bonus: if query entities appear in memory text
            if query_entities:
                text_lower = text.lower()
                entity_hits = sum(1 for e in query_entities if e.lower() in text_lower)
                kw = min(1.0, kw + entity_hits * 0.15)

            # --- Importance (normalized to [0,1]) ---
            imp = meta.get("importance", 5) / max(max_importance, 1)

            # --- Recency (normalized by max session_id) ---
            sid = meta.get("session_id", 0)
            rec = (sid / max(max_session, 1)) if max_session > 0 else 0.0

            # Weighted combination
            r["score"] = (
                self.semantic_weight * sem
                + self.keyword_weight * kw
                + self.importance_weight * imp
                + self.recency_weight * rec
            )

        # Sort by combined score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:self.final_k]

    # ------------------------------------------------------------------
    # Multi-hop expansion
    # ------------------------------------------------------------------

    def expand_by_entity(self, top_results: list[dict], k_extra: int = 5) -> list[dict]:
        """Follow entity links from top results to find related memories.

        Extracts named entities from the top results and searches for other
        memories mentioning the same entities — useful for multi-hop questions.
        """
        if not top_results or len(self.store) == 0:
            return []

        # Collect entities from top results
        all_entities = set()
        for r in top_results[:3]:
            text = r.get("text", "")
            entities = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", text)
            all_entities.update(entities)

        if not all_entities:
            return []

        # Build a query from found entities
        entity_query = " ".join(sorted(all_entities)[:10])
        extra = self.retrieve(entity_query)

        # Filter out results already in top_results
        seen_ids = {r["mem_id"] for r in top_results}
        new_results = [r for r in extra if r["mem_id"] not in seen_ids]
        return new_results[:k_extra]

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context(self, memories: list[dict]) -> str:
        """Format retrieved memories into a structured prompt context.

        Groups memories by category for better LLM comprehension.
        """
        if not memories:
            return "No relevant memories found."

        # Group by category
        by_category: dict[str, list[dict]] = {}
        for m in memories:
            cat = m["metadata"].get("category", "knowledge")
            by_category.setdefault(cat, []).append(m)

        lines = ["Relevant memories:"]

        # Order categories by relevance (more specific first)
        cat_order = ["personal_info", "preference", "relationship",
                     "plan", "event", "knowledge"]
        for cat in cat_order:
            if cat in by_category:
                cat_label = cat.replace("_", " ").title()
                lines.append(f"\n[{cat_label}]")
                for i, m in enumerate(by_category[cat], 1):
                    date = m["metadata"].get("date_time", "unknown date")
                    text = m.get("text", "")
                    imp = m["metadata"].get("importance", "?")
                    lines.append(f"  {i}. [{date}] (importance={imp}) {text}")

        # Any remaining uncategorized
        for cat, mems in by_category.items():
            if cat not in cat_order:
                lines.append(f"\n[{cat.replace('_', ' ').title()}]")
                for i, m in enumerate(mems, 1):
                    date = m["metadata"].get("date_time", "unknown date")
                    text = m.get("text", "")
                    lines.append(f"  {i}. [{date}] {text}")

        return "\n".join(lines)
