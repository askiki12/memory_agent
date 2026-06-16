"""Hybrid Memory Agent v4 — multi-dimensional memory scoring.

v4 adds importance and recency scoring on top of v3's proven single-index
architecture. Memories carry metadata that feeds into retrieval ranking.
"""

import json
import os
import re
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
torch.set_num_threads(4)

from sentence_transformers import SentenceTransformer

from eval_kit.llm_client import LLMClient
from memory.store import MemoryStore
from memory.writer import MemoryWriter
from memory.updater import MemoryUpdater
from memory.retriever import MemoryRetriever


# ---------------------------------------------------------------------------
# Answer prompt — simple, not overly restrictive
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are answering questions about a past conversation between two people. "
    "Use only the provided information to answer. "
    "Keep the answer short (a phrase or one sentence). "
    "If the information does not contain the answer, reply 'unknown'."
)

# V5: temporal-specific prompt — softer, just encourages date awareness
TEMPORAL_SYSTEM = (
    "You are answering a time-related question about a past conversation. "
    "The dialogue excerpts below include session dates in the headers. "
    "Use these dates to understand when events happened. "
    "If you see relative expressions like 'yesterday' or 'last week' "
    "in the dialogue, the session date tells you the absolute date. "
    "Use only the provided information to answer. "
    "Keep the answer short (a phrase or one sentence). "
    "If the information does not contain the answer, reply 'unknown'."
)

ANSWER_PROMPT = """{context}

=== Question ===
{question}

=== Answer ==="""



class MyMemoryAgent:
    """Hybrid memory agent v3: single-index retrieval + evidence-first formatting.

    ingest:  store raw turns + generate session summaries (same as v2)
    answer:  semantic search → raw-first context → LLM generation
    """

    def __init__(
        self,
        top_k: int = 10,
        similarity_threshold: float = 0.92,
        max_memories: int = 3000,
        importance_weight: float = 0.08,
        log_dir: str | None = None,
    ):
        self.llm = LLMClient()

        # Embedding model
        embed_path = os.getenv("EMBED_MODEL_PATH", "models/bge-small-en-v1.5")
        embed_model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        if embed_path and Path(embed_path).exists():
            self.embed_model = SentenceTransformer(embed_path, device="cpu")
        else:
            self.embed_model = SentenceTransformer(embed_model_name, device="cpu")

        try:
            embed_dim = self.embed_model.get_embedding_dimension()
        except AttributeError:
            embed_dim = self.embed_model.get_sentence_embedding_dimension()

        self.top_k = top_k
        self.store = MemoryStore(dim=embed_dim)
        self.writer = MemoryWriter(self.llm)
        self.updater = MemoryUpdater(
            self.embed_model, similarity_threshold, max_memories
        )
        self.importance_weight = importance_weight
        self.retriever = MemoryRetriever(
            self.embed_model, self.store, top_k=top_k,
            importance_weight=importance_weight,
        )

        # Logging
        self._log_dir = log_dir
        self._conv_log: dict = {
            "num_raw_turns": 0,
            "num_summaries": 0,
            "total_indexed": 0,
            "ghost_vectors": 0,
            "qa_log": [],
        }
        self._speaker_a = "A"
        self._speaker_b = "B"

    # ------------------------------------------------------------------
    # Importance scoring (heuristic, zero LLM cost)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_importance(
        text: str, turn_index: int = 0, total_turns: int = 1
    ) -> float:
        """Heuristic importance score in [0, 1]. Zero extra LLM calls.

        Five equally-weighted signals:
          1. Entity presence (proper nouns)
          2. Date mentions
          3. Number mentions
          4. Turn length (normalized)
          5. Turn position (first/last 20% of session → higher)
        """
        score = 0.0
        words = text.split()

        # 1. Entity presence: proper nouns (exclude common stopwords)
        stop_lower = {
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "should", "may", "might", "shall", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "about", "like", "just", "so", "that", "this", "and", "but",
            "or", "not", "no", "yes", "if", "then", "than", "too", "very",
            "also", "up", "out", "when", "where", "who", "what", "how",
            "all", "there", "here", "go", "got", "get", "hi", "hey",
            "oh", "well", "yeah", "ok", "okay", "um", "uh", "really",
            "still", "back", "see", "know", "think", "one", "time",
            "good", "great", "nice", "love", "much", "way", "lot",
        }
        has_entity = any(
            len(w) > 1 and w[0].isupper() and w.lower() not in stop_lower
            for w in words
        )
        if has_entity:
            score += 0.2

        # 2. Date mentions
        date_patterns = [
            r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b',
            r'\b(20\d{2})\b',
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\b',
        ]
        has_date = any(re.search(p, text, re.IGNORECASE) for p in date_patterns)
        if has_date:
            score += 0.2

        # 3. Number mentions (excluding years which are caught above)
        has_number = bool(re.search(r'\b\d+\b', text))
        if has_number:
            score += 0.2

        # 4. Turn length (normalized, cap at 200 chars)
        length_norm = min(len(text) / 200.0, 1.0)
        score += 0.2 * length_norm

        # 5. Position in session (first/last 20% → potentially important)
        if total_turns > 1:
            pos_ratio = turn_index / (total_turns - 1)
            if pos_ratio < 0.2 or pos_ratio > 0.8:
                score += 0.2
        else:
            score += 0.2

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    # Recency scoring (time-decay based on date distance)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str) -> date | None:
        """Parse various date formats into a date object.

        Handles: '1:56 pm on 8 May, 2023', '8 May, 2023',
                 '10 July 2023', 'May 2023', '2023', etc.
        Returns None for unparseable strings.
        """
        if not date_str or date_str == "unknown":
            return None

        text = str(date_str).strip()

        # Try ISO-ish: '2023-07-10'
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            pass

        # Try '1:56 pm on 8 May, 2023' or '8 May, 2023'
        m = re.search(
            r'(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
            r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
            r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+(\d{4})',
            text, re.IGNORECASE
        )
        if m:
            day = int(m.group(1))
            month_str = m.group(2)[:3].lower()
            month_map = {
                "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
            }
            month = month_map.get(month_str, 1)
            year = int(m.group(3))
            try:
                return date(year, month, day)
            except ValueError:
                pass

        # Try 'May 2023' or 'June 2023' (no day)
        m = re.search(
            r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
            r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
            r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})',
            text, re.IGNORECASE
        )
        if m:
            month_str = m.group(1)[:3].lower()
            month_map = {
                "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
            }
            month = month_map.get(month_str, 1)
            year = int(m.group(2))
            return date(year, month, 15)  # mid-month default

        # Try bare year: '2022' or '2023'
        m = re.search(r'\b(20\d{2})\b', text)
        if m:
            return date(int(m.group(1)), 6, 15)  # mid-year default

        return None

    @staticmethod
    def _compute_recency(
        memory_date_str: str, latest_date: date | None, decay_lambda: float = 0.005
    ) -> float:
        """Compute recency score using exponential decay.

        recency = e^(-λ * days_diff)
        λ = 0.005 → half-life ~140 days.
        Returns 1.0 for today, ~0.5 for 140 days ago, ~0.25 for 280 days ago.
        """
        if latest_date is None:
            return 0.5  # neutral if no date reference

        mem_date = MyMemoryAgent._parse_date(memory_date_str)
        if mem_date is None:
            return 0.5  # neutral if unparseable

        days_diff = max(0.0, (latest_date - mem_date).days)
        import math
        return math.exp(-decay_lambda * days_diff)

    # ------------------------------------------------------------------

    def ingest(self, conversation: dict) -> None:
        """Store raw turns + generate and index session summaries."""
        self._speaker_a = conversation.get("speaker_a", "A")
        self._speaker_b = conversation.get("speaker_b", "B")
        sessions = conversation.get("sessions", [])

        # --- Compute latest date as reference for recency ---
        latest_date: date | None = None
        for sess in sessions:
            d = self._parse_date(sess.get("date_time", ""))
            if d and (latest_date is None or d > latest_date):
                latest_date = d

        # --- Raw dialogue turns ---
        raw_items = []
        for sess in sessions:
            date_time = sess.get("date_time", "unknown")
            session_id = sess.get("session_id", -1)
            turns_in_sess = sess.get("turns", [])
            total = len(turns_in_sess)
            for ti, turn in enumerate(turns_in_sess):
                text = f"[{date_time}] {turn['speaker']}: {turn['text']}"
                importance = self._compute_importance(text, ti, total)
                recency = self._compute_recency(date_time, latest_date)
                raw_items.append({
                    "text": text,
                    "category": "raw_turn",
                    "session_id": session_id,
                    "date_time": date_time,
                    "importance": importance,
                    "recency": recency,
                })

        if raw_items:
            raw_texts = [item["text"] for item in raw_items]
            raw_embeds = self.embed_model.encode(
                raw_texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.store.add(np.array(raw_embeds, dtype=np.float32), raw_items)

        # --- Session summaries ---
        new_summaries = self.writer.extract_from_sessions(
            sessions, self._speaker_a, self._speaker_b
        )

        if new_summaries:
            existing_summaries = [
                m for m in self.store.get_all()
                if m.get("category") == "session_summary"
            ]
            to_add, to_delete = self.updater.merge(new_summaries, existing_summaries)

            for mem_id in to_delete:
                self.store.delete(mem_id)

            if to_add:
                for m in to_add:
                    m["importance"] = 0.5
                    m["recency"] = self._compute_recency(
                        m.get("date_time", ""), latest_date
                    )
                summary_texts = [m["text"] for m in to_add]
                summary_embeds = self.embed_model.encode(
                    summary_texts, normalize_embeddings=True, show_progress_bar=False
                )
                self.store.add(
                    np.array(summary_embeds, dtype=np.float32), to_add
                )

        # --- Prune if over capacity ---
        pruned = self.updater.prune(self.store.get_all())
        for mem_id in pruned:
            self.store.delete(mem_id)

        # Logging
        self._conv_log["num_raw_turns"] = len(raw_items)
        self._conv_log["num_summaries"] = len(new_summaries)
        self._conv_log["total_indexed"] = len(self.store)
        self._conv_log["ghost_vectors"] = self.store.ghost_count

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    @staticmethod
    def _is_temporal(question: str) -> bool:
        """Detect temporal questions (V5)."""
        q = question.strip().lower()
        return q.startswith("when ") or q.startswith("when did")

    def answer(self, question: str) -> str:
        """Retrieve relevant items, format with evidence first, generate.

        V5: uses temporal-specific system prompt for 'when' questions.
        """
        memories = self.retriever.retrieve(question)
        context = self.retriever.format_context(memories, question)

        prompt = ANSWER_PROMPT.format(context=context, question=question)

        # V5: adaptive system prompt based on question type
        system = TEMPORAL_SYSTEM if self._is_temporal(question) else ANSWER_SYSTEM

        try:
            answer = self.llm.generate(
                prompt, max_tokens=64, temperature=0.0, system=system
            )
        except Exception as e:
            answer = f"error: {e}"

        # Log
        self._conv_log["qa_log"].append({
            "question": question,
            "answer": answer.strip(),
            "retrieved_summaries": sum(
                1 for m in memories
                if m["metadata"].get("category") == "session_summary"
            ),
            "retrieved_turns": sum(
                1 for m in memories
                if m["metadata"].get("category") == "raw_turn"
            ),
            "full_prompt": prompt,
        })

        return answer.strip()

    def save_log(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._conv_log, f, ensure_ascii=False, indent=2)
