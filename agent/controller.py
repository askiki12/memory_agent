"""Hybrid Memory Agent v3 — two-stage retrieval with session routing.

Architecture (v3):
  SummaryStore (plain list) — summaries as "router" to find sessions
  TurnStore (FAISS) — raw dialogue turns as "evidence" for generation

  ingest:  generate session summaries → store in SummaryStore
           store raw turns in TurnStore
  answer:  Stage 1: query summaries → find relevant sessions
           Stage 2: retrieve raw turns with session-aware boosting
           Generate answer from raw turns only (no lossy summaries in context)

Key insight:
  Summaries have clean, query-like prose → excellent for semantic routing.
  Raw turns have exact facts and wording → essential for precise answers.
  Mixing them in one FAISS index (v2) caused summaries to crowd out raw
  turns, giving topically-relevant but factually-imprecise context.
  Separating their roles fixes this.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
torch.set_num_threads(4)

from sentence_transformers import SentenceTransformer

from eval_kit.llm_client import LLMClient
from memory import SummaryStore, TurnStore, MemoryWriter, MemoryUpdater, MemoryRetriever


# ---------------------------------------------------------------------------
# Answer prompt — raw turns only, no summaries
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are answering questions about a past conversation between two people. "
    "Below are relevant excerpts from that conversation. "
    "Use only the provided dialogue to answer. "
    "Keep the answer short (a phrase or one sentence). "
    "If the dialogue does not contain the answer, reply 'unknown'."
)

ANSWER_PROMPT = """{context}

=== Question ===
{question}

=== Answer ==="""


class MyMemoryAgent:
    """Hybrid memory agent v3: session-summary routing + raw-turn evidence.

    ingest:  store raw turns (TurnStore) + generate session summaries (SummaryStore)
    answer:  summaries route to sessions → raw turns provide evidence → generate
    """

    def __init__(
        self,
        top_k: int = 8,
        summary_k: int = 2,
        max_turns: int = 3000,
        log_dir: str | None = None,
    ):
        self.llm = LLMClient()

        # Embedding model (CPU, ~100MB)
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
        self.summary_k = summary_k

        # v3: separate stores for summaries (router) and turns (evidence)
        self.summary_store = SummaryStore(dim=embed_dim)
        self.turn_store = TurnStore(dim=embed_dim)

        self.writer = MemoryWriter(self.llm)
        self.updater = MemoryUpdater(max_turns=max_turns)
        self.retriever = MemoryRetriever(
            self.embed_model,
            summary_store=self.summary_store,
            turn_store=self.turn_store,
            top_k=top_k,
            summary_k=summary_k,
        )

        # Logging
        self._log_dir = log_dir
        self._conv_log: dict = {
            "num_raw_turns": 0,
            "num_summaries": 0,
            "qa_log": [],
        }
        self._speaker_a = "A"
        self._speaker_b = "B"

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, conversation: dict) -> None:
        """Store raw turns + generate session summaries in separate stores."""
        self._speaker_a = conversation.get("speaker_a", "A")
        self._speaker_b = conversation.get("speaker_b", "B")
        sessions = conversation.get("sessions", [])

        # --- Layer 0: Raw dialogue turns (evidence layer) ---
        raw_items = []
        for sess in sessions:
            date_time = sess.get("date_time", "unknown")
            session_id = sess.get("session_id", -1)
            for turn in sess.get("turns", []):
                raw_items.append({
                    "text": f"[{date_time}] {turn['speaker']}: {turn['text']}",
                    "category": "raw_turn",
                    "session_id": session_id,
                    "date_time": date_time,
                })

        if raw_items:
            raw_texts = [item["text"] for item in raw_items]
            raw_embeds = self.embed_model.encode(
                raw_texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.turn_store.add(np.array(raw_embeds, dtype=np.float32), raw_items)

        # --- Layer 1: Session summaries (router layer) ---
        new_summaries = self.writer.extract_from_sessions(
            sessions, self._speaker_a, self._speaker_b
        )

        if new_summaries:
            summary_texts = [m["text"] for m in new_summaries]
            summary_embeds = self.embed_model.encode(
                summary_texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.summary_store.add(
                np.array(summary_embeds, dtype=np.float32), new_summaries
            )

        # --- Prune turn store if over capacity (rare) ---
        pruned = self.updater.prune(self.turn_store.get_all())
        for mem_id in pruned:
            self.turn_store.delete(mem_id)

        # Logging
        self._conv_log["num_raw_turns"] = len(raw_items)
        self._conv_log["num_summaries"] = len(new_summaries)
        self._conv_log["total_turns_in_store"] = len(self.turn_store)
        self._conv_log["total_summaries_in_store"] = len(self.summary_store)

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        """Two-stage retrieval: summaries route → raw turns provide evidence."""
        retrieved = self.retriever.retrieve(question)
        context = self.retriever.format_context(retrieved)

        prompt = ANSWER_PROMPT.format(context=context, question=question)

        try:
            answer = self.llm.generate(
                prompt, max_tokens=64, temperature=0.0, system=ANSWER_SYSTEM
            )
        except Exception as e:
            answer = f"error: {e}"

        # Log
        num_summaries = len(retrieved.get("summaries", []))
        num_turns = len(retrieved.get("turns", []))
        matched_sessions = sorted(set(
            s["session_id"] for s in retrieved.get("summaries", [])
        ))
        self._conv_log["qa_log"].append({
            "question": question,
            "answer": answer.strip(),
            "num_retrieved_summaries": num_summaries,
            "num_retrieved_turns": num_turns,
            "matched_sessions": matched_sessions,
            "full_prompt": prompt,
        })

        return answer.strip()

    def save_log(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._conv_log, f, ensure_ascii=False, indent=2)
