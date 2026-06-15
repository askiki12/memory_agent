"""Hybrid Memory Agent — raw dialogue chunks + session summaries.

Architecture (post-evaluation redesign):
  Layer 0 — Raw dialogue turns (zero information loss, like Vanilla RAG)
  Layer 1 — Session summaries (natural-language, 3-5 sentences per session)

Why this design:
  Our evaluation showed the 3B model cannot reliably produce structured JSON
  facts. When JSON extraction fails, facts are silently lost and the system
  becomes worse than Vanilla RAG (which keeps everything).

  Instead, we:
    1. Store every dialogue turn as a raw chunk (RAG's strength: no loss)
    2. Generate simple NL summaries per session (easy for 3B model)
    3. Retrieve from both sources with pure semantic search
    4. Use a simple answer prompt (no CoT — confuses small models)

  The summaries act as "dense retrieval targets" that help surface relevant
  sessions. The raw turns provide the exact wording for answering.
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
from memory.store import MemoryStore
from memory.writer import MemoryWriter
from memory.updater import MemoryUpdater
from memory.retriever import MemoryRetriever


# ---------------------------------------------------------------------------
# Simple answer prompt — no CoT (CoT confuses 3B models)
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are answering questions about a past conversation between two people. "
    "Use only the provided information to answer. "
    "Keep the answer short (a phrase or one sentence). "
    "If the information does not contain the answer, reply 'unknown'."
)

ANSWER_PROMPT = """{context}

=== Question ===
{question}

=== Answer ==="""


class MyMemoryAgent:
    """Hybrid memory agent: raw dialogue chunks + session summaries.

    ingest:  store raw turns (like RAG) + generate session summaries (like memory)
    answer:  semantic search over combined index → simple LLM generation
    """

    def __init__(
        self,
        top_k: int = 8,
        similarity_threshold: float = 0.92,
        max_memories: int = 3000,
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
        self.updater = MemoryUpdater(self.embed_model, similarity_threshold, max_memories)
        self.retriever = MemoryRetriever(self.embed_model, self.store, top_k=top_k)

        # Logging
        self._log_dir = log_dir
        self._conv_log: dict = {
            "num_raw_turns": 0,
            "num_summaries": 0,
            "total_indexed": 0,
            "qa_log": [],
        }
        self._speaker_a = "A"
        self._speaker_b = "B"

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, conversation: dict) -> None:
        """Store raw turns + generate and index session summaries."""
        self._speaker_a = conversation.get("speaker_a", "A")
        self._speaker_b = conversation.get("speaker_b", "B")
        sessions = conversation.get("sessions", [])

        # --- Layer 0: Raw dialogue turns (zero information loss) ---
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
            self.store.add(np.array(raw_embeds, dtype=np.float32), raw_items)

        # --- Layer 1: Session summaries (dense retrieval targets) ---
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
                summary_texts = [m["text"] for m in to_add]
                summary_embeds = self.embed_model.encode(
                    summary_texts, normalize_embeddings=True, show_progress_bar=False
                )
                self.store.add(np.array(summary_embeds, dtype=np.float32), to_add)

        # --- Prune if over capacity ---
        pruned = self.updater.prune(self.store.get_all())
        for mem_id in pruned:
            self.store.delete(mem_id)

        # Logging
        self._conv_log["num_raw_turns"] = len(raw_items)
        self._conv_log["num_summaries"] = len(new_summaries)
        self._conv_log["total_indexed"] = len(self.store)

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        """Retrieve relevant chunks + summaries, then generate answer."""
        memories = self.retriever.retrieve(question)
        context = self.retriever.format_context(memories)

        prompt = ANSWER_PROMPT.format(context=context, question=question)

        try:
            answer = self.llm.generate(
                prompt, max_tokens=64, temperature=0.0, system=ANSWER_SYSTEM
            )
        except Exception as e:
            answer = f"error: {e}"

        # Log
        self._conv_log["qa_log"].append({
            "question": question,
            "answer": answer.strip(),
            "retrieved_summaries": sum(
                1 for m in memories if m["metadata"].get("category") == "session_summary"
            ),
            "retrieved_turns": sum(
                1 for m in memories if m["metadata"].get("category") == "raw_turn"
            ),
            "full_prompt": prompt,
        })

        return answer.strip()

    def save_log(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._conv_log, f, ensure_ascii=False, indent=2)
