import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ..llm_client import LLMClient
from ..memory.store import MemoryStore
from ..memory.writer import MemoryWriter
from ..memory.updater import MemoryUpdater
from ..memory.retriever import MemoryRetriever

ANSWER_SYSTEM = (
    "You are answering questions about a past conversation between two people. "
    "You will be given relevant extracted memories. Use only the provided memories "
    "to answer. Keep the answer short (a phrase or one sentence). "
    "If the memories do not contain the answer, reply 'unknown'."
)

ANSWER_PROMPT = """{context}

=== Question ===
{question}

=== Answer ==="""


class MyMemoryAgent:
    """Long-term memory dialog agent with memory extraction, storage, and retrieval."""

    def __init__(
        self,
        top_k: int = 10,
        similarity_threshold: float = 0.90,
        recency_weight: float = 0.2,
        max_memories: int = 500,
        log_dir: str | None = None,
    ):
        self.llm = LLMClient()

        # Embedding model: use local model dir; fall back to HuggingFace name
        embed_path = os.getenv("EMBED_MODEL_PATH", "")
        embed_model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
        if embed_path and Path(embed_path).exists():
            self.embed_model = SentenceTransformer(embed_path)
        else:
            self.embed_model = SentenceTransformer(embed_model_name)

        self.top_k = top_k
        self.store = MemoryStore(dim=512)
        self.writer = MemoryWriter(self.llm)
        self.updater = MemoryUpdater(self.embed_model, similarity_threshold, max_memories)
        self.retriever = MemoryRetriever(self.embed_model, self.store, top_k, recency_weight)

        # Logging
        self._log_dir = log_dir
        self._conv_log: dict = {"memories_added": 0, "qa_log": []}
        self._speaker_a = "A"
        self._speaker_b = "B"

    def ingest(self, conversation: dict) -> None:
        """Extract memories from all sessions and index them."""
        self._speaker_a = conversation.get("speaker_a", "A")
        self._speaker_b = conversation.get("speaker_b", "B")
        sessions = conversation.get("sessions", [])

        # Step 1: Extract candidate memories from each session
        new_memories = self.writer.extract_from_sessions(
            sessions, self._speaker_a, self._speaker_b
        )

        if not new_memories:
            return

        # Step 2: Merge with existing memories (dedup)
        existing = self.store.get_all()
        to_add, to_delete = self.updater.merge(new_memories, existing)

        # Step 3: Delete duplicates
        for mem_id in to_delete:
            self.store.delete(mem_id)

        # Step 4: Encode and add new memories
        if to_add:
            texts = [m["text"] for m in to_add]
            embeds = self.embed_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.store.add(np.array(embeds, dtype=np.float32), to_add)

        # Step 5: Prune if over capacity
        pruned = self.updater.prune(self.store.get_all())
        for mem_id in pruned:
            self.store.delete(mem_id)

        self._conv_log["memories_added"] = len(to_add)
        self._conv_log["memories_pruned"] = len(pruned)
        self._conv_log["total_memories"] = len(self.store)

    def answer(self, question: str) -> str:
        """Answer a question using retrieved memories."""
        # Retrieve relevant memories
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
            "retrieved_memories": [m["text"] for m in memories],
            "full_prompt": prompt,
        })

        return answer.strip()

    def save_log(self, output_path: str) -> None:
        """Save the QA log for this conversation to a JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._conv_log, f, ensure_ascii=False, indent=2)
