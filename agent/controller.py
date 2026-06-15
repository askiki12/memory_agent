import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

# CRITICAL: Prevent PyTorch from touching GPU — avoids CUDA driver version
# mismatch crashes that kill the co-located vLLM Docker container.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
# Limit CPU threads to reduce memory pressure on WSL2 (7.6GB system RAM shared
# with Docker vLLM engine). 32 cores × full threading = high peak memory.
torch.set_num_threads(4)

from sentence_transformers import SentenceTransformer

from eval_kit.llm_client import LLMClient
from memory.store import MemoryStore
from memory.writer import MemoryWriter
from memory.updater import MemoryUpdater
from memory.retriever import MemoryRetriever
from memory.reflection import ReflectionModule


# ---------------------------------------------------------------------------
# Answer prompts — improved with chain-of-thought for complex questions
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = (
    "You are answering questions about a past conversation between two people. "
    "You will be given relevant memories extracted from the conversation. "
    "Use only the provided memories to answer. "
    "IMPORTANT: Your final output must be ONLY the answer itself — a short phrase "
    "or one sentence. Do NOT include memory numbers, dates, importance scores, "
    "or reasoning steps in your final output. "
    "If the memories do not contain enough information to answer, reply 'unknown'."
)

ANSWER_PROMPT_SIMPLE = """{context}

=== Question ===
{question}

Answer (short phrase or sentence only):"""

ANSWER_PROMPT_COT = """{context}

=== Question ===
{question}

Think step by step using only the provided memories, then give a final answer.
Your FINAL answer must be ONLY a short phrase or one sentence (or 'unknown').

Final answer:"""


def _clean_answer(raw: str) -> str:
    """Post-process LLM output to extract the final answer.

    Handles common 3B-model artifacts:
      - Strips context-format leakage (leading date/importance patterns)
      - Extracts text after 'Final answer:' markers
      - Removes reasoning prefixes
    """
    text = raw.strip()

    # If the model wrote 'Final answer:' or similar, take only what follows
    for marker in ["Final answer:", "final answer:", "Answer:", "=== Answer ==="]:
        if marker in text:
            parts = text.split(marker, 1)
            text = parts[-1].strip()

    # Strip leading lines that look like memory context (date + importance patterns)
    # e.g., "[10:37 am on 27 June, 2023] (importance=7) Melanie..."
    text = re.sub(r"^\[[\d:].*?\]\s*(\(importance=\d+\))?\s*", "", text)

    # Remove leading numbered list items from CoT leakage
    text = re.sub(r"^\d+\.\s*(Relevant facts|Final|Step).*?\n", "", text, flags=re.IGNORECASE)

    # If there are still newlines, take the last substantive line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) > 1:
        # Take the last line that doesn't look like a reasoning step
        for line in reversed(lines):
            if not line.startswith(("1.", "2.", "3.", "Step", "Relevant", "Based on", "From the")):
                text = line
                break

    # If the result is empty, return "unknown"
    if not text or len(text) < 2:
        text = "unknown"

    return text.strip()


# ---------------------------------------------------------------------------
# Query type heuristics — determine retrieval strategy without extra LLM calls
# ---------------------------------------------------------------------------

def _analyze_query_type(question: str) -> dict:
    """Quick rule-based query analysis to guide retrieval strategy.

    Returns dict with:
      - is_temporal: question asks about time/date/when
      - is_multi_hop: question likely requires combining multiple facts
      - is_preference: question asks about likes/wants/opinions
      - entities: set of capitalized words (likely names/places)
      - temporal_refs: time-related words found in the question
    """
    q_lower = question.lower()

    # Temporal indicators
    temporal_words = {
        "when", "what year", "what month", "what day", "what date",
        "how long", "how many years", "how old", "what time",
        "before", "after", "earlier", "later", "recently",
        "last", "first", "next", "previous",
    }
    is_temporal = any(tw in q_lower for tw in temporal_words)

    # Multi-hop indicators
    multi_hop_words = {
        "would", "might", "could", "likely", "probably", "consider",
        "what would", "what might", "what could", "how would",
        "based on", "given that", "infer", "suggest",
        "common", "both", "shared", "similar", "difference",
        "compare", "relationship", "between",
    }
    is_multi_hop = any(mw in q_lower for mw in multi_hop_words)

    # Preference indicators
    pref_words = {"prefer", "like", "enjoy", "favorite", "love", "hate",
                  "interest", "opinion", "think of", "feel about"}
    is_preference = any(pw in q_lower for pw in pref_words)

    # Entity extraction (proper nouns)
    entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question))
    # Filter out sentence-start words (first word capitalized)
    first_word = question.split()[0] if question.split() else ""
    entities.discard(first_word)

    return {
        "is_temporal": is_temporal,
        "is_multi_hop": is_multi_hop,
        "is_preference": is_preference,
        "entities": entities,
    }


class MyMemoryAgent:
    """Long-term memory dialog agent with memory extraction, storage, and retrieval.

    Architecture (3-layer):
      Layer 0 — Raw dialogue index (for fallback verification)
      Layer 1 — Atomic fact memories (extracted per session, importance-scored)
      Layer 2 — Reflection memories (entity profiles, relationships, inferences)

    Key mechanisms:
      - Multi-strategy retrieval (semantic + keyword + importance + recency)
      - Ebbinghaus forgetting (strengthen-on-access, decay-based pruning)
      - Reflection synthesis (Generative Agents style)
      - Chain-of-thought answering for complex questions
    """

    def __init__(
        self,
        top_k: int = 15,
        final_k: int = 10,
        similarity_threshold: float = 0.88,
        recency_weight: float = 0.15,
        max_memories: int = 500,
        reflection_threshold: int = 150,
        log_dir: str | None = None,
    ):
        self.llm = LLMClient()

        # Embedding model: use local model dir; fall back to HuggingFace name
        embed_path = os.getenv("EMBED_MODEL_PATH", "models/bge-small-en-v1.5")
        embed_model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        if embed_path and Path(embed_path).exists():
            self.embed_model = SentenceTransformer(embed_path, device="cpu")
        else:
            self.embed_model = SentenceTransformer(embed_model_name, device="cpu")

        self.top_k = top_k
        self.final_k = final_k

        # Auto-detect embedding dimension from the loaded model
        try:
            embed_dim = self.embed_model.get_embedding_dimension()
        except AttributeError:
            embed_dim = self.embed_model.get_sentence_embedding_dimension()
        self.store = MemoryStore(dim=embed_dim)
        self.writer = MemoryWriter(self.llm)
        self.updater = MemoryUpdater(
            self.embed_model,
            similarity_threshold=similarity_threshold,
            max_memories=max_memories,
        )
        self.retriever = MemoryRetriever(
            self.embed_model,
            self.store,
            top_k=top_k,
            final_k=final_k,
            recency_weight=recency_weight,
        )
        self.reflection = ReflectionModule(
            self.llm,
            importance_threshold=reflection_threshold,
        )

        # Logging
        self._log_dir = log_dir
        self._conv_log: dict = {
            "memories_added": 0,
            "reflections_added": 0,
            "memories_pruned": 0,
            "total_memories": 0,
            "qa_log": [],
        }
        self._speaker_a = "A"
        self._speaker_b = "B"
        self._last_session_date = ""

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, conversation: dict) -> None:
        """Extract memories from all sessions, merge, reflect, and index."""
        self._speaker_a = conversation.get("speaker_a", "A")
        self._speaker_b = conversation.get("speaker_b", "B")
        sessions = conversation.get("sessions", [])

        # Track the last session date for reflection memories
        if sessions:
            self._last_session_date = sessions[-1].get("date_time", "")

        # Step 1: Extract candidate memories from each session (Layer 1)
        new_memories = self.writer.extract_from_sessions(
            sessions, self._speaker_a, self._speaker_b
        )

        if not new_memories:
            return

        # Step 2: Merge with existing memories (dedup + conflict resolution)
        existing = self.store.get_all()
        to_add, to_delete = self.updater.merge(new_memories, existing)

        # Step 3: Delete replaced/duplicate memories
        for mem_id in to_delete:
            self.store.delete(mem_id)

        # Step 4: Encode and add new atomic memories (Layer 1)
        if to_add:
            texts = [m["text"] for m in to_add]
            embeds = self.embed_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.store.add(np.array(embeds, dtype=np.float32), to_add)

        # Step 5: Reflection — generate higher-level insights (Layer 2)
        reflection_memories = []
        if self.reflection.should_reflect(to_add):
            all_mems = self.store.get_all()
            reflection_memories = self.reflection.reflect(
                all_mems,
                self._speaker_a,
                self._speaker_b,
                session_id=len(sessions),  # place after last session
                date_time=self._last_session_date,
            )
            if reflection_memories:
                ref_texts = [m["text"] for m in reflection_memories]
                ref_embeds = self.embed_model.encode(
                    ref_texts, normalize_embeddings=True, show_progress_bar=False
                )
                self.store.add(np.array(ref_embeds, dtype=np.float32), reflection_memories)

        # Step 6: Prune if over capacity (importance-aware)
        pruned = self.updater.prune(self.store.get_all())
        for mem_id in pruned:
            self.store.delete(mem_id)

        # Logging
        self._conv_log["memories_added"] = len(to_add)
        self._conv_log["reflections_added"] = len(reflection_memories)
        self._conv_log["memories_pruned"] = len(pruned)
        self._conv_log["total_memories"] = len(self.store)
        self._conv_log["total_importance"] = sum(
            m.get("importance", 5) for m in self.store.get_all()
        )

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        """Answer a question using multi-strategy retrieval + optional CoT."""
        # Step 1: Analyze query type
        qa = _analyze_query_type(question)

        # Step 2: Retrieve relevant memories (multi-strategy)
        memories = self.retriever.retrieve(question)

        # Step 3: Multi-hop expansion for complex questions
        if qa["is_multi_hop"] and memories:
            extra = self.retriever.expand_by_entity(memories, k_extra=5)
            # Merge, deduplicate, and re-sort
            seen = {m["mem_id"] for m in memories}
            for e in extra:
                if e["mem_id"] not in seen:
                    memories.append(e)
                    seen.add(e["mem_id"])
            memories.sort(key=lambda x: x["score"], reverse=True)
            memories = memories[:self.final_k]

        # Step 4: Format context
        context = self.retriever.format_context(memories)

        # Step 5: Choose prompt (chain-of-thought for multi-hop/inferential questions)
        use_cot = qa["is_multi_hop"] or qa["is_preference"]
        prompt_template = ANSWER_PROMPT_COT if use_cot else ANSWER_PROMPT_SIMPLE
        prompt = prompt_template.format(context=context, question=question)

        max_tokens = 256 if use_cot else 96

        # Step 6: Generate answer
        try:
            raw = self.llm.generate(
                prompt, max_tokens=max_tokens, temperature=0.0, system=ANSWER_SYSTEM
            )
            answer = _clean_answer(raw)
        except Exception as e:
            answer = f"error: {e}"

        # Step 7: Strengthen accessed memories (Ebbinghaus — used = stronger)
        for m in memories:
            mid = m.get("mem_id", "")
            if mid:
                self.store.touch(mid)

        # Log
        self._conv_log["qa_log"].append({
            "question": question,
            "query_type": {k: v for k, v in qa.items() if k != "entities"},
            "query_entities": list(qa["entities"]),
            "answer": answer.strip(),
            "num_retrieved": len(memories),
            "retrieved_memories": [
                {"text": m["text"], "score": round(m["score"], 4),
                 "importance": m["metadata"].get("importance", "?")}
                for m in memories
            ],
            "used_cot": use_cot,
            "full_prompt": prompt,
        })

        return answer.strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self, output_path: str) -> None:
        """Save the QA log for this conversation to a JSON file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self._conv_log, f, ensure_ascii=False, indent=2)
