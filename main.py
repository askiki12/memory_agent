"""Demo: run MyMemoryAgent on a sample conversation."""

import json
import os
import sys
from pathlib import Path

# Ensure the memory_agent package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_agent.agent.controller import MyMemoryAgent


def main():
    # Sample conversation
    conversation = {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "sessions": [
            {
                "session_id": 1,
                "date_time": "10:00 am on 1 May 2023",
                "turns": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "Hi Bob! How was your weekend?"},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "Great! I went hiking at Bear Mountain with my dog."},
                    {"speaker": "Alice", "dia_id": "D1:3", "text": "That sounds fun. I've never been there. Is it difficult?"},
                    {"speaker": "Bob", "dia_id": "D1:4", "text": "Not too bad, about 5 miles round trip. My dog Bruno loved it."},
                    {"speaker": "Alice", "dia_id": "D1:5", "text": "I should go sometime. By the way, are you still working at Google?"},
                    {"speaker": "Bob", "dia_id": "D1:6", "text": "Actually, I left last month. I joined a startup called DataFlow as a senior engineer."},
                ],
            },
            {
                "session_id": 2,
                "date_time": "2:30 pm on 15 May 2023",
                "turns": [
                    {"speaker": "Bob", "dia_id": "D2:1", "text": "Hey Alice, remember I told you about DataFlow? We're hiring!"},
                    {"speaker": "Alice", "dia_id": "D2:2", "text": "Oh nice! What kind of roles?"},
                    {"speaker": "Bob", "dia_id": "D2:3", "text": "We need a backend engineer. I think you'd be perfect for it."},
                    {"speaker": "Alice", "dia_id": "D2:4", "text": "I'm interested. I've been wanting to leave my current job for a while."},
                    {"speaker": "Bob", "dia_id": "D2:5", "text": "Great! I'll refer you. The office is in downtown, near the Central Park."},
                ],
            },
        ],
    }

    print("=== Long-Term Memory Dialog Agent Demo ===\n")
    agent = MyMemoryAgent()

    print("Ingesting conversation...")
    agent.ingest(conversation)
    print(f"  Extracted {len(agent.store)} memories\n")

    print("Stored memories:")
    for i, m in enumerate(agent.store.get_all(), 1):
        print(f"  {i}. [{m.get('category', '')}] {m.get('text', '')}")

    print("\n---\n")

    questions = [
        "What is Bob's dog's name?",
        "Where does Bob work now?",
        "Where is the DataFlow office located?",
        "What did Bob do on the weekend?",
        "What is Alice's favorite color?",
    ]

    for q in questions:
        ans = agent.answer(q)
        print(f"Q: {q}")
        print(f"A: {ans}\n")


if __name__ == "__main__":
    main()
