"""
No-Memory 基线：
  - ingest() 不做任何记忆构建
  - answer() 直接把问题喂给 LLM，不带任何对话上下文

这是理论上最弱的基线，用于验证记忆模块是否真的带来了增益。
如果你的系统比 No-Memory 还差，说明记忆在帮倒忙。

使用方式：
    python run_generation.py --eval_set eval_set.json \
        --agent no_memory_agent:NoMemoryAgent \
        --output predictions_nomem.json
"""

from llm_client import LLMClient


class NoMemoryAgent:
    """只把当前问题喂给 LLM，完全不看对话历史。"""

    def __init__(self):
        self.llm = LLMClient()

    def ingest(self, conversation: dict) -> None:
        """不做任何事情——这个基线故意不使用对话。"""
        pass

    def answer(self, question: str) -> str:
        prompt = (
            "You are answering a question. "
            "Keep the answer short (a phrase or one sentence). "
            "If you don't know the answer, reply 'unknown'.\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.generate(prompt, max_tokens=64).strip()
