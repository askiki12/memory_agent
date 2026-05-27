"""Evaluation entry point for MyMemoryAgent.

Usage (from memory_agent/):
    source .venv/bin/activate
    python eval/run_eval.py --eval_set ../eval_kit/eval_set.json --output predictions.json

Or run with limit for quick testing:
    python eval/run_eval.py --eval_set ../eval_kit/eval_set.json --output predictions.json --limit 2
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Add eval_kit to path so we can import run_generation's utilities
EVAL_KIT = Path(__file__).resolve().parent.parent.parent / "eval_kit"
sys.path.insert(0, str(EVAL_KIT))

# Also add memory_agent's parent so the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llm_client import LLMClient  # noqa: E402
from agent.controller import MyMemoryAgent  # noqa: E402


def run_generation(eval_set_path, output_path, limit=None, resume=False):
    """Run MyMemoryAgent on the eval set, saving predictions."""
    with open(eval_set_path) as f:
        eval_set = json.load(f)
    if limit:
        eval_set = eval_set[:limit]

    done_ids = set()
    predictions = []
    out_path = Path(output_path)
    if resume and out_path.exists():
        with open(out_path) as f:
            predictions = json.load(f)
        done_ids = {p["qa_id"] for p in predictions}

    total_convs = len(eval_set)
    total_qas = sum(len(s["qa_list"]) for s in eval_set)
    print(f"[初始化] 共 {total_convs} 段对话，{total_qas} 题")

    qa_done = 0
    for i, sample in enumerate(eval_set):
        sample_id = sample["sample_id"]
        remaining_qas = [qa for qa in sample["qa_list"] if qa["qa_id"] not in done_ids]
        if not remaining_qas:
            qa_done += len(sample["qa_list"])
            continue

        print(f"[{i+1}/{total_convs}] {sample_id}：正在 ingest "
              f"{len(sample['conversation']['sessions'])} 个 session ...")
        t0 = time.time()
        try:
            agent = MyMemoryAgent()
            agent.ingest(sample["conversation"])
        except Exception as e:
            print(f"  [错误] ingest 失败：{e}")
            traceback.print_exc()
            for qa in remaining_qas:
                predictions.append({
                    "qa_id": qa["qa_id"],
                    "question": qa["question"],
                    "reference": qa["answer"],
                    "category": qa["category"],
                    "category_name": qa["category_name"],
                    "prediction": "",
                    "error": f"ingest_failed: {e}",
                    "latency_sec": 0.0,
                })
                qa_done += 1
            _save(out_path, predictions)
            continue
        ingest_time = time.time() - t0

        for qa in remaining_qas:
            t1 = time.time()
            try:
                pred = agent.answer(qa["question"])
                err = None
            except Exception as e:
                pred = ""
                err = f"answer_failed: {e}"
                traceback.print_exc()
            latency = time.time() - t1
            predictions.append({
                "qa_id": qa["qa_id"],
                "question": qa["question"],
                "reference": qa["answer"],
                "category": qa["category"],
                "category_name": qa["category_name"],
                "prediction": str(pred).strip(),
                "error": err,
                "latency_sec": round(latency, 3),
            })
            qa_done += 1

        _save(out_path, predictions)
        print(f"  ingest 耗时 {ingest_time:.1f}s，"
              f"回答了 {len(remaining_qas)} 题，进度 {qa_done}/{total_qas}")

    print(f"[完成] 共保存 {len(predictions)} 条预测 -> {out_path}")


def _save(path, preds):
    with open(path, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", required=True, help="评测集 JSON 路径")
    parser.add_argument("--output", default="predictions.json", help="输出文件路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 段对话")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    args = parser.parse_args()

    run_generation(args.eval_set, args.output, args.limit, args.resume)


if __name__ == "__main__":
    main()
