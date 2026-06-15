"""One-click evaluation pipeline with existence checks and auto-generation.

Workflow:
1. Ensure eval_kit/eval_set.json exists; create it when missing.
2. Ensure three baseline prediction files exist; generate missing ones.
3. Ensure three baseline result files exist; judge missing ones.
4. Always rerun MyMemoryAgent generation and judge.
5. Store outputs in experiments/results/predictions and experiments/results/evals.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineConfig:
	name: str
	agent_spec: str
	prediction_filename: str
	result_filename: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_KIT_DIR = PROJECT_ROOT / "eval_kit"
EVAL_SET_PATH = EVAL_KIT_DIR / "eval_set.json"

PREDICTIONS_DIR = PROJECT_ROOT / "experiments" / "results" / "predictions"
EVALS_DIR = PROJECT_ROOT / "experiments" / "results" / "evals"

BASELINES: tuple[BaselineConfig, ...] = (
	BaselineConfig(
		name="fullctx",
		agent_spec="agent_template:FullContextAgent",
		prediction_filename="predictions_fullctx.json",
		result_filename="results_fullctx.json",
	),
	BaselineConfig(
		name="nomemory",
		agent_spec="no_memory_agent:NoMemoryAgent",
		prediction_filename="predictions_nomemory.json",
		result_filename="results_nomemory.json",
	),
	BaselineConfig(
		name="rag",
		agent_spec="vanilla_rag_agent:VanillaRAGAgent",
		prediction_filename="predictions_rag.json",
		result_filename="results_rag.json",
	),
)

MY_AGENT_SPEC = "agent.controller:MyMemoryAgent"
MY_PREDICTION_FILENAME = "predictions_mine.json"
MY_RESULT_FILENAME = "results_mine.json"


def _rel(path: Path) -> str:
	"""Return path relative to project root for stable command args."""
	return str(path.relative_to(PROJECT_ROOT))


def run_cmd(cmd: list[str], cwd: Path) -> None:
	print("\n$", " ".join(cmd))
	subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_eval_set() -> None:
	if EVAL_SET_PATH.exists():
		print(f"[OK] eval set exists: {EVAL_SET_PATH}")
		return

	print("[MISSING] eval_set.json not found, preparing dataset...")
	cmd = [
		sys.executable,
		"prepare_eval_set.py",
		"--output",
		"eval_set.json",
		"--per_category",
		"40",
		"--seed",
		"42",
	]
	run_cmd(cmd, cwd=EVAL_KIT_DIR)


def ensure_predictions_for_baselines() -> None:
	for baseline in BASELINES:
		pred_path = PREDICTIONS_DIR / baseline.prediction_filename
		if pred_path.exists():
			print(f"[OK] baseline prediction exists: {pred_path}")
			continue

		print(f"[MISSING] generating baseline prediction: {baseline.name}")
		cmd = [
			sys.executable,
			"eval_kit/run_generation.py",
			"--eval_set",
			_rel(EVAL_SET_PATH),
			"--agent",
			baseline.agent_spec,
			"--output",
			_rel(pred_path),
		]
		run_cmd(cmd, cwd=PROJECT_ROOT)


def ensure_results_for_baselines() -> None:
	for baseline in BASELINES:
		pred_path = PREDICTIONS_DIR / baseline.prediction_filename
		result_path = EVALS_DIR / baseline.result_filename
		if result_path.exists():
			print(f"[OK] baseline result exists: {result_path}")
			continue

		if not pred_path.exists():
			raise FileNotFoundError(
				f"Cannot run judge for {baseline.name}: missing predictions file {pred_path}"
			)

		print(f"[MISSING] judging baseline prediction: {baseline.name}")
		cmd = [
			sys.executable,
			"eval_kit/run_judge.py",
			"--predictions",
			_rel(pred_path),
			"--output",
			_rel(result_path),
			"--num_workers",
			"4",
		]
		run_cmd(cmd, cwd=PROJECT_ROOT)


def rerun_my_agent() -> None:
	pred_path = PREDICTIONS_DIR / MY_PREDICTION_FILENAME
	result_path = EVALS_DIR / MY_RESULT_FILENAME

	print("[RUN] regenerating predictions for MyMemoryAgent...")
	gen_cmd = [
		sys.executable,
		"eval_kit/run_generation.py",
		"--eval_set",
		_rel(EVAL_SET_PATH),
		"--agent",
		MY_AGENT_SPEC,
		"--output",
		_rel(pred_path),
	]
	run_cmd(gen_cmd, cwd=PROJECT_ROOT)

	print("[RUN] re-judging predictions for MyMemoryAgent...")
	judge_cmd = [
		sys.executable,
		"eval_kit/run_judge.py",
		"--predictions",
		_rel(pred_path),
		"--output",
		_rel(result_path),
		"--num_workers",
		"4",
	]
	run_cmd(judge_cmd, cwd=PROJECT_ROOT)


def main() -> None:
	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	EVALS_DIR.mkdir(parents=True, exist_ok=True)

	ensure_eval_set()
	ensure_predictions_for_baselines()
	ensure_results_for_baselines()
	rerun_my_agent()

	print("\n[Done] Evaluation workflow completed.")
	print(f"Predictions dir: {PREDICTIONS_DIR}")
	print(f"Results dir: {EVALS_DIR}")


if __name__ == "__main__":
	main()
