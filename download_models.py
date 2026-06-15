"""Download required models to models/ directory.

Two models are needed:
  - Qwen2.5-3B-Instruct-AWQ (2.6G) — LLM for vLLM serving
  - bge-small-zh-v1.5 (184M) — Chinese text embeddings

Strategy: try huggingface_hub (with hf-mirror.com fallback), then modelscope.
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS = [
    {
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "ms_repo": "qwen/Qwen2.5-3B-Instruct-AWQ",
        "local_dir": "Qwen2.5-3B-Instruct-AWQ",
    },
    {
        "hf_repo": "BAAI/bge-small-en-v1.5",
        "ms_repo": "AI-ModelScope/bge-small-en-v1.5",
        "local_dir": "bge-small-en-v1.5",
    },
]


def download_hf(repo_id: str, target: str) -> bool:
    """Try huggingface_hub (direct, then mirror)."""
    from huggingface_hub import snapshot_download

    endpoints = [None, "https://hf-mirror.com"]
    for ep in endpoints:
        try:
            env = os.environ.copy()
            if ep:
                env["HF_ENDPOINT"] = ep
            # spawn a subprocess so env var takes effect for the whole download
            import subprocess

            cmd = [
                sys.executable, "-c",
                f"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{repo_id}', local_dir='{target}')",
            ]
            subprocess.run(cmd, env=env, check=True)
            return True
        except Exception as e:
            print(f"  HF{' mirror' if ep else ''} failed: {e}", file=sys.stderr)
    return False


def download_ms(repo_id: str, target: str) -> bool:
    """Try modelscope."""
    try:
        from modelscope import snapshot_download

        snapshot_download(repo_id, local_dir=target)
        return True
    except ImportError:
        print("  modelscope not installed; run: pip install modelscope", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  ModelScope failed: {e}", file=sys.stderr)
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for model in MODELS:
        target = os.path.join(MODELS_DIR, model["local_dir"])
        if os.path.exists(target) and os.listdir(target):
            print(f"[SKIP] {model['local_dir']} already exists")
            continue

        print(f"[DOWNLOAD] {model['local_dir']}")

        success = download_hf(model["hf_repo"], target)
        if not success:
            print("  Falling back to ModelScope...")
            success = download_ms(model["ms_repo"], target)

        if not success:
            print(f"[FAIL] {model['local_dir']} — download failed", file=sys.stderr)
            sys.exit(1)

        print(f"[OK] {model['local_dir']}")

    print("All models ready.")


if __name__ == "__main__":
    main()
