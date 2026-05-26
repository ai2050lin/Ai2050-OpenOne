from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

from huggingface_hub import snapshot_download

from model_registry import MODEL_ROOT, all_model_keys, get_model_spec


def has_model_files(path: Path) -> bool:
    if not path.exists() or not (path / "config.json").exists():
        return False
    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        data = json.loads(index_file.read_text(encoding="utf-8"))
        required = sorted(set(data.get("weight_map", {}).values()))
        return bool(required) and all((path / name).exists() for name in required)
    return any(path.glob("*.safetensors"))


def snapshot_with_retries(spec, attempts: int = 6) -> str:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return snapshot_download(
                repo_id=spec.repo_id,
                local_dir=str(spec.local_dir),
                max_workers=1,
                allow_patterns=[
                    "*.json",
                    "*.model",
                    "*.txt",
                    "*.py",
                    "*.safetensors",
                    "tokenizer.*",
                    "generation_config.json",
                    "configuration*.py",
                    "modeling*.py",
                ],
            )
        except Exception as exc:  # noqa: BLE001 - download layer must resume robustly
            last_error = exc
            print(f"[download] attempt {attempt}/{attempts} failed for {spec.key}: {exc}")
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"Failed to download {spec.key} after {attempts} attempts") from last_error


def download_one(model_key: str, force: bool = False) -> dict:
    spec = get_model_spec(model_key)
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    if has_model_files(spec.local_dir) and not force:
        return {
            "model": spec.key,
            "repo_id": spec.repo_id,
            "local_dir": str(spec.local_dir),
            "status": "exists",
        }

    resolved = snapshot_with_retries(spec)
    return {
        "model": spec.key,
        "repo_id": spec.repo_id,
        "local_dir": str(spec.local_dir),
        "resolved_dir": resolved,
        "status": "downloaded",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        nargs="*",
        default=all_model_keys(),
        help="Model keys: qwen3 glm4 deepseek7b. Default: all.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for model_key in args.models:
        print(f"[download] {model_key}")
        results.append(download_one(model_key, force=args.force))
        print(json.dumps(results[-1], ensure_ascii=False, indent=2))

    manifest = MODEL_ROOT / "download_manifest.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[download] manifest: {manifest}")


if __name__ == "__main__":
    main()
