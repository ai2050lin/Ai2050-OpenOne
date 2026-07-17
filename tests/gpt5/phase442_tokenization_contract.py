#!/usr/bin/env python3
"""Phase442 tokenizer-only contract audit for Qwen3, GLM4, and DeepSeek7B."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from model_registry import MODEL_SPECS


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase442_static_sample_contract" / "phase442_samples.jsonl"
OUT_PATH = ROOT / "tests" / "gpt5" / "result" / "phase442_static_sample_contract" / "phase442_tokenization_report.json"
MANIFEST_PATH = ROOT / "tests" / "gpt5" / "result" / "phase442_static_sample_contract" / "phase442_artifact_manifest.json"


def load_samples() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SAMPLES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def audit_model(model_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
    )
    prompt_lengths = []
    answer_failures = []
    alias_lengths = []
    for row in rows:
        prompt_lengths.append(encode_len(tokenizer, row["input_text"]))
        aliases = row["answer_aliases"]
        encoded_aliases = [encode_len(tokenizer, alias) for alias in aliases]
        alias_lengths.extend(encoded_aliases)
        if not aliases or any(length <= 0 for length in encoded_aliases):
            answer_failures.append(row["sample_id"])

    max_prompt = max(prompt_lengths)
    return {
        "model": model_key,
        "tokenizer_path": str(spec.local_dir.relative_to(ROOT)),
        "samples_checked": len(rows),
        "alias_strings_checked": len(alias_lengths),
        "max_prompt_tokens": max_prompt,
        "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
        "max_alias_tokens": max(alias_lengths),
        "empty_alias_failures": answer_failures[:20],
        "empty_alias_failure_count": len(answer_failures),
        "prompt_length_limit": 1024,
        "prompt_over_limit_count": sum(length > 1024 for length in prompt_lengths),
        "status": "pass" if not answer_failures and max_prompt <= 1024 else "fail",
    }


def file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def update_manifest() -> None:
    if not MANIFEST_PATH.exists():
        return
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["artifacts"][str(OUT_PATH.relative_to(ROOT))] = file_sha256(OUT_PATH)
    manifest["joint_sha256"] = __import__("hashlib").sha256(
        json.dumps(manifest["artifacts"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_samples()
    reports = [audit_model(model_key, rows) for model_key in ("qwen3", "glm4", "deepseek7b")]
    out = {
        "schema_version": "phase442_tokenization_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(report["status"] == "pass" for report in reports) else "fail",
        "cuda_used": False,
        "model_weights_loaded": False,
        "reports": reports,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_manifest()
    print(OUT_PATH)


if __name__ == "__main__":
    main()
