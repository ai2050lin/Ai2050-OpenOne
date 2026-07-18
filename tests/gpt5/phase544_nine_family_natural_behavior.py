#!/usr/bin/env python3
"""Run the frozen Phase544 natural behavior matrix on one CUDA model at a time."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")
OUT_DIR = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
CASES_PATH = OUT_DIR / "phase544_registered_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase544_frozen_protocol.json"
STATIC_AUDIT_PATH = OUT_DIR / "phase544_static_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_protocol() -> dict[str, Any]:
    audit = read_json(STATIC_AUDIT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if audit["status"] != "static_pass_no_model_run" or not audit["valid"]:
        raise RuntimeError("Phase544 static gate has not passed")
    if protocol["registered_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase544 registered case bank drift")
    if protocol["evidence_policy"]["sealed_split_read"]:
        raise RuntimeError("Phase544 behavior protocol must not read a sealed split")
    return protocol


def candidate_position(text: str, candidate: str, *, case_sensitive: bool) -> int | None:
    if not candidate:
        return None
    flags = 0 if case_sensitive else re.IGNORECASE
    if any("\u4e00" <= char <= "\u9fff" for char in candidate):
        haystack = text if case_sensitive else text.casefold()
        needle = candidate if case_sensitive else candidate.casefold()
        position = haystack.find(needle)
        return position if position >= 0 else None
    pattern = re.compile(rf"(?<!\w){re.escape(candidate)}(?!\w)", flags=flags)
    match = pattern.search(text)
    return match.start() if match else None


def classify_semantic(row: dict[str, Any], generated: str) -> dict[str, Any]:
    case_sensitive = row["mechanism_id"] == "transform"
    target_positions = [
        position
        for alias in row["target_aliases"]
        if (position := candidate_position(generated, alias, case_sensitive=case_sensitive)) is not None
    ]
    distractor_positions = [
        position
        for distractor in row["distractors"]
        if (position := candidate_position(generated, distractor, case_sensitive=case_sensitive)) is not None
    ]
    target_position = min(target_positions) if target_positions else None
    distractor_position = min(distractor_positions) if distractor_positions else None
    recoverable = target_position is not None or distractor_position is not None
    correct = target_position is not None and (
        distractor_position is None or target_position < distractor_position
    )
    if correct:
        event = "target"
    elif distractor_position is not None:
        event = "registered_distractor"
    else:
        event = "unrecoverable"
    normalized = " ".join(generated.strip().split())
    protocol_valid = False
    if row["strict_kind"] == "json_answer":
        try:
            parsed = json.loads(generated.strip())
            protocol_valid = parsed == {"answer": row["target"]}
        except (json.JSONDecodeError, TypeError):
            protocol_valid = False
    else:
        protocol_valid = normalized == row["strict_expected"]
    return {
        "semantic_event": event,
        "semantic_event_recoverable": recoverable,
        "semantic_correct": correct,
        "strict_sequence_correct": normalized == row["strict_expected"],
        "protocol_valid": protocol_valid,
        "normalized_generated": normalized,
    }


def run_contract_path(model: str) -> Path:
    return OUT_DIR / f"phase544_{model}_run_contract.json"


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase544_{model}_behavior_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase544_{model}_execution_summary.json"


def prepare_resume(model: str, restart: bool) -> tuple[set[str], Path]:
    output = rows_path(model)
    contract_path = run_contract_path(model)
    frozen = {
        "schema_version": "phase544_model_run_contract.v1",
        "model": model,
        "created_at": now(),
        "registered_cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "do_sample": False,
        "sealed_split_read": False,
    }
    if restart:
        output.unlink(missing_ok=True)
        contract_path.unlink(missing_ok=True)
        summary_path(model).unlink(missing_ok=True)
    if contract_path.exists():
        existing = read_json(contract_path)
        for key in ("model", "registered_cases_sha256", "protocol_sha256", "do_sample"):
            if existing[key] != frozen[key]:
                raise RuntimeError(f"Phase544 resume contract drift for {model}: {key}")
    else:
        write_json(contract_path, frozen)
    completed: set[str] = set()
    if output.exists():
        for row in read_jsonl(output):
            completed.add(row["case_id"])
    return completed, output


def run_model(
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    use_8bit: bool,
    restart: bool,
) -> Path:
    verify_protocol()
    all_rows = [row for row in read_jsonl(CASES_PATH) if row["model"] == model_name]
    completed, output = prepare_resume(model_name, restart)
    pending = [row for row in all_rows if row["case_id"] not in completed]
    started = time.monotonic()
    model = None
    if not pending:
        payload = {
            "schema_version": "phase544_model_execution_summary.v1",
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "registered_case_count": len(all_rows),
            "completed_case_count": len(completed),
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "cuda_used": True,
            "sealed_split_read": False,
            "resumed_without_generation": True,
        }
        write_json(summary_path(model_name), payload)
        return summary_path(model_name)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase544 model execution requires CUDA")
        model, tokenizer, device = load_model(
            model_name, use_8bit=True if use_8bit else None
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        generated_count = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            )
            prompt_width = int(encoded["input_ids"].shape[1])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_rows = []
            for index, source in enumerate(batch):
                text = tokenizer.decode(generated[index, prompt_width:], skip_special_tokens=True)
                row = dict(source)
                row["generated_text"] = text
                row.update(classify_semantic(row, text))
                output_rows.append(row)
            append_jsonl(output, output_rows)
            generated_count += len(output_rows)
            del generated, encoded
            done = len(completed) + generated_count
            if start == 0 or done == len(all_rows) or (start // batch_size) % 16 == 15:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} {done}/{len(all_rows)}",
                    flush=True,
                )
        final_rows = read_jsonl(output)
        if len(final_rows) != len(all_rows) or len({row["case_id"] for row in final_rows}) != len(all_rows):
            raise RuntimeError(
                f"Phase544 incomplete or duplicate rows for {model_name}: {len(final_rows)}/{len(all_rows)}"
            )
        payload = {
            "schema_version": "phase544_model_execution_summary.v1",
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "registered_case_count": len(all_rows),
            "completed_case_count": len(final_rows),
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "cuda_used": True,
            "sealed_split_read": False,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "resumed_case_count": len(completed),
            "newly_generated_case_count": generated_count,
        }
        write_json(summary_path(model_name), payload)
        print(summary_path(model_name))
        return summary_path(model_name)
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run_model(
        args.model, args.batch_size, args.max_new_tokens,
        args.use_8bit, args.restart,
    )


if __name__ == "__main__":
    main()
