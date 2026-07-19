#!/usr/bin/env python3
"""Run Phase556 open behavior qualification on one CUDA model at a time."""

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
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
CASES_PATH = OUT_DIR / "phase556_open_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase556_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase556_static_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol() -> dict[str, Any]:
    audit = read_json(AUDIT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if not audit["valid"] or audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase556 static protocol did not pass")
    if protocol["open_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase556 open case bank drift")
    if protocol["evidence_policy"]["sealed_split_read"]:
        raise RuntimeError("Phase556 behavior must not read sealed rows")
    return protocol


def candidate_position(text: str, candidate: str) -> int | None:
    if not candidate:
        return None
    pattern = re.compile(rf"(?<!\w){re.escape(candidate)}(?!\w)", flags=re.IGNORECASE)
    match = pattern.search(text)
    return match.start() if match else None


def classify(row: dict[str, Any], generated: str) -> dict[str, Any]:
    target_positions = [
        position for alias in row["target_aliases"]
        if (position := candidate_position(generated, alias)) is not None
    ]
    distractor_positions = [
        position for value in row["distractors"]
        if (position := candidate_position(generated, value)) is not None
    ]
    target_position = min(target_positions) if target_positions else None
    distractor_position = min(distractor_positions) if distractor_positions else None
    semantic_correct = target_position is not None and (
        distractor_position is None or target_position < distractor_position
    )
    normalized = " ".join(generated.strip().split())
    strict_correct = normalized.casefold() in {alias.casefold() for alias in row["target_aliases"]}
    if semantic_correct:
        event = "target"
    elif distractor_position is not None:
        event = "registered_distractor"
    else:
        event = "unrecoverable"
    return {
        "generated_text": generated,
        "normalized_generated": normalized,
        "semantic_event": event,
        "semantic_correct": semantic_correct,
        "strict_sequence_correct": strict_correct,
        "semantic_event_recoverable": event != "unrecoverable",
    }


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase556_{model}_behavior_rows.jsonl"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase556_{model}_behavior_run_contract.json"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase556_{model}_behavior_execution_summary.json"


def prepare(model: str, restart: bool, use_8bit: bool) -> tuple[set[str], Path]:
    output = rows_path(model)
    contract = {
        "schema_version": "phase556_behavior_run_contract.v1",
        "created_at": now(),
        "model": model,
        "open_cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "use_8bit": use_8bit,
        "sealed_split_read": False,
    }
    if restart:
        for path in (output, contract_path(model), summary_path(model)):
            path.unlink(missing_ok=True)
    if contract_path(model).exists():
        existing = read_json(contract_path(model))
        for key in (
            "model", "open_cases_sha256", "protocol_sha256", "do_sample",
            "torch_dtype_requested", "use_8bit", "sealed_split_read",
        ):
            if existing[key] != contract[key]:
                raise RuntimeError(f"Phase556 resume contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), contract)
    completed = {row["case_id"] for row in read_jsonl(output)} if output.exists() else set()
    return completed, output


def run(model_name: str, batch_size: int, max_new_tokens: int, use_8bit: bool, restart: bool) -> Path:
    verify_protocol()
    all_rows = [row for row in read_jsonl(CASES_PATH) if row["model"] == model_name]
    if len(all_rows) != 3872 or any(row["sealed"] for row in all_rows):
        raise RuntimeError(f"Unexpected Phase556 open denominator for {model_name}")
    completed, output = prepare(model_name, restart, use_8bit)
    pending = [row for row in all_rows if row["case_id"] not in completed]
    started = time.monotonic()
    model = None
    if not pending:
        write_json(summary_path(model_name), {
            "schema_version": "phase556_behavior_execution_summary.v1",
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
        })
        return summary_path(model_name)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase556 behavior requires CUDA")
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=use_8bit
        )
        run_dtype = str(next(model.parameters()).dtype)
        quantized_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        generated_count = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            prompt_width = int(encoded["input_ids"].shape[1])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            result_rows = []
            for index, source in enumerate(batch):
                text = tokenizer.decode(generated[index, prompt_width:], skip_special_tokens=True)
                result_rows.append({
                    **source,
                    **classify(source, text),
                    "torch_dtype": run_dtype,
                    "quantized_8bit": quantized_8bit,
                })
            append_jsonl(output, result_rows)
            generated_count += len(result_rows)
            del generated, encoded, result_rows
            done = len(completed) + generated_count
            if start == 0 or done == len(all_rows) or (start // batch_size) % 16 == 15:
                print(f"[{time.strftime('%H:%M:%S')}] {model_name} Phase556 {done}/{len(all_rows)}", flush=True)
        final_rows = read_jsonl(output)
        if len(final_rows) != len(all_rows) or len({row["case_id"] for row in final_rows}) != len(all_rows):
            raise RuntimeError(f"Incomplete Phase556 behavior for {model_name}")
        payload = {
            "schema_version": "phase556_behavior_execution_summary.v1",
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "registered_case_count": len(all_rows),
            "completed_case_count": len(final_rows),
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "resumed_case_count": len(completed),
            "newly_generated_case_count": generated_count,
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "cuda_used": True,
            "torch_dtype": run_dtype,
            "quantized_8bit": quantized_8bit,
            "sealed_split_read": False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.max_new_tokens, args.use_8bit, args.restart)
