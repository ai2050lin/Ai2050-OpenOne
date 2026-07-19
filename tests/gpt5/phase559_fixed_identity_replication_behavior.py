#!/usr/bin/env python3
"""Run Phase559 independent behavior replication on one CUDA model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase558_fixed_identity_color_behavior import classify  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")
BEHAVIOR_SPLITS = ("behavior_discovery", "behavior_confirmation")
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CASES_PATH = OUT_DIR / "phase559_open_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase559_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase559_static_audit.json"
EXPECTED_MODEL_ROWS = 8192


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


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase559_{model}_behavior_rows.jsonl"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase559_{model}_behavior_run_contract.json"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase559_{model}_behavior_execution_summary.json"


def verify_protocol() -> dict[str, Any]:
    audit = read_json(AUDIT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if not audit["valid"] or protocol["open_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase559 frozen protocol failed or drifted")
    if protocol["phase558_contract_changes"] != {
        "behavior_thresholds_changed": False,
        "classifier_changed": False,
        "fact_order_conditions_changed": False,
        "only_new_disjoint_objects_and_larger_denominator": True,
        "surface_templates_changed": False,
    }:
        raise RuntimeError("Phase559 exact-contract replication drift")
    return protocol


def prepare(model: str, restart: bool) -> tuple[set[str], Path]:
    output = rows_path(model)
    contract = {
        "schema_version": "phase559_behavior_run_contract.v1",
        "created_at": now(),
        "model": model,
        "open_cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "selected_splits": list(BEHAVIOR_SPLITS),
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "use_8bit": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (output, contract_path(model), summary_path(model)):
            path.unlink(missing_ok=True)
    if contract_path(model).exists():
        old = read_json(contract_path(model))
        for key in (
            "model", "open_cases_sha256", "protocol_sha256", "selected_splits",
            "do_sample", "torch_dtype_requested", "use_8bit", "sealed_split_read",
        ):
            if old[key] != contract[key]:
                raise RuntimeError(f"Phase559 resume contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), contract)
    completed = {row["case_id"] for row in read_jsonl(output)} if output.exists() else set()
    return completed, output


def run(model_name: str, batch_size: int, max_new_tokens: int, restart: bool) -> Path:
    protocol = verify_protocol()
    rows = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_name and row["split"] in BEHAVIOR_SPLITS
    ]
    if len(rows) != EXPECTED_MODEL_ROWS or any(row["sealed"] for row in rows):
        raise RuntimeError(f"Unexpected Phase559 behavior denominator for {model_name}")
    completed, output = prepare(model_name, restart)
    pending = [row for row in rows if row["case_id"] not in completed]
    started = time.monotonic()
    model = None
    if not pending:
        return summary_path(model_name)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase559 behavior requires CUDA")
        model, tokenizer, device = load_model(model_name, dtype=torch.bfloat16, use_8bit=False)
        run_dtype = str(next(model.parameters()).dtype)
        quantized = bool(getattr(model, "is_loaded_in_8bit", False))
        if run_dtype != "torch.bfloat16" or quantized:
            raise RuntimeError(f"Phase559 precision drift: {run_dtype}/{quantized}")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        generated_count = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch], return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            width = int(encoded["input_ids"].shape[1])
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            output_rows = []
            for index, source in enumerate(batch):
                text = tokenizer.decode(generated[index, width:], skip_special_tokens=True)
                output_rows.append({
                    **source, **classify(source, text),
                    "torch_dtype": run_dtype, "quantized_8bit": quantized,
                })
            append_jsonl(output, output_rows)
            generated_count += len(output_rows)
            del generated, encoded, output_rows
            done = len(completed) + generated_count
            if start == 0 or done == len(rows) or (start // batch_size) % 25 == 24:
                print(f"[{time.strftime('%H:%M:%S')}] {model_name} Phase559 {done}/{len(rows)}", flush=True)
        final_rows = read_jsonl(output)
        if len(final_rows) != len(rows) or len({row["case_id"] for row in final_rows}) != len(rows):
            raise RuntimeError(f"Incomplete Phase559 behavior for {model_name}")
        summary = {
            "schema_version": "phase559_behavior_execution_summary.v1",
            "phase_id": "Phase559",
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "completed_case_count": len(final_rows),
            "expected_case_count": protocol["behavior_case_count_per_model"],
            "runtime_seconds": time.monotonic() - started,
            "resumed_case_count": len(completed),
            "newly_generated_case_count": generated_count,
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "cuda_used": True,
            "torch_dtype": run_dtype,
            "quantized_8bit": quantized,
            "sealed_split_read": False,
        }
        write_json(summary_path(model_name), summary)
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
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.max_new_tokens, args.restart)
