#!/usr/bin/env python3
"""Run the frozen Phase559 path and unseen behavior denominator for Qwen3."""

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


MODEL = "qwen3"
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CASES_PATH = OUT_DIR / "phase559_open_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase559_frozen_protocol.json"
PATH_CONTRACT_PATH = OUT_DIR / "phase559_path_behavior_frozen_contract.json"
ROWS_PATH = OUT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
RUN_CONTRACT_PATH = OUT_DIR / "phase559_qwen3_path_behavior_run_contract.json"
SUMMARY_PATH = OUT_DIR / "phase559_qwen3_path_behavior_execution_summary.json"
EXPECTED_ROWS = 7168


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def prepare(restart: bool) -> set[str]:
    contract = read_json(PATH_CONTRACT_PATH)
    frozen = {
        "schema_version": "phase559_path_behavior_run_contract.v1",
        "created_at": now(),
        "model": MODEL,
        "parent_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "path_contract_sha256": sha256_file(PATH_CONTRACT_PATH),
        "selected_splits": contract["selected_splits"],
        "expected_rows": EXPECTED_ROWS,
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "use_8bit": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (ROWS_PATH, RUN_CONTRACT_PATH, SUMMARY_PATH):
            path.unlink(missing_ok=True)
    if RUN_CONTRACT_PATH.exists():
        old = read_json(RUN_CONTRACT_PATH)
        for key in (
            "model", "parent_protocol_sha256", "path_contract_sha256", "selected_splits",
            "expected_rows", "do_sample", "torch_dtype_requested", "use_8bit",
            "sealed_split_read",
        ):
            if old[key] != frozen[key]:
                raise RuntimeError(f"Phase559 path resume drift: {key}")
    else:
        write_json(RUN_CONTRACT_PATH, frozen)
    return {row["case_id"] for row in read_jsonl(ROWS_PATH)} if ROWS_PATH.exists() else set()


def run(batch_size: int, max_new_tokens: int, restart: bool) -> Path:
    path_contract = read_json(PATH_CONTRACT_PATH)
    if path_contract["authorized_models"] != [MODEL] or path_contract["evidence_policy"]["sealed_split_read"]:
        raise RuntimeError("Phase559 path behavior authorization drift")
    selected_splits = set(path_contract["selected_splits"])
    rows = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == MODEL and row["split"] in selected_splits
    ]
    if len(rows) != EXPECTED_ROWS or any(row["sealed"] for row in rows):
        raise RuntimeError("Phase559 path denominator mismatch")
    completed = prepare(restart)
    pending = [row for row in rows if row["case_id"] not in completed]
    model = None
    started = time.monotonic()
    if not pending:
        return SUMMARY_PATH
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase559 path behavior requires CUDA")
        model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16, use_8bit=False)
        run_dtype = str(next(model.parameters()).dtype)
        quantized = bool(getattr(model, "is_loaded_in_8bit", False))
        if run_dtype != "torch.bfloat16" or quantized:
            raise RuntimeError(f"Phase559 path precision drift: {run_dtype}/{quantized}")
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
            append_jsonl(ROWS_PATH, output_rows)
            generated_count += len(output_rows)
            del generated, encoded, output_rows
            done = len(completed) + generated_count
            if start == 0 or done == len(rows) or (start // batch_size) % 25 == 24:
                print(f"[{time.strftime('%H:%M:%S')}] qwen3 Phase559 path {done}/{len(rows)}", flush=True)
        final_rows = read_jsonl(ROWS_PATH)
        if len(final_rows) != len(rows) or len({row["case_id"] for row in final_rows}) != len(rows):
            raise RuntimeError("Phase559 path behavior incomplete")
        summary = {
            "schema_version": "phase559_path_behavior_execution_summary.v1",
            "phase_id": "Phase559",
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "completed_case_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "cuda_used": True,
            "torch_dtype": run_dtype,
            "quantized_8bit": quantized,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        print(SUMMARY_PATH)
        return SUMMARY_PATH
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.batch_size, args.max_new_tokens, args.restart)
