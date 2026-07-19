#!/usr/bin/env python3
"""Run Phase564 behavior or edge-denominator behavior on one CUDA model."""

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
OUT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
CASES_PATH = OUT_DIR / "phase564_open_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase564_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase564_static_audit.json"
BEHAVIOR_SUMMARY_PATH = OUT_DIR / "phase564_behavior_summary.json"
EDGE_CONTRACT_PATH = OUT_DIR / "phase564_edge_behavior_frozen_contract.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(mode: str, model: str) -> tuple[Path, Path, Path]:
    prefix = f"phase564_{model}_{mode}_behavior"
    return (
        OUT_DIR / f"{prefix}_rows.jsonl",
        OUT_DIR / f"{prefix}_run_contract.json",
        OUT_DIR / f"{prefix}_execution_summary.json",
    )


def selected_splits(mode: str, protocol: dict[str, Any]) -> tuple[str, ...]:
    if mode == "behavior":
        return tuple(protocol["behavior_splits"])
    contract = read_json(EDGE_CONTRACT_PATH)
    return tuple(contract["selected_splits"])


def verify(mode: str, model: str) -> tuple[dict[str, Any], tuple[str, ...], int]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit["valid"] or protocol["open_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase564 frozen protocol failed or drifted")
    splits = selected_splits(mode, protocol)
    expected = sum(int(protocol["split_world_counts"][split]) * 32 for split in splits)
    if mode == "edge":
        contract = read_json(EDGE_CONTRACT_PATH)
        if model not in contract["authorized_models"] or contract["sealed_split_read"]:
            raise RuntimeError(f"Phase564 edge behavior is not authorized for {model}")
        if contract["parent_behavior_summary_sha256"] != sha256_file(BEHAVIOR_SUMMARY_PATH):
            raise RuntimeError("Phase564 edge behavior authorization drift")
    return protocol, splits, expected


def prepare(
    mode: str,
    model: str,
    splits: tuple[str, ...],
    expected: int,
    restart: bool,
) -> tuple[set[str], Path, Path]:
    rows_path, run_contract_path, summary_path = output_paths(mode, model)
    frozen = {
        "schema_version": f"phase564_{mode}_behavior_run_contract.v1",
        "phase_id": "Phase564",
        "created_at": now(),
        "mode": mode,
        "model": model,
        "open_cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "edge_contract_sha256": sha256_file(EDGE_CONTRACT_PATH) if mode == "edge" else None,
        "selected_splits": list(splits),
        "expected_rows": expected,
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "use_8bit": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (rows_path, run_contract_path, summary_path):
            path.unlink(missing_ok=True)
    if run_contract_path.exists():
        old = read_json(run_contract_path)
        checked = (
            "mode", "model", "open_cases_sha256", "protocol_sha256", "edge_contract_sha256",
            "selected_splits", "expected_rows", "do_sample", "torch_dtype_requested", "use_8bit",
            "sealed_split_read",
        )
        for key in checked:
            if old[key] != frozen[key]:
                raise RuntimeError(f"Phase564 resume contract drift: {mode}/{model}/{key}")
    else:
        write_json(run_contract_path, frozen)
    completed = {row["case_id"] for row in read_jsonl(rows_path)}
    return completed, rows_path, summary_path


def run(
    mode: str,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    restart: bool,
) -> Path:
    protocol, splits, expected = verify(mode, model_name)
    rows = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_name and row["split"] in set(splits)
    ]
    if len(rows) != expected or any(row["sealed"] for row in rows):
        raise RuntimeError(f"Unexpected Phase564 {mode} denominator for {model_name}")
    completed, output, summary_path = prepare(mode, model_name, splits, expected, restart)
    pending = [row for row in rows if row["case_id"] not in completed]
    if not pending:
        return summary_path
    model = None
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase564 behavior requires CUDA")
        model, tokenizer, device = load_model(model_name, dtype=torch.bfloat16, use_8bit=False)
        run_dtype = str(next(model.parameters()).dtype)
        quantized = bool(getattr(model, "is_loaded_in_8bit", False))
        if run_dtype != "torch.bfloat16" or quantized or not str(device).startswith("cuda"):
            raise RuntimeError(f"Phase564 precision/device drift: {run_dtype}/{quantized}/{device}")
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
                max_length=256,
            )
            width = int(encoded["input_ids"].shape[1])
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
                text = tokenizer.decode(generated[index, width:], skip_special_tokens=True)
                output_rows.append({
                    **source,
                    **classify(source, text),
                    "torch_dtype": run_dtype,
                    "quantized_8bit": quantized,
                })
            append_jsonl(output, output_rows)
            generated_count += len(output_rows)
            del generated, encoded, output_rows
            done = len(completed) + generated_count
            if start == 0 or done == len(rows) or (start // batch_size) % 25 == 24:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} Phase564 {mode} {done}/{len(rows)}",
                    flush=True,
                )
        final_rows = read_jsonl(output)
        if len(final_rows) != len(rows) or len({row["case_id"] for row in final_rows}) != len(rows):
            raise RuntimeError(f"Incomplete Phase564 {mode} behavior for {model_name}")
        summary = {
            "schema_version": f"phase564_{mode}_behavior_execution_summary.v1",
            "phase_id": "Phase564",
            "created_at": now(),
            "status": "complete",
            "mode": mode,
            "model": model_name,
            "selected_splits": list(splits),
            "completed_case_count": len(final_rows),
            "expected_case_count": expected,
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
        write_json(summary_path, summary)
        print(summary_path)
        return summary_path
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("behavior", "edge"))
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.model, args.batch_size, args.max_new_tokens, args.restart)
