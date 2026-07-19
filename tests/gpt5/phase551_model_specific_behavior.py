#!/usr/bin/env python3
"""Run one Phase551 calibration or validation behavior bank on CUDA."""

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
from phase544_nine_family_natural_behavior import classify_semantic  # noqa: E402
from phase551_model_specific_route_protocol import (  # noqa: E402
    CALIBRATION_AUDIT_PATH,
    CALIBRATION_CASES_PATH,
    MODELS,
    OUT_DIR,
    PROTOCOL_PATH,
    VALIDATION_PROTOCOL_PATH,
    VALIDATION_AUDIT_PATH,
    VALIDATION_CASES_PATH,
)


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


def stage_paths(stage: str) -> tuple[Path, Path]:
    if stage == "calibration":
        return CALIBRATION_CASES_PATH, CALIBRATION_AUDIT_PATH
    return VALIDATION_CASES_PATH, VALIDATION_AUDIT_PATH


def rows_path(stage: str, model: str) -> Path:
    return OUT_DIR / f"phase551_{stage}_{model}_behavior_rows.jsonl"


def summary_path(stage: str, model: str) -> Path:
    return OUT_DIR / f"phase551_{stage}_{model}_behavior_execution.json"


def verify(stage: str) -> tuple[Path, Path, dict[str, Any]]:
    cases_path, audit_path = stage_paths(stage)
    audit = read_json(audit_path)
    if not audit["valid"] or not audit["status"].startswith("static_pass"):
        raise RuntimeError(f"Phase551 {stage} static gate failed")
    protocol_path = PROTOCOL_PATH if stage == "calibration" else VALIDATION_PROTOCOL_PATH
    protocol = read_json(protocol_path)
    if stage == "calibration":
        expected_hash = protocol["calibration_cases_sha256"]
    else:
        expected_hash = protocol["validation_cases_sha256"]
    if expected_hash != sha256_file(cases_path):
        raise RuntimeError(f"Phase551 {stage} cases drift")
    if protocol["evidence_boundaries"]["new_sealed_split_read"]:
        raise RuntimeError("Phase551 cannot read a sealed split")
    return cases_path, protocol_path, protocol


def prepare(stage: str, model: str, cases_path: Path, protocol_path: Path, restart: bool) -> set[str]:
    output = rows_path(stage, model)
    contract = OUT_DIR / f"phase551_{stage}_{model}_behavior_contract.json"
    frozen = {
        "schema_version": "phase551_behavior_contract.v1",
        "phase_id": "Phase551",
        "created_at": now(),
        "stage": stage,
        "model": model,
        "cases_sha256": sha256_file(cases_path),
        "protocol_sha256": sha256_file(protocol_path),
        "do_sample": False,
        "new_sealed_split_read": False,
    }
    if restart:
        output.unlink(missing_ok=True)
        contract.unlink(missing_ok=True)
        summary_path(stage, model).unlink(missing_ok=True)
    if contract.exists():
        old = read_json(contract)
        for key in ("stage", "model", "cases_sha256", "protocol_sha256", "do_sample"):
            if old[key] != frozen[key]:
                raise RuntimeError(f"Phase551 resume drift: {stage}/{model}/{key}")
    else:
        write_json(contract, frozen)
    return {row["case_id"] for row in read_jsonl(output)} if output.exists() else set()


def run(
    stage: str, model_name: str, batch_size: int, max_new_tokens: int,
    use_8bit: bool, restart: bool,
) -> Path:
    cases_path, protocol_path, _protocol = verify(stage)
    source = [row for row in read_jsonl(cases_path) if row["model"] == model_name]
    completed = prepare(stage, model_name, cases_path, protocol_path, restart)
    pending = [row for row in source if row["case_id"] not in completed]
    loaded = None
    started = time.monotonic()
    generated_count = 0
    try:
        if pending:
            if not torch.cuda.is_available():
                raise RuntimeError("Phase551 behavior execution requires CUDA")
            loaded, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                encoded = tokenizer(
                    [row["prompt"] for row in batch], return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )
                width = int(encoded["input_ids"].shape[1])
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    generated = loaded.generate(
                        **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    )
                rows_out = []
                for index, source_row in enumerate(batch):
                    text = tokenizer.decode(generated[index, width:], skip_special_tokens=True)
                    row = dict(source_row)
                    row["generated_text"] = text
                    row.update(classify_semantic(row, text))
                    rows_out.append(row)
                append_jsonl(rows_path(stage, model_name), rows_out)
                generated_count += len(rows_out)
                del generated, encoded
                if start == 0 or generated_count % (batch_size * 16) == 0 or generated_count == len(pending):
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {model_name} {stage} "
                        f"{len(completed) + generated_count}/{len(source)}",
                        flush=True,
                    )
        final_rows = read_jsonl(rows_path(stage, model_name)) if rows_path(stage, model_name).exists() else []
        if len(final_rows) != len(source) or len({row["case_id"] for row in final_rows}) != len(source):
            raise RuntimeError(f"Incomplete Phase551 behavior run: {stage}/{model_name}")
        payload = {
            "schema_version": "phase551_behavior_execution.v1",
            "phase_id": "Phase551",
            "created_at": now(),
            "status": "complete" if source else "skipped_no_authorized_contract",
            "stage": stage,
            "model": model_name,
            "registered_case_count": len(source),
            "completed_case_count": len(final_rows),
            "rows_path": str(rows_path(stage, model_name).relative_to(ROOT)) if source else None,
            "rows_sha256": sha256_file(rows_path(stage, model_name)) if source else None,
            "cuda_used": bool(source),
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "resumed_case_count": len(completed),
            "newly_generated_case_count": generated_count,
            "new_sealed_split_read": False,
        }
        write_json(summary_path(stage, model_name), payload)
        print(summary_path(stage, model_name))
        return summary_path(stage, model_name)
    finally:
        if loaded is not None:
            release_model(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--stage", choices=("calibration", "validation"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.stage, args.model, args.batch_size, args.max_new_tokens, args.use_8bit, args.restart)


if __name__ == "__main__":
    main()
