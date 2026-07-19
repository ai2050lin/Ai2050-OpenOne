#!/usr/bin/env python3
"""Run the Phase548 matched natural-behavior gate on one CUDA model."""

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
from phase548_shared_attention_compute_protocol import MODELS  # noqa: E402


OUT_DIR = ROOT / "tests/gpt5/result/phase548_shared_attention_compute"
CASES_PATH = OUT_DIR / "phase548_registered_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase548_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase548_static_audit.json"


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


def verify() -> dict[str, Any]:
    audit = read_json(AUDIT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if audit["status"] != "static_pass_no_model_run" or not audit["valid"]:
        raise RuntimeError("Phase548 static protocol did not pass")
    if protocol["registered_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase548 case-bank hash drift")
    if protocol["evidence_boundaries"]["new_sealed_split_read"]:
        raise RuntimeError("Phase548 cannot read a sealed split")
    return protocol


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase548_{model}_behavior_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase548_{model}_behavior_execution.json"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase548_{model}_behavior_contract.json"


def prepare(model: str, restart: bool) -> set[str]:
    output = rows_path(model)
    contract = contract_path(model)
    frozen = {
        "schema_version": "phase548_behavior_contract.v1",
        "phase_id": "Phase548",
        "created_at": now(),
        "model": model,
        "cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "do_sample": False,
        "new_sealed_split_read": False,
    }
    if restart:
        output.unlink(missing_ok=True)
        contract.unlink(missing_ok=True)
        summary_path(model).unlink(missing_ok=True)
    if contract.exists():
        old = read_json(contract)
        for key in ("model", "cases_sha256", "protocol_sha256", "do_sample"):
            if old[key] != frozen[key]:
                raise RuntimeError(f"Phase548 behavior resume drift: {model}/{key}")
    else:
        write_json(contract, frozen)
    return {row["case_id"] for row in read_jsonl(output)} if output.exists() else set()


def run(model_name: str, batch_size: int, max_new_tokens: int, use_8bit: bool, restart: bool) -> Path:
    verify()
    source_rows = [row for row in read_jsonl(CASES_PATH) if row["model"] == model_name]
    completed = prepare(model_name, restart)
    pending = [row for row in source_rows if row["case_id"] not in completed]
    started = time.monotonic()
    loaded = None
    generated_count = 0
    try:
        if pending:
            if not torch.cuda.is_available():
                raise RuntimeError("Phase548 behavior execution requires CUDA")
            loaded, tokenizer, device = load_model(
                model_name, use_8bit=True if use_8bit else None,
            )
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
                output_rows = []
                for index, source in enumerate(batch):
                    text = tokenizer.decode(generated[index, width:], skip_special_tokens=True)
                    row = dict(source)
                    row["generated_text"] = text
                    row.update(classify_semantic(row, text))
                    output_rows.append(row)
                append_jsonl(rows_path(model_name), output_rows)
                generated_count += len(output_rows)
                del generated, encoded
                if start == 0 or generated_count % (batch_size * 16) == 0 or generated_count == len(pending):
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {model_name} "
                        f"{len(completed) + generated_count}/{len(source_rows)}",
                        flush=True,
                    )
        final_rows = read_jsonl(rows_path(model_name))
        if len(final_rows) != len(source_rows) or len({row["case_id"] for row in final_rows}) != len(source_rows):
            raise RuntimeError(f"Incomplete Phase548 behavior rows: {model_name}")
        summary = {
            "schema_version": "phase548_behavior_execution.v1",
            "phase_id": "Phase548",
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "registered_case_count": len(source_rows),
            "completed_case_count": len(final_rows),
            "rows_path": str(rows_path(model_name).relative_to(ROOT)),
            "rows_sha256": sha256_file(rows_path(model_name)),
            "cuda_used": True,
            "new_sealed_split_read": False,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "resumed_case_count": len(completed),
            "newly_generated_case_count": generated_count,
        }
        write_json(summary_path(model_name), summary)
        print(summary_path(model_name))
        return summary_path(model_name)
    finally:
        if loaded is not None:
            release_model(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.max_new_tokens, args.use_8bit, args.restart)


if __name__ == "__main__":
    main()
