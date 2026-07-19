#!/usr/bin/env python3
"""Run one Phase549 factorial natural-behavior model on CUDA."""

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
from phase549_route_answer_factorial_protocol import (  # noqa: E402
    AUDIT_PATH, CASES_PATH, MODELS, OUT_DIR, PROTOCOL_PATH,
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


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase549_{model}_behavior_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase549_{model}_behavior_execution.json"


def verify() -> None:
    audit = read_json(AUDIT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if not audit["valid"] or audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase549 static gate failed")
    if protocol["registered_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase549 registered cases drift")
    if protocol["evidence_boundaries"]["new_sealed_split_read"]:
        raise RuntimeError("Phase549 cannot read a sealed split")


def prepare(model: str, restart: bool) -> set[str]:
    output = rows_path(model)
    contract = OUT_DIR / f"phase549_{model}_behavior_contract.json"
    frozen = {
        "schema_version": "phase549_behavior_contract.v1", "phase_id": "Phase549",
        "created_at": now(), "model": model, "cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "do_sample": False,
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
                raise RuntimeError(f"Phase549 resume drift: {model}/{key}")
    else:
        write_json(contract, frozen)
    return {row["case_id"] for row in read_jsonl(output)} if output.exists() else set()


def run(model_name: str, batch_size: int, max_new_tokens: int, use_8bit: bool, restart: bool) -> Path:
    verify()
    source = [row for row in read_jsonl(CASES_PATH) if row["model"] == model_name]
    completed = prepare(model_name, restart)
    pending = [row for row in source if row["case_id"] not in completed]
    loaded = None
    started = time.monotonic()
    generated_count = 0
    try:
        if pending:
            if not torch.cuda.is_available():
                raise RuntimeError("Phase549 behavior execution requires CUDA")
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
                out_rows = []
                for index, source_row in enumerate(batch):
                    text = tokenizer.decode(generated[index, width:], skip_special_tokens=True)
                    row = dict(source_row)
                    row["generated_text"] = text
                    row.update(classify_semantic(row, text))
                    out_rows.append(row)
                append_jsonl(rows_path(model_name), out_rows)
                generated_count += len(out_rows)
                del generated, encoded
                if start == 0 or generated_count % (batch_size * 16) == 0 or generated_count == len(pending):
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {model_name} "
                        f"{len(completed) + generated_count}/{len(source)}", flush=True,
                    )
        final_rows = read_jsonl(rows_path(model_name))
        if len(final_rows) != len(source) or len({row["case_id"] for row in final_rows}) != len(source):
            raise RuntimeError(f"Incomplete Phase549 behavior run: {model_name}")
        payload = {
            "schema_version": "phase549_behavior_execution.v1", "phase_id": "Phase549",
            "created_at": now(), "status": "complete", "model": model_name,
            "registered_case_count": len(source), "completed_case_count": len(final_rows),
            "rows_path": str(rows_path(model_name).relative_to(ROOT)),
            "rows_sha256": sha256_file(rows_path(model_name)), "cuda_used": True,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "resumed_case_count": len(completed), "newly_generated_case_count": generated_count,
            "new_sealed_split_read": False,
        }
        write_json(summary_path(model_name), payload)
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.max_new_tokens, args.use_8bit, args.restart)


if __name__ == "__main__":
    main()
