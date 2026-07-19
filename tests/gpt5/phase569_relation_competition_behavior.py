#!/usr/bin/env python3
"""Run the frozen Phase569 relation-competition behavior bank on one CUDA model."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402


PHASE_ID = "Phase569"
SCHEMA_PREFIX = "phase569"
MODELS = ("qwen3", "glm4", "deepseek7b")
EXPECTED_MODEL_ROWS = 48384
OUT_DIR = ROOT / "tests/gpt5/result/phase569_relation_competition"
CASES_PATH = OUT_DIR / "phase569_open_cases.jsonl.gz"
PROTOCOL_PATH = OUT_DIR / "phase569_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase569_static_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
        raise RuntimeError("Phase569 static protocol did not pass")
    if protocol["open_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase569 open case bank drift")
    if protocol["evidence_policy"]["sealed_split_read"]:
        raise RuntimeError("Phase569 behavior must not read sealed rows")
    if protocol["open_semantic_case_count"] != EXPECTED_MODEL_ROWS:
        raise RuntimeError("Phase569 open denominator drift")
    return protocol


def candidate_position(text: str, candidate: str) -> int | None:
    match = re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text, flags=re.IGNORECASE)
    return match.start() if match else None


def classify(row: dict[str, Any], generated: str) -> dict[str, Any]:
    observed = [
        (position, candidate)
        for candidate in row["all_candidates"]
        if (position := candidate_position(generated, candidate)) is not None
    ]
    selected = min(observed)[1] if observed else None
    semantic_correct = selected == row["target"]
    relation_confusion = selected == row["other_relation_target"]
    normalized = " ".join(generated.strip().split())
    strict = normalized.casefold() == row["target"].casefold()
    if semantic_correct:
        event = "target"
    elif relation_confusion:
        event = "same_object_other_relation"
    elif selected is not None:
        event = "registered_other"
    else:
        event = "unrecoverable"
    return {
        "generated_text": generated,
        "normalized_generated": normalized,
        "selected_candidate": selected,
        "semantic_event": event,
        "semantic_correct": semantic_correct,
        "strict_sequence_correct": strict,
        "semantic_event_recoverable": selected is not None,
        "relation_confusion": relation_confusion,
    }


def compact_source(row: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": "phase569_relation_competition_behavior.v1",
        "phase_id": PHASE_ID,
        "case_id": f"{model}__{row['semantic_case_id']}",
        "semantic_case_id": row["semantic_case_id"],
        "model": model,
        "split": row["split"],
        "anchor_id": row["anchor_id"],
        "triplet_id": row["triplet_id"],
        "factorial_cell": row["factorial_cell"],
        "binding": row["binding"],
        "query_object_index": row["query_object_index"],
        "query_object": row["query_object"],
        "query_relation": row["query_relation"],
        "other_relation": row["other_relation"],
        "surface_id": row["surface_id"],
        "fact_order": row["fact_order"],
        "value_regime": row["value_regime"],
        "values": row["values"],
        "target": row["target"],
        "other_relation_target": row["other_relation_target"],
        "all_candidates": row["all_candidates"],
        "raw_prompt": row["raw_prompt"],
        "registered_prompt_token_count": row["prompt_token_count_by_model"][model],
        "sealed": False,
    }


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase569_{model}_behavior_rows.jsonl"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase569_{model}_behavior_run_contract.json"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase569_{model}_behavior_execution_summary.json"


def prepare(model: str, restart: bool) -> tuple[set[str], Path]:
    output = rows_path(model)
    contract = {
        "schema_version": "phase569_behavior_run_contract.v1",
        "created_at": now(),
        "model": model,
        "open_cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "use_8bit": False,
        "max_prompt_tokens": 512,
        "sealed_split_read": False,
    }
    if restart:
        for path in (output, contract_path(model), summary_path(model)):
            path.unlink(missing_ok=True)
    if contract_path(model).exists():
        existing = read_json(contract_path(model))
        for key in (
            "model", "open_cases_sha256", "protocol_sha256", "do_sample",
            "torch_dtype_requested", "use_8bit", "max_prompt_tokens",
            "sealed_split_read",
        ):
            if existing[key] != contract[key]:
                raise RuntimeError(f"Phase569 resume contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), contract)
    completed = {row["case_id"] for row in iter_jsonl(output)} if output.exists() else set()
    return completed, output


def load_pending(model: str, completed: set[str]) -> tuple[int, list[dict[str, Any]]]:
    count = 0
    pending = []
    for row in iter_jsonl(CASES_PATH):
        count += 1
        if row["sealed"]:
            raise RuntimeError("Phase569 open bank contains a sealed row")
        source = compact_source(row, model)
        if source["case_id"] not in completed:
            pending.append(source)
    return count, pending


def completed_audit(path: Path) -> tuple[int, int]:
    count = 0
    case_ids = set()
    for row in iter_jsonl(path):
        count += 1
        case_ids.add(row["case_id"])
    return count, len(case_ids)


def run(model_name: str, batch_size: int, max_new_tokens: int, restart: bool) -> Path:
    verify_protocol()
    completed, output = prepare(model_name, restart)
    registered_count, pending = load_pending(model_name, completed)
    if registered_count != EXPECTED_MODEL_ROWS:
        raise RuntimeError(f"Unexpected Phase569 denominator: {registered_count}")
    started = time.monotonic()
    model = None
    if not pending:
        count, unique = completed_audit(output)
        if count != EXPECTED_MODEL_ROWS or unique != EXPECTED_MODEL_ROWS:
            raise RuntimeError("Phase569 completed output is not a full unique denominator")
        return summary_path(model_name)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase569 behavior requires CUDA")
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=False
        )
        run_dtype = str(next(model.parameters()).dtype)
        quantized_8bit = bool(getattr(model, "is_loaded_in_8bit", False))
        if run_dtype != "torch.bfloat16" or quantized_8bit:
            raise RuntimeError(
                f"Phase569 precision drift: dtype={run_dtype}, 8bit={quantized_8bit}"
            )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        generated_count = 0
        maximum_prompt_tokens = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            prompts = [render_chat(tokenizer, model_name, row["raw_prompt"]) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
            prompt_width = int(encoded["input_ids"].shape[1])
            maximum_prompt_tokens = max(maximum_prompt_tokens, prompt_width)
            if prompt_width > 512:
                raise RuntimeError(f"Phase569 prompt exceeded 512 tokens: {prompt_width}")
            actual_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
            for source, actual in zip(batch, actual_lengths):
                if actual != source["registered_prompt_token_count"]:
                    raise RuntimeError(
                        f"Phase569 tokenizer drift: {source['case_id']} "
                        f"registered={source['registered_prompt_token_count']} actual={actual}"
                    )
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
            result_rows = []
            for index, source in enumerate(batch):
                text = tokenizer.decode(
                    generated[index, prompt_width:], skip_special_tokens=True
                )
                result_rows.append({
                    **source,
                    **classify(source, text),
                    "torch_dtype": run_dtype,
                    "quantized_8bit": quantized_8bit,
                })
            append_jsonl(output, result_rows)
            generated_count += len(result_rows)
            del generated, encoded, result_rows, prompts
            done = len(completed) + generated_count
            if start == 0 or done == registered_count or (start // batch_size) % 25 == 24:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} Phase569 "
                    f"{done}/{registered_count}",
                    flush=True,
                )
        final_count, unique_count = completed_audit(output)
        if final_count != registered_count or unique_count != registered_count:
            raise RuntimeError(
                f"Incomplete Phase569 output: rows={final_count}, unique={unique_count}"
            )
        payload = {
            "schema_version": "phase569_behavior_execution_summary.v1",
            "phase_id": PHASE_ID,
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "registered_semantic_case_count": registered_count,
            "completed_case_count": final_count,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "resumed_case_count": len(completed),
            "newly_generated_case_count": generated_count,
            "maximum_prompt_tokens": maximum_prompt_tokens,
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "cuda_used": True,
            "torch_dtype": run_dtype,
            "quantized_8bit": quantized_8bit,
            "sealed_split_read": False,
        }
        write_json(summary_path(model_name), payload)
        print(summary_path(model_name), flush=True)
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
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.batch_size, args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
