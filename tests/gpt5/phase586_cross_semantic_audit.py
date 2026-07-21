#!/usr/bin/env python3
"""Run one local CUDA judge over all blinded Phase585 open responses."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
import phase585_object_swap_behavior as source_behavior  # noqa: E402
import phase585_object_swap_protocol as source  # noqa: E402
import phase586_cross_semantic_audit_protocol as protocol  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths(judge_model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase586_{judge_model}_cross_semantic_audit"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def judge_prompt(row: dict[str, Any]) -> str:
    relation_description = {
        "ordinary_origin": (
            "the broad ordinary process or source by which the subject comes into existence; "
            "a merely geographic, historical, categorical, or energy-source answer does not answer it"
        ),
        "primary_function": (
            "the subject's ordinary practical purpose or common real-world role; valid specific "
            "uses and valid nutritional or culinary roles count"
        ),
    }[row["relation"]]
    return (
        "You are auditing a short-answer dataset. Judge semantic correctness only. "
        "Your FIRST WORD must be exactly YES, NO, or AMBIGUOUS. Put the label before any reasoning.\n"
        f"Original question: {row['raw_prompt']}\n"
        f"Question meaning: {relation_description}.\n"
        f"Reference fact: {row['canonical_answer']}\n"
        f"Candidate response: {row['normalized_generated']}\n"
        "A valid paraphrase or a more specific compatible answer counts as YES. "
        "A subject echo without an answer, a different aspect, or a contradiction counts as NO. "
        "Use AMBIGUOUS only when the response cannot be reliably decided.\n"
        "Do not explain. Begin the response now with exactly one label: YES, NO, or AMBIGUOUS."
    )


def parse_judgment(text: str) -> str | None:
    matches = re.findall(r"(?<!\w)(YES|NO|AMBIGUOUS)(?!\w)", text.upper())
    if not matches:
        return None
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def load_source_rows() -> list[dict[str, Any]]:
    rows = []
    for source_model in source.MODELS:
        source_rows = list(iter_jsonl(source_behavior.paths(source_model)["rows"]))
        by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in source_rows:
            by_case[row["case_id"]][row["execution_repeat"]] = row
        for case_id, repeats in by_case.items():
            if set(repeats) != set(source.NOOP_REPEATS):
                raise RuntimeError(f"Phase586 malformed source repeats {source_model}:{case_id}")
            first = repeats[source.NOOP_REPEATS[0]]
            second = repeats[source.NOOP_REPEATS[1]]
            if first["normalized_generated"] != second["normalized_generated"]:
                raise RuntimeError(f"Phase586 unstable source response {source_model}:{case_id}")
            rows.append(
                {
                    **first,
                    "source_model": source_model,
                    "source_repeat_exact": True,
                }
            )
    return sorted(rows, key=lambda row: (row["case_id"], row["source_model"]))


def generate_batch(
    loaded: Any,
    judge_model: str,
    rows: list[dict[str, Any]],
    judge_repeat: str,
) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, judge_model, judge_prompt(row)) for row in rows]
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=protocol.MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    output = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(generated[index, width:], skip_special_tokens=True)
        output.append(
            {
                "schema_version": "phase586_cross_semantic_judgment.v1",
                "phase_id": protocol.PHASE,
                "created_at": now(),
                "judge_model": judge_model,
                "judge_repeat": judge_repeat,
                "source_model": row["source_model"],
                "case_id": row["case_id"],
                "split": row["split"],
                "object_id": row["object_id"],
                "semantic_group": row["semantic_group"],
                "relation": row["relation"],
                "surface_id": row["surface_id"],
                "judge_text": text,
                "judgment": parse_judgment(text),
                "source_model_identity_visible_to_judge": False,
                "split_visible_to_judge": False,
                "previous_alias_score_visible_to_judge": False,
                "sealed": False,
            }
        )
    del encoded, generated
    return output


def run(judge_model: str, restart: bool) -> Path:
    output = paths(judge_model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
    for model, artifact in frozen["source_artifacts"].items():
        if sha256_file(ROOT / artifact["rows_path"]) != artifact["rows_sha256"]:
            raise RuntimeError(f"Phase586 source drift after freeze: {model}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase586 cross semantic audit requires CUDA")
    source_rows = load_source_rows()
    if any(row["sealed"] for row in source_rows):
        raise RuntimeError("Phase586 received sealed source rows")
    write_json(
        output["contract"],
        {
            "schema_version": "phase586_cross_semantic_contract.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "judge_model": judge_model,
            "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
            "source_case_count": len(source_rows),
            "judge_repeats": list(protocol.JUDGE_REPEATS),
            "source_model_identity_visible_to_judge": False,
            "split_visible_to_judge": False,
            "sealed_split_read": False,
            "torch_dtype_requested": "torch.bfloat16",
        },
    )
    loaded = None
    output_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(judge_model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase586 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase586 requires BF16, got {dtype}")
        for judge_repeat in protocol.JUDGE_REPEATS:
            for start in range(0, len(source_rows), protocol.FIXED_BATCH_SIZE):
                output_rows.extend(
                    generate_batch(
                        loaded,
                        judge_model,
                        source_rows[start : start + protocol.FIXED_BATCH_SIZE],
                        judge_repeat,
                    )
                )
            print(
                f"[{time.strftime('%H:%M:%S')}] {judge_model} Phase586 "
                f"{judge_repeat} {len(source_rows)}/{len(source_rows)}",
                flush=True,
            )
        write_jsonl(output["rows"], output_rows)
        by_case: dict[tuple[str, str], dict[str, str | None]] = defaultdict(dict)
        for row in output_rows:
            by_case[(row["source_model"], row["case_id"])][row["judge_repeat"]] = row[
                "judgment"
            ]
        repeat_exact = sum(
            set(values) == set(protocol.JUDGE_REPEATS)
            and values[protocol.JUDGE_REPEATS[0]] is not None
            and values[protocol.JUDGE_REPEATS[0]] == values[protocol.JUDGE_REPEATS[1]]
            for values in by_case.values()
        ) / len(by_case)
        parse_rate = sum(row["judgment"] is not None for row in output_rows) / len(
            output_rows
        )
        summary = {
            "schema_version": "phase586_cross_semantic_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "judge_model": judge_model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "source_case_count": len(source_rows),
            "row_count": len(output_rows),
            "parse_rate": parse_rate,
            "repeat_exact_rate": repeat_exact,
            "judgment_counts": dict(Counter(row["judgment"] or "UNPARSEABLE" for row in output_rows)),
            "judge_quality_gate_passes": bool(
                parse_rate >= protocol.MIN_JUDGE_PARSE_RATE
                and repeat_exact >= protocol.MIN_JUDGE_REPEAT_EXACT_RATE
            ),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "sealed_split_read": False,
        }
        write_json(output["summary"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return output["summary"]
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judge_model", choices=protocol.MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.judge_model, args.restart)


if __name__ == "__main__":
    main()
