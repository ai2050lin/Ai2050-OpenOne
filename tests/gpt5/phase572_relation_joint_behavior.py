#!/usr/bin/env python3
"""Run Phase572 fresh Qwen3 behavior and freeze exact matched donor pairs."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_relation_competition_behavior import classify  # noqa: E402
import phase572_relation_joint_protocol as protocol  # noqa: E402


OUT_DIR = protocol.OUT_DIR
ROWS_PATH = OUT_DIR / "phase572_qwen3_behavior_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase572_qwen3_behavior_summary.json"
REGISTRY_PATH = OUT_DIR / "phase572_joint_donor_registry.json"
CONTRACT_PATH = OUT_DIR / "phase572_qwen3_behavior_contract.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["source_factorial_cell"], row["target"], row["other_relation_target"]


def matched_pairs(
    correct: list[dict[str, Any]], confusion: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    banks: dict[str, dict[tuple[str, str, str], deque[dict[str, Any]]]] = {
        "correct": defaultdict(deque),
        "confusion": defaultdict(deque),
    }
    for row in sorted(correct, key=lambda item: item["case_id"]):
        banks["correct"][stratum(row)].append(row)
    for row in sorted(confusion, key=lambda item: item["case_id"]):
        banks["confusion"][stratum(row)].append(row)
    keys = sorted(set(banks["correct"]) & set(banks["confusion"]))
    pairs = []
    while keys:
        remaining = []
        for key in keys:
            left = banks["correct"][key]
            right = banks["confusion"][key]
            if left and right:
                pairs.append((left.popleft(), right.popleft()))
            if left and right:
                remaining.append(key)
        keys = remaining
    return pairs


def generate_batch(
    loaded: Any,
    rows: list[dict[str, Any]],
    repeat: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts = [render_chat(loaded.tokenizer, protocol.MODEL, row["raw_prompt"]) for row in rows]
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    prompt_width = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    output = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(
            generated[index, prompt_width:], skip_special_tokens=True
        )
        output.append(
            {
                **row,
                **classify(row, text),
                "execution_repeat": repeat,
                "observer_only": True,
                "causal": False,
            }
        )
    del encoded, generated
    return output


def run(max_new_tokens: int, restart: bool) -> Path:
    frozen = read_json(protocol.PROTOCOL_PATH)
    audit = read_json(protocol.AUDIT_PATH)
    if not audit["valid"] or audit["cases_sha256"] != sha256_file(protocol.CASES_PATH):
        raise RuntimeError("Phase572 frozen cases drift")
    if restart:
        for path in (ROWS_PATH, SUMMARY_PATH, REGISTRY_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    contract = {
        "schema_version": "phase572_behavior_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": protocol.MODEL,
        "cases_sha256": sha256_file(protocol.CASES_PATH),
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "fixed_batch_size": frozen["fixed_batch_size"],
        "noop_repeats": frozen["noop_repeats"],
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "sealed_split_read": False,
    }
    if CONTRACT_PATH.exists():
        existing = read_json(CONTRACT_PATH)
        for key in (
            "model", "cases_sha256", "protocol_sha256", "fixed_batch_size",
            "noop_repeats", "do_sample", "torch_dtype_requested", "sealed_split_read",
        ):
            if existing[key] != contract[key]:
                raise RuntimeError(f"Phase572 behavior contract drift: {key}")
    else:
        write_json(CONTRACT_PATH, contract)

    cases = sorted(iter_jsonl(protocol.CASES_PATH), key=lambda row: row["case_id"])
    if len(cases) != frozen["candidate_case_count"] or any(row["sealed"] for row in cases):
        raise RuntimeError("Phase572 behavior denominator drift")
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(protocol.MODEL)
        loaded.tokenizer.padding_side = "left"
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase572 requires BF16, got {run_dtype}")
        output_rows = []
        for repeat in ("noop1", "noop2"):
            for start in range(0, len(cases), frozen["fixed_batch_size"]):
                batch = cases[start:start + frozen["fixed_batch_size"]]
                output_rows.extend(generate_batch(loaded, batch, repeat, max_new_tokens))
                done = min(start + frozen["fixed_batch_size"], len(cases))
                if start == 0 or done == len(cases) or start // 8 % 32 == 31:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] qwen3 Phase572 {repeat} "
                        f"{done}/{len(cases)}",
                        flush=True,
                    )
        write_jsonl(ROWS_PATH, output_rows)
        by_case_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output_rows
        }
        stable_correct = []
        stable_confusion = []
        exact_mismatch = 0
        semantic_mismatch = 0
        for case in cases:
            first = by_case_repeat[(case["case_id"], "noop1")]
            second = by_case_repeat[(case["case_id"], "noop2")]
            exact_mismatch += int(first["normalized_generated"] != second["normalized_generated"])
            semantic_mismatch += int(first["semantic_event"] != second["semantic_event"])
            if first["semantic_event"] != second["semantic_event"]:
                continue
            if first["semantic_correct"] and second["semantic_correct"]:
                stable_correct.append(first)
            if first["relation_confusion"] and second["relation_confusion"]:
                stable_confusion.append(first)
        pairs = matched_pairs(stable_correct, stable_confusion)
        if len(pairs) < frozen["minimum_candidate_pairs"]:
            raise RuntimeError(f"Phase572 has only {len(pairs)} exact matched pairs")

        correct_rows = [left for left, _right in pairs]
        correct_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in correct_rows:
            correct_by_stratum[stratum(row)].append(row)
        ordered_correct = sorted(correct_rows, key=lambda row: row["case_id"])

        def matched_donor(receiver: dict[str, Any], paired_correct: dict[str, Any]) -> dict[str, Any]:
            if receiver["case_id"] != paired_correct["case_id"]:
                return paired_correct
            options = correct_by_stratum[stratum(receiver)]
            if len(options) == 1:
                return options[0]
            position = next(i for i, row in enumerate(options) if row["case_id"] == receiver["case_id"])
            return options[(position + 1) % len(options)]

        def wrong_donor(receiver: dict[str, Any]) -> dict[str, Any]:
            preferred = [
                row for row in ordered_correct
                if row["target"] == receiver["other_relation_target"]
                and row["target"] != receiver["target"]
            ]
            fallback = [row for row in ordered_correct if row["target"] != receiver["target"]]
            options = preferred or fallback
            offset = int(hashlib.sha256(receiver["case_id"].encode()).hexdigest()[:8], 16)
            return options[offset % len(options)]

        entries = []
        for pair_index, (correct, confusion) in enumerate(pairs):
            for receiver in (correct, confusion):
                donor = matched_donor(receiver, correct)
                wrong = wrong_donor(receiver)
                entries.append(
                    {
                        "pair_index": pair_index,
                        "receiver_case_id": receiver["case_id"],
                        "receiver_phenotype": (
                            "stable_correct"
                            if receiver["case_id"] == correct["case_id"]
                            else "stable_relation_confusion"
                        ),
                        "matched_correct_donor_case_id": donor["case_id"],
                        "wrong_target_donor_case_id": wrong["case_id"],
                        "receiver_target": receiver["target"],
                        "receiver_other_relation_target": receiver["other_relation_target"],
                        "matched_donor_target": donor["target"],
                        "wrong_donor_target": wrong["target"],
                        "matched_stratum_equal": stratum(receiver) == stratum(donor),
                        "wrong_target_differs": wrong["target"] != receiver["target"],
                    }
                )
        if any(
            not row["matched_stratum_equal"] or not row["wrong_target_differs"]
            for row in entries
        ):
            raise RuntimeError("Phase572 donor assignment identity failed")
        registry = {
            "schema_version": "phase572_joint_donor_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": protocol.MODEL,
            "candidate_pair_count": len(pairs),
            "final_pair_count": frozen["final_pair_count"],
            "entries": entries,
            "selection_uses_intervention_outcomes": False,
            "sealed_split_read": False,
        }
        write_json(REGISTRY_PATH, registry)
        summary = {
            "schema_version": "phase572_behavior_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": protocol.MODEL,
            "torch_dtype": run_dtype,
            "case_count": len(cases),
            "behavior_row_count": len(output_rows),
            "stable_correct_count": len(stable_correct),
            "stable_relation_confusion_count": len(stable_confusion),
            "exact_matched_pair_count": len(pairs),
            "matched_stratum_count": len({stratum(left) for left, _right in pairs}),
            "matched_target_other_pair_count": len({
                (left["target"], left["other_relation_target"]) for left, _right in pairs
            }),
            "noop_exact_text_mismatch_count": exact_mismatch,
            "noop_semantic_event_mismatch_count": semantic_mismatch,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "qualified_for_joint_causal": len(pairs) >= frozen["minimum_candidate_pairs"],
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return SUMMARY_PATH
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
