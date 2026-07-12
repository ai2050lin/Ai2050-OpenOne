#!/usr/bin/env python3
"""Qualify only mechanically accepted Phase353 contracts on true batch-one execution."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase331_refined_mechanism_audit import target_match  # noqa: E402
from phase339_cross_task_boundary_audit import generate_batch  # noqa: E402
from phase348_adjusted_block_screen import score_case_b1  # noqa: E402
from phase353_family_contract_case_bank import OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def evaluate(case: dict[str, Any], generated: dict[str, Any]) -> tuple[bool, bool]:
    text = generated["answer_head_text"].strip()
    target = case["target"]
    demanded = case["operation_demanded"]
    semantic = target_match(text, case["target_aliases"])
    protocol = semantic
    if case["mechanism_id"] == "json_format" and demanded:
        try:
            payload = json.loads(text)
            semantic = str(payload.get("answer", "")).strip().lower() == target.lower()
            protocol = semantic and set(payload) == {"answer"}
        except (json.JSONDecodeError, AttributeError):
            semantic = protocol = False
    elif case["mechanism_id"] in {"target_vs_continue", "multi_token_stop", "continue_suppression"}:
        if demanded:
            protocol = semantic
        else:
            starts = text.lower().startswith(target.lower())
            semantic = starts
            protocol = starts and len(text) > len(target) + 1
    elif case["mechanism_id"] == "answer_only":
        if demanded:
            protocol = semantic
        else:
            semantic = text.lower().startswith(target.lower())
            protocol = semantic
    return bool(semantic), bool(protocol)


def base(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "model": case["model"], "case_id": case["case_id"],
        "contract_group_id": case["contract_group_id"], "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"], "manipulated_variable": case["manipulated_variable"],
        "contrast_condition": case["contrast_condition"], "operation_demanded": case["operation_demanded"],
        "split": case["split"], "template_id": case["template_id"], "target": case["target"],
        "execution_mode": "b1_left_cache0", "internal_intervention": False,
    }


def finite_score(score: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in score.values())


def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    accepted = {
        (row["family_id"], row["mechanism_id"])
        for row in read_jsonl(root / "phase353_contract_registry.jsonl")
        if row["strict_contract_gate_pass"]
    }
    cases = [
        row for row in read_jsonl(root / "phase353_registered_cases.jsonl")
        if row["model"] == model and (row["family_id"], row["mechanism_id"]) in accepted
    ]
    phrase_rows, rollout_rows = [], []
    loaded = None
    try:
        loaded = load_probe_model(model)
        for index, case in enumerate(cases, 1):
            scored_case = {
                **case,
                "target": f" {case['target']}",
                "distractors": [f" {value}" for value in case["distractors"]],
            }
            score = score_case_b1(loaded, scored_case, None, None)
            generated = generate_batch(loaded, [case], None, None, 16)[0]
            valid = finite_score(score)
            semantic, protocol = evaluate(case, generated)
            phrase_rows.append({
                **base(case), **{key: round(float(value), 7) if math.isfinite(float(value)) else None for key, value in score.items()},
                "score_valid": valid, "target_wins": bool(valid and score["phrase_margin"] > 0),
                "initial_score_valid": valid, "score_retry_count": 0,
                "actual_model_batch_size": 1,
            })
            rollout_rows.append({
                **base(case), **generated, "semantic_correct": semantic,
                "protocol_correct": protocol, "contract_outcome_correct": semantic and protocol,
                "actual_model_batch_size": 1,
            })
            if index % 96 == 0 or index == len(cases):
                print(f"[{model}] {index}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        write_jsonl(model_root / "phase353_phrase_rows.jsonl", phrase_rows)
        write_jsonl(model_root / "phase353_rollout_rows.jsonl", rollout_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "accepted_contract_count": len(accepted),
            "registered_case_count": len(cases), "phrase_row_count": len(phrase_rows),
            "rollout_row_count": len(rollout_rows),
            "invalid_phrase_row_count": sum(not row["score_valid"] for row in phrase_rows),
            "actual_model_batch_size": 1,
            "valid": (
                len(accepted) == 11 and len(cases) == 1056 and len(phrase_rows) == 1056
                and len(rollout_rows) == 1056 and all(row["score_valid"] for row in phrase_rows)
            ),
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def repair_invalid(model: str, round_name: str = ROUND_DEFAULT, max_retries: int = 3) -> dict[str, Any]:
    root = OUT / round_name
    model_root = root / "models" / model
    phrase_rows = read_jsonl(model_root / "phase353_phrase_rows.jsonl")
    invalid_ids = {row["case_id"] for row in phrase_rows if not row["score_valid"]}
    cases = {
        row["case_id"]: row for row in read_jsonl(root / "phase353_registered_cases.jsonl")
        if row["model"] == model and row["case_id"] in invalid_ids
    }
    loaded = None
    try:
        if invalid_ids:
            loaded = load_probe_model(model)
        replacements: dict[str, dict[str, Any]] = {}
        for case_id in sorted(invalid_ids):
            score: dict[str, float] = {}
            retry_count = 0
            while retry_count < max_retries:
                retry_count += 1
                case = cases[case_id]
                scored_case = {
                    **case,
                    "target": f" {case['target']}",
                    "distractors": [f" {value}" for value in case["distractors"]],
                }
                score = score_case_b1(loaded, scored_case, None, None)
                if finite_score(score):
                    break
            valid = bool(score) and finite_score(score)
            original = next(row for row in phrase_rows if row["case_id"] == case_id)
            replacements[case_id] = {
                **original,
                **{key: round(float(value), 7) if math.isfinite(float(value)) else None for key, value in score.items()},
                "score_valid": valid,
                "target_wins": bool(valid and score["phrase_margin"] > 0),
                "initial_score_valid": False,
                "score_retry_count": retry_count,
                "repaired_at": now(),
            }
        phrase_rows = [replacements.get(row["case_id"], row) for row in phrase_rows]
        write_jsonl(model_root / "phase353_phrase_rows.jsonl", phrase_rows)
        complete = read_json(model_root / "complete.json")
        complete["invalid_phrase_row_count"] = sum(not row["score_valid"] for row in phrase_rows)
        complete["initial_invalid_phrase_row_count"] = len(invalid_ids)
        complete["retried_phrase_row_count"] = len(replacements)
        complete["valid"] = (
            complete["registered_case_count"] == 1056 and complete["phrase_row_count"] == 1056
            and complete["rollout_row_count"] == 1056 and complete["invalid_phrase_row_count"] == 0
        )
        complete["updated_at"] = now()
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--repair-invalid", action="store_true")
    args = parser.parse_args()
    result = repair_invalid(args.model, args.round) if args.repair_invalid else run_model(args.model, args.round)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
