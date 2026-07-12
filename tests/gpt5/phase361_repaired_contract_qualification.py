#!/usr/bin/env python3
"""Run repaired Phase361 contracts sequentially with true batch-one execution."""

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
from phase361_repaired_contract_case_bank import (  # noqa: E402
    MODELS, OUT, PHASE, ROUND_DEFAULT, SCHEMA_VERSION,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def finite_score(score: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in score.values())


def base(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "model": case["model"], "case_id": case["case_id"],
        "contract_group_id": case["contract_group_id"],
        "family_id": case["family_id"], "mechanism_id": case["mechanism_id"],
        "manipulated_variable": case["manipulated_variable"],
        "contrast_condition": case["contrast_condition"],
        "operation_demanded": case["operation_demanded"],
        "split": case["split"], "template_id": case["template_id"],
        "target": case["target"], "execution_mode": "b1_left_cache0",
        "actual_model_batch_size": 1, "internal_intervention": False,
    }


@torch.inference_mode()
def run_model(model: str, round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    contracts = read_jsonl(root / "phase361_repaired_contract_registry.jsonl")
    if len(contracts) != 7 or not all(row["strict_contract_gate_pass"] for row in contracts):
        raise RuntimeError("All seven repaired contracts must pass before model execution")
    cases = [
        row for row in read_jsonl(root / "phase361_registered_cases.jsonl")
        if row["model"] == model
    ]
    if len(cases) != 672:
        raise RuntimeError(f"Invalid model denominator for {model}: {len(cases)}")
    phrase_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
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
            semantic = target_match(generated["answer_head_text"].strip(), case["target_aliases"])
            phrase_rows.append({
                **base(case),
                **{key: round(float(value), 7) if math.isfinite(float(value)) else None for key, value in score.items()},
                "score_valid": valid,
                "target_wins": bool(valid and score["phrase_margin"] > 0),
            })
            rollout_rows.append({
                **base(case), **generated,
                "semantic_correct": bool(semantic),
                "protocol_correct": bool(semantic),
                "contract_outcome_correct": bool(semantic),
            })
            if index % 48 == 0 or index == len(cases):
                print(f"[{model}] {index}/{len(cases)}", flush=True)
        model_root = root / "models" / model
        write_jsonl(model_root / "phase361_phrase_rows.jsonl", phrase_rows)
        write_jsonl(model_root / "phase361_rollout_rows.jsonl", rollout_rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "accepted_contract_count": len(contracts),
            "registered_case_count": len(cases), "phrase_row_count": len(phrase_rows),
            "rollout_row_count": len(rollout_rows),
            "invalid_phrase_row_count": sum(not row["score_valid"] for row in phrase_rows),
            "semantic_correct_count": sum(row["semantic_correct"] for row in rollout_rows),
            "actual_model_batch_size": 1,
            "valid": (
                len(contracts) == 7 and len(cases) == 672
                and len(phrase_rows) == 672 and len(rollout_rows) == 672
                and all(row["score_valid"] for row in phrase_rows)
            ),
        }
        write_json(model_root / "complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(run_model(args.model, args.round), ensure_ascii=False, indent=2))
