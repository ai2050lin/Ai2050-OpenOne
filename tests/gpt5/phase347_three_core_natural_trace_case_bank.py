#!/usr/bin/env python3
"""Freeze cross-model-qualified cases for natural physical trajectory mapping."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase347"
SCHEMA_VERSION = "23.0.0"
ROUND_DEFAULT = "three_core_natural_physical_trace"
OUT = ROOT / "tests/gpt5/result/phase347_three_core_natural_trace"
PHASE345 = ROOT / "tests/gpt5/result/phase345_three_core_protocol/three_core_protocol_qualification"
PHASE346 = ROOT / "tests/gpt5/result/phase346_protocol_repair/three_core_protocol_repair"
ITEMS = (0, 6, 12, 16, 17, 20, 21, 23)
PHASE345_TASKS = (
    "context_relation_binding", "parameter_knowledge_retrieval", "explicit_copy_control",
    "two_hop_entailment", "direct_fact_control",
    "sentence_past_tense", "no_morphology_control", "answer_only_protocol",
)
PHASE346_TASKS = ("contiguous_multi_token_answer", "simple_no_source_answer")


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


def register(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    rows = []
    sources = (
        (PHASE345 / "phase345_registered_cases.jsonl", PHASE345_TASKS, "phase345"),
        (PHASE346 / "phase346_registered_cases.jsonl", PHASE346_TASKS, "phase346"),
    )
    for path, tasks, source_phase in sources:
        for row in read_jsonl(path):
            if row["mechanism_id"] not in tasks or row["item_index"] not in ITEMS:
                continue
            rows.append({
                **row, "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "created_at": now(), "source_phase": source_phase,
                "source_case_id": row["case_id"],
                "case_id": row["case_id"].replace(f"{source_phase}_", "phase347_", 1),
                "natural_trace_only": True, "internal_intervention_allowed": False,
                "single_unit_intervention_allowed": False,
            })
    if len(rows) != 720 or len({row["case_id"] for row in rows}) != 720:
        raise RuntimeError(f"Invalid Phase347 denominator: {len(rows)}")
    protocol = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "purpose": "Map natural layer/component/position trajectories for ten cross-model-qualified tasks.",
        "registered_case_count": len(rows), "case_count_per_model": 240,
        "selected_item_indices": list(ITEMS),
        "selected_tasks": list(PHASE345_TASKS + PHASE346_TASKS),
        "position_roles": ["source", "query", "answer_start"],
        "components": ["attention_output", "mlp_output", "residual_increment"],
        "metrics": ["component_l2_norm", "target_first_token_projection"],
        "execution_mode": "b1_left_cache0",
        "claim_boundaries": [
            "Phase347 records natural trajectories only; no component or neuron is selected.",
            "Unembedding projection is a descriptive readout, not a causal contribution.",
            "Physical trace coverage does not count as mechanism closure.",
        ],
    }
    validation = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "registered_case_count": len(rows), "task_count": 10,
        "model_case_count": {
            model: sum(row["model"] == model for row in rows)
            for model in ("qwen3", "glm4", "deepseek7b")
        },
        "split_case_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("discovery", "calibration", "heldout", "private_heldout")
        },
        "valid": True,
    }
    root = OUT / round_name
    write_jsonl(root / "phase347_registered_cases.jsonl", rows)
    write_json(root / "phase347_registered_protocol.json", protocol)
    write_json(root / "phase347_case_bank_validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(register(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
