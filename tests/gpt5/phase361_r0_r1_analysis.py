#!/usr/bin/env python3
"""Aggregate R0/R1 quality without opening semantic labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace"
ROUND = "four_admitted_balanced_trace"
MODELS = ("qwen3", "glm4", "deepseek7b")
BANNED = {"model", "family_id", "mechanism_id", "target", "case_id", "source_case_id"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def main() -> None:
    root = OUT / ROUND
    case_summary = read_json(root / "phase361_r0_r1_case_summary.json")
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    rows = [
        row for model in MODELS
        for row in read_jsonl(root / "models" / model / "phase361_r0_r1_ledger_rows.jsonl")
    ]
    blind_key_leaks = sum(bool(BANNED & set(row)) for row in rows)
    gate_names = ("attention", "mlp", "block", "input_norm", "post_attention_norm", "probability")
    max_errors = {
        name: max(float(row["errors"][name]) for row in rows)
        for name in gate_names
    }
    role_exact_counts = {
        role: sum(row["role_position_exact"][role] for row in rows)
        for role in rows[0]["role_names"]
    }
    summary = {
        "schema_version": "38.0.0", "phase_id": "Phase361", "created_at": now(),
        "denominator": {
            **case_summary["denominator"],
            "ledger_row_count": len(rows),
            "sealed_byte_count": sum(row["sealed_byte_count"] for row in completions),
            "model_layer_counts": {row["model"]: row["layer_count"] for row in completions},
        },
        "quality": {
            "all_model_completions_valid": all(row["valid"] for row in completions),
            "all_component_gates_pass": all(row["all_component_gates_pass"] for row in completions),
            "blind_ledger_label_leak_row_count": blind_key_leaks,
            "max_errors": max_errors,
            "role_exact_counts": role_exact_counts,
            "r1_shard_counts": case_summary["denominator"]["r1_shard_counts"],
        },
        "recorded_views": {
            "r0_role_state_count": 9,
            "role_count": 4,
            "all_attention_head_scalar_ledgers": True,
            "attention_source_probability_rows": True,
            "all_mlp_shard_scalar_ledgers": True,
            "one_balanced_raw_mlp_shard_per_case": True,
            "full_vocabulary_logits_per_case": True,
            "generation_time_count": 1,
        },
        "claim_boundary": {
            "r0_r1_recording_complete_for_admitted_sample": True,
            "all_eighteen_mechanisms_recorded": False,
            "multi_step_generation_recorded": False,
            "public_backbone_separated": False,
            "blind_operation_specific_motif_discovered": False,
            "future_prediction_tested": False,
            "physical_heldout_revealed": False,
            "causal_intervention_executed": False,
        },
        "next_decision": "build_blind_public_backbone_and_test_next_layer_prediction",
    }
    write_json(root / "phase361_r0_r1_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
