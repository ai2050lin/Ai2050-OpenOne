#!/usr/bin/env python3
"""Validate Phase358 component ledgers on blind discovery/calibration samples."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase358_multiresolution_full_trace/format_development_component_conservation"
PHASE = "Phase358"
SCHEMA_VERSION = "34.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
STAGES = ("blind_discovery", "blind_calibration")


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


def blind_model_id(model: str) -> str:
    return "physical_system_" + hashlib.sha256(f"phase358:{model}".encode()).hexdigest()[:12]


def aggregate() -> dict[str, Any]:
    completions, layer_rows = [], []
    for stage in STAGES:
        for model in MODELS:
            root = BASE / "stages" / stage / "models" / model
            completions.append(read_json(root / "complete.json"))
            layer_rows.extend(read_jsonl(root / "phase358_layer_rows.jsonl"))
    blind_rows = []
    layer_counts = {model: next(row["layer_count"] for row in completions if row["model"] == model) for model in MODELS}
    for row in layer_rows:
        stage = next(item["stage"] for item in completions if item["model"] == row["model"] and row["anchor_id"].startswith(item["stage"]))
        blind_rows.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
            "blind_case_id": row["anchor_id"],
            "blind_model_id": blind_model_id(row["model"]),
            "blind_split": stage,
            "relative_layer": round(row["layer_index"] / max(layer_counts[row["model"]] - 1, 1), 7),
            "execution_dtype": row["execution_dtype"],
            "input_normalized_state_norm": row["input_normalized_state_norm"],
            "post_attention_normalized_state_norm": row["post_attention_normalized_state_norm"],
            "attention_relative_reconstruction_error": row["attention_relative_reconstruction_error"],
            "mlp_relative_reconstruction_error": row["mlp_relative_reconstruction_error"],
            "block_relative_reconstruction_error": row["block_relative_reconstruction_error"],
            "attention_head_count": row["attention_head_count"],
            "mlp_channel_count": row["mlp_channel_count"],
            "mlp_shard_count": row["mlp_shard_count"],
            "semantic_label_used": False,
        })
    write_jsonl(BASE / "phase358_blind_multiview_ledger_rows.jsonl", blind_rows)
    gates = {
        "block": all(row["block_gate_pass"] for row in completions),
        "attention": all(row["attention_gate_pass"] for row in completions),
        "attention_probability": all(row["attention_probability_gate_pass"] for row in completions),
        "mlp": all(row["mlp_gate_pass"] for row in completions),
        "normalization": all(row["normalization_gate_pass"] for row in completions),
    }
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "model_count": 3, "blind_discovery_case_count": sum(row["format_case_count"] for row in completions if row["stage"] == "blind_discovery"),
            "blind_calibration_case_count": sum(row["format_case_count"] for row in completions if row["stage"] == "blind_calibration"),
            "layer_row_count": len(layer_rows), "blind_multiview_row_count": len(blind_rows),
        },
        "results": {
            "expanded_ledger_gate_pass": all(gates.values()),
            "gate_results": gates, "semantic_label_used": False,
            "physical_heldout_revealed": False, "causal_intervention_executed": False,
            "single_unit_causal_count": 0,
        },
        "claim_boundary": {
            "expanded_ledger_is_blind_motif_discovery": False,
            "expanded_ledger_is_full_phase358": False,
            "full_vector_anchors_persisted": False,
            "physical_heldout_tested": False,
        },
        "next_decision": "freeze_storage_budget_and_full_vector_anchor_format" if all(gates.values()) else "repair_expanded_component_ledger",
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_json(BASE / "phase358_expanded_ledger_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
