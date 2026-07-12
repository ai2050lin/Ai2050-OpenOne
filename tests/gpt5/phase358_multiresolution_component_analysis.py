#!/usr/bin/env python3
"""Aggregate Phase358 format-development conservation gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase358_multiresolution_full_trace/format_development_component_conservation"
DESTINATIONS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)
PHASE = "Phase358"
SCHEMA_VERSION = "34.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def aggregate() -> dict[str, Any]:
    completions = [read_json(BASE / "models" / model / "complete.json") for model in MODELS]
    layer_rows = [row for model in MODELS for row in read_jsonl(BASE / "models" / model / "phase358_layer_rows.jsonl")]
    head_rows = [row for model in MODELS for row in read_jsonl(BASE / "models" / model / "phase358_attention_head_rows.jsonl")]
    shard_rows = [row for model in MODELS for row in read_jsonl(BASE / "models" / model / "phase358_mlp_shard_rows.jsonl")]
    model_rows = []
    for model in MODELS:
        layers = [row for row in layer_rows if row["model"] == model]
        heads = [row for row in head_rows if row["model"] == model]
        model_rows.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "format_case_count": next(row["format_case_count"] for row in completions if row["model"] == model),
            "layer_row_count": len(layers), "attention_head_row_count": len(heads),
            "mean_attention_relative_error": round(mean(row["attention_relative_reconstruction_error"] for row in layers), 9),
            "max_attention_relative_error": round(max(row["attention_relative_reconstruction_error"] for row in layers), 9),
            "mean_mlp_relative_error": round(mean(row["mlp_relative_reconstruction_error"] for row in layers), 9),
            "max_mlp_relative_error": round(max(row["mlp_relative_reconstruction_error"] for row in layers), 9),
            "max_attention_probability_sum_error": round(max(row["max_probability_sum_error"] for row in heads), 9),
            "block_gate_pass": all(row["block_reconstruction_gate_pass"] for row in layers),
            "attention_gate_pass": all(row["attention_reconstruction_gate_pass"] for row in layers),
            "attention_probability_gate_pass": all(row["probability_normalization_gate_pass"] for row in heads),
            "mlp_gate_pass": all(row["mlp_reconstruction_gate_pass"] for row in layers),
            "normalization_gate_pass": all(row["input_normalization_finite"] for row in layers),
        })
    format_gate = all(
        row[gate]
        for row in model_rows
        for gate in ("block_gate_pass", "attention_gate_pass", "attention_probability_gate_pass", "mlp_gate_pass", "normalization_gate_pass")
    )
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "model_count": len(MODELS), "format_case_count": sum(row["format_case_count"] for row in completions),
            "layer_row_count": len(layer_rows), "attention_head_row_count": len(head_rows),
            "mlp_shard_row_count": len(shard_rows), "mlp_shard_count": 16,
            "all_model_completions_valid": all(row["valid"] for row in completions),
        },
        "results": {
            "format_development_gate_pass": format_gate,
            "block_gate_model_count": sum(row["block_gate_pass"] for row in model_rows),
            "attention_reconstruction_model_count": sum(row["attention_gate_pass"] for row in model_rows),
            "attention_probability_model_count": sum(row["attention_probability_gate_pass"] for row in model_rows),
            "mlp_reconstruction_model_count": sum(row["mlp_gate_pass"] for row in model_rows),
            "normalization_finite_model_count": sum(row["normalization_gate_pass"] for row in model_rows),
            "semantic_label_used": False, "top_k_selection_used": False,
            "all_attention_heads_recorded": True, "all_mlp_channels_partitioned": True,
            "physical_heldout_revealed": False, "causal_intervention_executed": False,
            "single_unit_causal_count": 0,
        },
        "claim_boundary": {
            "format_development_is_full_phase358": False,
            "component_conservation_is_language_mechanism": False,
            "attention_probability_is_value_flow": False,
            "mlp_shard_is_single_neuron": False,
            "full_vector_anchors_persisted": False,
        },
        "next_decision": "expand_to_blind_discovery_calibration_without_heldout" if format_gate else "repair_component_ledger",
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(BASE / "phase358_model_component_summary.jsonl", model_rows)
    write_json(BASE / "phase358_global_summary.json", summary)
    report = [
        "# Phase358 Component Conservation Format Development", "",
        f"- Cases/layers: {summary['denominator']['format_case_count']}/{len(layer_rows)}",
        f"- Head rows/MLP shard rows: {len(head_rows)}/{len(shard_rows)}",
        f"- Cross-model format gate: {format_gate}", "",
        "Physical heldout, causal intervention, and single-neuron claims remain closed.",
    ]
    (BASE / "phase358_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for destination in DESTINATIONS:
        write_jsonl(destination / "phase358_model_component_summary.jsonl", model_rows)
        write_json(destination / "phase358_component_conservation_summary.json", summary)
        manifest_path = destination / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"schema_version": "pattern_family_atlas.v2"}
        manifest["updated_at"] = now()
        manifest["phase358"] = {
            "status": "format_development_pass" if format_gate else "component_ledger_repair_required",
            "format_cases": summary["denominator"]["format_case_count"],
            "layer_rows": len(layer_rows), "attention_head_rows": len(head_rows),
            "mlp_shard_rows": len(shard_rows), "format_gate_pass": format_gate,
            "full_phase358_complete": False, "physical_heldout_revealed": False,
            "single_unit_causal_count": 0,
            "files": ["phase358_model_component_summary.jsonl", "phase358_component_conservation_summary.json"],
        }
        write_json(manifest_path, manifest)
    return summary


if __name__ == "__main__":
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
