#!/usr/bin/env python3
"""Aggregate the pre-registered block reconstruction audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/gpt5/result/phase357_residual_reconstruction_audit/pre_registered_anchor_reconstruction"
DESTINATIONS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)
PHASE = "Phase357"
SCHEMA_VERSION = "33.0.0"
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
    rows = [row for model in MODELS for row in read_jsonl(BASE / "models" / model / "phase357_reconstruction_rows.jsonl")]
    model_rows = []
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        increment_errors = [row["relative_increment_reconstruction_error"] for row in selected]
        output_errors = [row["relative_output_reconstruction_error"] for row in selected]
        model_rows.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "row_count": len(selected),
            "mean_relative_increment_error": round(mean(increment_errors), 9),
            "max_relative_increment_error": round(max(increment_errors), 9),
            "mean_relative_output_error": round(mean(output_errors), 9),
            "max_relative_output_error": round(max(output_errors), 9),
            "increment_gate_pass_rate": round(mean(row["increment_gate_pass"] for row in selected), 7),
            "output_gate_pass_rate": round(mean(row["output_gate_pass"] for row in selected), 7),
            "reconstruction_gate_pass": all(row["increment_gate_pass"] and row["output_gate_pass"] for row in selected),
        })
    reconstruction_valid = all(row["reconstruction_gate_pass"] for row in model_rows)
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "model_count": len(MODELS), "anchor_case_count": sum(row["anchor_case_count"] for row in completions),
            "reconstruction_row_count": len(rows),
            "shape_mismatch_count": sum(not row["shape_match"] for row in rows),
            "nonfinite_row_count": sum(not row["finite"] for row in rows),
            "all_model_completions_valid": all(row["valid"] for row in completions),
        },
        "results": {
            "model_reconstruction_gate_count": sum(row["reconstruction_gate_pass"] for row in model_rows),
            "cross_model_reconstruction_valid": reconstruction_valid,
            "target_direction_used": False, "semantic_label_used_for_selection": False,
            "physical_heldout_revealed": False, "causal_intervention_executed": False,
            "single_unit_causal_count": 0,
        },
        "next_decision": "design_full_vector_coarse_trace" if reconstruction_valid else "repair_hook_decomposition_before_full_trace",
        "claim_boundary": {
            "block_reconstruction_validates_full_trace_schema": False,
            "block_reconstruction_validates_attention_edges": False,
            "block_reconstruction_is_causal": False,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(BASE / "phase357_model_reconstruction_summary.jsonl", model_rows)
    write_json(BASE / "phase357_global_summary.json", summary)
    report = [
        "# Phase357 Residual Reconstruction Audit", "",
        f"- Anchors/rows: {summary['denominator']['anchor_case_count']}/{len(rows)}",
        f"- Cross-model reconstruction valid: {reconstruction_valid}",
        f"- Passing models: {summary['results']['model_reconstruction_gate_count']}/3", "",
        "This validates only the block-update decomposition at the captured hook points.",
    ]
    (BASE / "phase357_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for destination in DESTINATIONS:
        write_jsonl(destination / "phase357_model_reconstruction_summary.jsonl", model_rows)
        write_json(destination / "phase357_reconstruction_summary.json", summary)
        manifest_path = destination / "manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"schema_version": "pattern_family_atlas.v2"}
        manifest["updated_at"] = now()
        manifest["phase357"] = {
            "status": "block_reconstruction_valid",
            "anchor_cases": summary["denominator"]["anchor_case_count"],
            "reconstruction_rows": summary["denominator"]["reconstruction_row_count"],
            "passing_models": summary["results"]["model_reconstruction_gate_count"],
            "cross_model_reconstruction_valid": summary["results"]["cross_model_reconstruction_valid"],
            "full_trace_schema_complete": False,
            "physical_heldout_revealed": False,
            "single_unit_causal_count": 0,
            "files": [
                "phase357_model_reconstruction_summary.jsonl",
                "phase357_reconstruction_summary.json",
            ],
        }
        write_json(manifest_path, manifest)
    return summary


if __name__ == "__main__":
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
