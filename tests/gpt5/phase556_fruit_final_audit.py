#!/usr/bin/env python3
"""Build the evidence-preserving final Phase556 audit without reading sealed rows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
OUTPUT = OUT_DIR / "phase556_final_audit.json"
EXPECTED_READOUT = "first_non_whitespace_candidate_content_token_v2"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def boundary_result_dir(model: str) -> Path:
    path = OUT_DIR / "layer_input_boundary"
    return path if model == "qwen3" else path / model


def parent_result_dir(model: str) -> Path:
    path = OUT_DIR / "direct_parent_decomposition"
    return path if model == "qwen3" else path / model


def build() -> dict[str, Any]:
    protocol = read_json(OUT_DIR / "phase556_frozen_protocol.json")
    static = read_json(OUT_DIR / "phase556_static_audit.json")
    behavior = read_json(OUT_DIR / "phase556_behavior_summary.json")
    events = [
        read_json(OUT_DIR / "event_collection" / model / split / "phase556_event_summary.json")
        for model in ("qwen3", "glm4")
        for split in ("discovery", "independent_confirmation")
    ]
    causal = read_json(OUT_DIR / "phase556_causal_analysis_summary.json")
    boundaries = {
        "qwen3": read_json(OUT_DIR / "phase556_layer_input_boundary_analysis.json"),
        "glm4": read_json(OUT_DIR / "phase556_glm4_layer_input_boundary_analysis.json"),
    }
    parents = {
        "qwen3": read_json(OUT_DIR / "phase556_direct_parent_analysis.json"),
        "glm4": read_json(OUT_DIR / "phase556_glm4_direct_parent_analysis.json"),
    }
    publish = read_json(OUT_DIR / "phase556_atlas_publish_summary.json")

    checks = {
        "static_protocol_valid": bool(static["valid"]),
        "registered_denominator_17424": static["registered_case_count"] == 17424,
        "open_executed_denominator_11616": behavior["open_case_count"] == 11616,
        "sealed_denominator_5808": protocol["sealed_case_count"] == 5808,
        "sealed_never_read": not any([
            behavior["sealed_split_read"],
            causal["sealed_split_read"],
            *(row["sealed_split_read"] for row in boundaries.values()),
            publish["sealed_split_read"],
        ]),
        "event_ledgers_complete": all(row["status"] == "complete" for row in events),
        "causal_readout_contract_v2": causal["restricted_readout_contract"] == EXPECTED_READOUT,
        "boundary_readout_contract_v2": all(
            row["restricted_readout_contract"] == EXPECTED_READOUT
            for row in boundaries.values()
        ),
        "parent_readout_contract_v2": all(
            row["restricted_readout_contract"] == EXPECTED_READOUT
            for row in parents.values()
        ),
        "all_internal_runs_bfloat16": all(
            row.get("torch_dtype") == "torch.bfloat16" for row in events
        ) and all(
            row.get("torch_dtypes") == ["torch.bfloat16"]
            for row in (*boundaries.values(), *parents.values())
        ),
        "parameter_scan_stopped": not causal["parameter_localization_authorized"],
        "closure_not_claimed": not causal["closure_claim_authorized"],
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"Phase556 final audit failed: {failed}")

    behavior_by_model = {row["model"]: row for row in behavior["model_reports"]}
    qualified = [row for row in causal["candidate_reports"] if row["causal_qualified"]]
    boundary_summary = {
        model: {
            mechanism: {
                "earliest_replicated_layer": report["earliest_replicated_layer"],
                "earliest_replicated_relative_depth": report["earliest_replicated_relative_depth"],
                "replicated_pass_layers": report["replicated_pass_layers"],
            }
            for mechanism, report in boundary["mechanism_reports"].items()
        }
        for model, boundary in boundaries.items()
    }
    parent_summary = {
        model: {
            mechanism: {
                "parent_layer": report["parent_layer"],
                "qualified_conditions": report["qualified_conditions"],
                "qualified_writer_conditions": report["qualified_writer_conditions"],
                "parameter_localization_authorized": report["parameter_localization_authorized"],
            }
            for mechanism, report in parent["mechanism_reports"].items()
        }
        for model, parent in parents.items()
    }
    payload = {
        "schema_version": "phase556_fruit_final_audit.v1",
        "phase_id": "Phase556",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_open_evidence_audit",
        "checks": checks,
        "denominators": {
            "registered_cases": static["registered_case_count"],
            "open_cases_executed": behavior["open_case_count"],
            "sealed_cases_unread": protocol["sealed_case_count"],
            "event_rows": sum(row["event_row_count"] for row in events),
            "causal_intervention_rows": sum(
                read_json(
                    OUT_DIR / "causal_intervention" / model / "phase556_causal_execution_summary.json"
                )["row_count"]
                for model in ("qwen3", "glm4")
            ),
            "layer_input_boundary_rows": {
                model: sum(
                    read_json(
                        boundary_result_dir(model)
                        / split / "phase556_boundary_execution_summary.json"
                    )["row_count"]
                    for split in ("boundary_discovery", "boundary_confirmation")
                )
                for model in boundaries
            },
            "direct_parent_rows": {
                model: read_json(
                    parent_result_dir(model)
                    / "phase556_direct_parent_execution_summary.json"
                )["row_count"]
                for model in parents
            },
        },
        "behavior": {
            model: {
                "semantic_accuracy": report["semantic_accuracy"],
                "strict_sequence_accuracy": report["strict_sequence_accuracy"],
                "controlled_factorial_gate_pass": report["controlled_factorial_gate_pass"],
                "authorized_natural_relations": report["authorized_natural_relations"],
            }
            for model, report in behavior_by_model.items()
        },
        "events": [
            {
                key: row.get(key)
                for key in (
                    "model", "split", "torch_dtype", "anchor_count", "layer_count",
                    "component_count", "event_row_count", "max_component_ledger_relative_error",
                )
            }
            for row in events
        ],
        "qualified_causal_state_or_component_candidates": [
            {
                key: row[key]
                for key in (
                    "candidate_id", "model", "mechanism", "component", "layer",
                    "relative_depth", "causal_state_carrier", "causal_component_update",
                )
            }
            for row in qualified
        ],
        "replicated_mechanisms": causal["replicated_mechanisms"],
        "layer_input_boundaries": boundary_summary,
        "direct_parents": parent_summary,
        "parent_intervention_semantics": {
            model: parent["parent_intervention_semantics"]
            for model, parent in parents.items()
        },
        "evidence_boundary": {
            "controlled_context_category_or_binding_state": bool(qualified),
            "natural_fruit_parameter_storage_recovered": False,
            "local_writer_recovered": any(
                report["qualified_writer_conditions"]
                for parent in parents.values()
                for report in parent["mechanism_reports"].values()
            ),
            "parameter_support_recovered": False,
            "sealed_replication": False,
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
            "global_physical_atlas_management_estimate_percent": 32,
            "scientific_maturity_management_estimate_percent": 29,
        },
        "hard_limitations": [
            "The controlled factorial tests contextual lookup, not natural fruit parameter storage.",
            "The frozen factors are entity, category, query, and binding; attribute content is not an independent factor.",
            "Matched bidirectional state replacement establishes scoped causal control, not a complete necessary writer circuit.",
            "Parent-component deltas are attributed at the child state; they do not establish source-module necessity or a direct compute edge.",
            "No parameter, head, channel, neuron, or sealed scan is authorized.",
            "Small-model external-validity risk remains approximately 30%-50% and does not relax evidence gates.",
        ],
    }
    write_json(OUTPUT, payload)
    print(OUTPUT)
    return payload


if __name__ == "__main__":
    build()
