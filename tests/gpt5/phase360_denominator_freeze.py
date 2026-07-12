#!/usr/bin/env python3
"""Freeze the nine-family denominator and blind-discovery admission states."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P353 = ROOT / "tests/gpt5/result/phase353_family_contracts/family_specific_contract_compiler"
P358 = ROOT / "tests/gpt5/result/phase358_multiresolution_full_trace/format_development_component_conservation"
P359 = ROOT / "tests/gpt5/result/phase359_full_vector_anchor"
OUT = ROOT / "tests/gpt5/result/phase360_denominator_freeze"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
FRONTEND = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def update_manifest(
    path: Path,
    phase358: dict[str, Any],
    phase359: dict[str, Any],
    summary: dict[str, Any],
    updated_at: str,
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["updated_at"] = updated_at
    manifest["phase358"].update({
        "status": "format_and_expanded_component_ledger_pass",
        "expanded_cases": (
            phase358["denominator"]["blind_discovery_case_count"]
            + phase358["denominator"]["blind_calibration_case_count"]
        ),
        "expanded_layer_rows": phase358["denominator"]["layer_row_count"],
        "expanded_ledger_gate_pass": phase358["results"]["expanded_ledger_gate_pass"],
        "full_vector_anchors_persisted": True,
        "blind_motif_discovery_complete": False,
    })
    manifest["phase358"]["files"] = list(dict.fromkeys([
        *manifest["phase358"]["files"],
        "phase358_expanded_ledger_summary.json",
    ]))
    manifest["phase359"] = {
        "status": "sealed_full_vector_anchor_replay_pass",
        "model_count": phase359["model_count"],
        "anchor_count": phase359["anchor_count"],
        "layer_file_count": phase359["layer_file_count"],
        "sealed_tensor_byte_count": phase359["total_byte_count"],
        "all_models_replay_pass": phase359["all_models_replay_pass"],
        "raw_tensors_frontend_exported": False,
        "physical_heldout_revealed": False,
        "single_unit_causal_count": 0,
        "files": ["phase359_full_vector_anchor_summary.json"],
    }
    manifest["phase360"] = {
        "status": "nine_family_denominator_frozen_not_ready_for_full_discovery",
        **summary["denominator"],
        "single_global_progress_percentage_valid": False,
        "physical_heldout_revealed": False,
        "single_unit_causal_count": 0,
        "files": ["phase360_denominator_summary.json"],
    }
    write_json(path, manifest)


def main() -> None:
    contracts = read_jsonl(P353 / "phase353_contract_registry.jsonl")
    cross_rows = {
        (row["family_id"], row["mechanism_id"]): row
        for row in read_jsonl(P353 / "phase353_cross_model_contract_summary.jsonl")
    }
    phase358 = json.loads((P358 / "phase358_expanded_ledger_summary.json").read_text(encoding="utf-8"))
    phase359 = json.loads((P359 / "phase359_replay_summary.json").read_text(encoding="utf-8"))
    mechanism_rows = []
    for contract in contracts:
        key = (contract["family_id"], contract["mechanism_id"])
        behavior = cross_rows.get(key)
        contract_pass = bool(contract["strict_contract_gate_pass"])
        trace_pass = bool(behavior and behavior.get("cross_model_trace_entry"))
        full_behavior_pass = bool(behavior and behavior.get("cross_model_full_behavior_gate_pass"))
        if not contract_pass:
            admission = "contract_repair_required"
        elif not trace_pass:
            admission = "behavior_rejected"
        else:
            admission = "blind_discovery_admitted"
        mechanism_rows.append({
            "family_id": contract["family_id"],
            "mechanism_id": contract["mechanism_id"],
            "contract_gate_pass": contract_pass,
            "contract_mapping_status": contract["mapping_status"],
            "mean_lexical_overlap": contract["mean_lexical_overlap"],
            "target_visibility_match_rate": contract["target_visibility_match_rate"],
            "behavior_measured_on_three_models": behavior is not None,
            "cross_model_trace_entry": trace_pass,
            "cross_model_full_behavior_gate_pass": full_behavior_pass,
            "blind_discovery_admission": admission,
            "physical_heldout_opened": False,
            "causal_sealed_opened": False,
        })
    family_ids = sorted({row["family_id"] for row in mechanism_rows})
    admitted = [row for row in mechanism_rows if row["blind_discovery_admission"] == "blind_discovery_admitted"]
    summary = {
        "schema_version": "36.0.0",
        "phase_id": "Phase360",
        "created_at": now(),
        "denominator": {
            "family_count": len(family_ids),
            "mechanism_count": len(mechanism_rows),
            "contract_gate_pass_count": sum(row["contract_gate_pass"] for row in mechanism_rows),
            "three_model_behavior_measured_count": sum(row["behavior_measured_on_three_models"] for row in mechanism_rows),
            "blind_discovery_admitted_count": len(admitted),
            "full_behavior_gate_pass_count": sum(row["cross_model_full_behavior_gate_pass"] for row in mechanism_rows),
            "phase358_expanded_case_count": (
                phase358["denominator"]["blind_discovery_case_count"]
                + phase358["denominator"]["blind_calibration_case_count"]
            ),
            "phase359_full_vector_anchor_count": phase359["anchor_count"],
            "phase359_replayed_layer_count": phase359["layer_file_count"],
        },
        "coverage_axes": {
            "registered_family_coverage": {"numerator": len(family_ids), "denominator": 9},
            "registered_mechanism_coverage": {"numerator": len(mechanism_rows), "denominator": 18},
            "contract_qualified_mechanism_coverage": {
                "numerator": sum(row["contract_gate_pass"] for row in mechanism_rows), "denominator": 18,
            },
            "blind_discovery_admission_coverage": {"numerator": len(admitted), "denominator": 18},
            "full_vector_format_model_coverage": {
                "numerator": phase359["anchor_count"] if phase359["all_models_replay_pass"] else 0,
                "denominator": 3,
            },
            "physical_heldout_mechanism_coverage": {"numerator": 0, "denominator": 18},
            "causal_sealed_mechanism_coverage": {"numerator": 0, "denominator": 18},
        },
        "mechanisms": mechanism_rows,
        "admitted_mechanisms": [
            {"family_id": row["family_id"], "mechanism_id": row["mechanism_id"]}
            for row in admitted
        ],
        "evidence_boundary": {
            "nine_family_denominator_frozen": True,
            "nine_family_blind_discovery_ready": len(admitted) == 18,
            "phase358_covers_all_nine_families": False,
            "full_vector_format_replayable_on_three_models": phase359["all_models_replay_pass"],
            "single_global_progress_percentage_valid": False,
            "language_encoding_closed": False,
            "intelligence_theory_closed": False,
        },
        "decision": "do_not_expand_r0_r1_to_nine_families",
        "next_large_stage": {
            "name": "repair_and_requalify_family_contract_denominator",
            "work_packages": [
                "repair seven structurally rejected contrast contracts without using model effects",
                "rerun repaired contracts sequentially on qwen3, glm4, and deepseek7b",
                "retain eight behavior-rejected contracts as negative atlas cells unless independent redesign is justified",
                "freeze admitted/rejected cells before balanced R0/R1 blind recording",
            ],
        },
    }
    write_json(OUT / "phase360_denominator_summary.json", summary)
    public = {
        key: summary[key]
        for key in ("schema_version", "phase_id", "created_at", "denominator", "coverage_axes", "admitted_mechanisms", "evidence_boundary", "decision", "next_large_stage")
    }
    manifest_updated_at = now()
    for directory in (ATLAS, FRONTEND):
        write_json(directory / "phase358_expanded_ledger_summary.json", phase358)
        write_json(directory / "phase359_full_vector_anchor_summary.json", {
            "schema_version": phase359["schema_version"],
            "phase_id": phase359["phase_id"],
            "model_count": phase359["model_count"],
            "anchor_count": phase359["anchor_count"],
            "layer_file_count": phase359["layer_file_count"],
            "total_byte_count": phase359["total_byte_count"],
            "all_models_replay_pass": phase359["all_models_replay_pass"],
            "models": phase359["models"],
            "evidence_boundary": phase359["evidence_boundary"],
        })
        write_json(directory / "phase360_denominator_summary.json", public)
        update_manifest(directory / "manifest.json", phase358, phase359, summary, manifest_updated_at)
    print(json.dumps({
        "denominator": summary["denominator"],
        "decision": summary["decision"],
        "admitted_mechanisms": summary["admitted_mechanisms"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
