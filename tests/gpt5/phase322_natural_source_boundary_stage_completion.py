#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase322"
SCHEMA_VERSION = "4.5.0"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def complete() -> dict[str, Any]:
    p317 = read_json(V2 / "phase317_natural_source_case_bank_summary.json")
    p318 = read_json(V2 / "phase318_natural_source_state_transfer_summary.json")
    p319 = read_json(V2 / "phase319_heldout_component_mediation_summary.json")
    p320 = read_json(V2 / "phase320_registered_edge_replication_summary.json")
    p321 = read_json(V2 / "phase321_natural_causal_edge_atlas_summary.json")
    p321_progress = read_json(V2 / "phase321_natural_causal_edge_progress.json")
    denominator = p317.get("scientific_denominator", {})
    claims = [
        {
            "claim_id": "phase322:balanced_denominator_complete",
            "evidence_level": "L1_registry",
            "status": "supported",
            "claim": "The frozen three-family denominator contains 144 base cases, 144 paired counterfactuals, three templates, and 432 model pairs.",
        },
        {
            "claim_id": "phase322:natural_source_transfer",
            "evidence_level": "L4_intervention_effect",
            "status": "supported_with_scope_limit",
            "claim": "Whole natural donor source states causally move output boundaries in explicit knowledge-value and some syntax routes.",
        },
        {
            "claim_id": "phase322:reasoning_source_transfer",
            "evidence_level": "L4_negative_result",
            "status": "not_supported",
            "claim": "Single critical-token state replacement does not recover a general reasoning route after controls.",
        },
        {
            "claim_id": "phase322:single_component_mediation",
            "evidence_level": "L4_negative_causal_audit",
            "status": "not_supported",
            "claim": "One attention head plus one MLP product group does not stably mediate the natural source-state effect.",
        },
        {
            "claim_id": "phase322:l5_promotion",
            "evidence_level": "L4_negative_replication",
            "status": "not_supported",
            "claim": "No candidate passes registered parallel-object and cross-model replication, so no edge is promoted to L5.",
        },
        {
            "claim_id": "phase322:language_mechanism_closure",
            "evidence_level": "L0_unresolved",
            "status": "not_complete",
            "claim": "The language encoding mechanism and strict natural closure remain unresolved.",
        },
    ]
    progress = {
        "schema_version": SCHEMA_VERSION,
        "last_phase": PHASE,
        "updated_at": now(),
        "engineering_progress": {
            "atlas_data_system": 0.94,
            "core_case_schema": 1.0,
            "provenance_fields": 1.0,
            "frontend_data_sync": 0.92,
        },
        "scientific_progress": {
            "controlled_core_independent_case_coverage": 1.0,
            "controlled_core_three_position_event_coverage": 1.0,
            "matched_control_analysis_coverage": 1.0,
            "same_template_heldout_prediction_coverage": 1.0,
            "template_heldout_prediction_coverage": 1.0,
            "natural_source_state_intervention_coverage_in_frozen_scope": 1.0,
            "source_to_query_last_propagation_coverage_in_frozen_scope": 1.0,
            "open_template_component_mediation_audit_coverage_in_frozen_scope": 1.0,
            "registered_candidate_replication_coverage_in_selected_scope": 1.0,
            "promoted_l5_edge_quality": 0.0,
            "latent_memory_retrieval_path_coverage": 0.0,
            "distributed_multinode_mediation_coverage": 0.0,
            "natural_gate_coverage": 0.0,
            "strict_clean_closure": 0.0,
        },
        "global_research_estimates": {
            "language_pattern_physical_atlas": {"lower": 0.48, "upper": 0.56},
            "language_encoding_mechanism": {"lower": 0.22, "upper": 0.30},
            "strict_natural_closure": {"lower": 0.05, "upper": 0.10},
            "verifiable_intelligence_theory": {"lower": 0.17, "upper": 0.25},
        },
        "legacy_management_estimates_are_not_mechanism_completion": True,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "stage_complete_no_replicated_l5_edge",
        "stage": "Phase317-322 Natural Source-to-Boundary Causal Edge Atlas",
        "frozen_denominator": denominator,
        "case_bank_hash": p317.get("case_bank_hash"),
        "pair_bank_hash": p317.get("pair_bank_hash"),
        "natural_source_intervention_model_pairs": p321.get("natural_source_intervention_pairs"),
        "open_template_heldout_model_pairs": p321.get("open_template_heldout_pairs"),
        "registered_replication_model_pairs": p321.get("registered_replication_model_pairs"),
        "discovery_scan_rows": p318.get("discovery_scan_rows"),
        "calibration_control_rows": p318.get("calibration_control_rows"),
        "source_to_query_last_propagation_rows": p318.get("propagation_rows"),
        "discovery_component_rows": p319.get("discovery_component_rows"),
        "heldout_condition_rows": p319.get("heldout_condition_rows"),
        "phrase_rollout_cases": p319.get("phrase_rollout_cases"),
        "calibration_control_corrected_transfer": p318.get("calibration_control_corrected_transfer"),
        "heldout_control_corrected_transfer": p319.get("heldout_control_corrected_transfer_mean"),
        "heldout_donor_win_rate": p319.get("heldout_donor_win_rate"),
        "phrase_transfer_shift_mean": p319.get("phrase_transfer_shift_mean"),
        "rollout_change_rate": p319.get("rollout_change_rate"),
        "patched_donor_start_rate": p319.get("patched_donor_start_rate"),
        "joint_mediation_loss_mean": p319.get("joint_mediation_loss_mean"),
        "screened_l5_candidate_count": p319.get("l5_candidate_count"),
        "registered_pass_count": p320.get("registered_pass_count"),
        "promoted_l5_edge_count": p320.get("promoted_l5_edge_count"),
        "family_aggregates": p321.get("family_aggregates"),
        "graph_nodes": p321.get("graph_nodes"),
        "graph_edges": p321.get("graph_edges"),
        "claims": claims,
        "progress": progress,
        "stage_conclusion": "Natural whole-state source effects are real in explicit value transport and some syntax routes, but the effect is not explained by one stable attention-head/MLP-group chain and does not generalize to reasoning. No L5 edge or closure is claimed.",
        "next_stage": {
            "phase_range": "Phase323-330",
            "title": "Distributed multi-node carrier set and latent-memory retrieval atlas",
            "same_stage": False,
            "priorities": [
                "Replace explicit answer-bearing knowledge records with latent-memory and counterfactual fact designs.",
                "Select sparse head sets and MLP channel sets by discovery-only ablation, not response norm alone.",
                "Trace rule clauses and variable bindings as multi-token source groups for reasoning.",
                "Require cross-template, cross-object, cross-language, phrase, rollout, and side-effect replication before L5 promotion.",
            ],
        },
    }
    report = f"""# Phase322 Natural Source-to-Boundary Stage Report

## Scope

- Base independent cases: {denominator.get('base_independent_cases', 0)}
- Base independent pairs: {denominator.get('base_independent_pairs', 0)}
- Planned model pairs: {denominator.get('planned_model_pairs', 0)}
- Open-template heldout pairs: {p321.get('open_template_heldout_pairs', 0)}
- Registered replication model pairs: {p321.get('registered_replication_model_pairs', 0)}

## Objective results

- Calibration control-corrected transfer: {p318.get('calibration_control_corrected_transfer', 0)}
- Heldout control-corrected transfer: {p319.get('heldout_control_corrected_transfer_mean', 0)}
- Heldout donor win rate: {p319.get('heldout_donor_win_rate', 0)}
- Joint single-head/single-group mediation loss: {p319.get('joint_mediation_loss_mean', 0)}
- Registered pass count: {p320.get('registered_pass_count', 0)}
- Promoted L5 edge count: {p320.get('promoted_l5_edge_count', 0)}

## Family split

{json.dumps(p321.get('family_aggregates', []), ensure_ascii=False, indent=2)}

## Strict conclusion

Whole natural source-state replacement has a reproducible L4 effect for explicit record-value transport and part of syntax. It does not establish latent knowledge retrieval, a general reasoning path, a stable single-head/single-channel mediator, an L5 causal edge, or strict closure.
"""
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase322_natural_source_boundary_stage_summary.json", summary)
        write_jsonl(base / "phase322_stage_claim_rows.jsonl", [{"schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(), **row} for row in claims])
        write_json(base / "phase322_evidence_progress.json", p321_progress)
        write_json(base / "progress.json", progress)
        (base / "phase322_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    complete()
