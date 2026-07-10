#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase321"
SCHEMA_VERSION = "4.4.0"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def aggregate(rows: list[dict[str, Any]], keys: list[str], values: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = {name: value for name, value in zip(keys, key)}
        record["case_count"] = len(members)
        for value in values:
            record[f"mean_{value}"] = mean_safe([safe_float(row.get(value)) for row in members])
        output.append(record)
    return output


def claim_rows(
    phase318: dict[str, Any],
    phase319: dict[str, Any],
    phase320: dict[str, Any],
    family_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    claims = [
        {
            "claim_id": "phase321:natural_source_state_changes_boundary",
            "claim": "Natural donor source-state replacement changes donor-versus-recipient output boundaries on calibration and open-template heldout pairs.",
            "evidence_level": "L4_intervention_effect",
            "support": {
                "calibration_control_corrected_transfer": phase318.get("calibration_control_corrected_transfer"),
                "heldout_control_corrected_transfer": phase319.get("heldout_control_corrected_transfer_mean"),
                "heldout_donor_win_rate": phase319.get("heldout_donor_win_rate"),
            },
            "limitation": "Whole residual replacement is high-dimensional and can carry lexical identity or explicit answer-value information.",
        },
        {
            "claim_id": "phase321:source_effect_reaches_query_last",
            "claim": "Patched source-state changes project in the donor direction at query and last positions more often than chance in the measured paths.",
            "evidence_level": "L3_observational_propagation",
            "support": {
                "propagation_rows": phase318.get("propagation_rows"),
                "model_summaries": [
                    {
                        "model": row.get("model"),
                        "query_positive_rate": row.get("query_positive_propagation_rate"),
                        "last_positive_rate": row.get("last_positive_propagation_rate"),
                    }
                    for row in phase318.get("model_summaries", [])
                ],
            },
            "limitation": "Projection is descriptive and does not identify the intervening causal carrier.",
        },
        {
            "claim_id": "phase321:single_head_channel_mediation_not_supported",
            "claim": "One discovery-selected attention head plus one MLP product group does not stably mediate the source-state transfer effect.",
            "evidence_level": "L4_negative_causal_audit",
            "support": {
                "joint_mediation_loss_mean": phase319.get("joint_mediation_loss_mean"),
                "registered_pass_count": phase320.get("registered_pass_count"),
                "promoted_l5_edge_count": phase320.get("promoted_l5_edge_count"),
            },
            "limitation": "A distributed head set, channel set, nonlinear gate, or different candidate-selection rule may still mediate the effect.",
        },
        {
            "claim_id": "phase321:knowledge_transfer_is_explicit_value_route",
            "claim": "Knowledge-family transfer is much stronger than reasoning transfer in this design, but the knowledge source token explicitly contains the answer value.",
            "evidence_level": "L4_task_specific_route",
            "support": {"family_rows": family_rows},
            "limitation": "The result demonstrates explicit record-value transport, not latent knowledge retrieval from model memory.",
        },
        {
            "claim_id": "phase321:no_l5_edge_promoted",
            "claim": "No source-to-boundary edge satisfies registered cross-object and cross-model promotion criteria.",
            "evidence_level": "L4_negative_replication",
            "support": {
                "registered_model_cases": phase320.get("registered_model_cases"),
                "registered_pass_rate": phase320.get("registered_pass_rate"),
            },
            "limitation": "The graph remains a causal-candidate atlas, not a closed mechanism graph.",
        },
    ]
    return [{"schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(), **row} for row in claims]


def graph_rows(
    source_selections: list[dict[str, Any]],
    component_selections: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def node(node_id: str, node_type: str, label: str, **attrs: Any) -> None:
        nodes[node_id] = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            **attrs,
        }

    def edge(source: str, target: str, relation: str, evidence_level: str, weight: float = 0.0, **attrs: Any) -> None:
        edges.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "edge_id": f"phase321:edge:{len(edges):04d}",
                "source": source,
                "target": target,
                "relation": relation,
                "evidence_level": evidence_level,
                "weight": round(weight, 6),
                **attrs,
            }
        )

    node("phase321:boundary", "boundary", "donor-versus-recipient boundary")
    node("phase321:phrase", "boundary", "complete target phrase likelihood")
    node("phase321:rollout", "generation", "natural rollout")
    for row in source_selections:
        model, family, mechanism = str(row["model"]), str(row["family_id"]), str(row["mechanism_id"])
        family_id = f"phase321:family:{family}"
        mechanism_id = f"phase321:mechanism:{family}:{mechanism}"
        source_id = f"phase321:source:{model}:{family}:{mechanism}:L{row['selected_source_layer']}"
        node(family_id, "family", family)
        node(mechanism_id, "mechanism", mechanism, family_id=family)
        node(source_id, "source_state", f"{model} {mechanism} source layer {row['selected_source_layer']}", model=model, layer=row["selected_source_layer"])
        edge(family_id, mechanism_id, "contains", "L1_registry")
        edge(mechanism_id, source_id, "discovery_selected_source", "L3_candidate", safe_float(row.get("discovery_control_corrected_transfer")))
        edge(source_id, "phase321:boundary", "natural_state_intervention_changes", "L4_intervention_effect", safe_float(row.get("discovery_control_corrected_transfer")))
    for row in component_selections:
        model, family, mechanism = str(row["model"]), str(row["family_id"]), str(row["mechanism_id"])
        source_candidates = [r for r in source_selections if r["model"] == model and r["family_id"] == family and r["mechanism_id"] == mechanism]
        if not source_candidates:
            continue
        source = source_candidates[0]
        source_id = f"phase321:source:{model}:{family}:{mechanism}:L{source['selected_source_layer']}"
        component_id = f"phase321:component:{model}:{family}:{mechanism}:{row['component_type']}:L{row['component_layer']}:{row['position_role']}:{row['component_index']}"
        node(
            component_id,
            "component_candidate",
            f"{row['component_type']} L{row['component_layer']} {row['position_role']} #{row['component_index']}",
            model=model,
            component_type=row["component_type"],
            layer=row["component_layer"],
            position_role=row["position_role"],
            component_index=row["component_index"],
        )
        edge(source_id, component_id, "naturally_changes", "L3_observational_response", safe_float(row.get("discovery_mean_relative_delta_norm")))
        edge(component_id, "phase321:boundary", "mediation_candidate_not_replicated", "L4_negative_causal_audit", 0.0)
    for row in claims:
        claim_id = str(row["claim_id"])
        node(claim_id, "claim", str(row["claim"]), evidence_level=row["evidence_level"])
    return list(nodes.values()), edges


def synthesize() -> dict[str, Any]:
    phase317 = read_json(V2 / "phase317_natural_source_case_bank_summary.json")
    phase318 = read_json(V2 / "phase318_natural_source_state_transfer_summary.json")
    phase319 = read_json(V2 / "phase319_heldout_component_mediation_summary.json")
    phase320 = read_json(V2 / "phase320_registered_edge_replication_summary.json")
    source_selections = read_jsonl(V2 / "phase318_source_layer_selection_rows.jsonl")
    component_selections = read_jsonl(V2 / "phase319_component_selection_rows.jsonl")
    calibration = read_jsonl(V2 / "phase318_calibration_control_rows.jsonl")
    mediation = read_jsonl(V2 / "phase319_heldout_mediation_rows.jsonl")
    rollouts = read_jsonl(V2 / "phase319_phrase_rollout_rows.jsonl")
    family_rows = aggregate(
        mediation,
        ["family_id"],
        ["source_transfer_shift", "control_corrected_transfer", "joint_mediation_loss", "joint_mediation_fraction"],
    )
    model_family_rows = aggregate(
        mediation,
        ["model", "family_id"],
        ["source_transfer_shift", "control_corrected_transfer", "joint_mediation_loss", "joint_mediation_fraction"],
    )
    mechanism_rows = aggregate(
        mediation,
        ["model", "family_id", "mechanism_id"],
        ["source_transfer_shift", "control_corrected_transfer", "attention_head_mediation_loss", "mlp_product_mediation_loss", "joint_mediation_loss"],
    )
    calibration_condition_rows = aggregate(calibration, ["model", "family_id", "condition"], ["donor_transfer_shift", "js_divergence"])
    rollout_family_rows = aggregate(rollouts, ["model", "family_id"], ["phrase_transfer_shift"])
    for row in rollout_family_rows:
        members = [r for r in rollouts if r["model"] == row["model"] and r["family_id"] == row["family_id"]]
        row["rollout_change_rate"] = mean_safe([1.0 if r["rollout_changed"] else 0.0 for r in members])
        row["patched_donor_start_rate"] = mean_safe([1.0 if r["patched_starts_with_donor"] else 0.0 for r in members])
    claims = claim_rows(phase318, phase319, phase320, family_rows)
    nodes, edges = graph_rows(source_selections, component_selections, family_rows, claims)
    planned_model_pairs = int(phase317.get("scientific_denominator", {}).get("planned_model_pairs", 0))
    progress = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "scope": "Phase317-320 frozen natural source-to-boundary denominator only",
        "engineering_coverage": {
            "planned_model_pairs": planned_model_pairs,
            "natural_source_intervention_pairs": 432,
            "natural_source_intervention_coverage": round(432 / planned_model_pairs, 6) if planned_model_pairs else 0.0,
            "matched_control_pairs": 288,
            "matched_control_coverage": round(288 / planned_model_pairs, 6) if planned_model_pairs else 0.0,
            "open_template_heldout_mediation_pairs": int(phase319.get("heldout_mediation_cases", 0)),
            "registered_replication_model_pairs": int(phase320.get("registered_model_cases", 0)),
            "phrase_rollout_heldout_pairs": int(phase319.get("phrase_rollout_cases", 0)),
        },
        "scientific_coverage": {
            "descriptive_source_to_query_last_path": 1.0,
            "natural_source_state_intervention": 1.0,
            "single_head_single_channel_mediation_audit": 1.0,
            "registered_candidate_replication": 1.0,
            "promoted_l5_edge_count": int(phase320.get("promoted_l5_edge_count", 0)),
            "promoted_l5_edge_quality": 0.0,
            "latent_memory_retrieval_coverage": 0.0,
            "distributed_multinode_mediation_coverage": 0.0,
            "strict_clean_closure": 0.0,
        },
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "source_phases": ["Phase317", "Phase318", "Phase319", "Phase320"],
        "planned_model_pairs": planned_model_pairs,
        "natural_source_intervention_pairs": 432,
        "open_template_heldout_pairs": int(phase319.get("heldout_mediation_cases", 0)),
        "registered_replication_model_pairs": int(phase320.get("registered_model_cases", 0)),
        "calibration_control_corrected_transfer": phase318.get("calibration_control_corrected_transfer"),
        "heldout_control_corrected_transfer": phase319.get("heldout_control_corrected_transfer_mean"),
        "heldout_donor_win_rate": phase319.get("heldout_donor_win_rate"),
        "joint_mediation_loss_mean": phase319.get("joint_mediation_loss_mean"),
        "registered_pass_count": phase320.get("registered_pass_count"),
        "promoted_l5_edge_count": phase320.get("promoted_l5_edge_count"),
        "phrase_transfer_shift_mean": phase319.get("phrase_transfer_shift_mean"),
        "rollout_change_rate": phase319.get("rollout_change_rate"),
        "patched_donor_start_rate": phase319.get("patched_donor_start_rate"),
        "family_aggregates": family_rows,
        "model_family_aggregates": model_family_rows,
        "mechanism_aggregates": mechanism_rows,
        "calibration_condition_aggregates": calibration_condition_rows,
        "rollout_family_aggregates": rollout_family_rows,
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "claim_count": len(claims),
        "stage_judgment": "strong_L4_natural_source_effect_but_no_replicated_L5_edge",
    }
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase321_natural_causal_edge_atlas_summary.json", summary)
        write_json(base / "phase321_natural_causal_edge_progress.json", progress)
        write_jsonl(base / "phase321_natural_causal_edge_claim_rows.jsonl", claims)
        write_jsonl(base / "phase321_natural_causal_edge_graph_nodes.jsonl", nodes)
        write_jsonl(base / "phase321_natural_causal_edge_graph_edges.jsonl", edges)
        write_jsonl(base / "phase321_family_aggregate_rows.jsonl", family_rows)
        write_jsonl(base / "phase321_model_family_aggregate_rows.jsonl", model_family_rows)
        write_jsonl(base / "phase321_mechanism_aggregate_rows.jsonl", mechanism_rows)
        write_jsonl(base / "phase321_calibration_condition_aggregate_rows.jsonl", calibration_condition_rows)
        write_jsonl(base / "phase321_rollout_family_aggregate_rows.jsonl", rollout_family_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    synthesize()
