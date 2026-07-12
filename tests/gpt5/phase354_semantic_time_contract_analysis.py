#!/usr/bin/env python3
"""Aggregate paired semantic-time traces without opening heldout or causal seals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace"
ROUND_DEFAULT = "qualified_contract_semantic_time"
PHASE = "Phase354"
SCHEMA_VERSION = "30.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("physical_discovery", "physical_calibration")
MODES = ("teacher_forced", "free_rollout")
PHASES = ("single", "first", "middle", "final")
COMPONENTS = ("attention_output", "mlp_output", "residual_increment")
DEPTHS = ("early", "middle", "late")
ROLES = ("source", "query", "answer_start", "current_generation")


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


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase354_protocol_summary.json")
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    case_rows = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase354_case_rows.jsonl")]
    traces = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase354_semantic_time_rows.jsonl")]
    contracts = sorted({(row["family_id"], row["mechanism_id"]) for row in traces})

    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in traces:
        key = (
            row["model"], row["family_id"], row["mechanism_id"], row["contract_group_id"],
            row["split"], row["template_id"], row["trajectory_mode"], row["semantic_phase"],
            row["layer_index"], row["depth_bin"], row["component"], row["position_role"],
        )
        grouped[key][row["contrast_condition"][0]].append(row["signed_competition_margin"])

    events, incomplete = [], 0
    for key, values in grouped.items():
        if set(values) != {"A", "B", "C", "D"}:
            incomplete += 1
            continue
        condition_mean = {condition: mean(entries) for condition, entries in values.items()}
        dx = condition_mean["A"] - condition_mean["B"]
        dy = condition_mean["C"] - condition_mean["D"]
        model, family, mechanism, group, split, template, mode, phase, layer, depth, component, role = key
        events.append({
            "model": model, "family_id": family, "mechanism_id": mechanism,
            "contract_group_id": group, "split": split, "template_id": template,
            "trajectory_mode": mode, "semantic_phase": phase, "layer_index": layer,
            "depth_bin": depth, "component": component, "position_role": role,
            "operation_delta_x": dx, "operation_delta_y": dy,
            "mean_operation_delta": (dx + dy) / 2,
            "mean_operation_magnitude": (abs(dx) + abs(dy)) / 2,
            "lexical_instability": abs(dx - dy),
            "lexical_sign_agreement": dx * dy > 0,
        })

    nodes = []
    for model in MODELS:
        for family, mechanism in contracts:
            for mode in MODES:
                for phase in PHASES:
                    for component in COMPONENTS:
                        for depth in DEPTHS:
                            for role in ROLES:
                                selected = [
                                    row for row in events
                                    if row["model"] == model and row["family_id"] == family
                                    and row["mechanism_id"] == mechanism and row["trajectory_mode"] == mode
                                    and row["semantic_phase"] == phase and row["component"] == component
                                    and row["depth_bin"] == depth and row["position_role"] == role
                                ]
                                split_gates, metrics = [], {}
                                for split in SPLITS:
                                    values = [row for row in selected if row["split"] == split]
                                    signal = mean(row["mean_operation_delta"] for row in values) if values else 0.0
                                    magnitude = mean(row["mean_operation_magnitude"] for row in values) if values else 0.0
                                    instability = mean(row["lexical_instability"] for row in values) if values else 0.0
                                    agreement = mean(row["lexical_sign_agreement"] for row in values) if values else 0.0
                                    template_signs = []
                                    for template in ("format_a", "format_b"):
                                        template_values = [row["mean_operation_delta"] for row in values if row["template_id"] == template]
                                        template_signs.append(sign(mean(template_values)) if template_values else 0)
                                    gate = bool(
                                        values and abs(signal) >= 0.005 and agreement >= 0.75
                                        and magnitude > instability and template_signs[0] == template_signs[1] != 0
                                    )
                                    split_gates.append(gate)
                                    metrics.update({
                                        f"{split}_event_count": len(values),
                                        f"{split}_mean_operation_delta": round(signal, 7),
                                        f"{split}_mean_operation_magnitude": round(magnitude, 7),
                                        f"{split}_mean_lexical_instability": round(instability, 7),
                                        f"{split}_lexical_sign_agreement_rate": round(agreement, 7),
                                        f"{split}_template_signs": template_signs,
                                        f"{split}_dynamic_gate_pass": gate,
                                    })
                                signal = mean(row["mean_operation_delta"] for row in selected) if selected else 0.0
                                node_id = f"phase354:{model}:{family}:{mechanism}:{mode}:{phase}:{component}:{depth}:{role}"
                                nodes.append({
                                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                                    "node_id": node_id, "node_type": "semantic_time_physical_region",
                                    "model": model, "family_id": family, "mechanism_id": mechanism,
                                    "trajectory_mode": mode, "semantic_phase": phase,
                                    "component": component, "depth_bin": depth, "position_role": role,
                                    "event_count": len(selected), "mean_operation_delta": round(signal, 7),
                                    "signed_direction": "positive" if signal > 0 else "negative" if signal < 0 else "zero",
                                    "dynamic_discovery_calibration_gate_pass": bool(selected) and all(split_gates),
                                    "mapping_status": "dynamic_signed_candidate" if bool(selected) and all(split_gates) else "descriptive_natural_trace",
                                    "causal_status": "not_tested", "single_unit_causal": False,
                                    **metrics,
                                })
    nodes.sort(key=lambda row: row["node_id"])

    dominant = []
    for model in MODELS:
        for family, mechanism in contracts:
            for mode in MODES:
                values = [
                    row for row in nodes if row["model"] == model and row["family_id"] == family
                    and row["mechanism_id"] == mechanism and row["trajectory_mode"] == mode
                    and row["event_count"] > 0
                ]
                gated = [row for row in values if row["dynamic_discovery_calibration_gate_pass"]]
                winner = max(
                    gated or values,
                    key=lambda row: (
                        row["dynamic_discovery_calibration_gate_pass"],
                        abs(row["mean_operation_delta"]),
                    ),
                )
                dominant.append({
                    "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                    "model": model, "family_id": family, "mechanism_id": mechanism,
                    "trajectory_mode": mode, "semantic_phase": winner["semantic_phase"],
                    "component": winner["component"], "depth_bin": winner["depth_bin"],
                    "position_role": winner["position_role"], "signed_direction": winner["signed_direction"],
                    "mean_operation_delta": winner["mean_operation_delta"],
                    "dynamic_gate_pass": winner["dynamic_discovery_calibration_gate_pass"],
                    "physical_heldout_tested": False, "causal_status": "not_tested",
                })

    convergence = []
    for family, mechanism in contracts:
        mode_rows = {}
        for mode in MODES:
            values = [row for row in dominant if row["family_id"] == family and row["mechanism_id"] == mechanism and row["trajectory_mode"] == mode]
            functional = {
                f"{row['semantic_phase']}:{row['component']}:{row['depth_bin']}:{row['position_role']}:{row['signed_direction']}"
                for row in values
            }
            pass_gate = all(row["dynamic_gate_pass"] for row in values) and len(functional) == 1
            mode_rows[mode] = (pass_gate, sorted(functional))
        teacher_gate, teacher_values = mode_rows["teacher_forced"]
        free_gate, free_values = mode_rows["free_rollout"]
        agreement = teacher_gate and free_gate and teacher_values == free_values
        convergence.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "family_id": family, "mechanism_id": mechanism,
            "teacher_cross_model_gate_pass": teacher_gate,
            "free_cross_model_gate_pass": free_gate,
            "teacher_free_functional_agreement": agreement,
            "teacher_functional_values": teacher_values,
            "free_functional_values": free_values,
            "physical_heldout_entry_open": agreement,
        })

    edges = []
    phase_order = {value: index for index, value in enumerate(PHASES)}
    for node in nodes:
        if node["event_count"] == 0:
            continue
        candidates = [
            other for other in nodes
            if other["model"] == node["model"] and other["family_id"] == node["family_id"]
            and other["mechanism_id"] == node["mechanism_id"] and other["trajectory_mode"] == node["trajectory_mode"]
            and other["component"] == node["component"] and other["depth_bin"] == node["depth_bin"]
            and other["position_role"] == node["position_role"]
            and phase_order[other["semantic_phase"]] == phase_order[node["semantic_phase"]] + 1
            and other["event_count"] > 0
        ]
        for other in candidates:
            edges.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE,
                "edge_id": f"{node['node_id']}->{other['node_id']}",
                "source_node_id": node["node_id"], "target_node_id": other["node_id"],
                "edge_type": "semantic_time_transition", "evidence_status": "natural_trace_only",
                "causal_status": "not_tested",
            })

    entry = [f"{row['family_id']}/{row['mechanism_id']}" for row in convergence if row["physical_heldout_entry_open"]]
    total_grouped = len(events) + incomplete
    summary = {
        **protocol,
        "created_at": now(),
        "denominator": {
            **protocol["denominator"],
            "case_row_count": len(case_rows), "raw_trace_row_count": len(traces),
            "paired_event_count": len(events), "incomplete_pair_count": incomplete,
            "complete_pair_rate": round(len(events) / total_grouped, 7) if total_grouped else 0.0,
            "fixed_node_count": len(nodes), "graph_edge_count": len(edges),
            "nonfinite_trace_row_count": sum(not row["finite"] for row in traces),
            "all_model_completions_valid": all(row["valid"] for row in completions),
        },
        "results": {
            "dynamic_gate_node_count": sum(row["dynamic_discovery_calibration_gate_pass"] for row in nodes),
            "dominant_model_mode_gate_count": sum(row["dynamic_gate_pass"] for row in dominant),
            "teacher_cross_model_contract_count": sum(row["teacher_cross_model_gate_pass"] for row in convergence),
            "free_cross_model_contract_count": sum(row["free_cross_model_gate_pass"] for row in convergence),
            "teacher_free_cross_model_agreement_count": len(entry),
            "physical_heldout_entry_contracts": entry,
            "physical_heldout_trace_revealed": False, "causal_sealed_trace_revealed": False,
            "internal_intervention_executed_count": 0, "single_unit_causal_count": 0,
        },
        "claim_boundary": {
            "natural_trace_is_causal": False,
            "teacher_forced_path_equals_free_rollout_path": False,
            "dominant_region_is_single_neuron_mechanism": False,
            "physical_heldout_tested": False,
        },
        "next_decision": "freeze_candidates_before_heldout" if entry else "repair_semantic_time_contracts_without_heldout",
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(root / "phase354_dynamic_nodes.jsonl", nodes)
    write_jsonl(root / "phase354_dominant_regions.jsonl", dominant)
    write_jsonl(root / "phase354_cross_model_convergence.jsonl", convergence)
    write_jsonl(root / "phase354_graph_nodes.jsonl", nodes)
    write_jsonl(root / "phase354_graph_edges.jsonl", edges)
    write_json(root / "phase354_global_summary.json", summary)
    report = [
        "# Phase354 Qualified Contract Semantic-Time Trace", "",
        f"- Registered cases: {len(case_rows)}",
        f"- Raw trace rows: {len(traces)}",
        f"- Complete paired events: {len(events)}/{total_grouped}",
        f"- Fixed graph nodes/edges: {len(nodes)}/{len(edges)}",
        f"- Teacher/free cross-model agreements: {len(entry)}/{len(contracts)}",
        f"- Physical heldout entries: {', '.join(entry) if entry else 'none'}", "",
        "Physical heldout, causal sealed cases, interventions, and neuron search remain closed.",
    ]
    (root / "phase354_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))
