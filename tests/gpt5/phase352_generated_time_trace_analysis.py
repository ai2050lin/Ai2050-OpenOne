#!/usr/bin/env python3
"""Aggregate teacher-forced generated-time paired trajectory events."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase352_generated_time_trace"
PHASE = "Phase352"
SCHEMA_VERSION = "28.0.0"
ROUND_DEFAULT = "generated_time_signed_trace"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("physical_discovery", "physical_calibration")
PHASES = ("first", "middle", "final", "first_final")
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


def aggregate(round_name: str = ROUND_DEFAULT) -> dict[str, Any]:
    root = OUT / round_name
    completions = [read_json(root / "models" / model / "complete.json") for model in MODELS]
    traces = [row for model in MODELS for row in read_jsonl(root / "models" / model / "phase352_generated_time_rows.jsonl")]
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in traces:
        key = (
            row["model"], row["family_id"], row["item_index"], row["split"], row["template_id"],
            row["generation_phase"], row["layer_index"], row["depth_bin"],
            row["component"], row["position_role"],
        )
        grouped[key][row["contrast_condition"][0]].append(row["signed_competition_margin"])
    events, incomplete = [], 0
    for key, values in grouped.items():
        if set(values) != {"A", "B", "C", "D"}:
            incomplete += 1
            continue
        model, family, item, split, template, phase, layer, depth, component, role = key
        condition_mean = {condition: mean(entries) for condition, entries in values.items()}
        dx = condition_mean["A"] - condition_mean["B"]
        dy = condition_mean["C"] - condition_mean["D"]
        events.append({
            "model": model, "family_id": family, "item_index": item, "split": split,
            "template_id": template, "generation_phase": phase, "layer_index": layer,
            "depth_bin": depth, "component": component, "position_role": role,
            "operation_delta_x": dx, "operation_delta_y": dy,
            "mean_operation_delta": (dx + dy) / 2,
            "mean_operation_magnitude": (abs(dx) + abs(dy)) / 2,
            "lexical_instability": abs(dx - dy), "lexical_sign_agreement": dx * dy > 0,
        })
    families = sorted({row["family_id"] for row in events})
    nodes = []
    for model in MODELS:
        for family in families:
            for phase in PHASES:
                for component in ("attention_output", "mlp_output", "residual_increment"):
                    for depth in ("early", "middle", "late"):
                        for role in ROLES:
                            values = [row for row in events if row["model"] == model and row["family_id"] == family and row["generation_phase"] == phase and row["component"] == component and row["depth_bin"] == depth and row["position_role"] == role]
                            split_metrics, split_gates = {}, []
                            for split in SPLITS:
                                selected = [row for row in values if row["split"] == split]
                                if not selected:
                                    signal = magnitude = instability = sign_rate = 0.0
                                    template_signs = [0, 0]
                                    gate = False
                                else:
                                    signal = mean(row["mean_operation_delta"] for row in selected)
                                    magnitude = mean(row["mean_operation_magnitude"] for row in selected)
                                    instability = mean(row["lexical_instability"] for row in selected)
                                    sign_rate = mean(row["lexical_sign_agreement"] for row in selected)
                                    template_signs = []
                                    for template in ("format_a", "format_b"):
                                        template_values = [row["mean_operation_delta"] for row in selected if row["template_id"] == template]
                                        template_mean = mean(template_values)
                                        template_signs.append(1 if template_mean > 0 else -1 if template_mean < 0 else 0)
                                    gate = bool(abs(signal) >= 0.005 and sign_rate >= 0.75 and magnitude > instability and template_signs[0] == template_signs[1] != 0)
                                split_gates.append(gate)
                                split_metrics.update({
                                    f"{split}_event_count": len(selected),
                                    f"{split}_mean_operation_delta": round(signal, 7),
                                    f"{split}_mean_operation_magnitude": round(magnitude, 7),
                                    f"{split}_mean_lexical_instability": round(instability, 7),
                                    f"{split}_lexical_sign_agreement_rate": round(sign_rate, 7),
                                    f"{split}_template_signs": template_signs,
                                    f"{split}_dynamic_gate_pass": gate,
                                })
                            signal = mean([row["mean_operation_delta"] for row in values]) if values else 0.0
                            magnitude = mean([row["mean_operation_magnitude"] for row in values]) if values else 0.0
                            instability = mean([row["lexical_instability"] for row in values]) if values else 0.0
                            nodes.append({
                                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                                "node_id": f"phase352:{model}:{family}:{phase}:{component}:{depth}:{role}",
                                "model": model, "family_id": family, "generation_phase": phase,
                                "component": component, "depth_bin": depth, "position_role": role,
                                "event_count": len(values), "mean_operation_delta": round(signal, 7),
                                "mean_operation_magnitude": round(magnitude, 7),
                                "mean_lexical_instability": round(instability, 7),
                                "signed_direction": "positive" if signal > 0 else "negative" if signal < 0 else "zero",
                                "dynamic_discovery_calibration_gate_pass": bool(values) and all(split_gates),
                                **split_metrics, "teacher_forced": True,
                                "mapping_status": "dynamic_signed_candidate" if bool(values) and all(split_gates) else "descriptive_generated_time_trace",
                                "causal_status": "not_tested", "single_unit_causal": False,
                            })
    nodes.sort(key=lambda row: row["node_id"])
    dominant = []
    for model in MODELS:
        for family in families:
            values = [row for row in nodes if row["model"] == model and row["family_id"] == family and row["event_count"] > 0]
            gated = [row for row in values if row["dynamic_discovery_calibration_gate_pass"]]
            winner = max(gated or values, key=lambda row: (row["dynamic_discovery_calibration_gate_pass"], row["mean_operation_magnitude"] - row["mean_lexical_instability"]))
            dominant.append({
                "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
                "model": model, "family_id": family, "generation_phase": winner["generation_phase"],
                "component": winner["component"], "depth_bin": winner["depth_bin"],
                "position_role": winner["position_role"], "signed_direction": winner["signed_direction"],
                "mean_operation_delta": winner["mean_operation_delta"],
                "mean_operation_magnitude": winner["mean_operation_magnitude"],
                "mean_lexical_instability": winner["mean_lexical_instability"],
                "dynamic_gate_pass": winner["dynamic_discovery_calibration_gate_pass"],
                "teacher_forced": True, "physical_heldout_tested": False, "causal_status": "not_tested",
            })
    convergence = []
    for family in families:
        values = [row for row in dominant if row["family_id"] == family]
        functional = {f"{row['generation_phase']}:{row['depth_bin']}:{row['position_role']}:{row['signed_direction']}" for row in values}
        all_gate = all(row["dynamic_gate_pass"] for row in values)
        convergence.append({
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "family_id": family, "model_dynamic_gate_count": sum(row["dynamic_gate_pass"] for row in values),
            "cross_model_functional_agreement": len(functional) == 1,
            "cross_model_dynamic_gate_pass": all_gate and len(functional) == 1,
            "functional_values": sorted(functional), "teacher_forced": True,
            "physical_heldout_entry_open": all_gate and len(functional) == 1,
        })
    expected_nodes = 3 * 3 * 4 * 3 * 3 * 4
    entry = [row["family_id"] for row in convergence if row["physical_heldout_entry_open"]]
    summary = {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "denominator": {
            "registered_case_count": sum(row["registered_case_count"] for row in completions),
            "target_step_count": sum(row["target_step_count"] for row in completions),
            "raw_trace_row_count": len(traces), "paired_event_count": len(events),
            "incomplete_pair_count": incomplete, "fixed_dynamic_node_count": len(nodes),
            "complete_dynamic_pair_rate": round(len(events) / (len(events) + incomplete), 7),
            "expected_fixed_dynamic_node_count": expected_nodes,
            "nonfinite_trace_row_count": sum(not row["finite"] for row in traces),
            "all_model_completions_valid": all(row["valid"] for row in completions),
        },
        "results": {
            "dynamic_gate_node_count": sum(row["dynamic_discovery_calibration_gate_pass"] for row in nodes),
            "model_family_dynamic_gate_count": sum(row["dynamic_gate_pass"] for row in dominant),
            "cross_model_dynamic_family_count": len(entry),
            "physical_heldout_entry_families": entry,
            "physical_heldout_trace_revealed": False, "causal_sealed_trace_revealed": False,
            "internal_intervention_executed_count": 0,
            "behavior_mechanism_closed_count": 0, "single_unit_causal_count": 0,
        },
        "next_decision": "selectively_reveal_physical_heldout" if entry else "repair_contrasts_without_heldout_reveal",
        "claim_boundary": {
            "teacher_forced_time_is_free_generation_path": False,
            "dynamic_signed_delta_is_causal": False,
            "explicit_shortcut_delta_is_pure_operation_delta": False,
            "cross_lexical_generation_phase_alignment_complete": incomplete == 0,
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    write_jsonl(root / "phase352_dynamic_signed_nodes.jsonl", nodes)
    write_jsonl(root / "phase352_dominant_dynamic_regions.jsonl", dominant)
    write_jsonl(root / "phase352_cross_model_convergence.jsonl", convergence)
    write_json(root / "phase352_global_summary.json", summary)
    report = [
        "# Phase352 Teacher-Forced Generated-Time Trace", "",
        f"- Cases/target steps: {summary['denominator']['registered_case_count']}/{summary['denominator']['target_step_count']}",
        f"- Raw trace rows: {len(traces)}", f"- Paired events: {len(events)}",
        f"- Fixed dynamic nodes: {len(nodes)}/{expected_nodes}",
        f"- Cross-model dynamic entry families: {', '.join(entry) if entry else 'none'}", "",
        "Physical heldout, causal sealed cases, interventions, and neuron search remain closed.",
    ]
    (root / "phase352_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.round), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
