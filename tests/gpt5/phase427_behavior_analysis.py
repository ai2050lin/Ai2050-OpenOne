#!/usr/bin/env python3
"""Audit Phase427 behavior gates without treating behavior as a physical path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase427_dual_route_protocol import (  # noqa: E402
    BLOCKS,
    MODELS,
    OUT,
    ROUTE_MODES,
    SCHEMA_VERSION,
    SCORABLE_CANDIDATE_ROUTES,
)


PHASE_ID = "Phase427-DualRouteBehaviorAnalysis"
VIS = ROOT / "frontend/public/vis_data/phase427_dual_route_behavior"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
GATE_SPLITS = ("behavior_calibration", "behavior_holdout")
COLORS = {
    "none": "#64748b",
    "source_only": "#22c55e",
    "query_only": "#06b6d4",
    "consistent": "#f59e0b",
    "conflict": "#ef4444",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase427 non-finite scalar: {value}")
    return round(float(value), 10)


def mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.fmean(rows)) if rows else 0.0


def median(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return clean(statistics.median(rows)) if rows else 0.0


def summarize_route(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate paired role measurements at the semantic-group unit."""
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["semantic_group_id"]].append(row)
    if not by_group:
        return {
            "independent_group_count": 0,
            "condition_count": 0,
            "paired_role_group_count": 0,
            "teacher_sequence_correct_fraction": 0.0,
            "teacher_sequence_margin_median": 0.0,
            "natural_target_first_fraction": 0.0,
            "natural_opposite_first_fraction": 0.0,
            "natural_other_fraction": 0.0,
            "natural_revision_fraction": 0.0,
            "natural_boundary_fraction": 0.0,
            "natural_stop_fraction": 0.0,
            "natural_censoring_fraction": 0.0,
            "natural_exact_contract_fraction": 0.0,
        }
    group_rows = []
    for group_id, values in sorted(by_group.items()):
        roles = {row["role"] for row in values}
        if len(values) != 2 or roles != {"a", "b"}:
            raise RuntimeError(
                f"Phase427 group is not a paired role unit: {group_id}: {roles}"
            )
        group_rows.append(
            {
                "teacher_sequence_correct_fraction": mean(
                    float(row["teacher_sequence_correct"]) for row in values
                ),
                "teacher_sequence_margin_median": median(
                    row["teacher_sequence_logprob_margin"] for row in values
                ),
                "natural_target_first_fraction": mean(
                    float(row["natural_target_first"]) for row in values
                ),
                "natural_opposite_first_fraction": mean(
                    float(row["natural_opposite_first"]) for row in values
                ),
                "natural_other_fraction": mean(
                    float(row["natural_other"]) for row in values
                ),
                "natural_revision_fraction": mean(
                    float(row["natural_revision"]) for row in values
                ),
                "natural_boundary_fraction": mean(
                    float(row["natural_boundary"]) for row in values
                ),
                "natural_stop_fraction": mean(
                    float(row["natural_stop"]) for row in values
                ),
                "natural_censoring_fraction": mean(
                    float(row["natural_censoring"]) for row in values
                ),
                "natural_exact_contract_fraction": mean(
                    float(row["natural_exact_contract"]) for row in values
                ),
            }
        )
    return {
        "independent_group_count": len(group_rows),
        "condition_count": len(rows),
        "paired_role_group_count": len(group_rows),
        "teacher_sequence_correct_fraction": mean(
            row["teacher_sequence_correct_fraction"] for row in group_rows
        ),
        "teacher_sequence_margin_median": median(
            row["teacher_sequence_margin_median"] for row in group_rows
        ),
        "natural_target_first_fraction": mean(
            row["natural_target_first_fraction"] for row in group_rows
        ),
        "natural_opposite_first_fraction": mean(
            row["natural_opposite_first_fraction"] for row in group_rows
        ),
        "natural_other_fraction": mean(
            row["natural_other_fraction"] for row in group_rows
        ),
        "natural_revision_fraction": mean(
            row["natural_revision_fraction"] for row in group_rows
        ),
        "natural_boundary_fraction": mean(
            row["natural_boundary_fraction"] for row in group_rows
        ),
        "natural_stop_fraction": mean(
            row["natural_stop_fraction"] for row in group_rows
        ),
        "natural_censoring_fraction": mean(
            row["natural_censoring_fraction"] for row in group_rows
        ),
        "natural_exact_contract_fraction": mean(
            row["natural_exact_contract_fraction"] for row in group_rows
        ),
    }


def behavior_gate(summary: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "group_count": int(summary["independent_group_count"])
        >= int(thresholds["groups_per_block_split"]),
        "teacher_correct": float(summary["teacher_sequence_correct_fraction"])
        >= float(thresholds["teacher_sequence_correct_fraction_min"]),
        "teacher_margin": float(summary["teacher_sequence_margin_median"])
        > float(thresholds["teacher_sequence_margin_median_min"]),
        "natural_target": float(summary["natural_target_first_fraction"])
        >= float(thresholds["natural_target_first_fraction_min"]),
        "natural_opposite": float(summary["natural_opposite_first_fraction"])
        <= float(thresholds["natural_opposite_first_fraction_max"]),
        "natural_revision": float(summary["natural_revision_fraction"])
        <= float(thresholds["natural_revision_fraction_max"]),
        "natural_boundary": float(summary["natural_boundary_fraction"])
        >= float(thresholds["natural_boundary_fraction_min"]),
        "natural_stop": float(summary["natural_stop_fraction"])
        >= float(thresholds["natural_stop_fraction_min"]),
        "natural_censoring": float(summary["natural_censoring_fraction"])
        <= float(thresholds["natural_censoring_fraction_max"]),
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def route_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["block_id"], row["split"], row["route_mode"])].append(row)
    output = []
    for (model, block_id, split, route_mode), values in sorted(grouped.items()):
        first = values[0]
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "model": model,
                "block_id": block_id,
                "family_id": first["family_id"],
                "mechanism_id": first["mechanism_id"],
                "candidate": first["candidate"],
                "matched_control_block_id": first["matched_control_block_id"],
                "split": split,
                "route_mode": route_mode,
                "normative": bool(not first["candidate"] or route_mode in SCORABLE_CANDIDATE_ROUTES),
                **summarize_route(values),
            }
        )
    return output


def lookup_summaries(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (row["model"], row["block_id"], row["split"], row["route_mode"]): row
        for row in rows
    }


def build_open_audits(
    summaries: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    lookup = lookup_summaries(summaries)
    thresholds = protocol["registered_thresholds"]
    candidates = [block for block in BLOCKS if block["candidate"]]
    audits = []
    for model in MODELS:
        for block in candidates:
            route_audits: dict[str, Any] = {}
            for route in SCORABLE_CANDIDATE_ROUTES:
                split_audits = {}
                for split in GATE_SPLITS:
                    candidate_summary = lookup[(model, block["block_id"], split, route)]
                    control_summary = lookup[
                        (model, block["matched_control_block_id"], split, route)
                    ]
                    candidate_gate = behavior_gate(candidate_summary, thresholds)
                    control_gate = behavior_gate(control_summary, thresholds)
                    split_audits[split] = {
                        "candidate_summary": candidate_summary,
                        "control_summary": control_summary,
                        "candidate_gate": candidate_gate,
                        "control_gate": control_gate,
                        "paired_gate_pass": bool(
                            candidate_gate["gate_pass"] and control_gate["gate_pass"]
                        ),
                    }
                route_audits[route] = {
                    "splits": split_audits,
                    "qualified_open": all(
                        split_audits[split]["paired_gate_pass"] for split in GATE_SPLITS
                    ),
                }
            block_pass = bool(
                route_audits["consistent"]["qualified_open"]
                and any(
                    route_audits[route]["qualified_open"]
                    for route in ("source_only", "query_only")
                )
            )
            descriptive = {}
            for route in ("none", "conflict"):
                descriptive[route] = {
                    split: lookup[(model, block["block_id"], split, route)]
                    for split in ("behavior_discovery", *GATE_SPLITS)
                }
            audits.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "model": model,
                    "block_id": block["block_id"],
                    "family_id": block["family_id"],
                    "mechanism_id": block["mechanism_id"],
                    "matched_control_block_id": block["matched_control_block_id"],
                    "routes": route_audits,
                    "descriptive_routes": descriptive,
                    "model_block_behavior_qualified": block_pass,
                    "physical_tested": False,
                    "predictive_tested": False,
                    "causal_tested": False,
                }
            )
    return audits


def cross_model_open_gate(
    audits: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    replication_min = int(
        protocol["registered_thresholds"]["cross_model_replication_min"]
    )
    authorized = []
    for block in (value for value in BLOCKS if value["candidate"]):
        routes = []
        route_models: dict[str, list[str]] = {}
        for route in ("source_only", "query_only"):
            models = [
                row["model"]
                for row in audits
                if row["block_id"] == block["block_id"]
                and row["routes"]["consistent"]["qualified_open"]
                and row["routes"][route]["qualified_open"]
            ]
            route_models[route] = models
            if len(models) >= replication_min:
                routes.append(route)
        if routes:
            authorized.append(
                {
                    "block_id": block["block_id"],
                    "matched_control_block_id": block["matched_control_block_id"],
                    "routes": routes,
                    "models_by_route": {route: route_models[route] for route in routes},
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_behavior_unlock": bool(authorized),
        "sealed_behavior_unlock_blocks": [row["block_id"] for row in authorized],
        "authorized_candidates": authorized,
        "cross_model_replication_min": replication_min,
        "physical_protocol_unlock": False,
        "physical_hooks_run": False,
        "thresholds_updated": False,
        "sealed_rows_read": False,
        "strict_human_double_blind": False,
    }


def load_completed_stage(stage: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    model_results = {}
    for model in MODELS:
        root = OUT / "models" / model / stage
        complete_path = root / "phase427_collection_complete.json"
        row_path = root / "phase427_behavior_rows.jsonl"
        if not complete_path.exists() or not row_path.exists():
            raise RuntimeError(f"Phase427 {stage} collection missing for {model}")
        complete = read_json(complete_path)
        model_rows = read_jsonl(row_path)
        gate = bool(
            complete.get("all_rows_complete")
            and complete.get("finite_sequence_scores")
            and complete.get("natural_parser_complete")
            and len(model_rows) == int(complete["condition_count"])
            and all(not row["physical"] and not row["causal"] for row in model_rows)
        )
        model_results[model] = {
            "condition_count": len(model_rows),
            "independent_group_count": complete["independent_group_count"],
            "gate_pass": gate,
            "execution_dtype": complete["execution_dtype"],
            "elapsed_seconds": complete["elapsed_seconds"],
        }
        rows.extend(model_rows)
    return rows, model_results


def instrument_audit() -> dict[str, Any]:
    protocol = read_json(OUT / "phase427_protocol.json")
    rows, model_results = load_completed_stage("instrument")
    expected_per_model = int(protocol["split_contract"]["instrument_condition_count"]) // len(MODELS)
    for model in MODELS:
        model_results[model]["expected_condition_count"] = expected_per_model
        model_results[model]["gate_pass"] = bool(
            model_results[model]["gate_pass"]
            and model_results[model]["condition_count"] == expected_per_model
        )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "instrument_gate_pass": all(row["gate_pass"] for row in model_results.values()),
        "executed_condition_count": len(rows),
        "expected_condition_count": protocol["split_contract"]["instrument_condition_count"],
        "model_results": model_results,
        "physical_hooks_installed": False,
        "sealed_rows_read": False,
        "thresholds_or_theory_updated": False,
    }
    write_json(OUT / "phase427_instrument_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["instrument_gate_pass"]:
        raise RuntimeError("Phase427 instrument gate failed")
    return audit


def graph_for_model(
    model: str,
    summaries: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    sealed_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    lookup = lookup_summaries(summaries)
    audit_lookup = {
        row["block_id"]: row for row in audits if row["model"] == model
    }
    nodes = []
    edges = []
    for block_index, block in enumerate(BLOCKS):
        for route_index, route in enumerate(ROUTE_MODES):
            summary = lookup[(model, block["block_id"], "behavior_holdout", route)]
            audit = audit_lookup.get(block["block_id"])
            route_pass = bool(
                audit
                and route in SCORABLE_CANDIDATE_ROUTES
                and audit["routes"][route]["qualified_open"]
            )
            node_id = f"phase427:{model}:{block['block_id']}:{route}"
            nodes.append(
                {
                    "id": node_id,
                    "label": f"{block['mechanism_id']} / {route}",
                    "type": "behavior_route_observation",
                    "model": model,
                    "block_id": block["block_id"],
                    "family_id": block["family_id"],
                    "mechanism_id": block["mechanism_id"],
                    "candidate": block["candidate"],
                    "route_mode": route,
                    "normative": summary["normative"],
                    "independent_group_count": summary["independent_group_count"],
                    "teacher_correct_fraction": summary["teacher_sequence_correct_fraction"],
                    "teacher_margin_median": summary["teacher_sequence_margin_median"],
                    "natural_target_first_fraction": summary["natural_target_first_fraction"],
                    "natural_opposite_first_fraction": summary["natural_opposite_first_fraction"],
                    "natural_revision_fraction": summary["natural_revision_fraction"],
                    "natural_stop_fraction": summary["natural_stop_fraction"],
                    "natural_censoring_fraction": summary["natural_censoring_fraction"],
                    "open_behavior_gate_pass": route_pass,
                    "sealed_behavior_tested": sealed_gate is not None,
                    "physical": False,
                    "observer": True,
                    "native_compute_event": False,
                    "compute_edge": False,
                    "predictive": False,
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "evidence_level": "independent_behavior_observation",
                    "score": max(0.05, float(summary["natural_target_first_fraction"])),
                    "size": 0.65 + 0.35 * float(route_pass),
                    "color": COLORS[route],
                    "position": [float(route_index * 8), float((3 - block_index) * 12), 0.0],
                    "show_label": True,
                }
            )
        for source_route in ("source_only", "query_only"):
            source = f"phase427:{model}:{block['block_id']}:{source_route}"
            target = f"phase427:{model}:{block['block_id']}:consistent"
            edges.append(
                {
                    "id": f"{source}->{target}",
                    "source": source,
                    "target": target,
                    "type": "experimental_condition_relation",
                    "physical": False,
                    "observer": True,
                    "compute_edge": False,
                    "predictive": False,
                    "causal": False,
                    "pipeline_sealed": False,
                    "strict_double_blind": False,
                    "evidence_level": "behavior_condition_comparison",
                    "color": "#64748b",
                    "weight": 1.0,
                }
            )
    return {
        "schema_version": "atlas_graph_v1",
        "phase_id": "Phase427-DualRouteBehaviorAtlas",
        "title": f"Phase427 {model} 双路线行为资格图谱",
        "model": model,
        "evidence_scope": (
            "independent teacher-forced and natural behavior qualification; "
            "no physical, predictive, causal, head, channel, or neuron claim"
        ),
        "graph": {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "phase": 427,
                "open_only": sealed_gate is None,
                "pipeline_sealed": False,
                "strict_double_blind": False,
                "physical_node_count": 0,
                "compute_edge_count": 0,
                "causal": False,
            },
        },
    }


def publish_visual(
    summaries: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    sealed_gate: dict[str, Any] | None = None,
) -> None:
    VIS.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase427_{model}_dual_route_behavior.json"
        write_json(VIS / filename, graph_for_model(model, summaries, audits, sealed_gate))
        items.append(
            {
                "id": f"phase427_{model}_dual_route_behavior",
                "label": f"Phase427 {model} 双路线行为资格图谱",
                "filename": filename,
                "model": model,
                "phase": 427,
                "evidence_scope": "independent behavior observation; non-physical and non-causal",
            }
        )
    write_json(
        VIS / "manifest.json",
        {
            "schema_version": "phase427_dual_route_behavior_manifest.v1",
            "generated_at": now(),
            "default_item_id": items[0]["id"],
            "items": items,
        },
    )
    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase427_dual_route_behavior",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase427 来源预绑定与查询门控行为资格",
        "description": "三模型双路线任务的完整教师序列、自然首答、改口、停止和截断分账。",
        "manifest_path": "/vis_data/phase427_dual_route_behavior/manifest.json",
        "manifest_schema": "phase427_dual_route_behavior_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase427_dual_route_behavior",
        "models": list(MODELS),
        "evidence_scope": "独立行为资格；非物理、非预测、非因果、非神经元机制",
        "color": "#0ea5e9",
    }
    registry["sources"] = [
        row for row in registry["sources"] if row["id"] != source["id"]
    ] + [source]
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)


def write_open_report(summary: dict[str, Any], gate: dict[str, Any]) -> None:
    report = [
        "# Phase427 双路线行为资格开放审计",
        "",
        f"- 正式注册条件：{summary['registered_formal_condition_count']}",
        f"- 已执行开放条件：{summary['executed_open_condition_count']}",
        f"- 独立开放语义组：{summary['executed_open_independent_group_count']}",
        f"- 跨模型行为候选：{summary['cross_model_behavior_candidate_count']}",
        f"- 密封行为解锁：{gate['sealed_behavior_unlock']}",
        "- 冲突和无角色条件仅作路线偏好描述，不计正确率。",
        "- 匹配负对照与候选必须同时通过。",
        "- 本阶段没有安装物理钩子，也没有产生预测边或因果边。",
        "",
        summary["conclusion"],
    ]
    (OUT / "phase427_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def analyze_open() -> dict[str, Any]:
    protocol = read_json(OUT / "phase427_protocol.json")
    rows, model_results = load_completed_stage("open")
    expected_per_model = int(protocol["split_contract"]["open_condition_count"]) // len(MODELS)
    if any(
        not result["gate_pass"] or result["condition_count"] != expected_per_model
        for result in model_results.values()
    ):
        raise RuntimeError("Phase427 open collection contract failed")
    summaries = route_summaries(rows)
    audits = build_open_audits(summaries, protocol)
    gate = cross_model_open_gate(audits, protocol)
    write_jsonl(OUT / "phase427_open_route_summaries.jsonl", summaries)
    write_jsonl(OUT / "phase427_open_candidate_audits.jsonl", audits)
    write_json(OUT / "phase427_open_gate_freeze.json", gate)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "registered_formal_condition_count": protocol["validation"]["formal_condition_count"],
        "registered_open_condition_count": protocol["validation"]["open_condition_count"],
        "registered_sealed_condition_count": protocol["validation"]["sealed_condition_count"],
        "executed_open_condition_count": len(rows),
        "executed_open_independent_group_count": sum(
            result["independent_group_count"] for result in model_results.values()
        ),
        "model_results": model_results,
        "route_position_mismatch_count": protocol["validation"]["route_position_mismatch_count"],
        "candidate_conflict_and_none_normative": False,
        "cross_model_behavior_candidate_count": len(gate["authorized_candidates"]),
        "sealed_behavior_unlock": gate["sealed_behavior_unlock"],
        "sealed_behavior_tested": False,
        "physical_tested": False,
        "predictive_tested": False,
        "causal_tested": False,
        "strict_mechanism_closure": "0/72",
        "overall_scientific_progress_percent": 21,
        "progress_interval_percent": [18, 24],
        "conclusion": (
            "At least one candidate route passed independent cross-model behavior qualification; only its pipeline-sealed behavior split is authorized."
            if gate["sealed_behavior_unlock"]
            else "No candidate route passed candidate, matched-control, calibration, holdout and cross-model behavior gates; sealed behavior and all physical stages remain closed."
        ),
    }
    write_json(OUT / "phase427_global_summary.json", summary)
    publish_visual(summaries, audits)
    write_open_report(summary, gate)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def analyze_sealed() -> dict[str, Any]:
    protocol = read_json(OUT / "phase427_protocol.json")
    open_gate = read_json(OUT / "phase427_open_gate_freeze.json")
    if not open_gate["sealed_behavior_unlock"]:
        raise RuntimeError("Phase427 sealed behavior was not authorized")
    rows, model_results = load_completed_stage("sealed")
    summaries = route_summaries(rows)
    lookup = lookup_summaries(summaries)
    thresholds = protocol["registered_thresholds"]
    candidates = []
    replication_min = int(thresholds["cross_model_replication_min"])
    for registered in open_gate["authorized_candidates"]:
        block_id = registered["block_id"]
        control_id = registered["matched_control_block_id"]
        routes = []
        models_by_route = {}
        for route in registered["routes"]:
            passing = []
            for model in registered["models_by_route"][route]:
                required = []
                for required_route in (route, "consistent"):
                    candidate_summary = lookup[(model, block_id, "sealed_behavior_holdout", required_route)]
                    control_summary = lookup[(model, control_id, "sealed_behavior_holdout", required_route)]
                    required.append(
                        behavior_gate(candidate_summary, thresholds)["gate_pass"]
                        and behavior_gate(control_summary, thresholds)["gate_pass"]
                    )
                if all(required):
                    passing.append(model)
            models_by_route[route] = passing
            if len(passing) >= replication_min:
                routes.append(route)
        if routes:
            candidates.append(
                {
                    "block_id": block_id,
                    "matched_control_block_id": control_id,
                    "routes": routes,
                    "models_by_route": {route: models_by_route[route] for route in routes},
                }
            )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "sealed_behavior_gate_pass": bool(candidates),
        "physical_protocol_unlock": bool(candidates),
        "authorized_physical_candidates": candidates,
        "cross_model_replication_min": replication_min,
        "sealed_rows_read": True,
        "physical_hooks_run": False,
        "predictive_tested": False,
        "causal_tested": False,
        "strict_human_double_blind": False,
    }
    write_jsonl(OUT / "phase427_sealed_route_summaries.jsonl", summaries)
    write_json(OUT / "phase427_behavior_gate_freeze.json", gate)
    global_summary = read_json(OUT / "phase427_global_summary.json")
    global_summary.update(
        {
            "created_at": now(),
            "executed_sealed_condition_count": len(rows),
            "sealed_behavior_tested": True,
            "sealed_behavior_gate_pass": gate["sealed_behavior_gate_pass"],
            "physical_protocol_unlock": gate["physical_protocol_unlock"],
            "conclusion": (
                "Pipeline-sealed cross-model behavior qualification passed; a separate frozen physical denominator is now permitted, but no physical mechanism has yet been tested."
                if gate["physical_protocol_unlock"]
                else "The pipeline-sealed behavior qualification failed; physical mapping and causal stages remain closed."
            ),
        }
    )
    write_json(OUT / "phase427_global_summary.json", global_summary)
    open_summaries = read_jsonl(OUT / "phase427_open_route_summaries.jsonl")
    open_audits = read_jsonl(OUT / "phase427_open_candidate_audits.jsonl")
    publish_visual(open_summaries, open_audits, gate)
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))
    return global_summary


def verify_frozen_implementation() -> None:
    protocol = read_json(OUT / "phase427_protocol.json")
    expected = protocol["implementation_commitments"][Path(__file__).name]
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("Phase427 analysis changed after protocol freeze")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("instrument", "open", "sealed"), required=True)
    args = parser.parse_args()
    verify_frozen_implementation()
    if args.stage == "instrument":
        instrument_audit()
    elif args.stage == "open":
        analyze_open()
    else:
        analyze_sealed()


if __name__ == "__main__":
    main()
