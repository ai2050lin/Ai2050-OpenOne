#!/usr/bin/env python3
"""Publish the Phase559-562 fixed-identity color route evidence atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE559 = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
PHASE560 = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PHASE561 = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
PUBLIC = ROOT / "frontend/public/vis_data/phase562_fixed_identity_color_route_atlas"
RESULT_ATLAS = PHASE561 / "atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱GLM4）",
    "deepseek7b": "DS7B（深度求索7B）",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def base_node(model: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"phase562:{model}:behavior_contract",
        "label": f"{LABELS[model]} / 固定身份颜色绑定合同",
        "model": model,
        "family_id": "semantic_knowledge",
        "mechanism_id": "fixed_identity_color_binding",
        "type": "behavior_contract",
        "position": [-48.0, 0.0, 0.0],
        "score": float(report["semantic_accuracy"]),
        "color": "#f59e0b",
        "observer": True,
        "physical": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "show_label": True,
        "semantic_accuracy": report["semantic_accuracy"],
        "failure_count": report["failure_count"],
        "behavior_gate_pass": report["authorized_for_path_behavior"],
    }


def stopped_graph(model: str, report: dict[str, Any]) -> dict[str, Any]:
    contract = base_node(model, report)
    stop = {
        "id": f"phase562:{model}:internal_stop",
        "label": f"{LABELS[model]} / 内部采集未授权",
        "model": model,
        "family_id": "semantic_knowledge",
        "mechanism_id": "fixed_identity_color_binding",
        "type": "physical_collection_stop",
        "position": [-24.0, 0.0, 0.0],
        "score": 0.0,
        "color": "#ef4444",
        "observer": True,
        "physical": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "show_label": True,
        "stop_reason": "frozen_behavior_gate_failed",
    }
    return {
        "schema_version": "phase562_fixed_identity_color_route_atlas.v1",
        "phase_id": "Phase562",
        "generated_at": now(),
        "model": model,
        "evidence_scope": "behavior stop only; no internal state was collected",
        "graph": {
            "meta": {
                "model": model,
                "internal_collection_authorized": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": [contract, stop],
            "edges": [{
                "id": f"phase562:{model}:behavior_stop",
                "source": contract["id"],
                "target": stop["id"],
                "type": "authorization_stop",
                "label": "冻结行为门失败",
                "score": 1.0,
                "predictive": False,
                "causal": False,
                "compute_edge": False,
                "evidence_level": "behavior",
            }],
        },
    }


def qwen_graph(
    behavior: dict[str, Any],
    source: dict[str, Any],
    parent: dict[str, Any],
    trace: dict[str, Any],
    reader: dict[str, Any],
) -> dict[str, Any]:
    nodes = [base_node("qwen3", behavior)]
    edges: list[dict[str, Any]] = []
    parent_by_layer = {row["layer"]: row for row in parent["candidate_reports"]}
    source_ids = []
    for report in sorted(source["candidate_reports"], key=lambda row: int(row["layer"])):
        layer = int(report["layer"])
        node_id = f"phase562:qwen3:source_color:L{layer}"
        source_ids.append(node_id)
        parent_report = parent_by_layer[layer]
        nodes.append({
            "id": node_id,
            "label": f"来源颜色词元 L{layer} / 粗因果内容截面",
            "model": "qwen3",
            "family_id": "semantic_knowledge",
            "mechanism_id": "fixed_identity_color_binding",
            "type": "coarse_source_color_causal_control",
            "position": [float(layer * 2 - 36), 18.0, 0.0],
            "layer": layer,
            "semantic_position": "source_color_end",
            "component": "layer_output",
            "score": float(report["correct_donor_win_rate"]),
            "color": "#16a34a",
            "observer": False,
            "physical": True,
            "causal": True,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "show_label": True,
            "correct_donor_win_rate": report["correct_donor_win_rate"],
            "minimum_factorial_cell_donor_win_rate": report[
                "minimum_factorial_cell_donor_win_rate"
            ],
            "heldout_donor_win_rate": report["heldout_donor_win_rate"],
            "mean_switch_effect": report["correct_donor_mean_switch_effect"],
            "residual_carry_dominant": parent_report["residual_carry_dominant"],
            "unique_local_writer": parent_report["current_layer_is_unique_color_writer"],
            "binding_operation_identified": False,
        })
        edges.append({
            "id": f"phase562:qwen3:contract_to_source_L{layer}",
            "source": "phase562:qwen3:behavior_contract",
            "target": node_id,
            "type": "behavior_authorized_physical_test",
            "label": "行为资格后独立因果验证",
            "score": float(report["correct_donor_win_rate"]),
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "evidence_level": "protocol",
        })
    for left, right in zip(source_ids, source_ids[1:]):
        edges.append({
            "id": f"phase562:qwen3:{left.rsplit(':', 1)[-1]}_to_{right.rsplit(':', 1)[-1]}",
            "source": left,
            "target": right,
            "type": "same_route_residual_cross_section",
            "label": "多个深度均可控制，深度不唯一",
            "score": 1.0,
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "evidence_level": "parent_diagnostic",
        })

    trace_specs = (
        ("query_onset", "query_object_end", trace["position_reports"]["query_object_end"]["first_causal_onset"], 0.0),
        ("answer_onset", "answer_boundary", trace["position_reports"]["answer_boundary"]["first_causal_onset"], -18.0),
        ("answer_stable", "answer_boundary", trace["position_reports"]["answer_boundary"]["first_stable_integration"], -18.0),
    )
    trace_ids = []
    for name, semantic_position, report, y in trace_specs:
        layer = int(report["layer"])
        node_id = f"phase562:qwen3:{name}:L{layer}"
        trace_ids.append(node_id)
        nodes.append({
            "id": node_id,
            "label": f"{semantic_position} L{layer} 注意力 / 干预传播响应",
            "model": "qwen3",
            "family_id": "semantic_knowledge",
            "mechanism_id": "fixed_identity_color_binding",
            "type": "intervention_conditioned_propagation_observation",
            "position": [float(layer * 2 - 36), y, 0.0],
            "layer": layer,
            "semantic_position": semantic_position,
            "component": report["component"],
            "score": float(report["mean_causal_projection_to_natural"]),
            "color": "#0891b2",
            "observer": True,
            "physical": True,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "show_label": True,
            "causal_to_natural_norm_ratio": report["mean_causal_to_natural_norm_ratio"],
            "causal_projection_to_natural": report["mean_causal_projection_to_natural"],
            "minimum_cell_norm_ratio": report[
                "minimum_factorial_cell_causal_to_natural_norm_ratio"
            ],
            "reader_validation_pass": False,
        })
        edges.append({
            "id": f"phase562:qwen3:source_L3_to_{name}_L{layer}",
            "source": "phase562:qwen3:source_color:L3",
            "target": node_id,
            "type": "intervention_conditioned_propagation",
            "label": "来源干预后的响应出现，非已验证运输边",
            "score": float(report["mean_causal_projection_to_natural"]),
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "evidence_level": "intervention_conditioned_observation",
        })

    output_id = "phase562:qwen3:restricted_color_readout"
    nodes.append({
        "id": output_id,
        "label": "受控颜色答案读出 / 来源状态可控制",
        "model": "qwen3",
        "family_id": "semantic_knowledge",
        "mechanism_id": "fixed_identity_color_binding",
        "type": "restricted_color_readout",
        "position": [36.0, -18.0, 0.0],
        "score": float(trace["source_patch_donor_win_rate"]),
        "color": "#0f766e",
        "observer": False,
        "physical": True,
        "causal": True,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "show_label": True,
        "source_patch_donor_win_rate": trace["source_patch_donor_win_rate"],
    })
    edges.append({
        "id": "phase562:qwen3:source_L3_to_readout",
        "source": "phase562:qwen3:source_color:L3",
        "target": output_id,
        "type": "coarse_causal_control",
        "label": "完整来源颜色状态替换控制答案",
        "score": float(trace["source_patch_donor_win_rate"]),
        "predictive": False,
        "causal": True,
        "compute_edge": False,
        "evidence_level": "coarse_causal_control",
    })

    reader_by_key = {
        (row["semantic_position"], int(row["layer"])): row
        for row in reader["candidate_reports"]
    }
    for node_id in trace_ids:
        node = next(row for row in nodes if row["id"] == node_id)
        report = reader_by_key[(node["semantic_position"], int(node["layer"]))]
        edges.append({
            "id": f"{node_id}:reader_rejected",
            "source": node_id,
            "target": output_id,
            "type": "rejected_static_reader_candidate",
            "label": "单位置状态替换未控制答案",
            "score": float(report["correct_donor_win_rate"]),
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "negative_evidence": True,
            "evidence_level": "independent_causal_rejection",
            "correct_donor_mean_switch_effect": report["correct_donor_mean_switch_effect"],
        })

    return {
        "schema_version": "phase562_fixed_identity_color_route_atlas.v1",
        "phase_id": "Phase562",
        "generated_at": now(),
        "model": "qwen3",
        "evidence_scope": (
            "fixed-identity source-color coarse causal control, intervention-conditioned propagation, "
            "and independently rejected static readers; no binding operator or compute closure"
        ),
        "graph": {
            "meta": {
                "model": "qwen3",
                "internal_collection_authorized": True,
                "coarse_source_color_edge_count": source["qualified_coarse_edge_count"],
                "qualified_static_reader_edge_count": reader["qualified_reader_edge_count"],
                "binding_operation_identified": False,
                "head_channel_parameter_neuron_scan_executed": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def publish() -> None:
    behavior_summary = read_json(PHASE559 / "phase559_behavior_summary.json")
    behavior = {row["model"]: row for row in behavior_summary["model_reports"]}
    source = read_json(PHASE560 / "phase560_semantic_color_unseen_analysis.json")
    parent = read_json(PHASE560 / "phase560_parent_decomposition_analysis.json")
    trace = read_json(PHASE561 / "phase561_source_to_query_trace_analysis.json")
    reader = read_json(PHASE561 / "phase562_reader_validation_analysis.json")
    items = []
    for model in MODELS:
        filename = f"phase562_{model}_fixed_identity_color_route.json"
        payload = (
            qwen_graph(behavior[model], source, parent, trace, reader)
            if model == "qwen3"
            else stopped_graph(model, behavior[model])
        )
        write_json(PUBLIC / filename, payload)
        write_json(RESULT_ATLAS / filename, payload)
        items.append({
            "id": f"phase562_{model}",
            "model": model,
            "label": f"{LABELS[model]} 固定身份颜色来源－查询路线",
            "path": filename,
        })
    manifest = {
        "schema_version": "phase562_fixed_identity_color_route_atlas_manifest.v1",
        "generated_at": now(),
        "route_id": "gpt5",
        "evidence_scope": (
            "固定身份颜色合同、来源颜色粗因果截面、干预传播观测与静态读出否定证据"
        ),
        "items": items,
    }
    write_json(PUBLIC / "manifest.json", manifest)
    write_json(RESULT_ATLAS / "manifest.json", manifest)

    registry = read_json(REGISTRY)
    source_entry = {
        "id": "gpt5_phase562_fixed_identity_color_route_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase559-562 固定身份颜色因果路线",
        "description": "固定对象身份下的颜色来源状态控制、跨位置传播观测及单位置读出否定结果。",
        "manifest_path": "/vis_data/phase562_fixed_identity_color_route_atlas/manifest.json",
        "manifest_schema": "phase562_fixed_identity_color_route_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase562_fixed_identity_color_route_atlas",
        "models": list(MODELS),
        "evidence_scope": "粗因果控制与传播观测分级；不包含绑定算子、计算闭合、参数或神经元证据",
        "color": "#16a34a",
    }
    positions = {row["id"]: index for index, row in enumerate(registry["sources"])}
    if source_entry["id"] in positions:
        registry["sources"][positions[source_entry["id"]]] = source_entry
    else:
        registry["sources"].append(source_entry)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    write_json(PHASE561 / "phase562_atlas_publish_summary.json", {
        "schema_version": "phase562_atlas_publish_summary.v1",
        "phase_id": "Phase562",
        "created_at": now(),
        "source_id": source_entry["id"],
        "model_count": len(MODELS),
        "authorized_internal_model_count": 1,
        "coarse_causal_control_edge_count": 1,
        "qualified_source_cross_section_count": source["qualified_coarse_edge_count"],
        "intervention_conditioned_propagation_edge_count": 3,
        "rejected_static_reader_edge_count": len(reader["candidate_reports"]),
        "qualified_static_reader_edge_count": reader["qualified_reader_edge_count"],
        "compute_edge_count": 0,
        "single_neuron_node_count": 0,
        "strict_closed_mechanisms": 0,
        "sealed_split_read": False,
    })
    print(PHASE561 / "phase562_atlas_publish_summary.json")


if __name__ == "__main__":
    publish()
