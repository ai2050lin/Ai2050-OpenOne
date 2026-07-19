#!/usr/bin/env python3
"""Publish Phase564-565 source-edge and residual-operator evidence."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PARENT_PUBLIC = ROOT / "frontend/public/vis_data/phase563_fixed_identity_color_route_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase565_fixed_identity_color_residual_atlas"
RESULT_ATLAS = ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator/atlas"
PHASE564_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
PHASE565_DIR = ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator"
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


def publish() -> Path:
    behavior = read_json(PHASE564_DIR / "phase564_behavior_summary.json")
    edge = read_json(PHASE564_DIR / "phase564_source_edge_confirmation_analysis.json")
    residual = read_json(PHASE565_DIR / "phase565_residual_operator_analysis.json")
    behavior_by_model = {row["model"]: row for row in behavior["model_reports"]}
    items = []
    for model in MODELS:
        parent_name = f"phase563_{model}_fixed_identity_color_route.json"
        payload = copy.deepcopy(read_json(PARENT_PUBLIC / parent_name))
        payload["schema_version"] = "phase565_fixed_identity_color_residual_atlas.v1"
        payload["phase_id"] = "Phase565"
        payload["generated_at"] = now()
        graph = payload["graph"]
        report = behavior_by_model[model]
        graph["meta"].update({
            "phase564_independent_behavior_accuracy": report["semantic_accuracy"],
            "phase564_internal_authorized": report["authorized_for_edge_behavior"],
            "phase564_source_edge_compute_count": 0,
            "phase565_residual_operator_count": (
                residual["qualified_operator_count"] if model == "qwen3" else 0
            ),
            "strict_closed_mechanisms": 0,
        })
        behavior_node_id = f"phase564:{model}:independent_behavior"
        graph["nodes"].append({
            "id": behavior_node_id,
            "label": (
                f"独立行为门 {'通过' if report['authorized_for_edge_behavior'] else '停止'} / "
                f"{report['semantic_accuracy']:.4f}"
            ),
            "model": model,
            "family_id": "semantic_knowledge",
            "mechanism_id": "fixed_identity_color_binding",
            "type": "behavior_authorization" if report["authorized_for_edge_behavior"] else "behavior_stop",
            "position": [-32.0, -10.0, 0.0],
            "layer": -1,
            "component": "behavior_contract",
            "score": float(report["semantic_accuracy"]),
            "color": "#15803d" if report["authorized_for_edge_behavior"] else "#d97706",
            "observer": True,
            "physical": False,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "show_label": True,
            "authorized_for_internal_collection": report["authorized_for_edge_behavior"],
        })
        if model == "qwen3":
            for edge_report in edge["candidate_reports"]:
                layer = int(edge_report["layer"])
                target_role = edge_report["target_role"]
                node_id = f"phase564:qwen3:source_edge:L{layer}:{target_role}"
                donor = edge_report["conditions"]["paired_donor_edge_replace"]
                graph["nodes"].append({
                    "id": node_id,
                    "label": f"L{layer} 来源颜色值边否定 / {target_role}",
                    "model": "qwen3",
                    "family_id": "semantic_knowledge",
                    "mechanism_id": "fixed_identity_color_binding",
                    "type": "rejected_source_conditioned_value_edge",
                    "position": [float(layer * 2 - 36), 18.0 if target_role == "query_object_end" else 21.0, 0.0],
                    "layer": layer,
                    "component": "aggregate_attention_source_contribution",
                    "score": float(donor["donor_win_rate"]),
                    "color": "#dc2626",
                    "observer": True,
                    "physical": True,
                    "causal": True,
                    "compute_edge": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "show_label": True,
                    "negative_evidence": True,
                    "donor_win_rate": donor["donor_win_rate"],
                    "mean_donor_switch_effect": donor["mean_donor_switch_effect"],
                    "post_softmax_value_contribution_only": True,
                    "key_effect_identified": False,
                })
                graph["edges"].append({
                    "id": f"phase564:qwen3:source_L3_to_value_edge:L{layer}:{target_role}",
                    "source": "phase562:qwen3:source_color:L3",
                    "target": node_id,
                    "type": "rejected_source_value_contribution",
                    "label": "独立发现与确认均未支持聚合来源值边",
                    "score": float(donor["donor_win_rate"]),
                    "predictive": False,
                    "causal": True,
                    "compute_edge": False,
                    "negative_evidence": True,
                    "evidence_level": "independent_causal_rejection",
                })
            for operator in residual["candidate_reports"]:
                layer = int(operator["layer"])
                block = operator["position_block"]
                donor = operator["conditions"]["paired_donor_residual_replace"]
                node_id = f"phase565:qwen3:residual:L{layer}:{block}"
                graph["nodes"].append({
                    "id": node_id,
                    "label": f"L{layer} {'七角色' if block == 'semantic7' else '完整序列'}残差充分性",
                    "model": "qwen3",
                    "family_id": "semantic_knowledge",
                    "mechanism_id": "fixed_identity_color_binding",
                    "type": "distributed_residual_state_sufficiency",
                    "position": [float(layer * 2 - 36), 38.0 if block == "semantic7" else 44.0, 0.0],
                    "layer": layer,
                    "component": "layer_output_residual",
                    "score": float(donor["donor_win_rate"]),
                    "color": "#0f766e" if block == "semantic7" else "#2563eb",
                    "observer": False,
                    "physical": True,
                    "causal": True,
                    "compute_edge": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "show_label": True,
                    "donor_win_rate": donor["donor_win_rate"],
                    "minimum_factorial_cell_donor_win_rate": donor["minimum_factorial_cell_donor_win_rate"],
                    "minimum_color_regime_donor_win_rate": donor["minimum_color_regime_donor_win_rate"],
                    "mean_donor_switch_effect": donor["mean_donor_switch_effect"],
                    "distributed_state_sufficiency_only": True,
                    "natural_necessity_tested": False,
                })
                graph["edges"].append({
                    "id": f"phase565:qwen3:behavior_to_residual:L{layer}:{block}",
                    "source": behavior_node_id,
                    "target": node_id,
                    "type": "distributed_state_sufficiency",
                    "label": "独立未见世界中的残差块供体充分性",
                    "score": float(donor["donor_win_rate"]),
                    "predictive": False,
                    "causal": True,
                    "compute_edge": False,
                    "negative_evidence": False,
                    "evidence_level": "distributed_state_sufficiency",
                })
        payload["evidence_scope"] = (
            "Phase564 independent three-model behavior authorization and rejected aggregate source-color "
            "post-softmax value edges; Phase565 typed/full-sequence residual-state sufficiency; "
            "no compute-edge, natural-necessity, parameter, or neuron closure"
        )
        filename = f"phase565_{model}_fixed_identity_color_residual.json"
        write_json(PUBLIC / filename, payload)
        write_json(RESULT_ATLAS / filename, payload)
        items.append({
            "id": f"phase565_{model}",
            "model": model,
            "label": f"{LABELS[model]} 固定身份颜色来源边与残差路线",
            "path": filename,
        })
    manifest = {
        "schema_version": "phase565_fixed_identity_color_residual_atlas_manifest.v1",
        "generated_at": now(),
        "route_id": "gpt5",
        "evidence_scope": "来源颜色值贡献边否定与分布式残差状态充分性",
        "items": items,
    }
    write_json(PUBLIC / "manifest.json", manifest)
    write_json(RESULT_ATLAS / "manifest.json", manifest)

    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase565_fixed_identity_color_residual_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase564-565 固定身份颜色来源边与残差路线",
        "description": "三模型独立行为门、聚合来源颜色值边否定、七角色及完整序列残差状态充分性。",
        "manifest_path": "/vis_data/phase565_fixed_identity_color_residual_atlas/manifest.json",
        "manifest_schema": "phase565_fixed_identity_color_residual_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase565_fixed_identity_color_residual_atlas",
        "models": list(MODELS),
        "evidence_scope": "分布式状态充分性；不包含精确计算边、自然必要性、参数或神经元闭合",
        "color": "#0f766e",
    }
    positions = {row["id"]: index for index, row in enumerate(registry["sources"])}
    if source["id"] in positions:
        registry["sources"][positions[source["id"]]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    summary_path = PHASE565_DIR / "phase565_atlas_publish_summary.json"
    write_json(summary_path, {
        "schema_version": "phase565_atlas_publish_summary.v1",
        "phase_id": "Phase565",
        "created_at": now(),
        "source_id": source["id"],
        "model_count": len(MODELS),
        "rejected_source_edge_count": edge["candidate_count"],
        "qualified_residual_operator_count": residual["qualified_operator_count"],
        "compute_edge_count": 0,
        "single_neuron_node_count": 0,
        "strict_closed_mechanisms": 0,
        "sealed_split_read": False,
    })
    print(summary_path)
    return summary_path


if __name__ == "__main__":
    publish()
