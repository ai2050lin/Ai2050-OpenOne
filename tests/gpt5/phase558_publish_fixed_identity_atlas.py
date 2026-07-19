#!/usr/bin/env python3
"""Publish the behavior-gated Phase558 fixed-identity atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase558_fixed_identity_color"
PUBLIC = ROOT / "frontend/public/vis_data/phase558_fixed_identity_color_atlas"
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


def graph(model: str, behavior: dict[str, Any], failure: dict[str, Any]) -> dict[str, Any]:
    discovery = behavior["split_reports"]["behavior_discovery"]
    confirmation = behavior["split_reports"]["behavior_confirmation"]
    nodes = [
        {
            "id": f"phase558:{model}:fixed_identity_contract",
            "label": f"{LABELS[model]} / 固定对象颜色配对32格合同",
            "model": model,
            "family_id": "semantic_knowledge",
            "mechanism_id": "fixed_identity_color_binding",
            "type": "behavior_contract",
            "position": [-36.0, 0.0, 0.0],
            "score": float(min(discovery["all_32_correct_world_rate"], confirmation["all_32_correct_world_rate"])),
            "color": "#f59e0b",
            "observer": True,
            "physical": False,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "show_label": True,
            "discovery_world_all32_rate": discovery["all_32_correct_world_rate"],
            "confirmation_world_all32_rate": confirmation["all_32_correct_world_rate"],
            "discovery_min_cell_lcb": discovery["minimum_cell_wilson_95_lcb"],
            "confirmation_min_cell_lcb": confirmation["minimum_cell_wilson_95_lcb"],
            "behavior_gate_pass": False,
        },
        {
            "id": f"phase558:{model}:failure_boundary",
            "label": f"{LABELS[model]} / 行为门停止，内部采集未授权",
            "model": model,
            "family_id": "semantic_knowledge",
            "mechanism_id": "fixed_identity_color_binding",
            "type": "physical_collection_stop",
            "position": [0.0, 0.0, 0.0],
            "score": 0.0,
            "color": "#ef4444",
            "observer": True,
            "physical": False,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "show_label": True,
            "failure_count": failure["failure_count"],
            "failure_rate": failure["failure_rate"],
            "failure_event_counts": failure["failure_event_counts"],
            "stop_reason": "frozen_behavior_gate_failed",
        },
        {
            "id": f"phase558:{model}:next_boundary",
            "label": f"{LABELS[model]} / 仅允许独立大样本行为复制",
            "model": model,
            "family_id": "semantic_knowledge",
            "mechanism_id": "fixed_identity_color_binding",
            "type": "next_test_boundary",
            "position": [36.0, 0.0, 0.0],
            "score": 0.0,
            "color": "#64748b",
            "observer": True,
            "physical": False,
            "causal": False,
            "compute_edge": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "show_label": True,
            "next_action": "independent_large_denominator_replication_without_gate_change",
        },
    ]
    edges = [
        {
            "id": f"phase558:{model}:gate_stop",
            "source": f"phase558:{model}:fixed_identity_contract",
            "target": f"phase558:{model}:failure_boundary",
            "type": "authorization_stop",
            "label": "冻结行为门失败",
            "score": 1.0,
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "evidence_level": "behavior",
        },
        {
            "id": f"phase558:{model}:stop_next",
            "source": f"phase558:{model}:failure_boundary",
            "target": f"phase558:{model}:next_boundary",
            "type": "preregistered_next_step",
            "label": "新阶段独立复制",
            "score": 1.0,
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "evidence_level": "protocol",
        },
    ]
    return {
        "schema_version": "phase558_fixed_identity_color_atlas.v1",
        "phase_id": "Phase558",
        "generated_at": now(),
        "model": model,
        "evidence_scope": "fixed-identity behavior only; no physical collection or causal edge",
        "graph": {
            "meta": {
                "model": model,
                "open_case_count": 9216,
                "sealed_case_count_unread": 2048,
                "internal_collection_authorized": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def publish() -> None:
    behavior_summary = read_json(RESULT / "phase558_behavior_summary.json")
    failure_audit = read_json(RESULT / "phase558_failure_audit.json")
    behavior = {row["model"]: row for row in behavior_summary["model_reports"]}
    failures = {row["model"]: row for row in failure_audit["model_reports"]}
    items = []
    for model in MODELS:
        filename = f"phase558_{model}_fixed_identity_color.json"
        payload = graph(model, behavior[model], failures[model])
        write_json(PUBLIC / filename, payload)
        write_json(RESULT / "atlas" / filename, payload)
        items.append({
            "id": f"phase558_{model}",
            "model": model,
            "label": f"{LABELS[model]} 固定身份颜色绑定行为审计",
            "path": filename,
        })
    manifest = {
        "schema_version": "phase558_fixed_identity_color_atlas_manifest.v1",
        "generated_at": now(),
        "route_id": "gpt5",
        "evidence_scope": "固定对象身份颜色绑定行为门；三模型内部采集均未授权",
        "items": items,
    }
    write_json(PUBLIC / "manifest.json", manifest)
    write_json(RESULT / "atlas" / "manifest.json", manifest)

    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase558_fixed_identity_color_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase558 固定身份颜色绑定行为审计",
        "description": "固定对象与颜色集合、只翻转对象－颜色配对的三模型大样本行为门。",
        "manifest_path": "/vis_data/phase558_fixed_identity_color_atlas/manifest.json",
        "manifest_schema": "phase558_fixed_identity_color_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase558_fixed_identity_color_atlas",
        "models": list(MODELS),
        "evidence_scope": "行为资格与停止边界；不包含物理状态、因果边、参数或神经元证据",
        "color": "#f59e0b",
    }
    positions = {row["id"]: index for index, row in enumerate(registry["sources"])}
    if source["id"] in positions:
        registry["sources"][positions[source["id"]]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    write_json(RESULT / "phase558_atlas_publish_summary.json", {
        "schema_version": "phase558_atlas_publish_summary.v1",
        "phase_id": "Phase558",
        "created_at": now(),
        "source_id": source["id"],
        "model_count": len(MODELS),
        "authorized_internal_model_count": 0,
        "physical_node_count": 0,
        "causal_edge_count": 0,
        "strict_closed_mechanisms": 0,
        "sealed_split_read": False,
    })
    print(RESULT / "phase558_atlas_publish_summary.json")


if __name__ == "__main__":
    publish()
