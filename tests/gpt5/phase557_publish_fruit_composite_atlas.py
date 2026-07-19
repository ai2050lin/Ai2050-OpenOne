#!/usr/bin/env python3
"""Publish the evidence-preserving Phase557 fruit composite atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase557_fruit_composite"
PUBLIC = ROOT / "frontend/public/vis_data/phase557_fruit_composite_atlas"
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def node(model: str, suffix: str, label: str, node_type: str, x: float, y: float, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"phase557:{model}:{suffix}",
        "label": f"{LABELS[model]} / {label}",
        "model": model,
        "family_id": "semantic_knowledge",
        "mechanism_id": "fruit_object_category_attribute_binding",
        "type": node_type,
        "position": [x, y, 0.0],
        "score": float(extra.pop("score", 0.0)),
        "color": extra.pop("color", "#64748b"),
        "observer": bool(extra.get("observer", True)),
        "physical": bool(extra.get("physical", False)),
        "predictive": bool(extra.get("predictive", False)),
        "causal": bool(extra.get("causal", False)),
        "compute_edge": bool(extra.get("compute_edge", False)),
        "single_neuron": False,
        "pipeline_sealed": False,
        "show_label": True,
        **extra,
    }


def edge(model: str, suffix: str, source: str, target: str, relation: str, causal: bool = False) -> dict[str, Any]:
    return {
        "id": f"phase557:{model}:{suffix}",
        "source": f"phase557:{model}:{source}",
        "target": f"phase557:{model}:{target}",
        "type": relation,
        "label": relation,
        "score": 1.0,
        "predictive": causal,
        "causal": causal,
        "compute_edge": causal,
        "evidence_level": "source_recompute_unseen_replication" if causal else "behavior_or_observer",
    }


def graph(model: str, behavior: dict[str, Any], replicated: list[dict[str, Any]], upstream: dict[str, Any]) -> dict[str, Any]:
    discovery = behavior["controlled_split_reports"]["behavior_discovery"]
    confirmation = behavior["controlled_split_reports"]["behavior_confirmation"]
    contextual_rate = min(
        discovery["world_all_32_correct_rate"], confirmation["world_all_32_correct_rate"]
    )
    color_authorized = "color" in behavior["authorized_natural_relations"]
    nodes = [
        node(
            model, "contextual_gate", f"受控四因素32格门 {contextual_rate:.3f}",
            "query_readout_event", -48, 12,
            score=contextual_rate, color="#ef4444", observer=True,
            behavior_gate_pass=False, sample_count=3072,
            stop_reason="contextual_four_factor_gate_failed",
        ),
        node(
            model, "natural_color_gate", "自然颜色关系行为门" + ("通过" if color_authorized else "失败"),
            "natural_knowledge_behavior", -48, -8,
            score=1.0 if color_authorized else 0.0,
            color="#22c55e" if color_authorized else "#ef4444",
            observer=True, behavior_gate_pass=color_authorized,
        ),
        node(
            model, "answer", "受限颜色答案边界", "query_readout_event", 48, 0,
            score=float(behavior["semantic_accuracy"]), color="#38bdf8", observer=True,
        ),
        node(
            model, "encoding_stop", "尚未分离颜色编码与完整对象身份", "physical_collection_stop", 24, 19,
            score=0.0, color="#ef4444", observer=True,
            physical=False, causal=False, compute_edge=False,
            stop_reason="full_lexical_identity_confound",
        ),
    ]
    edges = [
        edge(model, "contextual_stop", "contextual_gate", "encoding_stop", "authorization_stop"),
        edge(model, "color_behavior", "natural_color_gate", "answer", "behavior_measurement"),
    ]
    for index, item in enumerate(sorted(replicated, key=lambda row: int(row["layer"]))):
        layer = int(item["layer"])
        suffix = f"source_L{layer}"
        nodes.append(node(
            model, suffix, f"对象词元完整状态 → 颜色读出 L{layer}",
            "object_identity_transport_event", -20 + layer * 1.5, -1 - index * 7,
            score=1.0, color="#22c55e", observer=True, physical=True,
            predictive=True, causal=True, compute_edge=True,
            source_position="object_source_end",
            evidence_level="confirmation_and_unseen_source_recompute",
            interpretation="完整对象身份运输；不是独立颜色向量",
        ))
        edges.append(edge(model, f"gate_source_{index}", "natural_color_gate", suffix, "authorized_source_probe"))
        edges.append(edge(model, f"source_answer_{index}", suffix, "answer", "coarse_object_state_compute_edge", True))
    model_upstream = next((row for row in upstream["model_reports"] if row["model"] == model), None)
    if model_upstream and model_upstream["embedding_boundary_reached"]:
        nodes.append(node(
            model, "lexical_boundary", "L0 词汇身份输入边界（非颜色存储）",
            "lexical_identity_source", -35, -24,
            score=1.0, color="#06b6d4", observer=True, physical=True,
            predictive=True, causal=True, compute_edge=False,
            replicated_layers=model_upstream["replicated_layer_input_edges"],
            interpretation="替换完整词汇身份后，网络重新执行供体对象知识路径",
        ))
        for index, item in enumerate(sorted(replicated, key=lambda row: int(row["layer"]))):
            edges.append(edge(
                model, f"lexical_transport_{index}", "lexical_boundary", f"source_L{int(item['layer'])}",
                "residual_identity_transport", True,
            ))
    return {
        "schema_version": "phase557_fruit_composite_atlas.v1",
        "phase_id": "Phase557",
        "generated_at": now(),
        "model": model,
        "evidence_scope": (
            "frozen contextual behavior gate, natural-color observer, source recompute and unseen "
            "replication; not isolated color/category/binding code or neuron closure"
        ),
        "graph": {
            "meta": {
                "model": model,
                "registered_case_count": 9728,
                "open_case_count": 8064,
                "sealed_case_count_unread": 1664,
                "contextual_internal_collection_authorized": False,
                "natural_color_internal_collection_authorized": color_authorized,
                "replicated_coarse_compute_edge_count": len(replicated),
                "replicated_writer_parent_count": 0,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def publish() -> None:
    behavior_summary = read_json(RESULT / "phase557_behavior_summary.json")
    behavior = {row["model"]: row for row in behavior_summary["model_reports"]}
    replicated = read_jsonl(RESULT / "phase557_replicated_natural_color_compute_edges.jsonl")
    upstream = read_json(RESULT / "phase557_natural_color_upstream_analysis.json")
    items = []
    for model in MODELS:
        model_edges = [row for row in replicated if row["model"] == model]
        filename = f"phase557_{model}_fruit_composite.json"
        payload = graph(model, behavior[model], model_edges, upstream)
        write_json(PUBLIC / filename, payload)
        write_json(RESULT / "atlas" / filename, payload)
        items.append({
            "id": f"phase557_{model}",
            "model": model,
            "label": f"{LABELS[model]} 水果复合编码审计",
            "path": filename,
        })
    manifest = {
        "schema_version": "phase557_fruit_composite_atlas_manifest.v1",
        "generated_at": now(),
        "route_id": "gpt5",
        "evidence_scope": "上下文四因素门失败；自然颜色对象身份粗边复制；未分离颜色编码",
        "items": items,
    }
    write_json(PUBLIC / "manifest.json", manifest)
    write_json(RESULT / "atlas" / "manifest.json", manifest)

    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase557_fruit_composite_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase557 水果对象－类别－属性－绑定复合审计",
        "description": "三模型冻结行为门、自然颜色多位置轨迹、对象来源重计算与L0身份边界。",
        "manifest_path": "/vis_data/phase557_fruit_composite_atlas/manifest.json",
        "manifest_schema": "phase557_fruit_composite_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase557_fruit_composite_atlas",
        "models": list(MODELS),
        "evidence_scope": "自然颜色粗对象身份路径；不等于颜色/类别/绑定编码、参数支持或闭合",
        "color": "#06b6d4",
    }
    positions = {row["id"]: index for index, row in enumerate(registry["sources"])}
    if source["id"] in positions:
        registry["sources"][positions[source["id"]]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)
    write_json(RESULT / "phase557_atlas_publish_summary.json", {
        "schema_version": "phase557_atlas_publish_summary.v1",
        "phase_id": "Phase557",
        "created_at": now(),
        "source_id": source["id"],
        "model_count": len(MODELS),
        "replicated_coarse_compute_edge_count": len(replicated),
        "replicated_writer_parent_count": 0,
        "strict_closed_mechanisms": 0,
        "sealed_split_read": False,
    })
    print(RESULT / "phase557_atlas_publish_summary.json")


if __name__ == "__main__":
    publish()
