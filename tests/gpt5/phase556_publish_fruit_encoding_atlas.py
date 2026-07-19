#!/usr/bin/env python3
"""Publish the evidence-preserving Phase556 fruit encoding atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
PUBLIC = ROOT / "frontend/public/vis_data/phase556_fruit_encoding_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LABELS = {
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


def behavior_by_model() -> dict[str, dict[str, Any]]:
    summary = read_json(RESULT / "phase556_behavior_summary.json")
    return {row["model"]: row for row in summary["model_reports"]}


def causal_by_id() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = RESULT / "phase556_causal_analysis_summary.json"
    if not path.exists():
        return {}, {"qualified_candidate_count": 0, "qualified_edge_count": 0}
    summary = read_json(path)
    return {row["candidate_id"]: row for row in summary["candidate_reports"]}, summary


def candidate_id(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['model']}__{candidate['mechanism']}__{candidate['component']}__"
        f"L{candidate['layer']}__rank{candidate['component_rank']}"
    )


def node(model: str, suffix: str, label: str, node_type: str, x: float, y: float, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"phase556:{model}:{suffix}",
        "label": f"{MODEL_LABELS[model]} / {label}",
        "model": model,
        "family_id": "semantic_knowledge",
        "mechanism_id": "fruit_reuse_difference_binding",
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


def edge(
    model: str,
    suffix: str,
    source: str,
    target: str,
    relation: str,
    causal: bool = False,
    compute_edge: bool | None = None,
) -> dict[str, Any]:
    return {
        "id": f"phase556:{model}:{suffix}",
        "source": f"phase556:{model}:{source}",
        "target": f"phase556:{model}:{target}",
        "type": relation,
        "label": relation,
        "score": 1.0,
        "predictive": causal,
        "causal": causal,
        "compute_edge": causal if compute_edge is None else compute_edge,
        "evidence_level": "held_out_component_intervention" if causal else "behavior_or_observer",
    }


def graph_payload(
    model: str,
    behavior: dict[str, Any],
    candidates: list[dict[str, Any]],
    causal_reports: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    boundary: dict[str, Any] | None,
    parents: dict[str, Any] | None,
) -> dict[str, Any]:
    discovery = behavior["split_reports"]["discovery"]
    confirmation = behavior["split_reports"]["independent_confirmation"]
    controlled_score = min(
        discovery["controlled_all_16_correct_rate"],
        confirmation["controlled_all_16_correct_rate"],
    )
    fruit_category = diagnostics["model_reports"][model]["splits"]
    confirmation_key = (
        "independent_confirmation"
        if "independent_confirmation" in fruit_category else "confirmation"
    )
    fruit_category_score = min(
        fruit_category["discovery"]["relations"]["category"]["fruits_only"]["rate"],
        fruit_category[confirmation_key]["relations"]["category"]["fruits_only"]["rate"],
    )
    nodes = [
        node(
            model, "behavior_gate", f"受控16单元门 {controlled_score:.3f}",
            "query_readout_event", -48, 0,
            score=controlled_score,
            color="#22c55e" if behavior["internal_collection_authorized"] else "#ef4444",
            observer=True,
            behavior_gate_pass=behavior["internal_collection_authorized"],
            sample_count=3072,
        ),
        node(
            model, "natural_fruit_category", f"自然水果类别最低 {fruit_category_score:.3f}",
            "natural_knowledge_behavior", -48, -15,
            score=fruit_category_score,
            color="#f59e0b",
            observer=True,
            physical=False,
            evidence_note="水果与歧义控制对象已拆账；原总门不追认",
        ),
        node(
            model, "answer_boundary", "受限候选答案边界", "query_readout_event", 48, 0,
            score=behavior["semantic_accuracy"], color="#38bdf8", observer=True,
        ),
    ]
    edges = [
        edge(model, "behavior_to_answer", "behavior_gate", "answer_boundary", "behavior_measurement"),
    ]
    if not behavior["internal_collection_authorized"]:
        nodes.append(node(
            model, "physical_stop", "行为门失败，内部与参数阶段停止",
            "physical_collection_stop", 0, 16,
            score=0.0, color="#ef4444", observer=True,
            stop_reason="controlled_all_16_gate_failed",
        ))
        edges.append(edge(model, "gate_stop", "behavior_gate", "physical_stop", "authorization_stop"))
    for index, candidate in enumerate(candidates):
        cid = candidate_id(candidate)
        report = causal_reports.get(cid)
        qualified = bool(report and report["causal_qualified"])
        mechanism = candidate["mechanism"]
        component = candidate["component"]
        node_type = "shared_category_event" if mechanism == "category_reuse" else "binding_event"
        y_base = -8 if mechanism == "category_reuse" else 8
        y = y_base + {"layer_input": -3, "attention_output": 0, "mlp_output": 3}[component]
        suffix = f"candidate_{index}"
        nodes.append(node(
            model, suffix,
            f"{mechanism} / {component} / L{candidate['layer']}",
            node_type,
            -28 + 58 * float(candidate["relative_depth"]), y,
            score=float(candidate["replicated_specificity_margin"]),
            color="#22c55e" if qualified else "#f59e0b",
            observer=True,
            physical=qualified,
            predictive=qualified,
            causal=qualified,
            compute_edge=qualified and component != "layer_input",
            sample_count=report["scenario_reports"]["matched_factor_delta"]["target_n"] if report else 0,
            candidate_id=cid,
            causal_qualified=qualified,
            sealed=False,
            raw_result="tests/gpt5/result/phase556_fruit_encoding/phase556_causal_analysis_summary.json",
        ))
        edges.append(edge(model, f"observe_{index}", "behavior_gate", suffix, "observer_candidate"))
        edges.append(edge(
            model, f"readout_{index}", suffix, "answer_boundary",
            "matched_factor_causal_edge" if qualified else "unqualified_observation_edge",
            causal=qualified,
            compute_edge=qualified and component != "layer_input",
        ))
    if boundary:
        for mechanism_index, (mechanism, report) in enumerate(
            sorted(boundary["mechanism_reports"].items())
        ):
            layer = report["earliest_replicated_layer"]
            if layer is None:
                continue
            suffix = f"boundary_{mechanism}"
            y = -13 if mechanism == "category_reuse" else 13
            nodes.append(node(
                model, suffix, f"{mechanism} 最早复制状态边界 L{layer}",
                "shared_category_event" if mechanism == "category_reuse" else "binding_event",
                22, y,
                score=1.0, color="#22c55e", observer=True, physical=True,
                predictive=True, causal=True, compute_edge=False,
                sample_count=384,
                target_sample_count=192,
                split_count=2,
                evidence_level="two_independent_heldout_layer_input_sweeps",
            ))
            edges.append(edge(
                model, f"boundary_readout_{mechanism_index}", suffix, "answer_boundary",
                "causal_state_boundary", causal=True, compute_edge=False,
            ))
    if parents:
        for mechanism_index, (mechanism, report) in enumerate(
            sorted(parents["mechanism_reports"].items())
        ):
            for condition_index, condition in enumerate(report["qualified_conditions"]):
                suffix = f"parent_{mechanism}_{condition_index}"
                is_writer = condition in report["qualified_writer_conditions"]
                y = (-20 if mechanism == "category_reuse" else 20) + condition_index * 3
                nodes.append(node(
                    model, suffix,
                    f"{mechanism} L{report['parent_layer']} 父组 {condition}",
                    "shared_category_event" if mechanism == "category_reuse" else "binding_event",
                    8, y,
                    score=1.0, color="#06b6d4", observer=True, physical=True,
                    predictive=True, causal=True, compute_edge=is_writer,
                    sample_count=128,
                    target_sample_count=64,
                    evidence_level="independent_additive_parent_contribution_holdout",
                    intervention_semantics=parents.get(
                        "parent_intervention_semantics",
                        "additive_parent_component_delta_at_child_state",
                    ),
                ))
                edges.append(edge(
                    model, f"parent_boundary_{mechanism_index}_{condition_index}",
                    suffix, f"boundary_{mechanism}",
                    (
                        "direct_parent_writer_support"
                        if is_writer else "composite_parent_contribution_support"
                    ),
                    causal=True, compute_edge=is_writer,
                ))
    return {
        "schema_version": "phase556_fruit_encoding_atlas.v1",
        "phase_id": "Phase556",
        "generated_at": now(),
        "model": model,
        "evidence_scope": (
            "controlled contextual category/binding behavior, factor-event observers, and held-out "
            "component interventions; not natural fruit parameter storage, neuron closure, or sealed evidence"
        ),
        "graph": {
            "meta": {
                "model": model,
                "registered_case_count": 5808,
                "open_case_count": 3872,
                "sealed_case_count_unread": 1936,
                "internal_collection_authorized": behavior["internal_collection_authorized"],
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def publish() -> None:
    behavior = behavior_by_model()
    diagnostics = read_json(RESULT / "phase556_natural_behavior_diagnostics.json")
    causal_reports, causal_summary = causal_by_id()
    boundary_paths = {
        "qwen3": RESULT / "phase556_layer_input_boundary_analysis.json",
        "glm4": RESULT / "phase556_glm4_layer_input_boundary_analysis.json",
    }
    parent_paths = {
        "qwen3": RESULT / "phase556_direct_parent_analysis.json",
        "glm4": RESULT / "phase556_glm4_direct_parent_analysis.json",
    }
    boundaries = {
        model: read_json(path) for model, path in boundary_paths.items() if path.exists()
    }
    parents = {
        model: read_json(path) for model, path in parent_paths.items() if path.exists()
    }
    candidate_path = RESULT / "phase556_causal_candidate_registry.json"
    candidates = read_json(candidate_path)["candidates"] if candidate_path.exists() else []
    items = []
    for model in MODELS:
        model_candidates = [row for row in candidates if row["model"] == model]
        filename = f"phase556_{model}_fruit_encoding.json"
        payload = graph_payload(
            model, behavior[model], model_candidates, causal_reports, diagnostics,
            boundaries.get(model), parents.get(model),
        )
        write_json(PUBLIC / filename, payload)
        write_json(RESULT / "atlas" / filename, payload)
        items.append({
            "id": f"phase556_{model}",
            "model": model,
            "label": f"{MODEL_LABELS[model]} 水果复用差分审计",
            "path": filename,
        })
    manifest = {
        "schema_version": "phase556_fruit_encoding_atlas_manifest.v1",
        "generated_at": now(),
        "route_id": "gpt5",
        "evidence_scope": "行为、观察候选和留出组件干预分层；未读密封集，非神经元闭合",
        "items": items,
    }
    write_json(PUBLIC / "manifest.json", manifest)
    write_json(RESULT / "atlas" / "manifest.json", manifest)

    registry = read_json(REGISTRY)
    source = {
        "id": "gpt5_phase556_fruit_encoding_atlas",
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase556 水果复用－差分编码审计",
        "description": "三模型行为门、类别/绑定因素事件、因果留出干预与停止边界。",
        "manifest_path": "/vis_data/phase556_fruit_encoding_atlas/manifest.json",
        "manifest_schema": "phase556_fruit_encoding_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase556_fruit_encoding_atlas",
        "models": list(MODELS),
        "evidence_scope": "受控上下文复用/绑定；不等于自然水果参数编码、单神经元机制或密封闭合",
        "color": "#22c55e",
    }
    positions = {row["id"]: index for index, row in enumerate(registry["sources"])}
    if source["id"] in positions:
        registry["sources"][positions[source["id"]]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = now()
    write_json(REGISTRY, registry)

    write_json(RESULT / "phase556_atlas_publish_summary.json", {
        "schema_version": "phase556_atlas_publish_summary.v1",
        "phase_id": "Phase556",
        "created_at": now(),
        "source_id": source["id"],
        "model_count": len(MODELS),
        "candidate_count": len(candidates),
        "qualified_candidate_count": causal_summary.get("qualified_candidate_count", 0),
        "qualified_edge_count": causal_summary.get("qualified_edge_count", 0),
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "sealed_split_read": False,
    })
    print(PUBLIC / "manifest.json")


if __name__ == "__main__":
    publish()
