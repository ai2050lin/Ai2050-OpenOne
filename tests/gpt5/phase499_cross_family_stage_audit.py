#!/usr/bin/env python3
"""Aggregate Phase494-498, export atlas graphs, and freeze the stage decision."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_DIR = ROOT / "tests" / "gpt5" / "result" / "phase495_cross_family_behavior_gate"
AUTH_PATH = ROOT / "tests" / "gpt5" / "result" / "phase496_behavior_authorization" / "phase496_open_physical_authorization.json"
PHYSICAL_DIR = ROOT / "tests" / "gpt5" / "result" / "phase497_498_cross_family_trajectory"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase499_cross_family_stage_audit"
ATLAS_DIR = ROOT / "frontend" / "public" / "vis_data" / "phase499_cross_family_relation_trajectory"
REGISTRY_PATH = ROOT / "frontend" / "public" / "vis_data" / "source_registry.json"
REPORT_PATH = ROOT / "research" / "MainAnalysis" / "20260717_03_Phase487-499跨关系族状态轨迹审计.md"
MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LABELS = {"qwen3": "Qwen3（通义千问3）", "glm4": "GLM4（智谱GLM4）", "deepseek7b": "DS7B（深度求索7B）"}
FAMILIES = (
    "marker_inheritance",
    "signal_assignment",
    "symmetric_pair",
    "directed_mentor",
    "transitive_precedence",
    "direct_nontransitive",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def functional_alignment(left_model: str, right_model: str) -> dict[str, Any] | None:
    left_path = PHYSICAL_DIR / f"phase498_{left_model}_functional_scores.jsonl"
    right_path = PHYSICAL_DIR / f"phase498_{right_model}_functional_scores.jsonl"
    if not left_path.exists() or not right_path.exists():
        return None
    left = {(row["world_case_id"], row["track"]): row for row in load_jsonl(left_path)}
    right = {(row["world_case_id"], row["track"]): row for row in load_jsonl(right_path)}
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None
    x = np.asarray([left[key]["primary_linear_score"] for key in keys], dtype=np.float64)
    y = np.asarray([right[key]["primary_linear_score"] for key in keys], dtype=np.float64)
    correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
    sign_agreement = float(((x > 0) == (y > 0)).mean())
    return {
        "models": [left_model, right_model],
        "n_shared_surface_cases": len(keys),
        "primary_score_correlation": correlation,
        "prediction_sign_agreement": sign_agreement,
        "neural_coordinate_alignment_claim": False,
        "interpretation": "functional observer-score agreement only",
    }


def physical_concise(summary: dict[str, Any]) -> dict[str, Any]:
    primary = summary["primary_train_family_observer"]
    return {
        "status": summary["status"],
        "prediction_families": summary["behavior_qualified_prediction_families"],
        "primary_window": summary["primary_window"],
        "primary_prediction": primary["prediction"],
        "primary_geometry": primary["geometry"],
        "legacy_phase492_observer": summary["legacy_phase492_observer"],
        "nonlinear_controls": summary["nonlinear_controls"],
        "prompt_end_trajectory": summary["trajectory_by_position_role"]["prompt_end"],
        "position_formation_layers": {
            role: payload["stable_formation_layer"]
            for role, payload in summary["trajectory_by_position_role"].items()
        },
        "gates": summary["gates"],
        "evidence_boundary": summary["evidence_boundary"],
    }


def atlas_payload(model: str, behavior: dict[str, Any], physical: dict[str, Any] | None) -> dict[str, Any]:
    nodes = []
    edges = []
    for index, family in enumerate(FAMILIES):
        report = behavior["families"][family]
        passed = bool(report["behavior_gate_pass"])
        node_id = f"phase499:{model}:behavior:{family}"
        nodes.append({
            "id": node_id,
            "label": f"{MODEL_LABELS[model]} / {family} 行为门",
            "type": "relation_family_behavior_gate",
            "model": model,
            "family_id": "reasoning_relation_binding",
            "mechanism_id": family,
            "layer": -1,
            "relative_depth": 0.0,
            "position_role": "direct_true_false_output_boundary",
            "position": [-14.0, index * 8.0 - 20.0, 0.0],
            "score": report["native_intersection"]["rate"],
            "lcb95": report["native_intersection"]["lcb95"],
            "n": report["native_intersection"]["n"],
            "count": report["native_intersection"]["count"],
            "gate_pass": passed,
            "physical": False,
            "observer": True,
            "predictive": False,
            "causal": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "evidence_level": "frozen_cross_family_behavior_qualification",
            "color": "#22c55e" if passed else "#ef4444",
            "size": 0.62,
            "show_label": True,
        })

    if physical is None:
        nodes.append({
            "id": f"phase499:{model}:physical_blocked",
            "label": f"{MODEL_LABELS[model]} / 物理轨迹未授权",
            "type": "physical_measurement_blocked",
            "model": model,
            "layer": -1,
            "relative_depth": 0.0,
            "position_role": "behavior_gate",
            "position": [0.0, 0.0, 0.0],
            "score": 0.0,
            "gate_pass": False,
            "physical": False,
            "observer": True,
            "predictive": False,
            "causal": False,
            "single_neuron": False,
            "pipeline_sealed": False,
            "evidence_level": "behavior_gate_failure",
            "color": "#ef4444",
            "size": 0.8,
            "show_label": True,
        })
    else:
        events = physical["trajectory_by_position_role"]["prompt_end"]["events"]
        prior_id = None
        for event in events:
            layer = int(event["layer_with_embedding"])
            relative = layer / max(1, len(events) - 1)
            node_id = f"phase499:{model}:prompt_end:L{layer}"
            passed = bool(event["gate_pass"])
            nodes.append({
                "id": node_id,
                "label": f"{MODEL_LABELS[model]} / 提示终端 L{layer}",
                "type": "observational_relation_readability_trajectory",
                "model": model,
                "family_id": "reasoning_relation_binding",
                "mechanism_id": "cross_family_relation_truth_observer",
                "layer": layer,
                "relative_depth": relative,
                "position_role": "prompt_end",
                "position": [0.0, relative * 100.0, 0.0],
                "score": event["accuracy"],
                "q_native": event["q_native"],
                "lcb95": event["lcb95"],
                "trajectory_event": event["event"],
                "gate_pass": passed,
                "physical": True,
                "observer": True,
                "predictive": passed,
                "causal": False,
                "compute_edge": False,
                "single_neuron": False,
                "pipeline_sealed": False,
                "evidence_level": "open_unseen_family_observational_prediction",
                "color": "#10b981" if passed else "#64748b",
                "size": 0.45 if not passed else 0.72,
                "show_label": passed or layer in {0, len(events) - 1},
            })
            if prior_id is not None:
                edges.append({
                    "id": f"{prior_id}->{node_id}",
                    "source": prior_id,
                    "target": node_id,
                    "type": "observational_depth_order",
                    "label": "层深顺序（非计算边）",
                    "score": event["accuracy"],
                    "evidence_level": "observational_sequence_only",
                    "predictive": False,
                    "compute_edge": False,
                    "causal": False,
                })
            prior_id = node_id
        primary = physical["primary_window"]
        primary_id = f"phase499:{model}:prompt_end:L{primary['layer_with_embedding']}"
        for family in physical["behavior_qualified_prediction_families"]:
            edges.append({
                "id": f"phase499:{model}:{family}:authorization",
                "source": f"phase499:{model}:behavior:{family}",
                "target": primary_id,
                "type": "measurement_authorization",
                "label": "行为合格后允许物理观察",
                "score": 1.0,
                "evidence_level": "protocol_gate",
                "predictive": False,
                "compute_edge": False,
                "causal": False,
            })
    return {
        "schema_version": "phase499_cross_family_relation_trajectory.v1",
        "model": model,
        "evidence_scope": "open cross-family behavior and projected state readability trajectory; no compute edge, causality, sealed evidence, or neuron closure",
        "graph": {
            "meta": {
                "model": model,
                "sealed_split_read": False,
                "causal": False,
                "single_neuron": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def downgrade_phase492_atlas() -> None:
    """Preserve Phase492 measurements while revoking their relation-state claim."""
    old_dir = ROOT / "frontend" / "public" / "vis_data" / "phase492_relation_state_atlas"
    manifest_path = old_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = load_json(manifest_path)
    scope = (
        "historical same-family measurement superseded by Phase499: fixed-claim paired worlds "
        "did not qualify the old fit families, so the late direction is not valid relation-state evidence"
    )
    manifest["superseded_by"] = "gpt5_phase499_cross_family_relation_trajectory"
    manifest["evidence_scope"] = scope
    for item in manifest.get("items", []):
        item["evidence_scope"] = scope
        item["superseded_by_phase499"] = True
        filename = item.get("filename") or item.get("path")
        if not filename:
            continue
        payload_path = old_dir / filename
        if not payload_path.exists():
            continue
        payload = load_json(payload_path)
        payload["evidence_scope"] = scope
        graph = payload.get("graph", {})
        graph.setdefault("meta", {}).update({
            "superseded_by_phase499": True,
            "relation_state_interpretation_valid": False,
            "invalidation_reason": "fixed-claim paired-world behavior qualification failed",
        })
        for node in graph.get("nodes", []):
            if node.get("type") in {
                "relation_semantic_behavior_gate",
                "observational_relation_state_window",
                "independent_observational_prediction_gate",
            }:
                node["historical_evidence_level"] = node.get("evidence_level")
                node["evidence_level"] = "superseded_claim_identity_confounded_measurement"
                node["gate_pass"] = False
                node["predictive"] = False
                node["color"] = "#64748b"
                node["superseded_by_phase499"] = True
        for edge in graph.get("edges", []):
            edge["historical_evidence_level"] = edge.get("evidence_level")
            edge["evidence_level"] = "superseded_historical_observation"
            edge["predictive"] = False
            edge["superseded_by_phase499"] = True
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_atlas(behavior: dict[str, Any], physical: dict[str, Any]) -> None:
    downgrade_phase492_atlas()
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        payload = atlas_payload(model, behavior[model], physical.get(model))
        filename = f"phase499_{model}_cross_family_trajectory.json"
        path = ATLAS_DIR / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        items.append({"id": f"phase499_{model}", "model": model, "path": filename, "label": f"{MODEL_LABELS[model]} 跨关系族轨迹"})
    manifest = {
        "schema_version": "phase499_cross_family_relation_trajectory_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": "gpt5",
        "evidence_scope": "open observational projected-state trajectory; non-causal and non-neuronal",
        "items": items,
    }
    (ATLAS_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    registry = load_json(REGISTRY_PATH)
    source_id = "gpt5_phase499_cross_family_relation_trajectory"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase499 跨关系族状态轨迹图谱",
        "description": "六个关系族在三模型上的冻结行为资格与未见关系族逐层可读性轨迹。",
        "manifest_path": "/vis_data/phase499_cross_family_relation_trajectory/manifest.json",
        "manifest_schema": "phase499_cross_family_relation_trajectory_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase499_cross_family_relation_trajectory",
        "models": list(MODELS),
        "evidence_scope": "开放投影状态的观察预测；层序边不是计算边，非因果、非神经元闭合",
        "color": "#0ea5e9",
    }
    existing = {item["id"]: index for index, item in enumerate(registry["sources"])}
    if source_id in existing:
        registry["sources"][existing[source_id]] = source
    else:
        registry["sources"].append(source)
    old_source_id = "gpt5_phase492_relation_state_atlas"
    if old_source_id in existing:
        old = registry["sources"][existing[old_source_id]]
        old["label"] = "Phase492 晚层方向（历史结果，Phase499已降级）"
        old["description"] = "旧同族预测保留原始数值；固定断言成对世界审计失败，已撤销抽象关系状态解释。"
        old["evidence_scope"] = "历史同族观测，受断言词元身份混杂；不得作为关系状态、机制、因果或神经元证据"
        old["superseded_by"] = source_id
    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_report(audit: dict[str, Any]) -> None:
    behavior_lines = []
    for model in MODELS:
        item = audit["behavior"][model]
        unseen = item["overall"]["unseen_native_intersection"]
        positive_count = sum(payload["by_claim_polarity"]["positive"]["count"] for payload in item["families"].values())
        negative_count = sum(payload["by_claim_polarity"]["negative"]["count"] for payload in item["families"].values())
        polarity_total = sum(payload["by_claim_polarity"]["positive"]["n"] for payload in item["families"].values())
        behavior_lines.append(
            f"- {MODEL_LABELS[model]}：未见族原生交集 {unseen['count']}/{unseen['n']} = {unseen['rate']:.4f}，"
            f"肯定断言 {positive_count}/{polarity_total}，否定断言 {negative_count}/{polarity_total}，"
            f"四族行为门={'通过' if item['gates']['all_unseen_families_behavior_pass'] else '未通过'}。"
        )
    physical_lines = []
    for model in MODELS:
        item = audit["physical"].get(model)
        if item is None:
            physical_lines.append(f"- {MODEL_LABELS[model]}：行为未授权，未采集物理轨迹。")
            continue
        pred = item["primary_prediction"]["overall"]
        geom = item["primary_geometry"]
        physical_lines.append(
            f"- {MODEL_LABELS[model]}：冻结窗口未见族预测 {pred['count']}/{pred['n']} = {pred['rate']:.4f}，"
            f"LCB95={pred['lcb95']:.4f}，Q={geom['q_native']:.4f}，"
            f"跨族门={'通过' if item['gates']['primary_cross_family_prediction_pass'] else '未通过'}。"
        )
    decision = audit["stage_decision"]
    text = f"""# Phase487-499 跨关系族状态轨迹系统审计

生成时间：{audit['created_at']}

## 一、材料正确性

“语言是动态模式网络”可以保留为候选实验框架，但当前证据只覆盖条件相关的行为与低维状态可读性。稀疏门控、高阶模式竞争、非交换算子、抽象层级、元模式和训练损失分解均未被本轮直接观测，不能写成已经建立的数学体系。Phase487-493 的严格审计结论基本正确：此前正结果是两个已见关系族的晚层提示终端读出，不是上游计算机制。

## 二、Phase494 协议修复

六族总分母在运行前冻结。每个真假世界保持断言、规则、词汇多重集、事实数量和长度相同，只交换证据连接；同时平衡三档长度、三种事实顺序和肯定/否定断言。密封分割未读取。

这一修复改变了对旧结果的解释：Phase492（阶段492）的高分不能继续被当成抽象关系状态，因为旧真假世界更换了断言属性词元；新协议去掉该捷径以后，两个旧拟合族也没有在任何模型上通过成对世界门。

## 三、行为结果

{chr(10).join(behavior_lines)}

## 四、物理结果

{chr(10).join(physical_lines)}

层—位置轨迹是观察器可读性顺序，不是来源写入、合法计算边或因果运输。固定随机特征与局部核只作为低容量非线性对照，不提升证据等级。

## 五、理论边界

本轮能够检验的是：两个训练关系族形成的观察器是否外推到四个未见关系结构，以及可读性在哪些层—角色连续出现。即使通过，也只支持“跨关系族功能状态候选”；不支持稀疏模式门、模式代数、注意力运输、神经元载体或机制闭合。

## 六、阶段决定

{decision['plain_language']}

严格闭合仍为 0/72。总体科学成熟度为 {audit['progress']['point_percent']}%，合理区间 {audit['progress']['range_percent'][0]}%-{audit['progress']['range_percent'][1]}%。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    authorization = load_json(AUTH_PATH)
    behavior = {}
    physical = {}
    for model in MODELS:
        behavior[model] = load_json(BEHAVIOR_DIR / f"phase495_{model}_summary.json")
        physical_path = PHYSICAL_DIR / f"phase498_{model}_summary.json"
        if physical_path.exists():
            physical[model] = load_json(physical_path)

    physical_concise_map = {model: physical_concise(payload) for model, payload in physical.items()}
    alignments = []
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1:]:
            result = functional_alignment(left, right)
            if result is not None:
                alignments.append(result)
    cross_pass_models = [
        model for model, payload in physical.items()
        if payload["gates"]["primary_cross_family_prediction_pass"]
    ]
    formation_models = [
        model for model, payload in physical.items()
        if payload["gates"]["stable_prompt_end_formation_found"]
    ]
    stage_success = len(cross_pass_models) >= 2 and all(model in formation_models for model in cross_pass_models)
    if stage_success:
        next_action = "freeze_new_path_protocol"
        plain = "至少两个模型通过跨关系族观察预测并出现稳定形成边界。下一步应另行冻结合法来源写入与路径干预协议；该工作属于新的因果资格阶段，不能用本阶段结果直接自动干预。"
        point = 26
        interval = [25, 27]
    else:
        next_action = "stop_abstract_relation_mechanism_claim_and_audit_failures"
        plain = "跨关系族或稳定形成门不足两个模型通过。当前路线应停留在任务族/终端读出审计，先解释失败族与否定、长度、顺序控制，不得自动进入路径干预或神经元扫描。"
        point = 25
        interval = [24, 26]

    audit = {
        "schema_version": "phase499_cross_family_stage_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stage_complete",
        "behavior": behavior,
        "physical": physical_concise_map,
        "functional_alignments": alignments,
        "cross_family_pass_models": cross_pass_models,
        "stable_formation_models": formation_models,
        "strict_closure": {"closed": 0, "denominator": 72, "rate": 0.0},
        "progress": {"point_percent": point, "range_percent": interval},
        "theory_audit": {
            "dynamic_conditional_state_trajectories": "experimentally_tested_at_observer_level",
            "sparse_gate": "not_observed",
            "higher_order_pattern_competition": "not_observed",
            "noncommutative_operator_algebra": "not_observed",
            "abstract_hierarchy_and_meta_patterns": "not_observed",
            "invented_auxiliary_training_loss_decomposition": "unsupported",
            "candidate_framework_may_be_retained": True,
            "established_new_mathematical_system": False,
        },
        "evidence_boundary": {
            "sealed_split_read": False,
            "compute_transport_measured": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
            "strict_mechanism_closure": False,
        },
        "stage_decision": {
            "phase494_objective_complete": True,
            "success_gate": stage_success,
            "next_action": next_action,
            "automatic_next_phase_executed": False,
            "reason_not_automatic": "Any next step changes the evidence class from observation to path causality and requires a separately frozen denominator and controls.",
            "plain_language": plain,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = OUT_DIR / "phase499_cross_family_stage_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_atlas(behavior, physical)
    write_report(audit)
    print(audit_path)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
