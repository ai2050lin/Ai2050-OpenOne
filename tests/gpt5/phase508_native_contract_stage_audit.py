#!/usr/bin/env python3
"""Aggregate Phase500-507 and export the native relation contract atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase500_native_relation_contract_protocol"
STAGE_A_AUTH = (
    ROOT
    / "tests/gpt5/result/phase502_staged_behavior_authorization"
    / "phase502_calibration_authorization.json"
)
STAGE_B_AUTH = (
    ROOT
    / "tests/gpt5/result/phase504_staged_behavior_authorization"
    / "phase504_contract_authorization.json"
)
STAGE_C_AUTH = (
    ROOT
    / "tests/gpt5/result/phase506_staged_behavior_authorization"
    / "phase506_confirmation_authorization.json"
)
PHYSICAL_GATE = (
    ROOT
    / "tests/gpt5/result/phase507_conditional_physical_gate"
    / "phase507_conditional_physical_gate.json"
)
OUT_DIR = ROOT / "tests/gpt5/result/phase508_native_contract_stage_audit"
OUT_PATH = OUT_DIR / "phase508_native_contract_stage_audit.json"
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase508_native_relation_contract_atlas"
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = (
    ROOT
    / "research/MainAnalysis/20260717_04_Phase500-508固定断言原生关系合同审计.md"
)

MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱GLM4）",
    "deepseek7b": "DS7B（深度求索7B）",
}
FUNCTIONS = (
    "direct_symmetric",
    "direct_directed",
    "single_step_rule",
    "transitive_closure",
    "nontransitive_exclusion",
)
POLARITIES = ("positive", "explicit_negative", "reverse_query")
OBSERVERS = ("true_false", "mapped_ab", "mapped_01")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def passed_cell_keys(summary: dict[str, Any]) -> list[str]:
    return [
        f"{item['function_class']}|{item['polarity']}"
        for item in summary["passed_function_polarity_cells"]
    ]


def concise_stage_a(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": summary["status"],
        "row_count": summary["row_count"],
        "cuda_used": summary["cuda_used"],
        "model_weights_loaded": summary["model_weights_loaded"],
        "passed_function_polarity_cells": summary["passed_function_polarity_cells"],
        "cells": summary["cells"],
    }


def concise_stage_b(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": summary["status"],
        "row_count": summary["row_count"],
        "cuda_used": summary["cuda_used"],
        "model_weights_loaded": summary["model_weights_loaded"],
        "passed_native_contracts": summary["passed_native_contracts"],
        "cells": summary["cells"],
    }


def closest_stage_b_contract(stage_b: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for key, cell in stage_b["cells"].items():
        score = cell["observer_consistency"]["consistent_and_correct"]
        candidates.append((score["lcb95"], score["rate"], key, cell))
    if not candidates:
        return None
    _, _, key, cell = max(candidates)
    return {
        "key": key,
        "function_class": cell["function_class"],
        "polarity": cell["polarity"],
        "vocab_system": cell["vocab_system"],
        "gate_pass": cell["gate_pass"],
        "observer_consistency": cell["observer_consistency"],
        "observers": cell["observers"],
    }


def graph_payload(
    model: str,
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    passed = set(passed_cell_keys(stage_a))
    row = 0
    for function_index, function_class in enumerate(FUNCTIONS):
        for polarity_index, polarity in enumerate(POLARITIES):
            key = f"{function_class}|{polarity}"
            cell = stage_a["cells"][key]
            node_id = f"phase508:{model}:A:{key}"
            gate_pass = key in passed
            nodes.append({
                "id": node_id,
                "label": f"{function_class} / {polarity}",
                "type": "native_behavior_calibration_cell",
                "model": model,
                "family_id": "reasoning_relation_binding",
                "mechanism_id": function_class,
                "polarity": polarity,
                "vocab_system": "natural_names",
                "observer_contract": "true_false",
                "layer": -1,
                "relative_depth": 0.0,
                "position_role": "output_contract",
                "position": [-18.0, function_index * 15.0 + polarity_index * 4.0, 0.0],
                "score": cell["surface_intersection"]["rate"],
                "lcb95": cell["surface_intersection"]["lcb95"],
                "paired_world_rate": cell["paired_world"]["rate"],
                "paired_world_lcb95": cell["paired_world"]["lcb95"],
                "gate_pass": gate_pass,
                "physical": False,
                "observer": True,
                "predictive": False,
                "causal": False,
                "compute_edge": False,
                "single_neuron": False,
                "pipeline_sealed": False,
                "evidence_level": "frozen_native_behavior_calibration",
                "color": "#22c55e" if gate_pass else "#64748b",
                "size": 0.68 if gate_pass else 0.32,
                "show_label": gate_pass,
            })
            if not gate_pass:
                continue

            natural_key = f"{key}|natural_names"
            contract = stage_b["cells"][natural_key]
            observer_ids = []
            for observer_index, observer_name in enumerate(OBSERVERS):
                observer = contract["observers"][observer_name]
                observer_id = f"phase508:{model}:B:{key}:{observer_name}"
                observer_ids.append(observer_id)
                nodes.append({
                    "id": observer_id,
                    "label": f"{function_class} / {polarity} / {observer_name}",
                    "type": "output_observer_behavior_cell",
                    "model": model,
                    "family_id": "reasoning_relation_binding",
                    "mechanism_id": function_class,
                    "polarity": polarity,
                    "vocab_system": "natural_names",
                    "observer_contract": observer_name,
                    "layer": -1,
                    "relative_depth": 0.0,
                    "position_role": "output_contract",
                    "position": [0.0, row * 13.0 + observer_index * 3.5, 0.0],
                    "score": observer["surface_intersection"]["rate"],
                    "lcb95": observer["surface_intersection"]["lcb95"],
                    "paired_world_rate": observer["paired_world"]["rate"],
                    "paired_world_lcb95": observer["paired_world"]["lcb95"],
                    "gate_pass": observer["gate_pass"],
                    "physical": False,
                    "observer": True,
                    "predictive": False,
                    "causal": False,
                    "compute_edge": False,
                    "single_neuron": False,
                    "pipeline_sealed": False,
                    "evidence_level": "frozen_observer_equivalence_calibration",
                    "color": "#22c55e" if observer["gate_pass"] else "#ef4444",
                    "size": 0.58,
                    "show_label": True,
                })
                edges.append({
                    "id": f"{node_id}->{observer_id}",
                    "source": node_id,
                    "target": observer_id,
                    "type": "measurement_authorization",
                    "label": "阶段A通过后允许观察器校准",
                    "score": 1.0,
                    "evidence_level": "protocol_gate",
                    "predictive": False,
                    "compute_edge": False,
                    "causal": False,
                })
            consistency = contract["observer_consistency"]["consistent_and_correct"]
            contract_id = f"phase508:{model}:B:{key}:contract"
            nodes.append({
                "id": contract_id,
                "label": f"{function_class} / {polarity} / 三观察器合同",
                "type": "observer_equivalence_contract",
                "model": model,
                "family_id": "reasoning_relation_binding",
                "mechanism_id": function_class,
                "polarity": polarity,
                "vocab_system": "natural_names",
                "observer_contract": "true_false+mapped_ab+mapped_01",
                "layer": -1,
                "relative_depth": 0.0,
                "position_role": "construct_gate",
                "position": [18.0, row * 13.0 + 3.5, 0.0],
                "score": consistency["rate"],
                "lcb95": consistency["lcb95"],
                "gate_pass": contract["gate_pass"],
                "physical": False,
                "observer": True,
                "predictive": False,
                "causal": False,
                "compute_edge": False,
                "single_neuron": False,
                "pipeline_sealed": False,
                "evidence_level": "frozen_construct_validity_gate",
                "color": "#22c55e" if contract["gate_pass"] else "#ef4444",
                "size": 0.72,
                "show_label": True,
            })
            for observer_id in observer_ids:
                edges.append({
                    "id": f"{observer_id}->{contract_id}",
                    "source": observer_id,
                    "target": contract_id,
                    "type": "observer_equivalence_requirement",
                    "label": "三观察器必须共同通过",
                    "score": consistency["rate"],
                    "evidence_level": "construct_gate_only",
                    "predictive": False,
                    "compute_edge": False,
                    "causal": False,
                })
            row += 1

    blocked_id = f"phase508:{model}:confirmation_blocked"
    nodes.append({
        "id": blocked_id,
        "label": f"{MODEL_LABELS[model]} / 独立确认未授权",
        "type": "independent_confirmation_blocked",
        "model": model,
        "layer": -1,
        "relative_depth": 0.0,
        "position_role": "construct_gate",
        "position": [36.0, 12.0, 0.0],
        "score": 0.0,
        "gate_pass": False,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "no_complete_native_contract",
        "color": "#ef4444",
        "size": 0.78,
        "show_label": True,
    })
    nodes.append({
        "id": f"phase508:{model}:physical_blocked",
        "label": f"{MODEL_LABELS[model]} / 物理图谱未授权",
        "type": "physical_measurement_blocked",
        "model": model,
        "layer": -1,
        "relative_depth": 0.0,
        "position_role": "physical_gate",
        "position": [52.0, 12.0, 0.0],
        "score": 0.0,
        "gate_pass": False,
        "physical": False,
        "observer": True,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "evidence_level": "behavior_construct_gate_failure",
        "color": "#ef4444",
        "size": 0.82,
        "show_label": True,
    })
    return {
        "schema_version": "phase508_native_relation_contract_atlas.v1",
        "model": model,
        "evidence_scope": (
            "fixed-claim behavior and output-observer qualification only; no hidden-state "
            "trajectory, compute edge, causality, neuron evidence, or sealed evidence"
        ),
        "graph": {
            "meta": {
                "model": model,
                "stage_a_pass_count": len(passed),
                "stage_b_pass_count": len(stage_b["passed_native_contracts"]),
                "sealed_split_read": False,
                "physical_measurement": False,
                "causal": False,
                "single_neuron": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def write_atlas(stage_a: dict[str, Any], stage_b: dict[str, Any]) -> None:
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase508_{model}_native_contract.json"
        payload = graph_payload(model, stage_a[model], stage_b[model])
        (ATLAS_DIR / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        items.append({
            "id": f"phase508_{model}",
            "model": model,
            "path": filename,
            "label": f"{MODEL_LABELS[model]} 原生关系合同资格链",
        })
    manifest = {
        "schema_version": "phase508_native_relation_contract_atlas_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": "gpt5",
        "evidence_scope": "固定断言行为与输出观察器资格链；无新增物理或神经元数据",
        "items": items,
    }
    (ATLAS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase508_native_relation_contract_atlas"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase508 固定断言原生关系合同图谱",
        "description": "五类关系、三种极性与三观察器在三模型上的分层资格链。",
        "manifest_path": "/vis_data/phase508_native_relation_contract_atlas/manifest.json",
        "manifest_schema": "phase508_native_relation_contract_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase508_native_relation_contract_atlas",
        "models": list(MODELS),
        "evidence_scope": "行为与构念资格门；非物理、非计算边、非因果、非神经元闭合",
        "color": "#f59e0b",
    }
    existing = {item["id"]: index for index, item in enumerate(registry["sources"])}
    if source_id in existing:
        registry["sources"][existing[source_id]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def result_line(model: str, stage_a: dict[str, Any], closest: dict[str, Any] | None) -> str:
    a_count = len(stage_a["passed_function_polarity_cells"])
    if closest is None:
        return f"- {MODEL_LABELS[model]}：阶段A通过 {a_count}/15；阶段B没有执行单元。"
    score = closest["observer_consistency"]["consistent_and_correct"]
    return (
        f"- {MODEL_LABELS[model]}：阶段A通过 {a_count}/15；阶段B完整合同 0 个；"
        f"最接近合同 `{closest['key']}` 的三观察器一致且正确为 "
        f"{score['count']}/{score['n']}（LCB95={score['lcb95']:.4f}）。"
    )


def write_report(audit: dict[str, Any]) -> None:
    lines = [
        result_line(model, audit["stage_a"][model], audit["closest_stage_b_contracts"][model])
        for model in MODELS
    ]
    text = f"""# Phase500-508 固定断言原生关系合同系统审计

生成时间：{audit['created_at']}

## 一、对 Phase494-499 材料的复核

材料的核心反向校准正确：同分布独立预测、构念有效性和机制有效性是三个不同证据等级；固定断言与固定事实词元多重集能拆除旧 Phase492（阶段492）的断言身份捷径；旧晚层方向的原始数值应保留，但“抽象关系真值状态”解释必须撤销。停止未经行为授权的物理轨迹、干预与神经元扫描也是正确的。

需要收紧三点。第一，96项账本混合了协议字段、单格数值和科学拼图，不能作为完成比例。第二，固定词袋不能排除局部字符串匹配，对称关系和非传递排除的确定性精确匹配基线已达到100%。第三，`D_world>D_vocab` 一类物理比较只能在行为合格、观察合同可比的单元内解释，不能先算几何再寻找合同。

## 二、Phase500-507 分层算法

本轮没有一次性运行完整笛卡尔积，而是冻结四级漏斗：

1. 阶段A：五类功能乘三种极性，只用自然名称和 true/false（真/假）观察器。
2. 阶段B：只对阶段A通过单元交叉三套词汇与 true/false（真/假）、A/B、0/1 三观察器。
3. 阶段C：只对阶段B完整合同运行独立确认。
4. 阶段D：只有相同合同被至少两个模型独立确认，才允许开放物理采集。

冻结门可写为：

$$
G_A(m,f,p)=1[LCB(I) >= 0.85, LCB(P) >= 0.85, LCB(I intersection P) >= 0.85, LCB(pair) >= 0.80]
$$

$$
G_B(m,f,p,v)=AND(o in (TF, AB, 01), G_o) AND 1[LCB(consistent and correct) >= 0.85]
$$

$$
A_physical(c)=1[SUM(m, G_C(m,c)) >= 2]
$$

## 三、客观结果

{chr(10).join(lines)}

阶段A跨模型共享的单元只有 `direct_directed|positive` 和 `direct_symmetric|positive`。阶段B没有任何模型、功能、极性、词汇组合通过三观察器完整合同，因此阶段C在三个模型上均以空授权集结束，没有加载权重；Phase507（阶段507）物理采集为0行，密封集未读取。

GLM4（智谱GLM4）的 `direct_directed|positive|natural_names` 最接近通过：true/false（真/假）和0/1均为48/48、成对世界24/24，A/B为46/48、成对世界22/24，三观察器一致且正确94/96。它说明自然真假关系行为与任意标签编译发生分叉，不能说明内部关系状态不存在，也不能把A/B失败直接解释成关系推理失败。

## 四、硬伤与理论边界

直接对称关系可被断言与事实精确匹配基线100%解决，所以 Qwen3（通义千问3）和 GLM4（智谱GLM4）的该项通过不是深层规则机制证据。有向直接关系的同一基线是50%，两个模型的稳定通过构成真实行为入口，但仍局限于自然名称和真假接口。DS7B（深度求索7B）没有阶段A合格单元；这限制当前模型证据，不能外推为语言机制不存在。

“语言是动态模式网络”继续只作为候选实验框架。当前新增结论是观察合同必须显式进入测量映射：

$$
r_hat = O(m, omega; X)
$$

$$
omega = (任务族, 反事实构造, 词汇, 极性, 标签映射, 评分器, 分割)
$$

本轮没有观测隐藏状态，因此没有新增形成层、运输边、模式算子、因果边或神经元载体。输出观察器不变性失败只否定“当前三标签合同下的可移植构念”，不否定模型可能以接口条件化状态完成任务。

## 五、图谱与闭合

可视化图谱新增的是阶段A行为资格、阶段B观察器分叉和后续物理阻断节点；全部边都是测量授权或构念要求，不是计算边。全局物理分布保持 Phase499（阶段499）后的状态，没有用行为数据伪造神经元路径。

严格机制闭合仍为0/72，总体科学成熟度保持25%，合理区间24%-26%。本轮提升的是测量与构念边界，不是语言物理图谱完成度。

## 六、下一阶段决定

Phase500（阶段500）的行为可识别性目标已经完成，结果为“自然真假接口存在局部行为入口，但跨任意标签观察器的完整合同失败”。不能自动进入物理轨迹或神经元干预。

下一阶段应另行冻结“语义关系求值”和“输出标签编译”两个子合同：先在自然真假接口复现直接有向关系，再用独立标签绑定任务测量接口编译；只有两者分别通过，才研究二者在隐藏状态中的汇合。该方案改变了构念和选择门，属于新的 Phase（阶段），不能用本轮数据事后放宽阈值。
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    frozen_contract = read_json(PROTOCOL_DIR / "phase500_frozen_contract.json")
    static_audit = read_json(PROTOCOL_DIR / "phase500_static_audit.json")
    stage_a_auth = read_json(STAGE_A_AUTH)
    stage_b_auth = read_json(STAGE_B_AUTH)
    stage_c_auth = read_json(STAGE_C_AUTH)
    physical_gate = read_json(PHYSICAL_GATE)
    stage_a = {
        model: concise_stage_a(stage_a_auth["model_summaries"][model])
        for model in MODELS
    }
    stage_b = {
        model: concise_stage_b(stage_b_auth["model_summaries"][model])
        for model in MODELS
    }
    closest = {model: closest_stage_b_contract(stage_b[model]) for model in MODELS}
    audit = {
        "schema_version": "phase508_native_contract_stage_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stage_complete_gate_stopped",
        "protocol": {
            "contract_sha256": static_audit["contract_sha256"],
            "stage_order": frozen_contract["stage_order"],
            "behavior_gate": frozen_contract["behavior_gate"],
            "static_audit": static_audit,
        },
        "stage_a": stage_a,
        "stage_a_shared_cells": stage_a_auth["shared_cells_or_contracts"],
        "stage_b": stage_b,
        "stage_b_shared_contracts": stage_b_auth["shared_cells_or_contracts"],
        "closest_stage_b_contracts": closest,
        "stage_c": {
            "model_summaries": stage_c_auth["model_summaries"],
            "shared_confirmed_contracts": stage_c_auth["shared_confirmed_contracts"],
        },
        "physical_gate": physical_gate,
        "denominators": {
            "stage_a_model_rows": sum(item["row_count"] for item in stage_a.values()),
            "stage_b_model_rows": sum(item["row_count"] for item in stage_b.values()),
            "stage_c_model_rows": sum(
                item["row_count"] for item in stage_c_auth["model_summaries"].values()
            ),
            "physical_rows": physical_gate["physical_rows_collected"],
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
        },
        "gates": {
            "fixed_claim_pair_static_pass": all(
                item["fixed_claim_pair_pass"]
                for item in static_audit["splits"].values()
            ),
            "fixed_fact_token_multiset_static_pass": all(
                item["fixed_fact_token_multiset_pair_pass"]
                for item in static_audit["splits"].values()
            ),
            "shared_stage_a_cell_found": bool(stage_a_auth["shared_cells_or_contracts"]),
            "complete_native_contract_found": bool(
                stage_b_auth["shared_cells_or_contracts"]
                or any(item["passed_native_contracts"] for item in stage_b.values())
            ),
            "independent_confirmation_authorized": stage_b_auth["authorization"][
                "independent_confirmation"
            ],
            "physical_authorized": stage_c_auth["authorization"][
                "open_conditional_physical"
            ],
        },
        "evidence_boundary": {
            "sealed_split_read": False,
            "hidden_state_collected": False,
            "compute_transport_measured": False,
            "causal_intervention": False,
            "head_channel_neuron_scan": False,
            "strict_mechanism_closure": False,
        },
        "theory_audit": {
            "same_distribution_prediction_implies_construct_validity": False,
            "observer_contract_is_explicit_measurement_parameter": True,
            "observer_failure_implies_no_internal_relation_state": False,
            "dynamic_pattern_network": "candidate_experimental_framework_only",
            "mode_operator_identified": False,
            "global_physical_atlas_advanced": False,
        },
        "progress": {
            "point_percent": 25,
            "range_percent": [24, 26],
            "strict_closure_rate": 0.0,
        },
        "stage_decision": {
            "phase500_objective_complete": True,
            "automatic_physical_phase_executed": False,
            "next_action": "freeze_semantic_evaluation_and_label_compiler_as_separate_contracts",
            "reason": (
                "The current complete native contract denominator is empty. A next study changes "
                "the construct and must be frozen separately rather than relaxing this gate."
            ),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_atlas(stage_a, stage_b)
    write_report(audit)
    print(OUT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
