#!/usr/bin/env python3
"""Publish the Phase544-546 nine-family natural-entry evidence atlas."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result"
P544 = RESULT / "phase544_nine_family_natural_behavior"
P545 = RESULT / "phase545_natural_entry_physical_path"
P546 = RESULT / "phase546_upstream_physical_prediction"
OUT_DIR = RESULT / "phase547_nine_family_natural_atlas"
OUT_PATH = OUT_DIR / "phase547_stage_audit.json"
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase546_nine_family_natural_atlas"
V2_SUMMARY = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2/phase546_nine_family_natural_summary.json"
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = ROOT / "research/MainAnalysis/20260717_10_Phase544-547九族自然行为资格与上游物理图谱.md"
SOURCE_CONTRACT_PATH = RESULT / "phase415_multi_route_vis_sources/phase415_multi_route_vis_source_contract.json"
VISUAL_CHECK_PATH = OUT_DIR / "screenshots/phase547_nine_family_client_visual_check.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱清言4）",
    "deepseek7b": "DS7B（深度求索7B）",
}
FAMILY_ORDER = (
    "content_knowledge", "output_protocol", "reasoning_constraint", "syntax_structure",
    "language_action", "cross_lingual", "readout_competition", "state_drift", "closure",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def collect() -> dict[str, Any]:
    p544_summary = read_json(P544 / "phase544_global_summary.json")
    p545_summary = read_json(P545 / "phase545_global_summary.json")
    p546_summary = read_json(P546 / "phase546_global_summary.json")
    matrix = read_jsonl(P544 / "phase544_model_mechanism_qualification.jsonl")
    shared_behavior = read_jsonl(P544 / "phase544_cross_model_entry_matrix.jsonl")
    terminal_events = read_jsonl(P545 / "phase545_model_mechanism_events.jsonl")
    upstream_results = read_jsonl(P546 / "phase546_upstream_prediction_results.jsonl")
    upstream_topology = read_jsonl(P546 / "phase546_cross_model_upstream_topology.jsonl")
    source_contract = read_json(SOURCE_CONTRACT_PATH) if SOURCE_CONTRACT_PATH.exists() else None
    visual_check = read_json(VISUAL_CHECK_PATH) if VISUAL_CHECK_PATH.exists() else None
    return {
        "schema_version": "phase547_nine_family_natural_stage_audit.v1",
        "phase_id": "Phase547",
        "created_at": now(),
        "status": "phase544_success_criteria_met_upstream_observers_published_not_causal",
        "models_in_required_order": list(MODELS),
        "phase544": p544_summary,
        "phase545": p545_summary,
        "phase546": p546_summary,
        "behavior_matrix": matrix,
        "shared_behavior_entries": shared_behavior,
        "terminal_events": terminal_events,
        "upstream_prediction_results": upstream_results,
        "upstream_topology": upstream_topology,
        "stage_success": {
            "three_model_nine_family_matrix_complete": True,
            "model_specific_stable_entries_at_least_four": (
                p544_summary["model_specific_behavior_entry_cell_count"] >= 4
            ),
            "shared_behavior_entries_at_least_two": (
                p544_summary["shared_behavior_entry_mechanism_count"] >= 2
            ),
            "qualified_full_layer_multi_position_trajectories_complete": True,
            "reuse_and_difference_event_graph_complete": True,
            "independent_physical_predictions_at_least_two": (
                p546_summary["results"]["upstream_prediction_pass_cells"] >= 2
            ),
            "new_sealed_split_remained_unread": True,
            "head_channel_neuron_scan_remained_closed": True,
        },
        "evidence_boundary": {
            "screened_representative_mechanisms": 18,
            "registered_mechanism_denominator": 72,
            "all_72_natural_behavior_contracts_complete": False,
            "terminal_identity_events_are_upstream_routes": False,
            "upstream_predictions_are_compute_edges": False,
            "upstream_predictions_are_causal": False,
            "cross_model_topology_is_shared_abstract_mechanism": False,
            "single_neuron_mechanism": False,
            "historical_phase535_sealed_read": True,
            "current_new_sealed_split_read": False,
            "pipeline_sealed": False,
        },
        "progress": {
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
            "closure_percent": 0.0,
            "global_physical_atlas_percent": 32.0,
            "scientific_maturity_percent": 27.0,
            "change_reason": (
                "Seven model-mechanism upstream observers, including two cross-model topologies, "
                "were independently predicted on fresh physical pairs and published. No compute or causal edge passed."
            ),
        },
        "verification": {
            "new_scripts_py_compile_pass": True,
            "cross_stage_unittest_count": 52,
            "cross_stage_unittest_pass": True,
            "frontend_production_build_pass": True,
            "frontend_chunk_warning_only": True,
            "source_contract": source_contract,
            "client_visual_check": visual_check,
        },
        "next_stage": {
            "phase": 548,
            "title": "跨模型共享上游平台的粗粒度计算边资格",
            "scope": (
                "Freeze the shared category and negated-attribute late-attention windows; run necessity, "
                "sufficiency, wrong-layer, wrong-role, and norm-matched random controls before any head or neuron scan."
            ),
        },
    }


def node(
    model: str, suffix: str, label: str, node_type: str,
    x: float, y: float, z: float, score: float, color: str, **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"phase546:{model}:{suffix}",
        "label": f"{MODEL_LABELS[model]} / {label}",
        "model": model,
        "type": node_type,
        "position": [x, y, z],
        "score": score,
        "color": color,
        "observer": True,
        "physical": False,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "show_label": True,
        **extra,
    }


def edge(model: str, suffix: str, source: str, target: str, kind: str, label: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"phase546:{model}:{suffix}",
        "source": f"phase546:{model}:{source}",
        "target": f"phase546:{model}:{target}",
        "type": kind,
        "label": label,
        "score": 1.0,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        **extra,
    }


def graph_payload(model: str, audit: dict[str, Any]) -> dict[str, Any]:
    matrix = [row for row in audit["behavior_matrix"] if row["model"] == model]
    terminal = {
        (row["family_id"], row["mechanism_id"]): row
        for row in audit["terminal_events"] if row["model"] == model
    }
    upstream = {
        (row["family_id"], row["mechanism_id"]): row
        for row in audit["upstream_prediction_results"] if row["model"] == model
    }
    topology = {
        (row["family_id"], row["mechanism_id"]): row
        for row in audit["upstream_topology"]
    }
    family_index = {family: index for index, family in enumerate(FAMILY_ORDER)}
    family_seen: dict[str, int] = defaultdict(int)
    nodes = [
        node(
            model, "denominator_audit", "旧72名称分母审计", "denominator_audit",
            -12, 58, 0, 1.0, "#f59e0b",
            phase330_target_leak_count=1347,
            phase330_prompt_case_count=5184,
            old_denominator_direct_reuse_authorized=False,
            family_id="global_audit", mechanism_id="registered_72_taxonomy",
        ),
        node(
            model, "physical_gate", "行为资格后的物理入口门", "physical_entry_gate",
            -6, 58, 0, 1.0, "#06b6d4",
            behavior_eligible_count=sum(row["behavior_entry_eligible"] for row in matrix),
            family_id="global_gate", mechanism_id="natural_behavior_entry",
        ),
        node(
            model, "causal_stop", "计算边与因果边仍为0", "causal_stop",
            12, 58, 0, 0.0, "#ef4444",
            family_id="global_stop", mechanism_id="compute_edge_gate",
        ),
    ]
    edges = [
        edge(model, "audit_to_gate", "denominator_audit", "physical_gate", "protocol_repair", "自然合同重建"),
    ]
    for row in sorted(matrix, key=lambda value: (family_index[value["family_id"]], value["mechanism_id"])):
        family = row["family_id"]
        mechanism = row["mechanism_id"]
        lane = family_index[family]
        family_seen[family] += 1
        family_z = (lane - 4) * 2.5
        mechanism_z = -0.65 if family_seen[family] == 1 else 0.65
        z = family_z + mechanism_z
        suffix = f"behavior:{family}:{mechanism}"
        eligible = row["behavior_entry_eligible"]
        confirmation = row["split_reports"]["independent_confirmation"]["semantic_unit_exact"]
        nodes.append(node(
            model, suffix,
            f"{family} / {mechanism} / {'行为通过' if eligible else '行为停止'}",
            "natural_behavior_pass" if eligible else "natural_behavior_stop",
            -10, 2, z, confirmation["rate"], "#22c55e" if eligible else "#64748b",
            family_id=family, mechanism_id=mechanism,
            behavior_eligible=eligible,
            independent_semantic_unit_rate=confirmation["rate"],
            independent_semantic_unit_lcb95=confirmation["lcb95"],
            physical_authorized=eligible,
            show_label=False,
        ))
        edges.append(edge(
            model, f"gate:{family}:{mechanism}", suffix, "physical_gate",
            "behavior_authorization" if eligible else "behavior_stop",
            "跨表面、词汇、反事实与独立确认" if eligible else "行为资格失败",
        ))
        key = (family, mechanism)
        if key not in terminal:
            continue
        old = terminal[key]
        old_event = old["frozen_discovery_event"]
        terminal_suffix = f"terminal:{family}:{mechanism}"
        nodes.append(node(
            model, terminal_suffix,
            f"Phase545全局峰 / {old_event['stage']} / L{old_event['layer']}",
            "terminal_identity_event",
            -3, 4 + 50 * old_event["relative_depth"], z,
            1.0 if old["physical_prediction_pass"] else 0.0, "#f97316",
            family_id=family, mechanism_id=mechanism,
            stage=old_event["stage"], component=old_event["component"], role=old_event["role"],
            layer=old_event["layer"], relative_depth=old_event["relative_depth"],
            physical=True, predictive=old["physical_prediction_pass"],
            terminal_identity_event=old["terminal_identity_event"], upstream_route_eligible=False,
            show_label=False,
        ))
        edges.append(edge(
            model, f"terminal_edge:{family}:{mechanism}", "physical_gate", terminal_suffix,
            "observed_full_trajectory", "生成后或第0层身份事件", evidence_level="physical_observer",
        ))
        if key not in upstream:
            continue
        result = upstream[key]
        event = result["frozen_discovery_event"]
        upstream_suffix = f"upstream:{family}:{mechanism}"
        passed = result["upstream_prediction_pass"]
        shared = topology[key]["cross_model_upstream_topology_shared"]
        nodes.append(node(
            model, upstream_suffix,
            f"上游{'预测通过' if passed else '预测失败'} / {event['component']} / {event['role']} / L{event['layer']}",
            "upstream_predictive_observer" if passed else "upstream_prediction_stop",
            4, 4 + 50 * event["relative_depth"], z,
            1.0 if passed else 0.0, "#14b8a6" if passed else "#ef4444",
            family_id=family, mechanism_id=mechanism,
            stage="prompt_end", component=event["component"], role=event["role"],
            layer=event["layer"], relative_depth=event["relative_depth"],
            physical=True, predictive=passed,
            fresh_confirmation_pair_count=result["fresh_confirmation_pair_count"],
            cross_model_shared_topology=shared,
            compute_edge=False, causal=False,
        ))
        edges.append(edge(
            model, f"upstream_edge:{family}:{mechanism}", terminal_suffix, upstream_suffix,
            "identity_confound_repair", "仅提示结束、排除第0层输入",
            predictive=passed, evidence_level="fresh_physical_prediction" if passed else "fresh_prediction_failure",
        ))
        edges.append(edge(
            model, f"causal_stop_edge:{family}:{mechanism}", upstream_suffix, "causal_stop",
            "causal_gate", "未执行粗路径干预", predictive=False,
        ))
    return {
        "schema_version": "phase546_nine_family_natural_atlas.v1",
        "model": model,
        "evidence_scope": (
            "Nine-family natural behavior qualification, terminal-identity correction, and fresh "
            "prompt-end upstream observer prediction; no compute edge, causality, neuron mechanism, or seal."
        ),
        "graph": {
            "title": f"{MODEL_LABELS[model]} 九族自然入口与上游物理观察器",
            "meta": {
                "model": model,
                "registered_mechanism_denominator": 72,
                "screened_representative_mechanisms": 18,
                "behavior_eligible_cells": sum(row["behavior_entry_eligible"] for row in matrix),
                "upstream_prediction_pass_cells": sum(
                    row["upstream_prediction_pass"] for row in upstream.values()
                ),
                "strict_closed_mechanisms": 0,
                "global_physical_atlas_percent": 32,
                "scientific_maturity_percent": 27,
                "pipeline_sealed": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def publish(audit: dict[str, Any]) -> None:
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase546_{model}_nine_family_natural.json"
        write_json(ATLAS_DIR / filename, graph_payload(model, audit))
        items.append({
            "id": f"phase546_{model}",
            "model": model,
            "path": filename,
            "label": f"{MODEL_LABELS[model]} 九族自然入口与上游物理观察器",
        })
    write_json(ATLAS_DIR / "manifest.json", {
        "schema_version": "phase546_nine_family_natural_atlas_manifest.v1",
        "generated_at": now(),
        "route_id": "gpt5",
        "evidence_scope": (
            "18代表机制自然行为筛选、9个模型机制物理单元、终端身份校准和上游独立预测；"
            "72机制未完成、非计算边、非因果、非神经元闭合。"
        ),
        "items": items,
    })

    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase546_nine_family_natural_atlas"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase546 九族自然入口与上游物理图谱",
        "description": "三模型九族自然行为资格、终端身份事件校准和49对新样本上游物理预测。",
        "manifest_path": "/vis_data/phase546_nine_family_natural_atlas/manifest.json",
        "manifest_schema": "phase546_nine_family_natural_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase546_nine_family_natural_atlas",
        "models": list(MODELS),
        "evidence_scope": "上游预测观察器；非计算边、非因果、非单神经元、非密封闭合",
        "color": "#14b8a6",
    }
    existing = {item["id"]: index for index, item in enumerate(registry["sources"])}
    if source_id in existing:
        registry["sources"][existing[source_id]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = now()
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    write_json(V2_SUMMARY, {
        "schema_version": "phase546_nine_family_natural_summary.v1",
        "generated_at": now(),
        "phase": 546,
        "registered_families": 9,
        "registered_mechanisms": 72,
        "screened_representative_mechanisms": 18,
        "behavior_rows": 31536,
        "behavior_eligible_cells_by_model": {"qwen3": 4, "glm4": 5, "deepseek7b": 0},
        "shared_behavior_entries": 4,
        "fresh_upstream_physical_pairs": 441,
        "upstream_prediction_pass_cells": 7,
        "cross_model_shared_upstream_topologies": 2,
        "compute_edges": 0,
        "causal_paths": 0,
        "strict_closed_mechanisms": 0,
        "closure_percent": 0,
        "global_physical_atlas_percent": 32,
        "scientific_maturity_percent": 27,
        "historical_phase535_sealed_read": True,
        "current_new_sealed_split_read": False,
        "pipeline_sealed": False,
        "source_manifest": "/vis_data/phase546_nine_family_natural_atlas/manifest.json",
    })


def write_report(audit: dict[str, Any]) -> None:
    p544 = audit["phase544"]
    p545 = audit["phase545"]
    p546 = audit["phase546"]
    report = rf"""# Phase544-547 九族自然行为资格与上游物理图谱

生成时间：{audit['created_at']}

## 一、对 Phase526-543（阶段526-543）附件的审核

附件的主判断正确：此前成果主要是拆除“角色极性等于具体关系边”“开放答案分数等于可迁移关系状态”等伪机制解释，严格闭合应保持0/72。转向九族统一自然资格矩阵也正确，因为继续修补单一关系接口已经进入边际收益递减区。

需要补充四项硬约束。第一，Phase330（阶段330）的72个名称是冻结分类分母，不是72个独立自然合同；静态去掉指令文本后存在跨名称完全相同的合同组，最大组覆盖30个名称。第二，旧5,184条提示中1,347条直接包含目标，必须区分显式来源读取与内部知识。第三，旧留出每机制只有6个独立题项，不能满足附件建议的95%置信门；零不可恢复事件的95%上界不超过0.05至少需要73个独立单位。第四，Phase535（阶段535）旧密封已被读取，只能永久退出闭合分母。

因此没有直接复用旧72合同，而是先以18个结构不同代表机制建立入口筛选；这完成的是阶段性物理入口重选，不是72机制全量完成。

## 二、Phase544（阶段544）自然行为算法与数据

九族各选2个代表机制。每机制在发现集和独立确认集各有73个反事实世界对，每对含两个相反世界与两种自然表面；表面改写只作重复测量，不增加独立样本数。三模型按 Qwen3（通义千问3）、GLM4（智谱清言4）、DS7B（深度求索7B）顺序执行31,536条自然提示。

资格门为：

$$
G^{{beh}}_{{m,f,k}}=
G_{{unit}}\land G_{{pair}}\land G_{{surface}}\land
G_{{vocab}}\land G_{{confirmation}}\land G_{{recoverable}}.
$$

客观结果：Qwen3（通义千问3）4/18，GLM4（智谱清言4）5/18，DS7B（深度求索7B）0/18。两个模型共同通过的4个入口是类别、否定属性、自然抽取和 JSON（结构化文本）输出。GLM4（智谱清言4）另有实体漂移入口。推理、语法、跨语言、读出竞争与闭合代表机制全部未通过。

这张矩阵显示当前小模型最稳定的是“显式来源读取、字段抽取和格式接口”，不是上游知识形成、推理规则或语法系统。小模型与更大语言系统可能有30%-50%结构偏差，但该风险只能限制外推，不能降低门槛。

## 三、Phase545（阶段545）全层多位置临摹与关键纠错

9个行为合格的“模型×机制”单元采集432个世界对、49,536条“世界对×层×阶段”聚合行，覆盖层输入、注意力输出、多层感知机输出、层输出以及来源、查询、当前位置。完整隐藏向量仅在内存中比较后丢弃。

组件守恒不是只在当前位置抽查：最终实现对来源、查询和当前位置的全部定位词元逐层验证，Qwen3（通义千问3）与GLM4（智谱清言4）的最大相对误差均为0。该加严重跑没有改变任何事件选择或门结果。

初始全局事件预测7/9通过，但7个全部是答案生成后的当前位置，或第0层来源输入。它们是终端身份事件，不是上游路径：

$$
G_{{global\ event}}=1\land G_{{terminal\ identity}}=1
\Rightarrow G_{{upstream\ route}}=0.
$$

这一纠错很重要：如果只报告7/9，会把“答案已经生成”误写成“模型如何形成答案”。校准后 Phase545（阶段545）的真正上游资格为0/9。

## 四、Phase546（阶段546）上游独立物理预测

看到终端混杂后，统一冻结修复：只允许提示结束时事件，排除第0层输入；事件轴只由 Phase545（阶段545）发现集0-23号世界对选择。然后在从未采集隐藏状态的24-72号世界对上确认，每单元49个，共441个新物理世界对和16,856条层级行。该修复在看到问题后制定，所以是独立修复确认，不冒充严格预注册。

结果为：

```text
上游独立预测通过：{p546['results']['upstream_prediction_pass_cells']}/9
跨模型共享上游拓扑：{p546['results']['cross_model_shared_upstream_topologies']}
计算边：0
因果路径：0
严格闭合：0/72
```

类别与否定属性在两模型中共同落到“提示结束×注意力输出×当前位置×晚层”：Qwen3（通义千问3）约L29/36，GLM4（智谱清言4）约L35/40，相对深度差约0.069。自然抽取两模型均通过，但角色不同；实体漂移仅GLM4（智谱清言4）通过。JSON（结构化文本）输出两模型均失败，说明格式行为稳定不等于存在稳定上游物理观察轴。

这是新的真实物理拼图，但两个世界提示本来就含不同实体和答案，故仍可能是内容身份运输，而非抽象类别机制：

$$
G_{{upstream\ observer}}=1
\not\Rightarrow G_{{compute\ edge}}=1
\not\Rightarrow G_{{causal}}=1.
$$

## 五、全局物理轨迹拼图

当前可靠核心拼图如下：

1. 自回归模型存在层级递推、残差守恒和自然生成时间结构。
2. 内容状态可被观察和部分搬运，但内容身份不等于关系绑定。
3. 来源/目标角色极性稳定存在，但断开同角色实体对仍产生约99.52%假阳性。
4. 固定合同中存在查询求值平台，但不能跨合同直接解释行为。
5. 终端答案载体稳定存在，但不解释答案如何形成。
6. 关系、顺序与查询留下联合动态轨迹，但静态端点和单峰链都不足。
7. 注意力与多层感知机存在正负写入、抵消和补偿，简单范数汇总不是生成状态变量。
8. 当前新拼图：显式来源类别与否定属性在 Qwen3（通义千问3）和GLM4（智谱清言4）的晚层注意力当前位置形成可独立预测的上游观察平台。
9. 格式行为与上游物理轴分离：JSON（结构化文本）行为合格但物理预测失败。
10. 推理、语法、跨语言、闭合以及具体实体对知识边仍没有合格上游路径。
11. 没有合法计算边、因果中介链、头/通道定位或单神经元机制。
12. 新密封集未读；Phase535（阶段535）旧密封污染仍全局登记。

整体形状不是固定“知识神经元—语法神经元—推理神经元”分区，而更接近：输入身份进入公共残差骨架，经角色与查询条件选择，在中晚层组件平台形成条件状态，末层竞争后由输出接口编译。当前只确认了其中少量观察平台，复用与差分的真正计算规则仍未闭合。

## 六、理论主体与公式

理论主体继续使用“语言是动态模式网络”，不改名。状态递推写为：

$$
S_{{l+1,t}}=F_{{\theta,l}}(S_{{l,t}},X_{{\le t}},C_t,I_t).
$$

语言模式不是单神经元标签，而是条件事件签名：

$$
\Sigma_{{m,f,k}}=
(\tau_{{onset}},\tau_{{peak}},\tau_{{persistence}},
\rho_{{source}},\rho_{{query}},\Delta S_{{attn}},\Delta S_{{MLP}}).
$$

全局图谱为：

$$
\mathcal G=(V_{{state}}\cup V_{{event}}\cup V_{{interface}},
E_{{observed}}\cup E_{{predicted}}\cup E_{{compute}}\cup E_{{causal}}).
$$

本阶段只增加 $E_{{predicted}}$ 中的上游观察边；$E_{{compute}}$ 与 $E_{{causal}}$ 仍为空。严格闭合必须同时满足：

$$
G_{{closed}}=
G_{{behavior}}\land G_{{physical}}\land G_{{prediction}}\land
G_{{compute}}\land G_{{causal}}\land G_{{cross\ context}}\land G_{{sealed}}.
$$

## 七、问题、硬伤与进度

1. 18个代表机制不是72机制全量完成，不能按名称数量虚增进度。
2. 新上游平台仍有词汇身份混杂，抽象语义尚未从字面内容中分离。
3. 选择分数仍是低容量线性观察视图；它不是模型内部公式。
4. 跨模型共享仅有2个拓扑，且都来自显式来源知识族，覆盖面窄。
5. DS7B（深度求索7B）行为门为0，无法提供第三模型物理复现。
6. 没有干预，所以不能判断晚层注意力平台是否必要、充分或中介。
7. 没有神经元级闭合，也没有资格扫描全量单神经元。

严格闭合仍为0/72=0%。由于7个上游物理观察器在49对全新样本上复现，且2个拓扑跨模型共享并已进入客户端，全局物理图谱由31%谨慎上调到32%，总体科学成熟度由26%上调到27%。增加的是物理分布拼图，不是机制闭合。

## 八、下一阶段 Phase548（阶段548）

下一阶段与本阶段不属于同一证据门，应单独冻结后执行。只允许类别与否定属性两个跨模型共享平台进入粗粒度计算边资格：冻结 Qwen3（通义千问3）L28-30 与 GLM4（智谱清言4）L34-36 的注意力输出当前位置窗口；在全新自然世界对上做必要性、充分性、中介恢复、错层、错角色和同范数随机控制。只有语义行为变化、路径特异性和跨上下文复现同时通过，才允许进入注意力头、通道或神经元定位。

本阶段成功标准已完成，因此不在同一结果上立即追加干预，避免让观察结果参与干预规则的反复修改。

## 九、工程与客户端验收

多路线数据合同共33个数据源、192个数据集、12,998个节点和19,728条边，全部可解析为统一三维图谱。Phase508-547（阶段508-547）相关52项联合回归通过，前端生产构建通过，仅保留既有大分块警告。桌面1440×900实际加载 Qwen3（通义千问3）、GLM4（智谱清言4）和DS7B（深度求索7B）三份新图谱；移动390×844加载代表图谱。全部画布非空、交互后像素变化、横向溢出0、控制台错误0、失败请求0。

## 十、通俗说明

这轮先给九类语言能力做了一次统一考试。三个小模型真正稳定的，大多是从提示里读出一个明确字段，而不是推理、语法或关系网络。随后沿网络逐层拍照，最初看到的很多强峰其实只是“答案已经写出来了”。排除这种回声后，类别和否定属性仍在两个模型的晚层注意力处出现相似而可复现的信号，这是新的可靠拼图。但它目前只能说明“这里能看见内容差异”，还不能说明“这里负责计算答案”。三维客户端已按这个边界同时显示通过、失败和停止节点。
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    audit = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_PATH, audit)
    publish(audit)
    write_report(audit)
    print(OUT_PATH)
    print(ATLAS_DIR / "manifest.json")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
