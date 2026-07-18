#!/usr/bin/env python3
"""Publish the Phase526-533 stage audit and client-loadable atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result"
MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱清言4）",
    "deepseek7b": "DS7B（深度求索7B）",
}

PHASE527 = RESULT / "phase527_world_geometry_behavior_qualification"
PHASE529 = RESULT / "phase529_relation_contract_factorial_behavior"
PHASE531 = RESULT / "phase531_glm4_fresh_world_geometry_behavior"
PHASE532 = RESULT / "phase532_glm4_role_normalized_world_geometry"
PHASE533_PATH = (
    RESULT
    / "phase533_world_geometry_role_binding_audit"
    / "phase533_world_geometry_role_binding_audit.json"
)
OUT_DIR = RESULT / "phase534_world_geometry_stage_audit"
OUT_PATH = OUT_DIR / "phase534_world_geometry_stage_audit.json"
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase533_world_geometry_role_binding_atlas"
V2_SUMMARY = (
    ROOT
    / "frontend/public/vis_data/pattern_family_atlas/v2"
    / "phase533_world_geometry_role_binding_summary.json"
)
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = (
    ROOT
    / "research/MainAnalysis/20260717_07_Phase526-533角色规范化世界几何与关系合同全因子审计.md"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def split_compact(summary: dict[str, Any], split: str) -> dict[str, Any]:
    report = summary["split_reports"][split]
    return {
        "overall_first_event": report["overall_first_event"],
        "four_way_source_pair": report["four_way_source_pair"],
        "strict_whole_response": report["strict_whole_response"],
        "gate_pass": report["gate_pass"],
    }


def collect_audit() -> dict[str, Any]:
    initial = {
        model: read_json(PHASE527 / f"phase527_{model}_behavior_summary.json")
        for model in MODELS
    }
    factorial = {
        model: read_json(PHASE529 / f"phase529_{model}_factorial_summary.json")
        for model in MODELS
    }
    factorial_authorization = read_json(PHASE529 / "phase529_factorial_authorization.json")
    fresh_authorization = read_json(PHASE531 / "phase531_fresh_physical_authorization.json")
    fresh_glm = read_json(PHASE531 / "phase531_glm4_fresh_behavior_summary.json")
    physical = read_json(PHASE532 / "phase532_glm4_world_geometry_summary.json")
    role_binding = read_json(PHASE533_PATH)
    glm_factors = factorial["glm4"]["stage_reports"]

    model_reports: dict[str, Any] = {}
    for model in MODELS:
        model_reports[model] = {
            "initial_geometry_behavior": {
                split: split_compact(initial[model], split)
                for split in (
                    "world_fit",
                    "world_entity_prediction",
                    "world_relation_prediction",
                    "bridge_open_prediction",
                )
            },
            "factorial_confirmed_conditions": factorial[model]["confirmed_conditions"],
            "fresh_physical_authorized": model in fresh_authorization["fresh_physical_authorized_models"],
            "fresh_behavior": fresh_glm["split_reports"] if model == "glm4" else None,
            "physical_world_geometry": physical if model == "glm4" else None,
            "role_binding_decomposition": role_binding if model == "glm4" else None,
        }

    return {
        "schema_version": "phase534_world_geometry_stage_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_stopped_before_prediction",
        "scope": "Phase526-533 role-normalized world geometry and relation-contract factorial stage",
        "models_in_required_order": list(MODELS),
        "model_reports": model_reports,
        "factorial_findings": {
            "confirmed_conditions_by_model": factorial_authorization["confirmed_conditions_by_model"],
            "shared_confirmed_conditions": factorial_authorization["shared_confirmed_conditions"],
            "glm4_calibration_factor_margins": glm_factors["calibration"]["factor_margins"],
            "glm4_confirmation_factor_margins": glm_factors["confirmation"]["factor_margins"],
            "dominant_behavioral_factor": "graph_shape",
        },
        "physical_findings": {
            "fresh_physical_authorized_models": fresh_authorization["fresh_physical_authorized_models"],
            "orientation_accuracy_by_projection": [
                item["orientation_accuracy"] for item in role_binding["selected_cells"]
            ],
            "node_role_accuracy_by_projection": [
                item["node_source_target_role_accuracy"] for item in role_binding["selected_cells"]
            ],
            "selected_layers_with_embedding": [
                item["layer_with_embedding"] for item in role_binding["selected_cells"]
            ],
            "mean_disconnected_source_to_target_false_positive": role_binding[
                "mean_source_to_target_disconnected_false_positive"
            ],
            "node_role_polarity_models": ["glm4"],
            "pair_specific_edge_binding_models": [],
            "world_relation_platform_models": [],
            "causal_models": [],
        },
        "stop_accounting": {
            "discovery_platform_count": physical["platform_count"],
            "entity_prediction_split_read": False,
            "relation_prediction_split_read": False,
            "permutation_replicates_run": 0,
            "permutation_stop_reason": "no discovery platform",
            "sealed_split_read": False,
        },
        "attachment_audit": {
            "overall_direction_correct": True,
            "accepted": [
                "role-normalized entity geometry is a valid replacement for the cancelled global mean direction",
                "world and query states must remain separated",
                "entity and relation holdouts must precede any physical claim",
                "prediction and permutation are conditional on a frozen discovery platform",
            ],
            "corrections": [
                "a fitted swap operator over a manually concatenated pair feature would test feature bookkeeping, not model equivariance",
                "only entity-register anchor tokens require stable token mapping; exact token identity for the complete prefix is invalid",
                "a trainable bilinear bridge requires strict low-capacity controls because it can memorize relation labels",
                "the entity register is an observer scaffold and cannot be called a spontaneous natural world state",
            ],
        },
        "evidence_boundary": {
            "observational_hidden_state": True,
            "scaffold_conditioned": True,
            "predictive_edge_binding": False,
            "compute_transport": False,
            "causal": False,
            "component_head_channel_neuron": False,
            "cross_model_shared_mechanism": False,
            "sealed": False,
        },
        "progress": {
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
            "closure_percent": 0,
            "global_physical_atlas_percent": 31,
            "overall_research_percent": 26,
            "change_reason": (
                "a scaffold-conditioned GLM4 node-role signature was added, but pair binding, "
                "natural world state, prediction, transport, causality, and sealed closure remain absent"
            ),
        },
        "next_stage": {
            "phase": 535,
            "title": "成对地址保持的世界关系绑定图谱",
            "priority": "distinguish pair identity from generic source/target role before any component or neuron scan",
        },
    }


def node(
    model: str,
    suffix: str,
    label: str,
    node_type: str,
    position: list[float],
    score: float,
    color: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"phase533:{model}:{suffix}",
        "label": f"{MODEL_LABELS[model]} / {label}",
        "model": model,
        "family_id": "reasoning_relation_binding",
        "mechanism_id": "world_relation_binding",
        "type": node_type,
        "position": position,
        "score": score,
        "color": color,
        "observer": True,
        "physical": bool(extra.pop("physical", False)),
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "show_label": True,
        **extra,
    }


def edge(model: str, suffix: str, source: str, target: str, edge_type: str, label: str) -> dict[str, Any]:
    return {
        "id": f"phase533:{model}:{suffix}",
        "source": f"phase533:{model}:{source}",
        "target": f"phase533:{model}:{target}",
        "type": edge_type,
        "label": label,
        "score": 1.0,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "evidence_level": "protocol_or_offline_observer_decomposition",
    }


def graph_payload(model: str, audit: dict[str, Any]) -> dict[str, Any]:
    report = audit["model_reports"][model]
    initial = report["initial_geometry_behavior"]["world_fit"]
    confirmed = report["factorial_confirmed_conditions"]
    fresh = report["fresh_behavior"]
    nodes = [
        node(
            model,
            "initial_behavior",
            "原始四边世界行为门失败",
            "behavior_gate_failure",
            [-34.0, -14.0, 0.0],
            initial["overall_first_event"]["rate"],
            "#ef4444",
            four_way_rate=initial["four_way_source_pair"]["rate"],
            evidence_level="behavior_gate_failure",
        ),
        node(
            model,
            "factorial_contract",
            f"全因子确认条件 {len(confirmed)} 个",
            "factorial_behavior_contract",
            [-18.0, -7.0, 0.0],
            1.0 if confirmed else 0.0,
            "#f59e0b" if confirmed else "#64748b",
            confirmed_condition_count=len(confirmed),
            evidence_level="independent_behavior_confirmation" if confirmed else "no_confirmed_condition",
        ),
    ]
    edges = [
        edge(
            model,
            "initial_to_factorial",
            "initial_behavior",
            "factorial_contract",
            "diagnostic_protocol_transition",
            "全因子拆分行为混杂",
        )
    ]

    if model == "glm4":
        fresh_discovery = fresh["discovery"]
        role = report["role_binding_decomposition"]
        nodes.extend([
            node(
                model,
                "fresh_behavior",
                "新鲜两边合同三分割通过",
                "fresh_behavior_authorization",
                [-2.0, 0.0, 0.0],
                fresh_discovery["overall"]["rate"],
                "#22c55e",
                entity_prediction_rate=fresh["entity_prediction"]["overall"]["rate"],
                relation_prediction_rate=fresh["relation_prediction"]["overall"]["rate"],
                evidence_level="fresh_behavior_authorization",
            ),
            node(
                model,
                "node_role_polarity",
                "来源/目标节点角色极性",
                "scaffold_conditioned_node_role_signature",
                [14.0, 0.0, 0.0],
                sum(item["node_source_target_role_accuracy"] for item in role["selected_cells"]) / 3,
                "#06b6d4",
                physical=True,
                selected_layers=[item["layer_with_embedding"] for item in role["selected_cells"]],
                projection_accuracies=[item["node_source_target_role_accuracy"] for item in role["selected_cells"]],
                evidence_level="scaffold_conditioned_observational_signature",
            ),
            node(
                model,
                "orientation_signal",
                "真边相对反向边方向准确率",
                "pair_orientation_observer",
                [28.0, -9.0, 0.0],
                1.0,
                "#06b6d4",
                physical=True,
                evidence_level="offline_oof_observational_signature",
            ),
            node(
                model,
                "disconnected_false_positive",
                "断开来源→目标假阳性 99.52%",
                "binding_confound",
                [28.0, 9.0, 0.0],
                role["mean_source_to_target_disconnected_false_positive"],
                "#ef4444",
                physical=True,
                evidence_level="qualified_binding_counterexample",
            ),
            node(
                model,
                "pair_binding_missing",
                "实体对特异绑定未恢复",
                "pair_specific_binding_missing",
                [44.0, 0.0, 0.0],
                0.0,
                "#ef4444",
                evidence_level="qualified_negative_result",
            ),
            node(
                model,
                "prediction_stopped",
                "预测与1024次置换按门停止",
                "prediction_stop",
                [58.0, 0.0, 0.0],
                0.0,
                "#64748b",
                evidence_level="preregistered_stop_rule",
            ),
        ])
        edges.extend([
            edge(model, "factorial_to_fresh", "factorial_contract", "fresh_behavior", "behavior_selection", "冻结最小可用合同"),
            edge(model, "fresh_to_role", "fresh_behavior", "node_role_polarity", "physical_authorization_gate", "行为门授权隐藏状态观察"),
            edge(model, "role_to_orientation", "node_role_polarity", "orientation_signal", "observer_decomposition", "角色极性产生方向分数"),
            edge(model, "role_to_false_positive", "node_role_polarity", "disconnected_false_positive", "observer_decomposition", "断开实体对反证"),
            edge(model, "orientation_to_binding", "orientation_signal", "pair_binding_missing", "evidence_downgrade", "方向正确不等于边绑定"),
            edge(model, "false_positive_to_binding", "disconnected_false_positive", "pair_binding_missing", "counterexample", "角色匹配制造假边"),
            edge(model, "binding_to_stop", "pair_binding_missing", "prediction_stopped", "preregistered_stop", "发现平台为零"),
        ])
    else:
        nodes.append(node(
            model,
            "physical_not_authorized",
            "新鲜物理测试未获授权",
            "physical_not_authorized",
            [0.0, 0.0, 0.0],
            0.0,
            "#64748b",
            evidence_level="behavior_authorization_failure",
        ))
        edges.append(edge(
            model,
            "factorial_to_noauth",
            "factorial_contract",
            "physical_not_authorized",
            "physical_authorization_gate",
            "无独立确认条件，不加载模型做物理扫描",
        ))

    return {
        "schema_version": "phase533_world_geometry_role_binding_atlas.v1",
        "model": model,
        "evidence_scope": (
            "factorial behavior qualification and scaffold-conditioned role-signature decomposition; "
            "no pair binding, prediction platform, compute edge, causality, neuron, or sealed closure"
        ),
        "graph": {
            "meta": {
                "model": model,
                "confirmed_factorial_condition_count": len(confirmed),
                "fresh_physical_authorized": report["fresh_physical_authorized"],
                "pair_specific_edge_binding": False,
                "world_relation_platform_count": 0,
                "permutation_replicates_run": 0,
                "sealed_split_read": False,
                "strict_closed_mechanisms": 0,
                "mechanism_denominator": 72,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def publish_atlas(audit: dict[str, Any]) -> None:
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase533_{model}_world_geometry_role_binding.json"
        write_json(ATLAS_DIR / filename, graph_payload(model, audit))
        items.append({
            "id": f"phase533_{model}",
            "model": model,
            "path": filename,
            "label": f"{MODEL_LABELS[model]} 世界几何—角色绑定审计",
        })
    write_json(ATLAS_DIR / "manifest.json", {
        "schema_version": "phase533_world_geometry_role_binding_atlas_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": "gpt5",
        "evidence_scope": "节点角色极性局部正结果与实体对绑定强负结果；非计算边、非因果、非神经元、未读密封集",
        "items": items,
    })

    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase533_world_geometry_role_binding_atlas"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase533 世界几何—角色绑定审计图谱",
        "description": "三模型关系合同全因子资格链、GLM4节点角色极性及实体对绑定反证。",
        "manifest_path": "/vis_data/phase533_world_geometry_role_binding_atlas/manifest.json",
        "manifest_schema": "phase533_world_geometry_role_binding_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase533_world_geometry_role_binding_atlas",
        "models": list(MODELS),
        "evidence_scope": "脚手架条件观察签名与绑定反证；非预测、非计算边、非因果、非神经元闭合",
        "color": "#06b6d4",
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

    write_json(V2_SUMMARY, {
        "schema_version": "phase533_world_geometry_role_binding_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 533,
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "closure_percent": 0,
        "overall_research_percent": 26,
        "global_physical_atlas_percent": 31,
        "node_role_polarity_models": ["glm4"],
        "pair_specific_edge_binding_models": [],
        "world_relation_platform_models": [],
        "causal_models": [],
        "permutation_replicates_run": 0,
        "sealed_split_read": False,
        "source_manifest": "/vis_data/phase533_world_geometry_role_binding_atlas/manifest.json",
    })


def write_report(audit: dict[str, Any]) -> None:
    glm = audit["model_reports"]["glm4"]
    factors = audit["factorial_findings"]
    physical = audit["physical_findings"]
    q_fit = audit["model_reports"]["qwen3"]["initial_geometry_behavior"]["world_fit"]
    g_fit = glm["initial_geometry_behavior"]["world_fit"]
    d_fit = audit["model_reports"]["deepseek7b"]["initial_geometry_behavior"]["world_fit"]
    fresh = glm["fresh_behavior"]
    factor_cal = factors["glm4_calibration_factor_margins"]["graph_shape"]
    factor_con = factors["glm4_confirmation_factor_margins"]["graph_shape"]
    role_acc = physical["node_role_accuracy_by_projection"]
    false_positive = physical["mean_disconnected_source_to_target_false_positive"]

    report = rf"""# Phase526-533 角色规范化世界几何与关系合同全因子审计

生成时间：{audit['created_at']}

## 一、总体判断

附件的主方向正确：Phase518-525（阶段518-525）没有证明世界状态不存在，只否定了单位置全局均值差观察器；下一步应当保持世界区与查询区分离，转向角色规范化的实体对几何，并把实体、关系词和表面形式留出。

但有四处必须修正。第一，对人工拼接的成对特征拟合交换算子，只会验证特征重排，不会证明模型内部等变。第二，要求追加后完整前缀的分词逐词元完全相同并不合理，只需冻结实体寄存器锚点。第三，高容量双线性桥容易记忆关系词，不能直接作为世界结构。第四，实体寄存器是观察脚手架，不是模型自然形成的世界状态。

本轮已依次完成三模型行为资格、全因子合同诊断、GLM4（智谱清言4）独立新鲜确认、角色规范化物理发现和离线绑定反证。最终结论是：

$$
\boxed{{\text{{来源/目标角色极性可见}}\;\ne\;\text{{具体实体对关系绑定可见}}}}
$$

## 二、协议和样本

1. Phase526（阶段526）构造四节点有向环：发现384行，实体留出768行，关系留出768行，桥接开放集768行，密封集768行。世界前缀不含查询，事实词元袋、入度、出度和实体寄存器槽位全部平衡。
2. Phase527（阶段527）按 Qwen3（通义千问3）、GLM4（智谱清言4）、DS7B（深度求索7B）顺序运行。三模型四分割均未通过行为门。
3. Phase528-529（阶段528-529）冻结五因素全因子设计：图形为两边/四环、伪边为反向/断开、查询为主动/被动、寄存器为有/无、表面为同形/释义；校准1024行、确认2048行、密封2048行。
4. Phase530-531（阶段530-531）从 GLM4（智谱清言4）独立确认条件中确定“两边、反向、主动、有寄存器”，再生成完全新鲜的发现384行、实体预测768行、关系预测768行和未读密封768行。
5. Phase532（阶段532）只读取发现分割，使用三个固定48维投影、四折组外验证和实体规范重排。发现平台为0后，实体预测、关系预测、1024次全流程置换和密封集全部按规则停止。

## 三、客观行为结果

原始四边世界的发现分割：

- Qwen3（通义千问3）：{q_fit['overall_first_event']['count']}/{q_fit['overall_first_event']['n']}，四联组 {q_fit['four_way_source_pair']['count']}/{q_fit['four_way_source_pair']['n']}。
- GLM4（智谱清言4）：{g_fit['overall_first_event']['count']}/{g_fit['overall_first_event']['n']}，四联组 {g_fit['four_way_source_pair']['count']}/{g_fit['four_way_source_pair']['n']}。
- DS7B（深度求索7B）：{d_fit['overall_first_event']['count']}/{d_fit['overall_first_event']['n']}，四联组 {d_fit['four_way_source_pair']['count']}/{d_fit['four_way_source_pair']['n']}。

全因子结果只有 GLM4（智谱清言4）在校准与确认共有五个条件通过，Qwen3（通义千问3）和DS7B（深度求索7B）均为0。GLM4（智谱清言4）中最强因素是图形规模：校准两边 {factor_cal['two_edge']['rate']:.4f}、四环 {factor_cal['four_cycle']['rate']:.4f}；确认两边 {factor_con['two_edge']['rate']:.4f}、四环 {factor_con['four_cycle']['rate']:.4f}。反向伪边与断开伪边差异远小于两边与四环差异，因此此前失败的主要来源是关系负载/结构形状，不是某一个寄存器或伪边类型。

GLM4（智谱清言4）新鲜合同通过：发现 {fresh['discovery']['overall']['count']}/{fresh['discovery']['overall']['n']}，实体留出 {fresh['entity_prediction']['overall']['count']}/{fresh['entity_prediction']['overall']['n']}，关系留出 {fresh['relation_prediction']['overall']['count']}/{fresh['relation_prediction']['overall']['n']}。这只授权物理观察，不直接增加机制证据。

## 四、特征算法与反证

实体寄存器位置的隐藏状态先经过固定投影：

$$
z_{{e,l,p}}=P_p h_{{e,l}}.
$$

按实体身份恢复规范顺序后，对有序实体对构造低容量特征：

$$
\phi_{{l,p}}(a,b)=\left[z_{{b,l,p}}-z_{{a,l,p}}\;;\;z_{{a,l,p}}\odot z_{{b,l,p}}\right].
$$

四折组外观察器只比较真实边与反向边：

$$
s_{{l,p}}(a,b)=w_{{l,p}}^\top\phi_{{l,p}}(a,b),
\qquad
s(a,b)>s(b,a).
$$

三个投影在 L14、L19、L16 的方向准确率均为1.0；节点来源/目标角色准确率分别为 {role_acc[0]:.4f}、{role_acc[1]:.4f}、{role_acc[2]:.4f}。如果只看这一步，很容易误写成“关系方向已恢复”。

关键反证来自断开实体对。对于不存在真实边、但第一个实体是任一来源节点、第二个实体是任一目标节点的组合，观察器仍判为边的比例是：

$$
\boxed{{\operatorname{{FPR}}_{{\mathrm{{source}}\to\mathrm{{target}},\,\mathrm{{disconnected}}}}={false_positive:.4%}}}.
$$

因此当前分数更接近：

$$
s(a,b)\approx r_{{\mathrm{{source}}}}(a)+r_{{\mathrm{{target}}}}(b),
$$

而不是包含实体对身份的关系绑定：

$$
s(a,b)\not\approx r_{{\mathrm{{source}}}}(a)+r_{{\mathrm{{target}}}}(b)+B(a,b\mid W).
$$

当前算法恢复了节点角色极性，没有恢复 $B(a,b\mid W)$。

## 五、新增核心拼图

1. 四边世界行为失败不能直接归因为模型没有关系能力。
2. GLM4（智谱清言4）的主要行为瓶颈是同时关系边数量/图形结构。
3. 寄存器有无和反向/断开伪边不是主导因素。
4. GLM4（智谱清言4）在完全新鲜的两边合同上跨实体和关系词留出通过。
5. 三个固定投影均能从实体寄存器恢复来源/目标角色极性。
6. 真实边相对反向边的方向排序可达100%。
7. 断开来源—目标实体对假阳性达到99.52%，严格否定实体对特异绑定。
8. 实体寄存器主要保留节点角色，不保留“谁与谁相连”的充分证据。
9. 世界关系发现平台仍为0。
10. 预测、置换和密封集均因发现门失败而未读取。
11. 三模型没有共享行为条件或共享物理机制。
12. 严格闭合仍为0/72。

## 六、问题与硬伤

1. 正信号依赖人工实体寄存器，不能称为自然世界状态。
2. 当前实体角色由两边图形清晰赋予，来源/目标极性可能是句法位置和局部上下文的合成结果。
3. 成对特征中的差分项天然强化方向角色，必须使用断开同角色实体对控制；此前没有这个控制会产生严重假阳性。
4. GLM4（智谱清言4）是唯一获新鲜物理授权的模型，不存在跨模型复制。
5. 物理发现只使用48维随机投影；虽然三投影结论一致，仍不能排除丢失高维绑定信息。
6. 关系预测集按停止规则未读，因此没有实体对绑定的开放预测结果。
7. 没有组件、注意力头、通道、单神经元、必要性或充分性证据。
8. 任务仍是合成两边关系，不覆盖自然知识网络、多步推理和语法系统。
9. 小模型编码可能与大型模型或真实语言机制相差30%-50%，本轮只适合作为算法校准。

## 七、理论更新

理论主体继续使用“语言是动态模式网络”，不改名。本轮不是新增统一理论，而是删除一个错误等价：

$$
\text{{方向角色可分}}
\;\not\Rightarrow\;
\text{{实体对绑定可分}}.
$$

更完整的候选状态仍应写成条件化动态递推：

$$
S_{{l+1,t}}=F_{{\theta,l}}(S_{{l,t}},X_{{\le t}},C_t,I_t),
$$

而具体关系至少需要三部分共同存在：

$$
R_l(a,b\mid W)=G_l\!\left(U_l(a),V_l(b),B_l(a,b\mid W)\right).
$$

当前仅观察到 $U_l$ 与 $V_l$ 的角色极性；$B_l$ 的存在、位置、运输和因果功能均未确认。

## 八、闭合与进度

$$
\boxed{{\text{{严格机制闭合}}=0/72=0\%}}
$$

$$
\boxed{{\text{{全局物理图谱}}=31\%}},
\qquad
\boxed{{\text{{总体科学成熟度}}=26\%}}.
$$

进度不提高。原因是新增签名来自观察脚手架，并被证明不能区分具体关系边；自然世界状态、实体对绑定、开放预测、计算运输、因果边和密封闭合均没有增加。

## 九、下一阶段大任务

下一阶段冻结为 Phase535（阶段535）：**成对地址保持的世界关系绑定图谱**。

核心改动不是继续调线性观察器，而是改变观测对象：在同一世界后附加完全平衡的中性“候选实体对地址账本”，让每个真实边、反向边和断开同角色实体对都有独立位置，同时设置无账本自然端点对照。发现门必须同时满足：真实边高于反向边、真实边高于断开来源—目标对、实体和关系词留出通过、位置基线与嵌入基线失败、自然端点至少出现一致趋势。只有恢复实体对特异绑定，才允许做1024次全流程置换和组件/神经元定位。

## 十、通俗总结

这轮看起来一度非常漂亮：模型能以100%准确率分清箭头朝哪边。但进一步检查发现，只要随便拿一个“来源角色”和一个“目标角色”拼起来，观察器几乎都会说它们之间有边。它学到的是“谁像起点、谁像终点”，不是“哪一个起点真的连到哪一个终点”。这项负结果非常关键，因为它阻止我们把角色标签误画成知识网络。下一步必须给每一对实体一个可追踪地址，再寻找真正保留配对身份的内部状态。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    audit = collect_audit()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_PATH, audit)
    publish_atlas(audit)
    write_report(audit)
    print(OUT_PATH)
    print(ATLAS_DIR / "manifest.json")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
