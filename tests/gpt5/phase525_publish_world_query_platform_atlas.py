#!/usr/bin/env python3
"""Publish the Phase518-524 audit and a client-loadable multi-model atlas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱清言4）",
    "deepseek7b": "DS7B（深度求索7B）",
}
ROLE_LABELS = {
    "target_evidence_end": "目标证据末端",
    "distractor_evidence_end": "干扰证据末端",
    "claim_entity_end": "断言实体末端",
    "claim_relation_end": "断言关系末端",
    "claim_end": "断言末端",
    "prompt_end": "提示末端",
}
ROLE_Y = {
    "target_evidence_end": -22.0,
    "distractor_evidence_end": -13.0,
    "claim_entity_end": -4.0,
    "claim_relation_end": 5.0,
    "claim_end": 14.0,
    "prompt_end": 23.0,
}

RESULT_DIR = ROOT / "tests/gpt5/result"
PHASE519_DIR = RESULT_DIR / "phase519_natural_relation_binding_calibration"
PHASE520_PATH = RESULT_DIR / "phase520_behavior_authorization/phase520_behavior_authorization.json"
PHASE521_DIR = RESULT_DIR / "phase521_natural_relation_binding_confirmation"
PHASE522_PATH = RESULT_DIR / "phase522_semantic_event_confirmation/phase522_physical_authorization.json"
PHASE523_DIR = RESULT_DIR / "phase523_world_query_platform_physical"
PHASE524_DIR = RESULT_DIR / "phase524_platform_permutation_audit"
OUT_DIR = RESULT_DIR / "phase525_world_query_stage_audit"
OUT_PATH = OUT_DIR / "phase525_world_query_stage_audit.json"
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase524_world_query_platform_atlas"
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
V2_SUMMARY_PATH = (
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2/phase524_world_query_platform_summary.json"
)
REPORT_PATH = (
    ROOT
    / "research/MainAnalysis/20260717_06_Phase518-524自然关系事件与世界查询平台系统审计.md"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_rate(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    return {
        key: report[key]
        for key in ("n", "count", "rate", "lcb95", "ucb95")
        if key in report
    }


def collect_audit() -> dict[str, Any]:
    phase520 = read_json(PHASE520_PATH)
    phase522 = read_json(PHASE522_PATH)
    models = {}
    for model in MODELS:
        calibration = read_json(PHASE519_DIR / f"phase519_{model}_summary.json")
        confirmation = read_json(PHASE521_DIR / f"phase521_{model}_summary.json")
        physical = read_json(PHASE523_DIR / f"phase523_{model}_world_query_platform_summary.json")
        permutation = read_json(PHASE524_DIR / f"phase524_{model}_platform_permutation_summary.json")
        discovery_event = phase520["model_reports"][model]
        confirmation_event = phase522["model_reports"][model]
        models[model] = {
            "calibration": calibration,
            "semantic_event_discovery": discovery_event,
            "confirmation": confirmation,
            "semantic_event_confirmation": confirmation_event,
            "physical": physical,
            "permutation": permutation,
        }
    return {
        "schema_version": "phase525_world_query_stage_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "Phase518-524 natural relation event and world-query observational platform stage",
        "models": models,
        "stage_findings": {
            "strict_whole_response_relation_models": [],
            "independently_confirmed_first_event_models": phase522["relation_models"],
            "strict_binding_models": phase522["binding_models"],
            "familywise_controlled_observational_query_platform_models": ["qwen3"],
            "supportive_query_platform_behavior_gate_miss_models": ["glm4"],
            "world_topology_platform_models": [],
            "causal_models": [],
            "sealed_split_read": False,
        },
        "progress": {
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
            "overall_research_percent": 26,
            "global_physical_atlas_percent": 31,
            "closure_percent": 0,
            "progress_change_reason": (
                "one familywise-controlled Qwen3 query-evaluation platform was added; "
                "world-state, label compilation, compute transport, causality, and sealed closure remain absent"
            ),
        },
        "evidence_boundary": {
            "observational": True,
            "predictive": True,
            "compute_transport": False,
            "causal": False,
            "component_head_channel_neuron": False,
            "shared_cross_model_mechanism": False,
            "sealed": False,
        },
    }


def base_node(
    model: str,
    node_id: str,
    label: str,
    position: list[float],
    node_type: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "model": model,
        "family_id": "reasoning_relation_binding",
        "mechanism_id": "relation_evaluation",
        "type": node_type,
        "position": position,
        "physical": bool(extra.pop("physical", False)),
        "observer": True,
        "predictive": bool(extra.pop("predictive", False)),
        "causal": False,
        "compute_edge": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        "show_label": bool(extra.pop("show_label", True)),
        **extra,
    }


def graph_payload(model: str, audit: dict[str, Any]) -> dict[str, Any]:
    details = audit["models"][model]
    discovery = details["semantic_event_discovery"]
    confirmation = details["semantic_event_confirmation"]
    physical = details["physical"]
    permutation = details["permutation"]
    first_pass = bool(discovery.get("relation_first_event", {}).get("first_event_gate_pass", False))
    confirmed = bool(
        confirmation.get("relation_first_event", {}).get("first_event_gate_pass", False)
    )
    strict_relation = bool(discovery.get("strict_whole_response_relation_gate_pass", False))
    strict_binding = bool(discovery.get("strict_binding_gate_pass", False))
    nodes = [
        base_node(
            model,
            f"phase524:{model}:semantic_first_event",
            f"{MODEL_LABELS[model]} / 首个自然关系事件",
            [-30.0, -12.0, 0.0],
            "semantic_event_contract",
            score=1.0 if confirmed else 0.0,
            gate_pass=confirmed,
            calibration_pass=first_pass,
            confirmation_pass=confirmed,
            evidence_level="independent_behavior_confirmation" if confirmed else "behavior_gate_failure",
            color="#22c55e" if confirmed else "#ef4444",
            size=0.82,
        ),
        base_node(
            model,
            f"phase524:{model}:strict_serialization",
            f"{MODEL_LABELS[model]} / 严格整段输出合同",
            [-30.0, 0.0, 0.0],
            "serialization_contract",
            score=1.0 if strict_relation else 0.0,
            gate_pass=strict_relation,
            evidence_level="strict_whole_response_failure" if not strict_relation else "strict_whole_response_pass",
            color="#ef4444" if not strict_relation else "#22c55e",
            size=0.68,
        ),
        base_node(
            model,
            f"phase524:{model}:binding",
            f"{MODEL_LABELS[model]} / 标签编译四轴合同",
            [-30.0, 12.0, 0.0],
            "binding_contract",
            mechanism_id="label_binding",
            score=1.0 if strict_binding else 0.0,
            gate_pass=strict_binding,
            evidence_level="binding_contract_failure" if not strict_binding else "binding_contract_pass",
            color="#ef4444" if not strict_binding else "#22c55e",
            size=0.68,
        ),
        base_node(
            model,
            f"phase524:{model}:world_topology_missing",
            f"{MODEL_LABELS[model]} / 世界拓扑持续平台未发现",
            [-12.0, -22.0, 0.0],
            "world_platform_missing",
            position_role="target_evidence_end",
            score=0.0,
            gate_pass=False,
            physical=True,
            evidence_level="qualified_negative_result" if confirmed else "behavior_not_qualified",
            color="#64748b",
            size=0.72,
        ),
    ]
    edges = [
        {
            "id": f"phase524:{model}:semantic_to_world_gate",
            "source": f"phase524:{model}:semantic_first_event",
            "target": f"phase524:{model}:world_topology_missing",
            "type": "physical_authorization_gate",
            "label": "行为资格允许观察，不保证发现平台",
            "score": 1.0 if confirmed else 0.0,
            "predictive": False,
            "causal": False,
            "compute_edge": False,
            "evidence_level": "protocol_gate",
        }
    ]

    if physical.get("status") == "complete" and permutation.get("status") == "complete":
        task = permutation["tasks"]["query_evaluation"]
        behavior_ok = permutation["behavior_qualification"]["both_splits_pass"]
        familywise = task["familywise_significant"]
        layer_count = int(physical["model_info"]["n_layers_with_embedding"])
        for platform in task["observed_platforms"]:
            role = platform["position_role"]
            passed = bool(platform["prediction_gate_pass"])
            margin = float(platform["prediction"]["familywise_gate_margin"])
            controlled = behavior_ok and familywise and passed
            supportive = familywise and passed and not behavior_ok
            evidence_level = (
                "familywise_controlled_observational_query_platform"
                if controlled
                else "supportive_query_platform_behavior_gate_miss"
                if supportive
                else "frozen_platform_prediction_failure"
            )
            color = "#16a34a" if controlled else "#f59e0b" if supportive else "#94a3b8"
            layers = platform["layers_with_embedding"]
            previous = None
            for offset, layer in enumerate(layers):
                relative = layer / max(1, layer_count - 1)
                node_id = f"phase524:{model}:{platform['platform_id']}:L{layer}"
                nodes.append(base_node(
                    model,
                    node_id,
                    f"{MODEL_LABELS[model]} / {ROLE_LABELS[role]} L{layer}",
                    [-2.0 + relative * 56.0, ROLE_Y[role], 0.0],
                    "query_evaluation_platform_layer",
                    layer=layer,
                    relative_depth=relative,
                    position_role=role,
                    platform_id=platform["platform_id"],
                    platform_layer_index=offset,
                    score=platform["prediction"]["aggregate_metrics"]["overall"]["rate"],
                    lcb95=platform["prediction"]["aggregate_metrics"]["overall"]["lcb95"],
                    paired_score=platform["prediction"]["aggregate_metrics"]["four_way_pair"]["rate"],
                    paired_lcb95=platform["prediction"]["aggregate_metrics"]["four_way_pair"]["lcb95"],
                    familywise_gate_margin=margin,
                    permutation_p_value=task["permutation_p_value"],
                    null_quantile=task["null_quantile"],
                    gate_pass=passed,
                    behavior_qualified=behavior_ok,
                    familywise_significant=familywise,
                    physical=True,
                    predictive=passed,
                    evidence_level=evidence_level,
                    color=color,
                    size=0.48 if controlled else 0.40,
                    show_label=offset == 0 or offset == len(layers) - 1,
                ))
                if previous is not None:
                    edges.append({
                        "id": f"phase524:{model}:{platform['platform_id']}:E{layer - 1}-{layer}",
                        "source": previous,
                        "target": node_id,
                        "type": "observational_platform_continuity",
                        "label": "同角色相邻层观察连续性",
                        "score": platform["prediction"]["aggregate_metrics"]["overall"]["rate"],
                        "predictive": passed,
                        "causal": False,
                        "compute_edge": False,
                        "evidence_level": evidence_level,
                    })
                previous = node_id

    return {
        "schema_version": "phase524_world_query_platform_atlas.v1",
        "model": model,
        "evidence_scope": (
            "first-event semantic behavior, strict serialization, binding, world-platform negative result, "
            "and familywise-controlled observational query platforms; no compute, causal, neuron, or sealed edge"
        ),
        "graph": {
            "meta": {
                "model": model,
                "strict_serialization_pass": strict_relation,
                "first_event_calibration_pass": first_pass,
                "first_event_confirmation_pass": confirmed,
                "binding_pass": strict_binding,
                "world_topology_platform_count": 0,
                "query_platform_count": (
                    permutation.get("tasks", {}).get("query_evaluation", {}).get("observed_platform_count", 0)
                ),
                "observational_platform_confirmed": permutation.get("observational_platform_confirmed", False),
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


def publish_atlas(audit: dict[str, Any]) -> None:
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase524_{model}_world_query_platform.json"
        write_json(ATLAS_DIR / filename, graph_payload(model, audit))
        items.append({
            "id": f"phase524_{model}",
            "model": model,
            "path": filename,
            "label": f"{MODEL_LABELS[model]} 世界—查询持续平台图谱",
        })
    manifest = {
        "schema_version": "phase524_world_query_platform_atlas_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": "gpt5",
        "evidence_scope": "世界平台强负结果与查询求值观察平台；非计算边、非因果、非神经元、未读密封集",
        "items": items,
    }
    write_json(ATLAS_DIR / "manifest.json", manifest)

    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase524_world_query_platform_atlas"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase524 世界—查询持续平台图谱",
        "description": "三模型自然关系事件资格链、世界平台负结果及查询求值连续平台。",
        "manifest_path": "/vis_data/phase524_world_query_platform_atlas/manifest.json",
        "manifest_schema": "phase524_world_query_platform_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase524_world_query_platform_atlas",
        "models": list(MODELS),
        "evidence_scope": "观察与预测平台；非计算边、非因果、非神经元闭合、未读密封集",
        "color": "#16a34a",
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

    write_json(V2_SUMMARY_PATH, {
        "schema_version": "phase524_world_query_platform_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 524,
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "overall_research_percent": 26,
        "global_physical_atlas_percent": 31,
        "confirmed_observational_query_platform_models": ["qwen3"],
        "supportive_platform_models": ["glm4"],
        "world_topology_platform_models": [],
        "causal_models": [],
        "sealed_split_read": False,
        "source_manifest": "/vis_data/phase524_world_query_platform_atlas/manifest.json",
    })


def write_report(audit: dict[str, Any]) -> None:
    q_perm = audit["models"]["qwen3"]["permutation"]
    g_perm = audit["models"]["glm4"]["permutation"]
    q_task = q_perm["tasks"]["query_evaluation"]
    g_task = g_perm["tasks"]["query_evaluation"]
    report = rf"""# Phase518-524 自然关系事件与世界—查询持续平台系统审计

生成时间：{audit['created_at']}

## 一、总体判断

附件对 Phase509-517（阶段509-517）的功能分解、物理边界和停止结论总体正确，但 Phase518（阶段518）方案有四处必须收紧：不能用简单距离相关证明世界状态；不同随机投影坐标不能比较方向夹角；持续平台只能连接同一角色内相邻层；规范句输出属于冻结语义接口，不是自发自然语言。

本轮完整执行了协议冻结、三模型校准、独立确认、物理发现、开放预测和128次全流程分组置换。最重要的新结果是：Qwen3（通义千问3）获得了经过行为确认、三投影、四折发现、独立预测和全流程置换控制的**查询求值观察平台**；世界拓扑平台、标签编译、计算运输和因果闭合仍未获得。

## 二、数据与执行

- Phase518（阶段518）：关系校准384行、确认768行、平台发现384行、平台预测768行；标签校准512行、确认1024行；密封分割未读取。
- Phase519（阶段519）：Qwen3（通义千问3）、GLM4（智谱清言4）、DS7B（深度求索7B）依次完成，每模型896行行为校准。
- Phase521（阶段521）：只对校准授权的Qwen3（通义千问3）和GLM4（智谱清言4）各运行768行独立确认；DS7B（深度求索7B）不加载权重。
- Phase523（阶段523）：只对独立确认通过的两个模型收集六角色、三投影、全层隐藏状态；每模型发现384行、预测768行。
- Phase524（阶段524）：每个任务执行128次保持四联组结构的完整流水线置换；密封集始终未读。

## 三、客观结果

### 1. 自然关系事件与严格序列化必须分账

冻结的“整段响应必须完全等于一句规范句”合同三模型均失败。离线原文审计发现，Qwen3（通义千问3）和GLM4（智谱清言4）通常先给出正确规范句，再追加解释。因此建立首个完整事件账本，但不改写旧结果。

- 校准：Qwen3（通义千问3）384/384，GLM4（智谱清言4）384/384；DS7B（深度求索7B）216/384，四联组0/96。
- 独立确认：Qwen3（通义千问3）768/768，GLM4（智谱清言4）768/768；两者四联组均192/192。
- 严格整段输出仍为失败，不能把首事件正确写成完整输出接口闭合。

### 2. 标签编译没有恢复

三模型的冻结标签编译合同全部失败。候选概率、映射理解、映射反转、候选外概率质量和严格自由输出没有同时过门，因此组合合同继续禁止。

### 3. 世界拓扑平台是强负结果

Qwen3（通义千问3）和GLM4（智谱清言4）在目标证据、干扰证据、断言实体、断言关系、断言整体和提示末端均未形成满足“三投影一致、至少三折通过、同角色连续四层”的世界拓扑平台。世界状态不能由当前全局均值差线性观察器恢复。

### 4. 查询求值平台

- Qwen3（通义千问3）：断言实体末端L18-L22与断言末端L19-L24在768行预测集、两个表面、四个世界—查询组合和三组投影上均为100%；四联组192/192。全流程置换值为 {q_task['permutation_p_value']:.6f}，128个零假设流水线没有形成合格平台。
- GLM4（智谱清言4）：四个查询平台在预测集表现很强，置换值为 {g_task['permutation_p_value']:.6f}；但平台发现分割的行为四联组为88/96，下界0.844，低于0.85门。因此只能作为支持性复现，不能与Qwen3（通义千问3）共同升级为跨模型机制。
- DS7B（深度求索7B）：首事件行为门失败，未获物理测试授权。

## 四、算法原理

固定投影只用于压缩，不跨投影比较方向：

$$
z_{{m,p,l,r}} = P_p h_{{m,l,r}}, \qquad p\in\{{1,2,3}}.
$$

每折观察方向由基本中心差构造：

$$
d_{{p,l,r}} = \mu_{{p,l,r}}^+ - \mu_{{p,l,r}}^-,
\qquad
\hat y = \mathbf 1\!\left[\left\langle z-\frac{{\mu^++\mu^-}}2,d\right\rangle>0\right].
$$

平台不是单点，而是同一角色中的连续层集合：

$$
\Pi=(r,[l_a,l_b]),\qquad l_b-l_a+1\ge4,
$$

并要求三投影一致、四折中至少三折过门。置换控制重新执行训练、平台发现和预测：

$$
T_0=\max_{{\Pi\in\mathcal P_0}}M(\Pi),\qquad
T_b=\max_{{\Pi\in\mathcal P_b}}M(\Pi),\quad b=1,\ldots,128.
$$

这只确认观察平台超出本轮零假设，不确认平台是信息运输边或因果机制。

## 五、新增核心拼图

1. 语义首事件与严格整段序列化是不同合同。
2. Qwen3（通义千问3）和GLM4（智谱清言4）的首事件关系求值获得独立确认。
3. DS7B（深度求索7B）未复制该关系行为。
4. 三模型标签编译继续失败。
5. 六角色、三投影、四折持续平台协议已落地。
6. 两个授权模型均没有世界拓扑持续平台。
7. Qwen3（通义千问3）查询求值在断言实体末端和断言末端形成连续平台。
8. Qwen3（通义千问3）平台跨两个表面和全部世界—查询匹配子组复现。
9. 128次全流程置换没有产生零假设平台。
10. GLM4（智谱清言4）出现相似查询侧形状，但行为发现门略低于阈值。
11. 当前正结果定位于查询条件化求值，不是上游世界知识结构。
12. 严格机制闭合仍为0/72。

## 六、问题与硬伤

1. 首事件解析器是在严格合同失败后形成的探索性诊断；其可信度来自未读取确认集的再次通过，不应倒写成预注册校准结果。
2. 任务仍是合成封闭世界中的直接有向关系，不代表真实知识网络、长程推理或通用语法。
3. 当前世界标签是全局A/B拓扑类别；若关系编码依赖实体角色间几何而非统一方向，均值差观察器会相消。
4. Qwen3（通义千问3）平台是观察与预测对象，没有组件消融、路径运输、交换干预或神经元证据。
5. GLM4（智谱清言4）没有同时通过两个物理行为分割，不能充当严格跨模型确认。
6. 固定48维随机投影仍可能损失结构；三投影一致降低但没有消除该风险。
7. 标签编译和严格输出失败，完整关系—标签—输出链仍断裂。
8. 密封集未读取，任何闭合说法均不成立。
9. 小模型的编码可能比大型语言模型粗糙30%-50%，当前形状不能直接外推为人脑语言结构。

## 七、理论更新

理论主体继续使用“语言是动态模式网络”，不更名。本轮只增加两个受证据约束的修正：

$$
S_{{t+1}}=F_\theta(S_t,x_t,C_t,I_t),
$$

$$
E_{{\mathrm{{sem}}}}=D_{{\mathrm{{sem}}}}(S,C),
\qquad
Y_{{\mathrm{{seq}}}}=D_{{\mathrm{{seq}}}}(E_{{\mathrm{{sem}}}},I,H).
$$

语义事件可以稳定，而序列化合同仍失败；两者不能再共用一个“输出正确”变量。Qwen3（通义千问3）的结果支持“查询条件化求值在中后层形成持续观察平台”，但上游世界状态怎样形成、运输和参与求值仍未知。

## 八、闭合与进度

- 严格机制闭合：0/72，0%。
- 全局物理图谱：31%。
- 总体研究进度：26%。
- 相比上一阶段只小幅上调1个百分点，因为新增的是一个关系族中的观察平台，不是世界状态、计算边或因果闭合。

## 九、下一阶段

下一阶段应冻结为一个完整任务：**Phase526（阶段526）：角色规范化世界状态几何与查询平台来源验证**。

核心不是继续给线性公式加补丁，而是检验世界状态是否存在于实体角色之间的成对关系几何中：使用来源端无查询前缀、角色规范化端点对、匹配世界交换、独立关系词和实体留出；先恢复世界拓扑，再测试它是否自然到达已确认查询平台。只有来源几何和查询平台之间获得合法计算方向上的预测与干预证据，才进入组件或神经元定位。

## 十、通俗总结

模型已经稳定算出了“这句话对不对”。Qwen3（通义千问3）中还能在句子读完附近连续多层看到这个判断，而且不是随机搜索碰巧找到的。但我们仍看不到模型如何在证据区先组织出“这个小世界是什么样”，也没有证明这些层把信息送到了下一层。现在找到的是可靠的结果站台，不是完整铁路网。
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
