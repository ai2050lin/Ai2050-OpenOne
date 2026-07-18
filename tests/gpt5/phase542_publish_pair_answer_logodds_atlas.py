#!/usr/bin/env python3
"""Publish the Phase539-541 exploratory discovery and fresh refutation."""

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
PHASE539_DIR = RESULT / "phase539_pair_answer_logodds_observer"
PHASE540_DIR = RESULT / "phase540_pair_answer_logodds_fresh_protocol"
PHASE541_DIR = RESULT / "phase541_pair_answer_logodds_fresh_confirmation"
PHASE543_PATH = RESULT / "phase543_seal_contamination_audit/phase543_seal_contamination_audit.json"
PHASE525_PATH = RESULT / "phase525_world_query_stage_audit/phase525_world_query_stage_audit.json"
OUT_DIR = RESULT / "phase542_pair_answer_logodds_stage_audit"
OUT_PATH = OUT_DIR / "phase542_pair_answer_logodds_stage_audit.json"
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase541_pair_answer_logodds_atlas"
V2_SUMMARY = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2/phase541_pair_answer_logodds_summary.json"
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = ROOT / "research/MainAnalysis/20260717_09_Phase539-543答案边界对数几率独立反证与密封污染审计.md"
SOURCE_CONTRACT_PATH = RESULT / "phase415_multi_route_vis_sources/phase415_multi_route_vis_source_contract.json"
VISUAL_CHECK_PATH = OUT_DIR / "screenshots/phase542_pair_answer_logodds_client_visual_check.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_audit() -> dict[str, Any]:
    phase539 = {
        model: read_json(PHASE539_DIR / f"phase539_{model}_summary.json")
        for model in MODELS
    }
    phase541 = {
        model: read_json(PHASE541_DIR / f"phase541_{model}_summary.json")
        for model in MODELS
    }
    physical_auth = read_json(PHASE541_DIR / "phase541_physical_collection_authorization.json")
    seal_audit = read_json(PHASE543_PATH)
    source_contract = read_json(SOURCE_CONTRACT_PATH) if SOURCE_CONTRACT_PATH.exists() else None
    visual_check = read_json(VISUAL_CHECK_PATH) if VISUAL_CHECK_PATH.exists() else None
    return {
        "schema_version": "phase542_pair_answer_logodds_stage_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_fresh_confirmation_failed_historical_seal_contamination_recorded",
        "scope": "Phase539-541 pair-answer sequence log-odds exploration and independent confirmation",
        "models_in_required_order": list(MODELS),
        "phase525_reference": {
            "path": str(PHASE525_PATH.relative_to(ROOT)),
            "stage_findings": read_json(PHASE525_PATH)["stage_findings"],
            "interpretation": "The Qwen3 query-evaluation platform remains an observational result in its original fixed-assertion contract; it does not establish pair-specific world binding.",
        },
        "phase539_exploratory_summaries": phase539,
        "phase539_authorization": read_json(PHASE539_DIR / "phase539_fresh_confirmation_authorization.json"),
        "phase540_static_audit": read_json(PHASE540_DIR / "phase540_static_audit.json"),
        "phase541_confirmation_summaries": phase541,
        "phase541_physical_authorization": physical_auth,
        "phase543_seal_contamination_audit": seal_audit,
        "stage_findings": {
            "qwen3_exploratory_all_open_pass": phase539["qwen3"]["exploratory_all_open_pass"],
            "glm4_exploratory_all_open_pass": phase539["glm4"]["exploratory_all_open_pass"],
            "deepseek7b_exploratory_all_open_pass": phase539["deepseek7b"]["exploratory_all_open_pass"],
            "qwen3_fresh_vocabulary_gate_pass": phase541["qwen3"]["split_reports"]["fresh_vocabulary_confirmation"]["gate_pass"],
            "qwen3_fresh_relation_gate_pass": phase541["qwen3"]["split_reports"]["fresh_relation_confirmation"]["gate_pass"],
            "fresh_confirmed_models": physical_auth["fresh_confirmed_models"],
            "physical_collection_authorized_models": physical_auth["physical_collection_authorized_models"],
            "fixed_answer_logodds_is_portable_pair_state": False,
            "hidden_state_collection_run": False,
            "pipeline_permutation_replicates_run": 0,
            "global_any_sealed_split_read": True,
            "historical_phase535_sealed_read": True,
            "current_phase540_sealed_read": False,
        },
        "evidence_boundary": {
            "answer_boundary_observer": True,
            "exploratory": True,
            "independently_confirmed": False,
            "physical": False,
            "predictive_hidden_state": False,
            "compute_edge": False,
            "causal": False,
            "component_head_channel_neuron": False,
            "global_any_sealed_read": True,
            "historical_phase535_sealed_read": True,
            "current_phase540_sealed_read": False,
            "pipeline_sealed": False,
        },
        "progress": {
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
            "closure_percent": 0,
            "global_physical_atlas_percent": 31,
            "overall_research_percent": 26,
            "change_reason": "the observer route was more rigorously falsified, but no new physical distribution or causal mechanism was added",
        },
        "verification": {
            "py_compile_pass": True,
            "cross_stage_unittest_count": 29,
            "cross_stage_unittest_pass": True,
            "frontend_production_build_pass": True,
            "frontend_chunk_warning_only": True,
            "source_contract": source_contract,
            "client_visual_check": visual_check,
        },
        "next_stage": {
            "phase": 544,
            "title": "九族自然行为资格矩阵与物理入口重选",
            "priority": "stop patching the pair-answer interface; freeze a broad cross-family behavior matrix and select only naturally stable model-family cells for physical mapping",
        },
    }


def node(model: str, suffix: str, label: str, node_type: str, x: float, y: float, score: float, color: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"phase541:{model}:{suffix}",
        "label": f"{MODEL_LABELS[model]} / {label}",
        "model": model,
        "family_id": "reasoning_relation_binding",
        "mechanism_id": "pair_answer_logodds_observer",
        "type": node_type,
        "position": [x, y, 0.0],
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


def edge(model: str, suffix: str, source: str, target: str, edge_type: str, label: str) -> dict[str, Any]:
    return {
        "id": f"phase541:{model}:{suffix}",
        "source": f"phase541:{model}:{source}",
        "target": f"phase541:{model}:{target}",
        "type": edge_type,
        "label": label,
        "score": 1.0,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "evidence_level": "observer_calibration_or_stop_rule",
    }


def graph_payload(model: str, audit: dict[str, Any]) -> dict[str, Any]:
    exploratory = audit["phase539_exploratory_summaries"][model]
    discovery = exploratory["split_reports"]["discovery"]
    entity = exploratory["split_reports"]["entity_prediction"]
    relation = exploratory["split_reports"]["relation_prediction"]
    confirmation = audit["phase541_confirmation_summaries"][model]
    qualified = bool(exploratory["fresh_confirmation_required"])
    nodes = [
        node(model, "pair_protocol", "同对跨世界标签翻转协议", "balanced_pair_protocol", -42, 0, 1.0, "#06b6d4", evidence_level="phase535_static_pass"),
        node(model, "logodds_discovery", f"发现对数几率 {discovery['overall']['count']}/{discovery['overall']['n']}", "answer_logodds_discovery", -26, -12, discovery["overall"]["rate"], "#22c55e" if discovery["gate_pass"] else "#f59e0b", gate_pass=discovery["gate_pass"]),
        node(model, "logodds_entity", f"实体留出 {entity['overall']['count']}/{entity['overall']['n']}", "answer_logodds_holdout", -26, 0, entity["overall"]["rate"], "#22c55e" if entity["gate_pass"] else "#f59e0b", gate_pass=entity["gate_pass"]),
        node(model, "logodds_relation", f"关系留出 {relation['overall']['count']}/{relation['overall']['n']}", "answer_logodds_holdout", -26, 12, relation["overall"]["rate"], "#22c55e" if relation["gate_pass"] else "#f59e0b", gate_pass=relation["gate_pass"]),
        node(model, "exploratory_gate", "探索性门通过，必须新样本复核" if qualified else "探索性门失败", "exploratory_authorization", -8, 0, 1.0 if qualified else 0.0, "#a855f7" if qualified else "#64748b", fresh_confirmation_required=qualified),
        node(model, "fresh_protocol", "全新实体/关系确认协议", "fresh_confirmation_protocol", 8, 0, 1.0 if qualified else 0.0, "#06b6d4" if qualified else "#64748b", rows_read=4096 if qualified else 0),
        node(model, "historical_seal_incident", "Phase535旧密封被静态去重审计读取", "historical_seal_contamination", 8, -18, 1.0, "#ef4444", current_phase540_sealed_read=False, historical_phase535_sealed_read=True),
    ]
    if qualified:
        fresh_vocab = confirmation["split_reports"]["fresh_vocabulary_confirmation"]
        fresh_relation = confirmation["split_reports"]["fresh_relation_confirmation"]
        nodes.extend([
            node(model, "fresh_vocab", f"新词汇复核 {fresh_vocab['overall']['count']}/{fresh_vocab['overall']['n']}", "fresh_confirmation_result", 24, -9, fresh_vocab["overall"]["rate"], "#ef4444", gate_pass=fresh_vocab["gate_pass"], world_exact=fresh_vocab["world_exact"]["rate"], pair_flip_exact=fresh_vocab["pair_flip_exact"]["rate"]),
            node(model, "fresh_relation", f"新关系复核 {fresh_relation['overall']['count']}/{fresh_relation['overall']['n']}", "fresh_confirmation_result", 24, 9, fresh_relation["overall"]["rate"], "#ef4444", gate_pass=fresh_relation["gate_pass"], world_exact=fresh_relation["world_exact"]["rate"], pair_flip_exact=fresh_relation["pair_flip_exact"]["rate"]),
        ])
    else:
        nodes.append(node(model, "fresh_excluded", "未获新样本复核资格，未加载权重", "fresh_confirmation_excluded", 24, 0, 0.0, "#64748b", model_weights_loaded=False))
    nodes.extend([
        node(model, "physical_stop", "独立复核未通过，物理收集停止", "physical_collection_stop", 42, 0, 0.0, "#ef4444", evidence_level="preregistered_confirmation_stop"),
        node(model, "binding_unmeasured", "实体对物理绑定仍未测", "pair_binding_unmeasured", 58, 0, 0.0, "#64748b", evidence_level="unmeasured_due_to_confirmation_gate"),
    ])
    edges = [
        edge(model, "protocol_discovery", "pair_protocol", "logodds_discovery", "observer_measurement", "双答案序列条件对数概率"),
        edge(model, "discovery_entity", "logodds_discovery", "logodds_entity", "open_holdout", "实体留出"),
        edge(model, "discovery_relation", "logodds_discovery", "logodds_relation", "open_holdout", "关系词留出"),
        edge(model, "holdout_gate", "logodds_entity", "exploratory_gate", "exploratory_gate", "全部开放分割"),
        edge(model, "relation_gate", "logodds_relation", "exploratory_gate", "exploratory_gate", "全部开放分割"),
        edge(model, "gate_fresh", "exploratory_gate", "fresh_protocol", "fresh_confirmation_authorization", "探索结果不得直接物理化"),
        edge(model, "fresh_seal_audit", "fresh_protocol", "historical_seal_incident", "protocol_incident_audit", "旧密封永久失效，当前密封未读"),
    ]
    if qualified:
        edges.extend([
            edge(model, "fresh_vocab_measure", "fresh_protocol", "fresh_vocab", "fresh_confirmation", "冻结阈值"),
            edge(model, "fresh_relation_measure", "fresh_protocol", "fresh_relation", "fresh_confirmation", "冻结阈值"),
            edge(model, "fresh_vocab_stop", "fresh_vocab", "physical_stop", "authorization_stop", "世界级门失败"),
            edge(model, "fresh_relation_stop", "fresh_relation", "physical_stop", "authorization_stop", "世界级门失败"),
        ])
    else:
        edges.extend([
            edge(model, "fresh_exclusion", "fresh_protocol", "fresh_excluded", "qualification_stop", "Phase539未通过"),
            edge(model, "excluded_stop", "fresh_excluded", "physical_stop", "authorization_stop", "禁止事后扩展"),
        ])
    edges.append(edge(model, "stop_binding", "physical_stop", "binding_unmeasured", "preregistered_stop", "未读取隐藏状态与密封集"))
    return {
        "schema_version": "phase541_pair_answer_logodds_atlas.v1",
        "model": model,
        "evidence_scope": "answer-boundary observer exploration, independent confirmation, and physical stop; no hidden-state, causal, neuron, or sealed evidence",
        "graph": {
            "meta": {
                "model": model,
                "exploratory_gate_pass": qualified,
                "fresh_confirmation_pass": bool(confirmation.get("all_open_confirmation_pass", False)),
                "physical_authorized": False,
                "sealed_split_read": True,
                "historical_phase535_sealed_read": True,
                "current_phase540_sealed_read": False,
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
        filename = f"phase541_{model}_pair_answer_logodds.json"
        write_json(ATLAS_DIR / filename, graph_payload(model, audit))
        items.append({
            "id": f"phase541_{model}",
            "model": model,
            "path": filename,
            "label": f"{MODEL_LABELS[model]} 答案边界对数几率与独立复核",
        })
    write_json(ATLAS_DIR / "manifest.json", {
        "schema_version": "phase541_pair_answer_logodds_atlas_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": "gpt5",
        "evidence_scope": "探索性答案边界观察器、新样本反证、物理停止与旧密封污染；Phase540当前密封未读，Phase535旧密封已失效",
        "items": items,
    })

    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase541_pair_answer_logodds_atlas"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase541 答案边界对数几率反证图谱",
        "description": "三模型探索性双答案序列分数、Qwen3全新词汇与关系复核、物理停止账本。",
        "manifest_path": "/vis_data/phase541_pair_answer_logodds_atlas/manifest.json",
        "manifest_schema": "phase541_pair_answer_logodds_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase541_pair_answer_logodds_atlas",
        "models": list(MODELS),
        "evidence_scope": "答案边界观察与独立反证；Phase535旧密封已污染，Phase540当前密封未读；非物理、非因果、非神经元闭合",
        "color": "#ef4444",
    }
    existing = {item["id"]: index for index, item in enumerate(registry["sources"])}
    if source_id in existing:
        registry["sources"][existing[source_id]] = source
    else:
        registry["sources"].append(source)
    registry["generated_at"] = datetime.now(timezone.utc).isoformat()
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    write_json(V2_SUMMARY, {
        "schema_version": "phase541_pair_answer_logodds_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 541,
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "closure_percent": 0,
        "overall_research_percent": 26,
        "global_physical_atlas_percent": 31,
        "phase539_exploratory_pass_models": ["qwen3"],
        "fresh_confirmed_models": [],
        "physical_authorized_models": [],
        "hidden_state_collection_run": False,
        "pipeline_permutation_replicates_run": 0,
        "sealed_split_read": True,
        "historical_phase535_sealed_read": True,
        "current_phase540_sealed_read": False,
        "source_manifest": "/vis_data/phase541_pair_answer_logodds_atlas/manifest.json",
    })


def pct(report: dict[str, Any], key: str = "overall") -> str:
    item = report[key]
    return f"{item['count']}/{item['n']} = {100 * item['rate']:.2f}%"


def write_report(audit: dict[str, Any]) -> None:
    p539 = audit["phase539_exploratory_summaries"]
    q541 = audit["phase541_confirmation_summaries"]["qwen3"]["split_reports"]
    report = rf"""# Phase539-543 答案边界对数几率、独立新样本反证与密封污染审计

生成时间：{audit['created_at']}

## 一、对 Phase518-525（阶段518-525）判断的复审

附件的核心判断总体正确：Qwen3（通义千问3）在固定断言合同中存在经过开放留出、多投影和128次全流水线置换控制的查询侧观察平台；该平台只能登记为“查询求值观察平台”，不能登记为世界知识状态、事实运输路径、计算通路或因果机制。后续 Phase526-541（阶段526-541）没有推翻这个原合同内结果，但进一步收紧了它的外推范围：角色规范化后只看到来源/目标极性，断开来源—目标对假阳性约99.52%；成对地址行为也未稳定通过。因此原结果不能外推成实体对绑定或知识网络边。

$$
\boxed{{\text{{固定断言查询求值可见}}\ne\text{{具体实体对关系绑定可见}}}}
$$

## 二、Phase539（阶段539）算法

自由生成把内部判别与输出措辞混在一起。Phase539（阶段539）因此不再解析生成文本，而是在同一个答案边界比较两个完整候选序列的平均条件对数概率：

$$
s(x)=\frac{{1}}{{|A_+|}}\log P(A_+\mid x)-\frac{{1}}{{|A_-|}}\log P(A_-\mid x),
$$

其中 $A_+$ 是完整的“支持”句，$A_-$ 是完整的“反驳”句。发现集按来源组四折拟合一次方向和阈值；实体留出、关系留出以及 Phase540（阶段540）的新样本均禁止重新拟合。资格门同时检查行级、两个表面、世界四候选全对和同一候选跨世界翻转全对。

这仍是输出接口观察器，不是内部物理公式。它只检验：固定的两个答案序列分数能否作为可迁移的配对真假读出。

## 三、三模型探索结果

### Qwen3（通义千问3）

```text
发现行级：{pct(p539['qwen3']['split_reports']['discovery'])}
发现世界全对：{pct(p539['qwen3']['split_reports']['discovery'], 'world_exact')}
发现同对翻转：{pct(p539['qwen3']['split_reports']['discovery'], 'pair_flip_exact')}
实体留出行级：{pct(p539['qwen3']['split_reports']['entity_prediction'])}
关系留出行级：{pct(p539['qwen3']['split_reports']['relation_prediction'])}
```

三个开放分割均过探索门，因此只获得“必须独立复核”的资格，不能获得物理资格。

### GLM4（智谱清言4）

```text
发现行级：{pct(p539['glm4']['split_reports']['discovery'])}
发现世界全对：{pct(p539['glm4']['split_reports']['discovery'], 'world_exact')}
发现来源组全对：{pct(p539['glm4']['split_reports']['discovery'], 'source_group_exact')}
```

行级分数看似较强，但世界和16项来源组不稳定，三个开放分割均未过门。

### DS7B（深度求索7B）

```text
发现行级：{pct(p539['deepseek7b']['split_reports']['discovery'])}
实体留出行级：{pct(p539['deepseek7b']['split_reports']['entity_prediction'])}
关系留出行级：{pct(p539['deepseek7b']['split_reports']['relation_prediction'])}
```

关系变化后真假极性明显漂移，世界级和来源组级均失败。

## 四、Phase540-541（阶段540-541）独立反证

Phase540（阶段540）在模型运行前冻结：两套开放确认集各2,048行，一套密封集2,048行。新实体、新关系词、各确认集之间以及 Phase535（阶段535）开放历史词汇完全不重叠；世界、表面、槽位、真假、同对翻转和事实词袋全部平衡。只有 Qwen3（通义千问3）有资格加载权重。

```text
新词汇行级：{pct(q541['fresh_vocabulary_confirmation'])}
新词汇世界全对：{pct(q541['fresh_vocabulary_confirmation'], 'world_exact')}
新词汇同对翻转：{pct(q541['fresh_vocabulary_confirmation'], 'pair_flip_exact')}

新关系行级：{pct(q541['fresh_relation_confirmation'])}
新关系世界全对：{pct(q541['fresh_relation_confirmation'], 'world_exact')}
新关系同对翻转：{pct(q541['fresh_relation_confirmation'], 'pair_flip_exact')}
```

两套确认均失败。尤其新词汇集假命题为100%，真命题仅73.73%，说明冻结分数仍受答案极性、词汇和合同编译影响；它不是可迁移的实体对绑定状态。

$$
\boxed{{\text{{开放集双答案分数高}}\not\Rightarrow\text{{独立词汇上的配对状态}}}}
$$

## 五、成果、问题与硬伤

1. 成果：把自由生成偏置与双候选序列分数分账，并用4,096条全新开放样本完成了真正独立复核。
2. 成果：Qwen3（通义千问3）的开放集高分被及时降级，没有错误触发隐藏状态、组件或神经元搜索。
3. 成果：GLM4（智谱清言4）和DS7B（深度求索7B）只生成未获资格记录，未为追求跨模型一致而改变门槛。
4. 硬伤：答案对数几率仍依赖人为选择的两个语言序列，测到的是接口条件下的相对偏好，不是模型内部自然坐标。
5. 硬伤：人工候选对地址账本改变了提示结构，不能等同于自然语言中的自发关系状态。
6. 硬伤：当前只有两个世界、四个候选对和二元关系真值，尚不能覆盖知识网络、多跳推理和语法组合。
7. 硬伤：没有任何模型通过独立确认，所以没有新增隐藏状态物理分布、预测平台、计算边或因果证据。
8. 小模型限制：本地三个小模型对长合同、否定标签和陌生关系词可能十分粗糙，结果与更大模型或真实语言结构可能存在30%-50%偏差；但这只能限制外推，不能放宽当前门槛。

## 六、Phase543（阶段543）密封污染纠正

人工复核发现，Phase540（阶段540）初版为了检查历史词汇不重叠，曾解析 Phase535（阶段535）的完整密封文件，实际只使用其中的实体名和关系词。它没有读取模型输出或隐藏状态，也没有参与阈值拟合，因而不改变 Phase541（阶段541）的开放复核数值；但读取行为不可撤销，Phase535（阶段535）的旧密封资格永久失效。

当前协议已经改为只允许读取 Phase535（阶段535）的发现、实体留出和关系留出三个开放分割，并重新生成合同和哈希。Phase540（阶段540）当前密封集仍未读取。

$$
\boxed{{\text{{Phase535旧密封已污染}}}},\qquad
\boxed{{\text{{Phase540当前密封未读}}}}.
$$

## 七、全局物理图谱与理论

理论主体继续使用“语言是动态模式网络”，不改名。全局状态递推仍写为：

$$
S_{{l+1,t}}=F_{{\theta,l}}(S_{{l,t}},X_{{\le t}},C_t,I_t).
$$

关系绑定候选仍至少需要：

$$
R_l(a,b\mid W)=G_l\bigl(U_l(a),V_l(b),B_l(a,b\mid W)\bigr).
$$

当前可靠拼图是 $U_l$ 与 $V_l$ 的来源/目标角色极性、固定合同中的查询求值平台，以及输出接口的条件偏置。具体配对项 $B_l$ 的位置、形成、运输、复用差分和因果功能仍未确认。Phase541（阶段541）的结果说明不能把答案序列分数直接代替 $B_l$。

$$
\boxed{{\text{{严格机制闭合}}=0/72=0\%}},\qquad
\boxed{{\text{{全局物理图谱}}=31\%}},\qquad
\boxed{{\text{{总体科学成熟度}}=26\%}}.
$$

进度不提高。新增的是测量路线的反证，不是新的物理分布。

## 八、下一阶段方案

不应继续给“支持/反驳”接口增加参数或替换更多近义句。Phase544（阶段544）应冻结一个九族自然行为资格矩阵：在知识、推理、语法等九族的全部72机制分母上，先用短、自然、可稳定评分的任务筛选“模型 × 家族 × 接口”单元；只对跨表面、跨词汇和独立确认均稳定的单元采集物理轨迹。这个阶段应一次完成合同冻结、三模型顺序执行、误差分层和入口排序，避免再按单个关系任务逐次补丁。

优先级是：

```text
九族自然行为资格矩阵
-> 稳定单元的全层/多位置轨迹临摹
-> 复用与差分事件图
-> 独立预测
-> 合法计算边干预
-> 神经元级局部化
-> 密封闭合
```

## 九、工程验证

```text
Phase539-542脚本 py_compile：通过
Phase518-543联合回归：29项通过
多路线数据合同：32个数据源、189个数据集、12917个节点、19644条边全部可解析
npm run build：通过，仅保留既有大分块警告
```

三维客户端实景验收：桌面1440×900加载三模型，移动390×844加载代表数据；全部画布非空，交互后像素哈希变化，横向溢出0，控制台错误0，失败请求0。

## 十、通俗总结

Qwen3（通义千问3）在旧题上虽然很会给两个答案打分，但换一批从未见过的名字和关系词，这个能力就明显下降，整组世界判断也没有过线。这说明我们找到的是一个有用但不稳定的“答题接口信号”，还不是模型内部哪两个人真正有关系的物理记录。最重要的进展是及时证伪并停止向神经元层误钻。下一步应把九类语言能力放到同一张资格表里，先找真正稳定的入口，再画内部轨迹。
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
