#!/usr/bin/env python3
"""Publish the Phase535-537 behavior stop and its client atlas."""

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
BEHAVIOR_DIR = RESULT / "phase536_pair_addressed_binding_behavior"
DIAGNOSTIC_PATH = (
    RESULT
    / "phase537_pair_addressed_behavior_diagnostics"
    / "phase537_pair_addressed_behavior_diagnostics.json"
)
PROTOCOL_AUDIT_PATH = (
    RESULT
    / "phase535_pair_addressed_binding_protocol"
    / "phase535_static_audit.json"
)
OUT_DIR = RESULT / "phase538_pair_addressed_binding_stage_audit"
OUT_PATH = OUT_DIR / "phase538_pair_addressed_binding_stage_audit.json"
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase537_pair_addressed_behavior_atlas"
V2_SUMMARY = (
    ROOT
    / "frontend/public/vis_data/pattern_family_atlas/v2"
    / "phase537_pair_addressed_behavior_summary.json"
)
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = (
    ROOT
    / "research/MainAnalysis/20260717_08_Phase535-537成对地址关系绑定行为审计.md"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect_audit() -> dict[str, Any]:
    static = read_json(PROTOCOL_AUDIT_PATH)
    authorization = read_json(BEHAVIOR_DIR / "phase536_physical_authorization.json")
    diagnostics = read_json(DIAGNOSTIC_PATH)
    summaries = {
        model: read_json(BEHAVIOR_DIR / f"phase536_{model}_behavior_summary.json")
        for model in MODELS
    }
    return {
        "schema_version": "phase538_pair_addressed_binding_stage_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_stopped_before_physical_collection",
        "scope": "Phase535-537 pair-addressed world-binding behavior qualification",
        "models_in_required_order": list(MODELS),
        "protocol_static_audit": static,
        "behavior_summaries": summaries,
        "behavior_diagnostics": diagnostics,
        "physical_authorization": authorization,
        "stage_findings": {
            "pair_address_world_flip_is_balanced": True,
            "physical_authorized_models": [],
            "qwen3_relation_prediction_gate_pass": summaries["qwen3"]["split_reports"]["relation_prediction"]["gate_pass"],
            "qwen3_all_split_gate_pass": False,
            "truth_polarity_bias_present": True,
            "pair_address_shortcut_detected": False,
            "hidden_state_collection_run": False,
            "prediction_hidden_states_read": False,
            "pipeline_permutation_replicates_run": 0,
            "sealed_split_read": False,
        },
        "evidence_boundary": {
            "behavior_only": True,
            "physical": False,
            "predictive_pair_binding": False,
            "compute_edge": False,
            "causal": False,
            "component_head_channel_neuron": False,
            "sealed": False,
        },
        "progress": {
            "strict_closed_mechanisms": 0,
            "mechanism_denominator": 72,
            "closure_percent": 0,
            "global_physical_atlas_percent": 31,
            "overall_research_percent": 26,
            "change_reason": "the pair-address protocol improved falsifiability but no model qualified for physical collection",
        },
        "next_stage": {
            "phase": 539,
            "title": "冻结答案边界对数几率观察器与新鲜确认",
            "priority": (
                "separate latent truth discrimination from supported/contradicted generation bias; "
                "positive exploration must be independently confirmed before hidden-state collection"
            ),
        },
    }


def node(
    model: str,
    suffix: str,
    label: str,
    node_type: str,
    x: float,
    y: float,
    score: float,
    color: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"phase537:{model}:{suffix}",
        "label": f"{MODEL_LABELS[model]} / {label}",
        "model": model,
        "family_id": "reasoning_relation_binding",
        "mechanism_id": "pair_specific_world_binding",
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
        "id": f"phase537:{model}:{suffix}",
        "source": f"phase537:{model}:{source}",
        "target": f"phase537:{model}:{target}",
        "type": edge_type,
        "label": label,
        "score": 1.0,
        "predictive": False,
        "causal": False,
        "compute_edge": False,
        "evidence_level": "behavior_protocol_or_stop_rule",
    }


def graph_payload(model: str, audit: dict[str, Any]) -> dict[str, Any]:
    summary = audit["behavior_summaries"][model]
    diagnostics = audit["behavior_diagnostics"]["model_reports"][model]
    discovery = summary["split_reports"]["discovery"]
    entity = summary["split_reports"]["entity_prediction"]
    relation = summary["split_reports"]["relation_prediction"]
    pair_diag = diagnostics["splits"]["discovery"]["pair_flip"]
    supported_rate = diagnostics["splits"]["discovery"]["event_distribution"]["supported"]["rate"]
    contradicted_rate = diagnostics["splits"]["discovery"]["event_distribution"]["contradicted"]["rate"]
    nodes = [
        node(model, "balanced_protocol", "同对跨世界地址与标签翻转", "balanced_pair_address_protocol", -36, 0, 1.0, "#06b6d4", evidence_level="static_protocol_pass"),
        node(model, "discovery_overall", f"发现整体 {discovery['overall']['count']}/{discovery['overall']['n']}", "behavior_overall", -20, -12, discovery["overall"]["rate"], "#f59e0b", gate_pass=discovery["gate_pass"]),
        node(model, "discovery_world_exact", f"发现世界全对 {discovery['world_exact']['count']}/{discovery['world_exact']['n']}", "world_exact_behavior", -20, 0, discovery["world_exact"]["rate"], "#ef4444", gate_pass=False),
        node(model, "discovery_pair_flip", f"发现同对翻转 {discovery['pair_flip_exact']['count']}/{discovery['pair_flip_exact']['n']}", "pair_flip_behavior", -20, 12, discovery["pair_flip_exact"]["rate"], "#f59e0b", observed_event_flip_rate=pair_diag["observed_event_value_flip"]["rate"]),
        node(model, "truth_bias", "支持/反驳输出偏置", "truth_polarity_bias", -2, 14, abs(supported_rate - contradicted_rate), "#ef4444", supported_rate=supported_rate, contradicted_rate=contradicted_rate),
        node(model, "entity_holdout", f"实体留出 {entity['overall']['count']}/{entity['overall']['n']}", "entity_holdout_behavior", -2, -8, entity["overall"]["rate"], "#f59e0b", gate_pass=entity["gate_pass"]),
        node(model, "relation_holdout", f"关系留出 {relation['overall']['count']}/{relation['overall']['n']}", "relation_holdout_behavior", -2, 2, relation["overall"]["rate"], "#22c55e" if relation["gate_pass"] else "#f59e0b", gate_pass=relation["gate_pass"]),
        node(model, "physical_stop", "三开放分割未齐，物理收集停止", "physical_collection_stop", 18, 0, 0.0, "#64748b", evidence_level="preregistered_behavior_stop"),
        node(model, "binding_missing", "实体对物理绑定仍未测试", "pair_binding_unmeasured", 36, 0, 0.0, "#64748b", evidence_level="unmeasured_due_to_behavior_gate"),
    ]
    edges = [
        edge(model, "protocol_discovery", "balanced_protocol", "discovery_overall", "behavior_measurement", "冻结协议进入行为测量"),
        edge(model, "discovery_world", "discovery_overall", "discovery_world_exact", "behavior_decomposition", "整体准确率拆为世界全对"),
        edge(model, "discovery_flip", "discovery_overall", "discovery_pair_flip", "behavior_decomposition", "整体准确率拆为同对翻转"),
        edge(model, "flip_bias", "discovery_pair_flip", "truth_bias", "error_diagnostic", "翻转失败受输出极性影响"),
        edge(model, "discovery_entity", "discovery_overall", "entity_holdout", "behavior_holdout", "实体留出"),
        edge(model, "discovery_relation", "discovery_overall", "relation_holdout", "behavior_holdout", "关系词留出"),
        edge(model, "entity_stop", "entity_holdout", "physical_stop", "authorization_gate", "全部开放分割必须通过"),
        edge(model, "relation_stop", "relation_holdout", "physical_stop", "authorization_gate", "单一留出通过不足以授权"),
        edge(model, "stop_binding", "physical_stop", "binding_missing", "preregistered_stop", "隐藏状态与置换均未运行"),
    ]
    return {
        "schema_version": "phase537_pair_addressed_behavior_atlas.v1",
        "model": model,
        "evidence_scope": "pair-addressed behavior qualification and stop accounting only; no hidden-state, predictive, causal, neuron, or sealed evidence",
        "graph": {
            "meta": {
                "model": model,
                "physical_authorized": False,
                "hidden_state_collection_run": False,
                "pipeline_permutation_replicates_run": 0,
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
        filename = f"phase537_{model}_pair_addressed_behavior.json"
        write_json(ATLAS_DIR / filename, graph_payload(model, audit))
        items.append({
            "id": f"phase537_{model}",
            "model": model,
            "path": filename,
            "label": f"{MODEL_LABELS[model]} 成对地址绑定行为审计",
        })
    write_json(ATLAS_DIR / "manifest.json", {
        "schema_version": "phase537_pair_addressed_behavior_atlas_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "route_id": "gpt5",
        "evidence_scope": "成对地址行为资格与物理停止；非隐藏状态、非计算边、非因果、非神经元、未读密封集",
        "items": items,
    })

    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase537_pair_addressed_behavior_atlas"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase537 成对地址关系绑定行为图谱",
        "description": "三模型同一实体对跨世界真假翻转、输出偏置、留出资格与物理停止账本。",
        "manifest_path": "/vis_data/phase537_pair_addressed_behavior_atlas/manifest.json",
        "manifest_schema": "phase537_pair_addressed_behavior_atlas_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase537_pair_addressed_behavior_atlas",
        "models": list(MODELS),
        "evidence_scope": "行为资格与停止账本；非物理、非预测、非计算边、非因果、非神经元闭合",
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

    write_json(V2_SUMMARY, {
        "schema_version": "phase537_pair_addressed_behavior_summary.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": 537,
        "strict_closed_mechanisms": 0,
        "mechanism_denominator": 72,
        "closure_percent": 0,
        "overall_research_percent": 26,
        "global_physical_atlas_percent": 31,
        "physical_authorized_models": [],
        "pair_binding_models": [],
        "hidden_state_collection_run": False,
        "pipeline_permutation_replicates_run": 0,
        "sealed_split_read": False,
        "source_manifest": "/vis_data/phase537_pair_addressed_behavior_atlas/manifest.json",
    })


def write_report(audit: dict[str, Any]) -> None:
    summaries = audit["behavior_summaries"]
    diagnostics = audit["behavior_diagnostics"]["model_reports"]
    report = rf"""# Phase535-537 成对地址关系绑定行为审计

生成时间：{audit['created_at']}

## 一、总体判断

Phase533（阶段533）证明实体寄存器只恢复来源/目标角色，不能恢复具体配对。Phase535（阶段535）因此改变观测对象：四个候选实体对在两个世界中保持同一地址，而每一对的真假标签严格翻转。这个设计成功消除了“任意来源加任意目标都像边”的旧混杂，但三模型没有同时通过发现、实体留出和关系词留出的行为门，物理状态收集必须停止。

$$
\boxed{{\text{{成对地址提高可证伪性}}\;\ne\;\text{{实体对绑定已经恢复}}}}
$$

## 二、数据和执行

两个世界保持所有节点的来源/目标角色不变：

$$
W_0=\{{0\to1,\;2\to3\}},
\qquad
W_1=\{{0\to3,\;2\to1\}}.
$$

候选对固定为 $(0,1),(0,3),(2,1),(2,3)$。同一表面下，候选对跨世界占用相同槽位，且边标签恰好翻转。发现集768行，实体留出1536行，关系词留出1536行，密封1536行未读。三模型按 Qwen3（通义千问3）、GLM4（智谱清言4）、DS7B（深度求索7B）顺序各运行3840行，模型间释放显存。

## 三、客观结果

### Qwen3（通义千问3）

```text
发现：{summaries['qwen3']['split_reports']['discovery']['overall']['count']}/768
  世界四候选全对：{summaries['qwen3']['split_reports']['discovery']['world_exact']['count']}/192
  同对跨世界翻转全对：{summaries['qwen3']['split_reports']['discovery']['pair_flip_exact']['count']}/384
实体留出：{summaries['qwen3']['split_reports']['entity_prediction']['overall']['count']}/1536
关系词留出：{summaries['qwen3']['split_reports']['relation_prediction']['overall']['count']}/1536
```

关系词留出单独通过冻结门，但发现集和实体留出未同时通过，因此不能从最强分割反向选择物理入口。

### GLM4（智谱清言4）

```text
发现：{summaries['glm4']['split_reports']['discovery']['overall']['count']}/768
  世界四候选全对：{summaries['glm4']['split_reports']['discovery']['world_exact']['count']}/192
  同对跨世界翻转全对：{summaries['glm4']['split_reports']['discovery']['pair_flip_exact']['count']}/384
实体留出：{summaries['glm4']['split_reports']['entity_prediction']['overall']['count']}/1536
关系词留出：{summaries['glm4']['split_reports']['relation_prediction']['overall']['count']}/1536
```

GLM4（智谱清言4）真命题在三个分割几乎全对，假命题只有39.58%-51.04%，表现为强肯定偏置。

### DS7B（深度求索7B）

```text
发现：{summaries['deepseek7b']['split_reports']['discovery']['overall']['count']}/768
  世界四候选全对：0/192
  同对跨世界翻转全对：16/384
实体留出：{summaries['deepseek7b']['split_reports']['entity_prediction']['overall']['count']}/1536
关系词留出：{summaries['deepseek7b']['split_reports']['relation_prediction']['overall']['count']}/1536
```

DS7B（深度求索7B）在关系词留出中未正确反驳任何假命题，并有约四分之一输出无法解析。

## 四、误差张量

Qwen3（通义千问3）的发现集同对事件值实际翻转276/384；GLM4（智谱清言4）152/384；DS7B（深度求索7B）20/384。候选槽位间差异小于真假极性和表面差异，因此没有证据表明某一个地址槽位制造了主要捷径。

对每个候选对 $p$，行为资格要求：

$$
\hat y(p,W_0)\ne\hat y(p,W_1),
\qquad
\hat y(p,W_i)=y(p,W_i).
$$

当前失败主要表现为：模型在两个世界中保持同一输出极性，而不是稳定追随世界中的配对变化。这个结果仍属于行为接口，不能反推内部没有配对状态。

## 五、硬伤与证据边界

1. 候选对地址账本是人工脚手架，增加了提示长度和指令负载。
2. 自由生成的 `supported/contradicted` 接口把内部判别、标签编译和输出偏置混在一起。
3. Qwen3（通义千问3）不同分割差异很大，不能只选择表现最好的关系词留出。
4. GLM4（智谱清言4）和DS7B（深度求索7B）的肯定偏置足以抬高真命题准确率并压低世界全对率。
5. 没有模型获物理授权，所以不存在隐藏状态、预测、置换、组件或神经元结果。
6. 密封集未读取，闭合仍为0。
7. 小模型对长合同和否定接口的粗糙性可能造成30%-50%偏差，不能把行为失败写成语言结构不存在。

## 六、理论与进度

理论主体“语言是动态模式网络”不修改。本轮只确认观察合同必须把配对变化和输出接口分层：

$$
Z_{{\mathrm{{pair}}}}=E(S,W,p),
\qquad
Y_{{\mathrm{{label}}}}=D(Z_{{\mathrm{{pair}}}},I,H).
$$

行为输出失败可能来自 $E$ 或 $D$，当前实验不能区分二者。

$$
\boxed{{\text{{严格机制闭合}}=0/72=0\%}},
\quad
\boxed{{\text{{全局物理图谱}}=31\%}},
\quad
\boxed{{\text{{总体科学成熟度}}=26\%}}.
$$

进度不提高，因为本轮只改善了数据可证伪性，没有新增物理分布。

## 七、下一阶段

Phase539（阶段539）应冻结答案边界的双候选对数几率观察器，直接比较完整答案序列的条件对数几率，先在开放校准集检验是否消除输出极性偏置；若出现正结果，必须重新生成实体和关系词新鲜确认集，确认前不得收集隐藏状态。这样改变的是测量接口，不是给旧线性物理公式继续加参数。

## 八、通俗总结

我们已经让每一对实体拥有固定座位，并让同一座位在两个世界里从“有边”变成“无边”。如果模型稳定跟踪具体关系，它的答案也应跟着翻转。Qwen3（通义千问3）有明显迹象，但三个分割没有一起过关；另外两个模型主要被“总想回答支持”拖住。因此现在不能往神经元层钻。下一步要先绕开支持/反驳文字输出的偏置，直接测量模型在答案边界上更倾向哪一个完整答案。
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
