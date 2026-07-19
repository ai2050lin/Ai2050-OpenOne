#!/usr/bin/env python3
"""Publish Phase551-552 model-specific route calibration to the atlas client."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result"
P551 = RESULT / "phase551_model_specific_route"
P552 = RESULT / "phase552_surface_route_answer"
OUT_DIR = RESULT / "phase553_model_specific_factorial_atlas"
AUDIT_PATH = OUT_DIR / "phase553_stage_audit.json"
VISUAL_CHECK_PATH = OUT_DIR / "client_visual_check/phase553_client_visual_check.json"
SOURCE_CONTRACT_PATH = (
    RESULT / "phase415_multi_route_vis_sources/phase415_multi_route_vis_source_contract.json"
)
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase552_model_specific_route_factorial"
SUMMARY_PATH = (
    ROOT
    / "frontend/public/vis_data/pattern_family_atlas/v2"
    / "phase552_model_specific_route_summary.json"
)
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = (
    ROOT
    / "research/MainAnalysis/20260717_12_Phase551-553模型专属合同与三因素路线校准.md"
)
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = (
    "category",
    "negated_attribute",
    "transitive_order",
    "subject_verb_agreement",
)
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱GLM4）",
    "deepseek7b": "DS7B（深度求索7B）",
}
MECHANISM_LABELS = {
    "category": "类别知识",
    "negated_attribute": "否定属性",
    "transitive_order": "传递顺序",
    "subject_verb_agreement": "主谓一致",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def collect() -> dict[str, Any]:
    phase551_events = read_jsonl(P551 / "phase551_confirmed_route_observer_events.jsonl")
    return {
        "phase551_calibration_audit": read_json(P551 / "phase551_calibration_static_audit.json"),
        "phase551_frozen_scaffolds": read_json(P551 / "phase551_frozen_scaffolds.json"),
        "phase551_validation_audit": read_json(P551 / "phase551_validation_static_audit.json"),
        "phase551_validation_summary": read_json(P551 / "phase551_validation_behavior_summary.json"),
        "phase551_validation_qualification": read_jsonl(
            P551 / "phase551_validation_behavior_qualification.jsonl"
        ),
        "phase551_observer_summary": read_json(P551 / "phase551_full_layer_route_summary.json"),
        "phase551_events": phase551_events,
        "phase551_event_counts": Counter(
            (row["model"], row["mechanism_id"]) for row in phase551_events
        ),
        "phase552_static_audit": read_json(P552 / "phase552_static_audit.json"),
        "phase552_behavior_summary": read_json(P552 / "phase552_behavior_summary.json"),
        "phase552_behavior_qualification": read_jsonl(
            P552 / "phase552_behavior_qualification.jsonl"
        ),
        "phase552_observer_summary": read_json(P552 / "phase552_full_layer_factorial_summary.json"),
    }


def ordered_source(source: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "route_id", "route_label", "label", "description", "manifest_path",
        "manifest_schema", "manifest_adapter", "payload_adapter", "data_base_path",
        "models", "evidence_scope", "color",
    )
    return {
        **{key: source[key] for key in keys if key in source},
        **{key: value for key, value in source.items() if key not in keys},
    }


def collect_verification() -> dict[str, Any]:
    visual = read_optional_json(VISUAL_CHECK_PATH)
    contract = read_optional_json(SOURCE_CONTRACT_PATH)
    source_result = None
    if contract:
        source_result = next(
            (
                row
                for row in contract.get("source_results", [])
                if row.get("source_id") == "gpt5_phase552_model_specific_route_factorial"
            ),
            None,
        )
    return {
        "source_contract": {
            "valid": contract.get("valid", False) if contract else False,
            "source_count": contract.get("source_count", 0) if contract else 0,
            "dataset_count": contract.get("dataset_count", 0) if contract else 0,
            "canonical_node_count": contract.get("canonical_node_count", 0) if contract else 0,
            "canonical_edge_count": contract.get("canonical_edge_count", 0) if contract else 0,
            "source_result": source_result,
            "evidence_path": str(SOURCE_CONTRACT_PATH.relative_to(ROOT)),
        },
        "client_visual_check": visual,
    }


def graph_node(
    model: str,
    suffix: str,
    label: str,
    kind: str,
    position: list[float],
    score: float,
    color: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"phase552:{model}:{suffix}",
        "label": f"{MODEL_LABELS[model]} / {label}",
        "model": model,
        "type": kind,
        "position": position,
        "score": score,
        "color": color,
        "show_label": True,
        "observer": True,
        "predictive": False,
        "compute_edge": False,
        "causal": False,
        "single_neuron": False,
        "pipeline_sealed": False,
        **extra,
    }


def graph_edge(
    model: str,
    suffix: str,
    source: str,
    target: str,
    kind: str,
    label: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": f"phase552:{model}:{suffix}",
        "source": f"phase552:{model}:{source}",
        "target": f"phase552:{model}:{target}",
        "type": kind,
        "label": label,
        "score": 1.0,
        "predictive": False,
        "compute_edge": False,
        "causal": False,
        **extra,
    }


def graph_payload(model: str, evidence: dict[str, Any]) -> dict[str, Any]:
    q551 = {
        row["mechanism_id"]: row
        for row in evidence["phase551_validation_qualification"]
        if row["model"] == model
    }
    q552 = {
        row["mechanism_id"]: row
        for row in evidence["phase552_behavior_qualification"]
        if row["model"] == model
    }
    overview = {
        row["mechanism_id"]: row
        for row in evidence["phase552_observer_summary"]["contract_overview"]
        if row["model"] == model
    }
    nodes = [
        graph_node(
            model,
            "denominator",
            "模型专属四机制行为分母",
            "protocol_denominator",
            [-13, 58, 0],
            1.0,
            "#06b6d4",
            calibration_worlds=24,
            validation_worlds_per_split=73,
            mechanism_count=4,
            physical=False,
        ),
        graph_node(
            model,
            "compute_stop",
            "三因素控制后计算边仍为0",
            "compute_route_stop",
            [14, 58, 0],
            0.0,
            "#ef4444",
            compute_edges=0,
            causal_edges=0,
            single_neuron_scans=0,
            physical=False,
        ),
    ]
    edges = []
    z_values = (-9.0, -3.0, 3.0, 9.0)
    for mechanism, z in zip(MECHANISMS, z_values):
        qualification = q551[mechanism]
        behavior_pass = qualification["behavior_gate_pass"]
        behavior_suffix = f"behavior:{mechanism}"
        nodes.append(
            graph_node(
                model,
                behavior_suffix,
                f"{MECHANISM_LABELS[mechanism]} / {'双分割通过' if behavior_pass else '行为停止'}",
                "model_specific_behavior_pass" if behavior_pass else "model_specific_behavior_stop",
                [-8, 22, z],
                1.0 if behavior_pass else 0.0,
                "#22c55e" if behavior_pass else "#64748b",
                mechanism_id=mechanism,
                family_id=qualification["family_id"],
                behavior_gate_pass=behavior_pass,
                validation_executed=qualification["validation_executed"],
                physical=False,
                show_label=False,
            )
        )
        edges.append(
            graph_edge(
                model,
                f"denominator_to_behavior:{mechanism}",
                "denominator",
                behavior_suffix,
                "behavior_gate",
                "24校准世界；73+73独立验证世界",
                evidence_level="model_specific_natural_behavior",
            )
        )
        if not behavior_pass:
            edges.append(
                graph_edge(
                    model,
                    f"behavior_to_stop:{mechanism}",
                    behavior_suffix,
                    "compute_stop",
                    "behavior_stop",
                    "行为资格失败，禁止内部搜索",
                    evidence_level="behavior_gate_stop",
                )
            )
            continue
        uncontrolled_count = evidence["phase551_event_counts"][(model, mechanism)]
        uncontrolled_suffix = f"uncontrolled:{mechanism}"
        nodes.append(
            graph_node(
                model,
                uncontrolled_suffix,
                f"旧路线差分事件 {uncontrolled_count} 个（混杂）",
                "contrast_confounded_observer",
                [-2, 38, z],
                1.0,
                "#f97316",
                mechanism_id=mechanism,
                uncontrolled_event_count=uncontrolled_count,
                physical=True,
                predictive=True,
                show_label=True,
            )
        )
        edges.append(
            graph_edge(
                model,
                f"behavior_to_uncontrolled:{mechanism}",
                behavior_suffix,
                uncontrolled_suffix,
                "uncontrolled_observation",
                "路线差异同时改变措辞、关系赋值和查询角色",
                predictive=True,
                evidence_level="independent_observer_reproduction",
            )
        )
        controlled = q552[mechanism]
        controlled_behavior_suffix = f"controlled_behavior:{mechanism}"
        nodes.append(
            graph_node(
                model,
                controlled_behavior_suffix,
                f"{MECHANISM_LABELS[mechanism]} / 八格行为 73/73×2",
                "factorial_behavior_pass",
                [3, 20, z],
                1.0,
                "#16a34a",
                mechanism_id=mechanism,
                surface0_scaffold_id=controlled["surface0_scaffold_id"],
                surface1_scaffold_id=controlled["surface1_scaffold_id"],
                physical=False,
                show_label=False,
            )
        )
        edges.append(
            graph_edge(
                model,
                f"uncontrolled_to_factorial:{mechanism}",
                uncontrolled_suffix,
                controlled_behavior_suffix,
                "factorial_deconfounding",
                "语义路线×表面形式×答案身份",
                evidence_level="fresh_eight_cell_behavior_gate",
            )
        )
        metrics = overview[mechanism]
        controlled_result_suffix = f"controlled_result:{mechanism}"
        nodes.append(
            graph_node(
                model,
                controlled_result_suffix,
                (
                    "严格候选0 / 最佳门进度 "
                    f"{metrics['best_minimum_gate_progress']:.1%}"
                ),
                "controlled_observer_stop",
                [9, 42, z],
                metrics["best_minimum_gate_progress"],
                "#eab308",
                mechanism_id=mechanism,
                tested_coordinate_count=metrics["tested_coordinate_count"],
                confirmed_event_count=0,
                best_minimum_gate_progress=metrics["best_minimum_gate_progress"],
                maximum_confirmation_route_to_control_ratio=(
                    metrics["maximum_confirmation_route_to_control_ratio"]
                ),
                maximum_confirmation_route_dominance_fraction=(
                    metrics["maximum_confirmation_route_dominance_fraction"]
                ),
                physical=True,
                show_label=True,
            )
        )
        edges.append(
            graph_edge(
                model,
                f"factorial_to_result:{mechanism}",
                controlled_behavior_suffix,
                controlled_result_suffix,
                "controlled_observation",
                "发现集与独立确认集均须超过两个对照",
                evidence_level="controlled_full_layer_observer_stop",
            )
        )
        edges.append(
            graph_edge(
                model,
                f"result_to_stop:{mechanism}",
                controlled_result_suffix,
                "compute_stop",
                "compute_gate_stop",
                "0个严格候选，禁止头/通道/神经元干预",
                evidence_level="zero_candidate_stop",
            )
        )
    return {
        "schema_version": "phase552_model_specific_route_factorial_atlas.v1",
        "model": model,
        "evidence_scope": (
            "Model-specific natural behavior qualification and full-layer scalar geometry under "
            "semantic-route by surface-form by answer-identity controls. Observational only."
        ),
        "graph": {
            "title": f"{MODEL_LABELS[model]} 模型专属路线与三因素控制图谱",
            "meta": {
                "model": model,
                "registered_mechanism_denominator": 72,
                "qualified_mechanism_count": sum(
                    row["behavior_gate_pass"] for row in q551.values()
                ),
                "phase551_uncontrolled_event_count": sum(
                    evidence["phase551_event_counts"][(model, mechanism)]
                    for mechanism in MECHANISMS
                ),
                "phase552_controlled_event_count": 0,
                "compute_edges": 0,
                "causal_edges": 0,
                "strict_closed_mechanisms": 0,
                "global_physical_atlas_percent": 32,
                "scientific_maturity_percent": 29,
                "pipeline_sealed": False,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def validation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 模型 | 机制 | 发现集全部四格正确 | 确认集全部四格正确 | 结果 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["validation_executed"]:
            discovery = row["split_reports"]["discovery"]["all_cells_exact"]["count"]
            confirmation = row["split_reports"]["independent_confirmation"]["all_cells_exact"]["count"]
            result = "通过" if row["behavior_gate_pass"] else "停止"
            lines.append(
                f"| {MODEL_LABELS[row['model']]} | {MECHANISM_LABELS[row['mechanism_id']]} "
                f"| {discovery}/73 | {confirmation}/73 | {result} |"
            )
        else:
            lines.append(
                f"| {MODEL_LABELS[row['model']]} | {MECHANISM_LABELS[row['mechanism_id']]} "
                "| 未获校准资格 | 未执行 | 停止 |"
            )
    return "\n".join(lines)


def report_markdown(evidence: dict[str, Any], verification: dict[str, Any]) -> str:
    p551 = evidence["phase551_observer_summary"]
    p552 = evidence["phase552_observer_summary"]
    top = p552["top_near_miss_coordinates"][0]
    source = verification["source_contract"].get("source_result") or {}
    visual = verification.get("client_visual_check") or {}
    return rf"""# Phase551-553 模型专属合同、全层路线观察与三因素校准

生成时间：{now()}

## 一、对 Phase544-547（阶段544-547）材料的复核

附件的主判断正确：先统一行为分母、再进入内部物理观察，比继续修补单一关系任务更可靠；Phase546（阶段546）的晚层注意力拓扑只能登记为观察坐标，不能写成计算边、因果边或神经元机制。

需要继续收紧三点：

1. 小模型与真实语言编码“相差30%-50%”没有可测真值，只能作为外推风险情景，不能作为误差条。
2. 整块状态差异可能同时携带表面形式、答案身份和任务接口，强差异不等于操作路线。
3. 统计门只能执行冻结停止规则，不能把观察相关性升级为机制证据。

## 二、Phase551（阶段551）：模型专属自然行为合同

对类别知识、否定属性、传递顺序、主谓一致各设计5种自然脚手架。校准集为24个世界、4个路线—答案单元，三模型共5760条提示；只用校准结果冻结每个模型的脚手架。随后在与校准实体零重叠的73+73个世界上验证，共4672条提示。

{validation_table(evidence['phase551_validation_qualification'])}

冻结的8个验证合同最终只有4个通过：Qwen3（通义千问3）与GLM4（智谱GLM4）的类别知识和否定属性。DS7B（深度求索7B）类别合同两个分割均为65/73，未获内部观察资格。模型专属脚手架提高了行为可用性，但没有让推理和语法自动获得物理资格。

## 三、Phase551（阶段551）：全层、全组件、三位置观察

只对4个合格合同采集提示结束状态，覆盖Qwen3（通义千问3）36层、GLM4（智谱GLM4）40层，每层4个组件与来源、查询、当前位置3种角色：

$$
R_{{l,c,p}}=\operatorname{{median}}_i
\frac{{\left\|h^{{(r=0)}}_{{i,l,c,p}}-h^{{(r=1)}}_{{i,l,c,p}}\right\|}}
{{\left(\left\|h^{{(r=0)}}_{{i,l,c,p}}\right\|+
\left\|h^{{(r=1)}}_{{i,l,c,p}}\right\|\right)/2}}.
$$

共测试{p551['tested_layer_feature_event_count']}个层—组件—角色坐标；发现集产生{p551['discovery_route_candidate_count']}个候选，独立确认{p551['independently_confirmed_route_event_count']}个，形成{p551['confirmed_contiguous_window_count']}个连续窗口和{p551['shared_cross_model_topology_count']}个跨模型同构外观。

这不是强正结果。1053/1824约为57.73%，并且从早层延伸到晚层，说明“路线差分”改变了大范围输入条件，不是一个已经局部化的运算路径。该路线同时混合措辞、关系赋值、查询极性与结构复杂度。

## 四、Phase552（阶段552）：语义路线×表面形式×答案身份

Phase552（阶段552）没有修改阈值，而是从 Phase551（阶段551）校准数据中冻结第二种可行脚手架，构造8格自然行为合同：

$$
E_{{\mathrm{{route}}}}=\frac14\sum_{{s,a}}
d\left(h_{{0,s,a}},h_{{1,s,a}}\right),
$$

$$
E_{{\mathrm{{surface}}}}=\frac14\sum_{{r,a}}
d\left(h_{{r,0,a}},h_{{r,1,a}}\right),
\qquad
E_{{\mathrm{{answer}}}}=\frac14\sum_{{r,s}}
d\left(h_{{r,s,A}},h_{{r,s,B}}\right).
$$

严格观察门要求发现集和独立确认集同时满足：

$$
\operatorname{{median}}E_{{\mathrm{{route}}}}\ge0.02,
\qquad
\operatorname{{median}}\frac{{E_{{\mathrm{{route}}}}}}
{{\max(E_{{\mathrm{{surface}}}},E_{{\mathrm{{answer}}}})}}\ge1.10,
\qquad
P\left(E_{{\mathrm{{route}}}}>\max(E_{{\mathrm{{surface}}}},E_{{\mathrm{{answer}}}})\right)\ge0.70.
$$

四个合同在发现集与确认集的8个单元全部为73/73，行为分母没有塌缩。Qwen3（通义千问3）得到10,512条层级记录，GLM4（智谱GLM4）得到11,680条，组件账本最大相对误差为0。

客观结果：{p552['tested_layer_feature_event_count']}个坐标中，发现候选0，独立确认候选0，跨模型共享拓扑0；Phase551（阶段551）的1053个事件全部被三因素门撤销。没有授权任何干预、注意力头、通道或神经元搜索。

最接近通过的是 {MODEL_LABELS[top['model']]} 的否定属性、L{top['layer']}、多层感知机输出查询位置：发现集路线/最大对照比{top['discovery']['route_to_max_control_ratio_median']:.4f}，确认集{top['independent_confirmation']['route_to_max_control_ratio_median']:.4f}，低于冻结的1.10门。它是可继续拆解的近失配，不是机制候选。

## 五、算法进展、问题与硬伤

可靠进展：

1. 行为成功与物理观察失败已被严格分开；零候选不能归因于模型不会做题。
2. 模型专属合同显著减少了不合格内部采集，DS7B（深度求索7B）按门停止。
3. 1053→0证明大范围差分观察器存在系统混杂，及时阻止了昂贵且无效的全量单神经元CUDA（并行计算平台）干预。
4. 负结果保留了24个近失配坐标及每合同的最佳门进度，可用于下一算法，而不是只保留一个零。

当前硬伤：

1. 归一化范数距离丢失方向、正负写入和抵消关系；不同向量可以得到同一标量。
2. “语义路线”仍同时改变关系赋值和查询角色，尚不是最小自然操作。
3. 表面形式改变会带来词元位置与长度变化，虽有角色定位，仍可能改变执行形状。
4. 只覆盖类别与否定属性、两个模型；推理、语法和DS7B（深度求索7B）没有合格入口。
5. 没有计算边、因果边、注意力头、通道、神经元或密封证据。

## 六、理论边界与进度

理论名称继续保持“语言是动态模式网络”，不改名：

$$
S_{{l+1,t}}=F_{{\theta,l}}(S_{{l,t}},X_{{\le t}},C_t,I_t).
$$

全局图谱仍为：

$$
\mathcal G=(V_{{\mathrm{{state}}}}\cup V_{{\mathrm{{event}}}}\cup V_{{\mathrm{{interface}}}},
E_{{\mathrm{{observed}}}}\cup E_{{\mathrm{{predicted}}}}\cup
E_{{\mathrm{{compute}}}}\cup E_{{\mathrm{{causal}}}}).
$$

本轮撤销的是混杂观察事件，不是否定动态模式网络。严格闭合仍为0/72=0%；全局物理图谱覆盖度保持32%；总体科学成熟度从28%谨慎上调到29%，原因是行为分母、全层账本和三因素反证链已完成。两个百分比均为项目管理估计，不是自然常数。

## 七、Phase554（阶段554）建议

下一阶段应改算法对象，而不是降低1.10门或继续修补当前范数公式：

1. 把关系赋值、查询角色、表面形式、答案身份拆成4个独立自然因子，先做16格行为合同。
2. 保存可复算的带方向差分摘要，至少同时记录差分方向一致性、正负写入和跨层传递；不能只记录范数。
3. 先在类别与否定属性的Qwen3（通义千问3）、GLM4（智谱GLM4）上发现，再用全新实体确认；DS7B（深度求索7B）重新走行为门。
4. 只有在两个分割都出现紧凑连续窗口且超过所有混杂对照后，才允许组件必要性干预；否则继续修改观察对象，不进入神经元。

这与当前阶段不是同一个证据门。Phase553（阶段553）已经完成当前阶段的停止、图谱同步与验证；Phase554（阶段554）应作为新的冻结研究阶段启动。

## 八、客户端同步

- 数据源：`gpt5_phase552_model_specific_route_factorial`。
- 固定数据集：3；本数据源节点：{source.get('canonical_node_count', 0)}；边：{source.get('canonical_edge_count', 0)}。
- 数据源合同：{'通过' if source.get('all_dataset_paths_resolved') and source.get('all_payloads_renderable') else '待验证'}。
- 桌面与移动3D画布、交互像素和溢出检查：{'通过' if visual.get('passed') else '待验证'}。

客户端只显示关键行为门、混杂事件规模、严格控制后的停止结论和近失配程度，不显示无关神经元状态。所有边保持 `compute_edge=false`、`causal=false`、`single_neuron=false`。
"""


def publish() -> dict[str, Any]:
    evidence = collect()
    verification = collect_verification()
    items = []
    for model in MODELS:
        filename = f"phase552_{model}_model_specific_route_factorial.json"
        write_json(ATLAS_DIR / filename, graph_payload(model, evidence))
        items.append(
            {
                "id": f"phase552_{model}",
                "model": model,
                "path": filename,
                "label": f"{MODEL_LABELS[model]} 模型专属路线与三因素控制图谱",
            }
        )
    write_json(
        ATLAS_DIR / "manifest.json",
        {
            "schema_version": "phase552_model_specific_route_factorial_manifest.v1",
            "generated_at": now(),
            "route_id": "gpt5",
            "evidence_scope": (
                "模型专属自然行为合同、全层路线观察和语义路线×表面形式×答案身份控制；"
                "非计算、非因果、非神经元闭合。"
            ),
            "items": items,
        },
    )
    summary = {
        "schema_version": "phase552_model_specific_route_summary.v1",
        "phase_id": "Phase553",
        "generated_at": now(),
        "models": list(MODELS),
        "phase551_validation": evidence["phase551_validation_summary"],
        "phase551_full_layer": evidence["phase551_observer_summary"],
        "phase552_behavior": evidence["phase552_behavior_summary"],
        "phase552_factorial": evidence["phase552_observer_summary"],
        "strict_closed_mechanisms": 0,
        "registered_mechanism_denominator": 72,
        "global_physical_atlas_percent": 32,
        "scientific_maturity_percent": 29,
        "compute_edges": 0,
        "causal_edges": 0,
        "single_neuron_mechanisms": 0,
    }
    write_json(SUMMARY_PATH, summary)
    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase552_model_specific_route_factorial"
    source = {
        "id": source_id,
        "route_id": "gpt5",
        "route_label": "GPT5 路线",
        "label": "Phase552 模型专属路线与三因素控制图谱",
        "description": (
            "三模型专属行为资格、全层混杂路线观察及语义路线×表面形式×答案身份分解。"
        ),
        "manifest_path": "/vis_data/phase552_model_specific_route_factorial/manifest.json",
        "manifest_schema": "phase552_model_specific_route_factorial_manifest.v1",
        "manifest_adapter": "items",
        "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase552_model_specific_route_factorial",
        "models": list(MODELS),
        "evidence_scope": (
            "四个行为合同通过；1053个未控制事件在三因素门后降为0；"
            "非计算、非因果、非神经元闭合。"
        ),
        "color": "#f59e0b",
    }
    sources = [
        *[row for row in registry["sources"] if row["id"] != source_id],
        ordered_source(source),
    ]
    registry = {
        "schema_version": registry["schema_version"],
        "generated_at": now(),
        "default_source_id": registry["default_source_id"],
        "sources": sources,
    }
    write_json(REGISTRY_PATH, registry)
    REPORT_PATH.write_text(report_markdown(evidence, verification), encoding="utf-8")
    audit = {
        "schema_version": "phase553_stage_audit.v1",
        "phase_id": "Phase553",
        "created_at": now(),
        "status": "model_specific_factorial_atlas_published",
        "source_id": source_id,
        "manifest_path": str((ATLAS_DIR / "manifest.json").relative_to(ROOT)),
        "graph_count": len(items),
        "phase551_calibration_behavior_case_count": evidence[
            "phase551_calibration_audit"
        ]["registered_case_count"],
        "phase551_validation_behavior_case_count": evidence[
            "phase551_validation_audit"
        ]["registered_case_count"],
        "phase552_factorial_behavior_case_count": evidence[
            "phase552_static_audit"
        ]["registered_case_count"],
        "phase551_validation_pass_contract_count": evidence[
            "phase551_validation_summary"
        ]["validation_pass_contract_count"],
        "phase552_behavior_pass_contract_count": evidence[
            "phase552_behavior_summary"
        ]["behavior_pass_contract_count"],
        "phase551_observer_row_count": sum(
            row.get("row_count", 0)
            for row in evidence["phase551_observer_summary"]["execution"].values()
        ),
        "phase552_observer_row_count": sum(
            row.get("row_count", 0)
            for row in evidence["phase552_observer_summary"]["execution"].values()
        ),
        "phase551_uncontrolled_event_count": evidence[
            "phase551_observer_summary"
        ]["independently_confirmed_route_event_count"],
        "phase552_controlled_event_count": evidence[
            "phase552_observer_summary"
        ]["independently_confirmed_semantic_route_event_count"],
        "intervention_authorized": False,
        "compute_edges": 0,
        "causal_edges": 0,
        "single_neuron_mechanisms": 0,
        "new_sealed_split_read": False,
        "progress": {
            "closure_percent": 0.0,
            "global_physical_atlas_percent": 32.0,
            "scientific_maturity_percent": 29.0,
        },
        "verification": verification,
    }
    write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    publish()
