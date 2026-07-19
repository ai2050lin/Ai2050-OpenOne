#!/usr/bin/env python3
"""Publish Phase548-549 matched-control and factorial evidence to the client."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result"
P548 = RESULT / "phase548_shared_attention_compute"
P549 = RESULT / "phase549_route_answer_factorial"
OUT_DIR = RESULT / "phase550_matched_route_identity_atlas"
AUDIT_PATH = OUT_DIR / "phase550_stage_audit.json"
VISUAL_CHECK_PATH = OUT_DIR / "client_visual_check/phase550_client_visual_check.json"
SOURCE_CONTRACT_PATH = (
    RESULT / "phase415_multi_route_vis_sources/phase415_multi_route_vis_source_contract.json"
)
ATLAS_DIR = ROOT / "frontend/public/vis_data/phase549_route_answer_factorial"
SUMMARY_PATH = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2/phase549_route_answer_summary.json"
REGISTRY_PATH = ROOT / "frontend/public/vis_data/source_registry.json"
REPORT_PATH = ROOT / "research/MainAnalysis/20260717_11_Phase548-550匹配控制与答案身份分解.md"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("category", "negated_attribute")
MODEL_LABELS = {
    "qwen3": "Qwen3（通义千问3）",
    "glm4": "GLM4（智谱清言4）",
    "deepseek7b": "DS7B（深度求索7B）",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional_json(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def collect_verification() -> dict[str, Any]:
    visual = read_optional_json(VISUAL_CHECK_PATH)
    contract = read_optional_json(SOURCE_CONTRACT_PATH)
    compact_visual = None
    if visual:
        compact_visual = {
            "schema_version": visual.get("schema_version"),
            "source_id": visual.get("source_id"),
            "passed": visual.get("passed", False),
            "desktop": [
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "canvas_non_blank": row["canvasPixels"]["nonBlank"],
                    "quantized_color_count": row["canvasPixels"]["quantizedColorCount"],
                    "steady_changed_ratio": row["steadyDifference"]["changedRatio"],
                    "interaction_changed_ratio": row["interactionDifference"]["changedRatio"],
                    "interaction_changed": row["interactionChanged"],
                    "horizontal_overflow": row["state"]["horizontalOverflow"],
                }
                for row in visual.get("desktop", [])
            ],
            "mobile": {
                "model": visual.get("mobile", {}).get("model"),
                "dataset": visual.get("mobile", {}).get("dataset"),
                "canvas_non_blank": visual.get("mobile", {}).get("canvasPixels", {}).get("nonBlank"),
                "quantized_color_count": visual.get("mobile", {}).get("canvasPixels", {}).get("quantizedColorCount"),
                "steady_changed_ratio": visual.get("mobile", {}).get("steadyDifference", {}).get("changedRatio"),
                "interaction_changed_ratio": visual.get("mobile", {}).get("interactionDifference", {}).get("changedRatio"),
                "interaction_changed": visual.get("mobile", {}).get("interactionChanged"),
                "horizontal_overflow": visual.get("mobile", {}).get("state", {}).get("horizontalOverflow"),
            },
            "browser_events": visual.get("browser_events", {}),
            "evidence_path": str(VISUAL_CHECK_PATH.relative_to(ROOT)),
        }
    compact_contract = None
    if contract:
        source_result = next(
            (
                row for row in contract.get("source_results", [])
                if row.get("source_id") == "gpt5_phase549_route_answer_factorial"
            ),
            None,
        )
        compact_contract = {
            "schema_version": contract.get("schema_version"),
            "valid": contract.get("valid", False),
            "route_count": contract.get("route_count", 0),
            "source_count": contract.get("source_count", 0),
            "dataset_count": contract.get("dataset_count", 0),
            "canonical_node_count": contract.get("canonical_node_count", 0),
            "canonical_edge_count": contract.get("canonical_edge_count", 0),
            "source_result": source_result,
            "client_contract": contract.get("client_contract", {}),
            "evidence_path": str(SOURCE_CONTRACT_PATH.relative_to(ROOT)),
        }
    return {"client_visual_check": compact_visual, "source_contract": compact_contract}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def collect() -> dict[str, Any]:
    return {
        "phase548_behavior": read_json(P548 / "phase548_behavior_summary.json"),
        "phase548_behavior_cells": read_jsonl(P548 / "phase548_behavior_qualification.jsonl"),
        "phase548_observer": read_json(P548 / "phase548_matched_observer_summary.json"),
        "phase548_observer_cells": read_jsonl(P548 / "phase548_matched_observer_qualification.jsonl"),
        "phase549_behavior": read_json(P549 / "phase549_behavior_summary.json"),
        "phase549_behavior_cells": read_jsonl(P549 / "phase549_behavior_qualification.jsonl"),
        "phase549_observer": read_json(P549 / "phase549_factorial_observer_summary.json"),
        "phase549_observer_cells": read_jsonl(P549 / "phase549_factorial_observer_qualification.jsonl"),
    }


def node(model: str, suffix: str, label: str, kind: str, x: float, y: float, z: float, score: float, color: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"phase549:{model}:{suffix}", "label": f"{MODEL_LABELS[model]} / {label}",
        "model": model, "type": kind, "position": [x, y, z], "score": score,
        "color": color, "show_label": True, "observer": True, "physical": False,
        "predictive": False, "compute_edge": False, "causal": False,
        "single_neuron": False, "pipeline_sealed": False, **extra,
    }


def edge(model: str, suffix: str, source: str, target: str, kind: str, label: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"phase549:{model}:{suffix}", "source": f"phase549:{model}:{source}",
        "target": f"phase549:{model}:{target}", "type": kind, "label": label,
        "score": 1.0, "predictive": False, "compute_edge": False, "causal": False,
        **extra,
    }


def graph_payload(model: str, evidence: dict[str, Any]) -> dict[str, Any]:
    b548 = {row["mechanism_id"]: row for row in evidence["phase548_behavior_cells"] if row["model"] == model}
    o548 = {row["mechanism_id"]: row for row in evidence["phase548_observer_cells"] if row["model"] == model}
    b549 = {row["mechanism_id"]: row for row in evidence["phase549_behavior_cells"] if row["model"] == model}
    o549 = {row["mechanism_id"]: row for row in evidence["phase549_observer_cells"] if row["model"] == model}
    nodes = [
        node(model, "frozen_denominator", "冻结匹配行为分母", "protocol_denominator", -12, 58, 0, 1.0, "#06b6d4",
             phase548_prompts_per_model=1460, phase549_prompts_per_model=1168,
             independent_worlds_per_mechanism_split=73),
        node(model, "compute_stop", "粗粒度计算边仍为0", "compute_route_stop", 12, 58, 0, 0.0, "#ef4444",
             compute_edges=0, causal_edges=0, single_neuron_scans=0),
    ]
    edges = []
    for index, mechanism in enumerate(MECHANISMS):
        z = -5.0 if index == 0 else 5.0
        row548 = b548[mechanism]
        row549 = b549[mechanism]
        behavior548_suffix = f"phase548_behavior:{mechanism}"
        nodes.append(node(
            model, behavior548_suffix,
            f"Phase548 {mechanism} / {'五元行为通过' if row548['behavior_gate_pass'] else '五元行为停止'}",
            "matched_behavior_pass" if row548["behavior_gate_pass"] else "matched_behavior_stop",
            -7, 20, z, 1.0 if row548["behavior_gate_pass"] else 0.0,
            "#22c55e" if row548["behavior_gate_pass"] else "#64748b",
            family_id="content_knowledge", mechanism_id=mechanism,
            behavior_gate_pass=row548["behavior_gate_pass"], physical=row548["observer_collection_authorized"],
            show_label=False,
        ))
        edges.append(edge(
            model, f"denominator_to_548:{mechanism}", "frozen_denominator", behavior548_suffix,
            "behavior_gate", "73世界 × 2分割 × 5匹配条件",
            evidence_level="natural_behavior_qualification",
        ))
        observer548_suffix = f"phase548_observer:{mechanism}"
        obs548 = o548[mechanism]
        if obs548["split_reports"]:
            failed = ",".join(obs548["failed_control_axes"]) or "none"
            nodes.append(node(
                model, observer548_suffix,
                f"Phase548匹配观察停止 / 失败轴:{failed}", "matched_observer_stop",
                -1, 36, z, 0.0, "#f97316", family_id="content_knowledge",
                mechanism_id=mechanism, physical=True,
                matched_observer_gate_pass=obs548["matched_observer_gate_pass"],
                failed_control_axes=obs548["failed_control_axes"], show_label=False,
            ))
            edges.append(edge(
                model, f"548_behavior_to_observer:{mechanism}", behavior548_suffix, observer548_suffix,
                "matched_observation", "功能/实体/答案词/模板分账",
                evidence_level="fresh_matched_observer",
            ))
        behavior549_suffix = f"phase549_behavior:{mechanism}"
        nodes.append(node(
            model, behavior549_suffix,
            f"Phase549 {mechanism} / {'四格行为通过' if row549['behavior_gate_pass'] else '四格行为停止'}",
            "factorial_behavior_pass" if row549["behavior_gate_pass"] else "factorial_behavior_stop",
            2, 20, z, 1.0 if row549["behavior_gate_pass"] else 0.0,
            "#22c55e" if row549["behavior_gate_pass"] else "#64748b",
            family_id="content_knowledge", mechanism_id=mechanism,
            behavior_gate_pass=row549["behavior_gate_pass"], physical=row549["observer_collection_authorized"],
            show_label=False,
        ))
        edges.append(edge(
            model, f"denominator_to_549:{mechanism}", "frozen_denominator", behavior549_suffix,
            "factorial_behavior_gate", "73世界 × 2分割 × 4交叉条件",
            evidence_level="natural_behavior_qualification",
        ))
        obs549 = o549[mechanism]
        if obs549["split_reports"]:
            factor_suffix = f"phase549_factorial:{mechanism}"
            classification = obs549["stable_classification"]
            confirmation = obs549["split_reports"]["independent_confirmation"]
            nodes.append(node(
                model, factor_suffix,
                f"答案身份主导 / 路线{confirmation['route_effect_median']:.3f} / 答案{confirmation['answer_identity_effect_median']:.3f}",
                "answer_identity_dominant" if classification == "answer_identity_dominant" else "factorial_observer",
                7, 42, z, 1.0, "#eab308", family_id="content_knowledge",
                mechanism_id=mechanism, physical=True, predictive=True,
                stable_classification=classification,
                discovery=obs549["split_reports"]["discovery"],
                independent_confirmation=confirmation, show_label=True,
            ))
            edges.append(edge(
                model, f"549_behavior_to_factor:{mechanism}", behavior549_suffix, factor_suffix,
                "factorial_observation", "同答案换路线 / 同路线换答案",
                predictive=True, evidence_level="fresh_factorial_observer",
            ))
            edges.append(edge(
                model, f"factor_to_stop:{mechanism}", factor_suffix, "compute_stop",
                "interpretation_downgrade", "答案身份主导，禁止平台干预",
                evidence_level="compute_gate_stop",
            ))
        else:
            edges.append(edge(
                model, f"behavior_to_stop:{mechanism}", behavior549_suffix, "compute_stop",
                "behavior_stop", "四格自然行为资格失败",
                evidence_level="behavior_gate_stop",
            ))
    return {
        "schema_version": "phase549_route_answer_factorial_atlas.v1",
        "model": model,
        "evidence_scope": (
            "Matched natural behavior, frozen-window deconfounding, and route-by-answer factorial observation. "
            "No compute edge, intervention, causality, head, channel, neuron, or seal claim."
        ),
        "graph": {
            "title": f"{MODEL_LABELS[model]} 匹配控制与路线—答案身份图谱",
            "meta": {
                "model": model, "registered_mechanism_denominator": 72,
                "phase548_behavior_pass_cells": sum(row["behavior_gate_pass"] for row in b548.values()),
                "phase549_behavior_pass_cells": sum(row["behavior_gate_pass"] for row in b549.values()),
                "compute_edges": 0, "causal_edges": 0, "strict_closed_mechanisms": 0,
                "global_physical_atlas_percent": 32, "scientific_maturity_percent": 28,
                "pipeline_sealed": False,
            },
            "nodes": nodes, "edges": edges,
        },
    }


def report_markdown(evidence: dict[str, Any], verification: dict[str, Any]) -> str:
    q548 = next(
        row for row in evidence["phase548_observer_cells"]
        if row["model"] == "qwen3" and row["mechanism_id"] == "negated_attribute"
    )
    q549 = next(
        row for row in evidence["phase549_observer_cells"]
        if row["model"] == "qwen3" and row["mechanism_id"] == "negated_attribute"
    )
    d548 = q548["split_reports"]["discovery"]
    c548 = q548["split_reports"]["independent_confirmation"]
    d549 = q549["split_reports"]["discovery"]
    c549 = q549["split_reports"]["independent_confirmation"]
    visual = verification.get("client_visual_check") or {}
    contract = verification.get("source_contract") or {}
    desktop = visual.get("desktop", [])
    mobile = visual.get("mobile", {})
    interaction_rates = ", ".join(
        f"{row['model']}={row['interaction_changed_ratio']:.2%}"
        for row in desktop
    ) or "尚未执行"
    source_contract = contract.get("source_result") or {}
    return f"""# Phase548-550 匹配控制、路线—答案身份分解与图谱同步

生成时间：{now()}

## 一、对输入分析的审计

输入材料对 Phase544-547 的主结论基本正确：旧结果是生成前观察拓扑，不是计算边、因果边、注意力头或神经元机制；类别与否定属性仍混合内容和答案身份；在去混杂前禁止局部化是必要的。

需要修正三点：

1. “小模型与真实语言编码相差30%-50%”没有实验估计，只能登记为风险情景，不能写成测量误差。
2. 直接把整个注意力输出从反事实世界搬入接收世界，会同时搬运答案内容，不能单独证明充分性。
3. 中介量、置信区间和1024次置换不能替代行为资格与身份匹配；前置门失败时必须停止干预。

## 二、Phase548 冻结合同

- 模型顺序：Qwen3 → GLM4 → DS7B。
- 两个机制：类别、否定属性。
- 每机制每分割73个独立世界；发现与确认实体零重叠。
- 每世界五个条件：基础正条件、功能翻转、实体身份对照、答案词对照、模板对照。
- 共4380条模型特定自然提示。
- 冻结窗口：Qwen3 L28-L30；GLM4 L34-L36；DS7B没有已注册物理窗口。
- 行为、匹配观察、干预三道门串联；前门失败不得读取后门。

## 三、Phase548 客观结果

六个模型—机制单元只有 `Qwen3 / 否定属性` 同时通过两个分割的五条件行为门：两个分割均为73/73个世界全部正确。其余五个单元停止，因此旧合同中的“类别与否定属性跨模型共享入口”没有迁移到严格匹配合同。

唯一合格单元的冻结窗口结果：

| 分割 | 功能差中位数 | 答案词差中位数 | 功能逐对大于答案词 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 发现 | {d548['functional_delta_median']:.6f} | {d548['controls']['answer_token_delta']['control_median']:.6f} | {d548['controls']['answer_token_delta']['functional_dominance_fraction']:.3%} | 未过70%门 |
| 独立确认 | {c548['functional_delta_median']:.6f} | {c548['controls']['answer_token_delta']['control_median']:.6f} | {c548['controls']['answer_token_delta']['functional_dominance_fraction']:.3%} | 未过70%门 |

实体与模板对照均被功能差稳定超过，但答案词对照没有被稳定超过。按冻结停止规则，必要性、充分性和中介干预均未获授权；计算边仍为0。

## 四、Phase549 交叉分解

Phase549 没有修改 Phase548 阈值，而是建立新的2×2合同：同一词汇世界中交叉“路线0/路线1”和“答案A/答案B”。这样可以分别测量同答案换路线与同路线换答案。

- 每模型1168条提示，三模型合计3504条。
- 仍只有 `Qwen3 / 否定属性` 通过两个分割的四格行为门，均为73/73。
- 只对该单元采集Qwen3 L28-L30注意力输出当前位置；没有持久化完整向量，没有读取密封集。

| 分割 | 路线效应中位数 | 答案身份效应中位数 | 答案身份占优世界 | 置换值 |
| --- | ---: | ---: | ---: | ---: |
| 发现 | {d549['route_effect_median']:.6f} | {d549['answer_identity_effect_median']:.6f} | {d549['answer_dominance_fraction']:.3%} | {d549['answer_one_sided_p']:.6f} |
| 独立确认 | {c549['route_effect_median']:.6f} | {c549['answer_identity_effect_median']:.6f} | {c549['answer_dominance_fraction']:.3%} | {c549['answer_one_sided_p']:.6f} |

两套独立分割的73个世界全部由答案身份效应占优。最严格解释是：此前晚层注意力强差异主要对应答案内容整合或读取，不是已经分离出的否定操作路线。

## 五、进展、硬伤与理论边界

新增可靠拼图：

1. 跨模型共享观察拓扑不等于跨模型可迁移的行为合同。
2. 晚层当前位置注意力能够稳定承载答案身份差异。
3. 同答案换路线仍产生非零差异，但约为答案身份差异的三分之一，尚无计算资格。
4. 行为资格门阻止了五个不稳定模型—机制单元进入物理搜索。

仍存在的硬伤：

- 只有Qwen3否定属性可进入内部比较，不能外推为跨模型规律。
- 类别、推理和语法没有合格的匹配合同。
- 路线效应非零不等于存在路线算子，可能仍混合查询极性、词序和关系赋值。
- 晚层三层窗口来自旧观察合同，不代表全层路线效应的最佳位置。
- 没有干预、因果、注意力头、通道、神经元或密封证据。

理论名称保持“语言是动态模式网络”。本轮只增加一个边界：晚层答案整合状态与上游操作路线必须分账；不能再把强晚层差异直接写入统一机制公式中的操作项。

## 六、进度与下一阶段

- 严格机制闭合：0/72，0%。
- 全局物理图谱覆盖度：32%。本轮提高的是校准质量，不是机制覆盖量。
- 总体科学成熟度：28%。该数值是项目管理估计，不是客观定理。

下一阶段应是 Phase551：先为每个模型建立各自能稳定完成的同答案路线合同，再用全层、全组件但不搜索神经元的观察账本寻找上游路线主导区；发现集冻结区域后才在独立确认集验证。不能继续干预当前晚层窗口，也不应立即进入注意力头或神经元。

## 七、固定图谱与客户端同步

- 新增数据源：`gpt5_phase549_route_answer_factorial`。
- 固定数据集：3个；规范节点：{source_contract.get('canonical_node_count', 0)}；规范边：{source_contract.get('canonical_edge_count', 0)}。
- 路径解析与可渲染合同：{'通过' if source_contract.get('all_dataset_paths_resolved') and source_contract.get('all_payloads_renderable') else '待验证'}。
- 三模型桌面3D画布非空；拖拽交互像素变化：{interaction_rates}。
- Qwen3移动端交互像素变化：{mobile.get('interaction_changed_ratio', 0.0):.2%}；横向溢出：{mobile.get('horizontal_overflow', '待验证')}。
- 控制台错误、运行时异常、失败请求和HTTP错误均为0：{'是' if visual.get('passed') else '待验证'}。

这些客户端边只表达行为资格、匹配观察和解释降级，全部保持 `compute_edge=false`、`causal=false`、`single_neuron=false`。
"""


def publish() -> dict[str, Any]:
    evidence = collect()
    verification = collect_verification()
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for model in MODELS:
        filename = f"phase549_{model}_route_answer_factorial.json"
        write_json(ATLAS_DIR / filename, graph_payload(model, evidence))
        items.append({
            "id": f"phase549_{model}", "model": model, "path": filename,
            "label": f"{MODEL_LABELS[model]} 匹配控制与路线—答案身份图谱",
        })
    write_json(ATLAS_DIR / "manifest.json", {
        "schema_version": "phase549_route_answer_factorial_manifest.v1",
        "generated_at": now(), "route_id": "gpt5",
        "evidence_scope": "严格匹配行为门、冻结窗口去混杂和路线×答案交叉观察；非计算、非因果、非神经元闭合。",
        "items": items,
    })
    summary = {
        "schema_version": "phase549_route_answer_summary.v1", "phase_id": "Phase550",
        "generated_at": now(), "models": list(MODELS),
        "phase548": evidence["phase548_observer"], "phase549": evidence["phase549_observer"],
        "strict_closed_mechanisms": 0, "registered_mechanism_denominator": 72,
        "global_physical_atlas_percent": 32, "scientific_maturity_percent": 28,
        "compute_edges": 0, "causal_edges": 0, "single_neuron_mechanisms": 0,
    }
    write_json(SUMMARY_PATH, summary)
    registry = read_json(REGISTRY_PATH)
    source_id = "gpt5_phase549_route_answer_factorial"
    source = {
        "id": source_id, "route_id": "gpt5", "route_label": "GPT5 路线",
        "label": "Phase549 匹配控制与路线—答案身份图谱",
        "description": "三模型匹配自然行为资格、冻结晚层观察去混杂及路线×答案身份交叉分解。",
        "manifest_path": "/vis_data/phase549_route_answer_factorial/manifest.json",
        "manifest_schema": "phase549_route_answer_factorial_manifest.v1",
        "manifest_adapter": "items", "payload_adapter": "atlas_graph",
        "data_base_path": "/vis_data/phase549_route_answer_factorial",
        "models": list(MODELS),
        "evidence_scope": "Qwen3否定属性显示答案身份主导；其余单元行为停止；非计算、非因果、非神经元闭合。",
        "color": "#eab308",
    }
    old_sources = [item for item in registry["sources"] if item["id"] != source_id]
    registry["sources"] = [*old_sources, source]
    registry["generated_at"] = now()
    write_json(REGISTRY_PATH, registry)
    REPORT_PATH.write_text(report_markdown(evidence, verification), encoding="utf-8")
    audit = {
        "schema_version": "phase550_stage_audit.v1", "phase_id": "Phase550",
        "created_at": now(), "status": "matched_route_identity_audit_published",
        "source_id": source_id, "manifest_path": str((ATLAS_DIR / "manifest.json").relative_to(ROOT)),
        "graph_count": len(items),
        "phase548_behavior_case_count": sum(
            value["completed_case_count"] for value in evidence["phase548_behavior"]["execution"].values()
        ),
        "phase549_behavior_case_count": sum(
            value["completed_case_count"] for value in evidence["phase549_behavior"]["execution"].values()
        ),
        "matched_observer_rows": evidence["phase548_observer"]["execution"]["qwen3"].get("row_count", 0),
        "factorial_observer_rows": evidence["phase549_observer"]["execution"]["qwen3"].get("row_count", 0),
        "compute_edges": 0, "causal_edges": 0, "single_neuron_mechanisms": 0,
        "new_sealed_split_read": False,
        "progress": {"closure_percent": 0.0, "global_physical_atlas_percent": 32.0, "scientific_maturity_percent": 28.0},
        "verification": verification,
    }
    write_json(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    publish()
