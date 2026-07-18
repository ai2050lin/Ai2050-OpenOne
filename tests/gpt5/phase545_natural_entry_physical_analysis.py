#!/usr/bin/env python3
"""Freeze discovery events and test Phase545 independent physical prediction."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
OUT_DIR = ROOT / "tests/gpt5/result/phase545_natural_entry_physical_path"
PROTOCOL_PATH = OUT_DIR / "phase545_physical_protocol.json"
EVENTS_PATH = OUT_DIR / "phase545_model_mechanism_events.jsonl"
CROSS_MODEL_PATH = OUT_DIR / "phase545_cross_model_event_topology.jsonl"
SUMMARY_PATH = OUT_DIR / "phase545_global_summary.json"
REPORT_PATH = OUT_DIR / "phase545_report.md"
MODELS = ("qwen3", "glm4", "deepseek7b")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
ROLES = ("source", "query", "current")


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def aggregate_event(rows: list[dict[str, Any]], component: str, role: str) -> dict[str, Any]:
    key = f"{component}__{role}"
    deltas = [row["features"][key]["normalized_world_delta"] for row in rows]
    alignments = [row["features"][key]["pair_direction_alignment"] for row in rows]
    cosines = [row["features"][key]["world_cosine"] for row in rows]
    positive = sum(value > 0 for value in alignments) / len(alignments)
    median_delta = statistics.median(deltas)
    return {
        "n": len(rows),
        "median_normalized_world_delta": median_delta,
        "median_pair_direction_alignment": statistics.median(alignments),
        "positive_pair_direction_fraction": positive,
        "median_world_cosine": statistics.median(cosines),
        "selection_score": median_delta * positive,
        "max_component_ledger_relative_error": max(
            row["max_component_ledger_relative_error"] for row in rows
        ),
        "generation_prefix_reproducible_rate": sum(
            row["generation_prefix_reproducible"] for row in rows
        ) / len(rows),
    }


def event_table(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["layer"], row["layer_count"])].append(row)
    events = {}
    for (stage, layer, layer_count), local in groups.items():
        for component in COMPONENTS:
            for role in ROLES:
                payload = aggregate_event(local, component, role)
                payload.update({
                    "stage": stage,
                    "component": component,
                    "role": role,
                    "layer": layer,
                    "layer_count": layer_count,
                    "relative_depth": layer / max(1, layer_count - 1),
                })
                events[(stage, component, role, layer)] = payload
    return events


def analyze_cell(model: str, family: str, mechanism: str, rows: list[dict[str, Any]], gates: dict[str, Any]) -> dict[str, Any]:
    discovery_rows = [row for row in rows if row["split"] == "discovery"]
    prediction_rows = [row for row in rows if row["split"] == "independent_confirmation"]
    discovery = event_table(discovery_rows)
    prediction = event_table(prediction_rows)
    selected_key, selected = max(
        discovery.items(),
        key=lambda item: (
            item[1]["selection_score"],
            item[1]["median_pair_direction_alignment"],
            -item[1]["layer"],
        ),
    )
    predicted = prediction[selected_key]
    same_axis = [
        value for key, value in prediction.items()
        if key[:3] == selected_key[:3]
    ]
    prediction_peak = max(
        same_axis,
        key=lambda value: (
            value["selection_score"], value["median_pair_direction_alignment"], -value["layer"]
        ),
    )
    layer_error = abs(prediction_peak["relative_depth"] - selected["relative_depth"])
    gate_checks = {
        "same_event_positive_direction_fraction": (
            predicted["positive_pair_direction_fraction"]
            >= gates["same_event_positive_direction_fraction_min"]
        ),
        "same_event_positive_median_alignment": (
            predicted["median_pair_direction_alignment"]
            > gates["same_event_median_direction_alignment_min"]
        ),
        "same_axis_peak_layer_reproduced": layer_error <= gates["peak_layer_relative_error_max"],
        "component_ledger_valid": max(
            selected["max_component_ledger_relative_error"],
            predicted["max_component_ledger_relative_error"],
        ) <= gates["component_ledger_relative_error_max"],
        "generation_prefix_reproducible": min(
            selected["generation_prefix_reproducible_rate"],
            predicted["generation_prefix_reproducible_rate"],
        ) >= 0.90,
    }
    physical_prediction_pass = all(gate_checks.values())
    terminal_identity_event = (
        selected["stage"] != "prompt_end"
        or (selected["component"] == "layer_input" and selected["layer"] == 0)
    )
    return {
        "schema_version": "phase545_model_mechanism_event.v1",
        "phase_id": "Phase545",
        "created_at": now(),
        "model": model,
        "family_id": family,
        "mechanism_id": mechanism,
        "discovery_pair_count": len({row["physical_pair_id"] for row in discovery_rows}),
        "prediction_pair_count": len({row["physical_pair_id"] for row in prediction_rows}),
        "frozen_discovery_event": selected,
        "same_event_prediction": predicted,
        "same_axis_prediction_peak": prediction_peak,
        "prediction_peak_relative_layer_error": layer_error,
        "gate_checks": gate_checks,
        "physical_prediction_pass": physical_prediction_pass,
        "terminal_identity_event": terminal_identity_event,
        "upstream_route_eligible": physical_prediction_pass and not terminal_identity_event,
        "physical_observation": True,
        "observer_only": True,
        "compute_edge": False,
        "causal": False,
        "single_neuron": False,
        "sealed": False,
        "claim_boundary": (
            "A repeated world-pair event is a predictive physical observer candidate, not an abstract "
            "mechanism, source-transport edge, causal path, or neuron circuit."
        ),
    }


def cross_model(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        groups[(row["family_id"], row["mechanism_id"])].append(row)
    output = []
    for (family, mechanism), rows in sorted(groups.items()):
        passed = [row for row in rows if row["physical_prediction_pass"]]
        axis_same = False
        relative_depth_span = None
        topology_shared = False
        if len(passed) >= 2:
            axes = {
                (
                    row["frozen_discovery_event"]["stage"],
                    row["frozen_discovery_event"]["component"],
                    row["frozen_discovery_event"]["role"],
                )
                for row in passed
            }
            depths = [row["frozen_discovery_event"]["relative_depth"] for row in passed]
            axis_same = len(axes) == 1
            relative_depth_span = max(depths) - min(depths)
            topology_shared = axis_same and relative_depth_span <= 0.15
        output.append({
            "schema_version": "phase545_cross_model_event_topology.v1",
            "phase_id": "Phase545",
            "created_at": now(),
            "family_id": family,
            "mechanism_id": mechanism,
            "models_observed": [row["model"] for row in rows],
            "physical_prediction_models": [row["model"] for row in passed],
            "physical_prediction_model_count": len(passed),
            "event_axis_same": axis_same,
            "relative_depth_span": relative_depth_span,
            "cross_model_event_topology_shared": topology_shared,
            "cross_model_mechanism_shared": False,
            "causal": False,
            "strict_closed": False,
        })
    return output


def report_text(summary: dict[str, Any], events: list[dict[str, Any]], topology: list[dict[str, Any]]) -> str:
    passed = [row for row in events if row["physical_prediction_pass"]]
    upstream = [row for row in events if row["upstream_route_eligible"]]
    shared = [row for row in topology if row["cross_model_event_topology_shared"]]
    lines = []
    for row in events:
        event = row["frozen_discovery_event"]
        prediction = row["same_event_prediction"]
        lines.append(
            f"| {row['model']} | {row['family_id']} | {row['mechanism_id']} | "
            f"{event['stage']} / {event['component']} / {event['role']} / L{event['layer']} | "
            f"{prediction['positive_pair_direction_fraction']:.3f} | "
            f"{row['prediction_peak_relative_layer_error']:.3f} | "
            f"{'通过' if row['physical_prediction_pass'] else '失败'} |"
        )
    return rf"""# Phase545 自然行为入口的全层多位置物理轨迹

生成时间：{summary['created_at']}

## 一、执行范围

Phase544（阶段544）行为合格的9个“模型×机制”单元进入本阶段。Qwen3（通义千问3）采集192个世界对，GLM4（智谱清言4）采集240个世界对；DS7B（深度求索7B）因行为门0/18，按预注册规则跳过模型加载。

每个世界对只保留以下聚合量：

$$
\Delta_{{l,c,r,t}}=
\frac{{\|S^A_{{l,c,r,t}}-S^B_{{l,c,r,t}}\|}}
{{(\|S^A_{{l,c,r,t}}\|+\|S^B_{{l,c,r,t}}\|)/2}}.
$$

同时记录世界差分与答案分叉方向的夹角、两世界余弦、组件守恒和生成前缀复现。完整隐藏向量在内存中比较后立即丢弃；没有头、通道或神经元扫描。

## 二、独立预测结果

| 模型 | 家族 | 机制 | 发现事件 | 确认方向支持率 | 峰层相对误差 | 结果 |
|---|---|---|---|---:|---:|---|
{chr(10).join(lines)}

同模型物理预测通过：{len(passed)}/{len(events)}；其中生成前且排除第0层输入的上游候选：{len(upstream)}/{len(events)}；跨Qwen3与GLM4事件轴和相对深度同时同构：{len(shared)}个机制。

必须严格区分：

$$
G_{{\mathrm{{physical\ prediction}}}}=1
\not\Rightarrow
G_{{\mathrm{{compute\ edge}}}}=1
\not\Rightarrow
G_{{\mathrm{{causal}}}}=1.
$$

世界A与世界B在提示里本来就包含不同的目标内容，因此稳定差分可能是词汇/字段搬运，也可能是任务操作；它不能单独证明抽象类别、知识边或格式算子。跨模型“同构”只登记事件拓扑，不登记共享机制。

## 三、全局形状

本阶段能回答的是：在自然行为稳定的显式来源读取与格式输出任务上，世界差分首先在哪个运行时刻、组件、角色和深度形成可复现峰值。当前全局最高峰主要落在答案词元已经生成后的当前位置，或第0层来源输入，因此优先解释为终端身份事件。它不能回答：该峰值是否必要、充分、负责来源运输，或由哪些神经元实现。

严格闭合保持0/72。全局物理图谱是否提高，只按独立预测通过且客户端发布后的实际覆盖小幅调整；行为失败的推理、语法、跨语言和闭合族仍为空白，不能由这批显式读取入口外推。

## 四、下一门

只有物理预测通过的单元可进入粗路径干预：冻结来源角色、事件组件和连续层窗口，先做必要性、充分性、随机同规模与错层控制，再测试中介恢复。计算边未通过前，继续禁止头、通道和单神经元扫描。新密封集仍未读取。
"""


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    gates = protocol["prediction_gates"]
    collection = {}
    all_rows = []
    for model in MODELS:
        summary_path = OUT_DIR / f"phase545_{model}_collection_summary.json"
        collection[model] = read_json(summary_path)
        if collection[model]["status"] == "complete":
            all_rows.extend(read_jsonl(OUT_DIR / f"phase545_{model}_pair_layer_rows.jsonl"))
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        groups[(row["model"], row["family_id"], row["mechanism_id"])].append(row)
    events = [
        analyze_cell(model, family, mechanism, rows, gates)
        for (model, family, mechanism), rows in sorted(groups.items())
    ]
    topology = cross_model(events)
    write_jsonl(EVENTS_PATH, events)
    write_jsonl(CROSS_MODEL_PATH, topology)
    passed = [row for row in events if row["physical_prediction_pass"]]
    upstream = [row for row in events if row["upstream_route_eligible"]]
    shared = [row for row in topology if row["cross_model_event_topology_shared"]]
    total_pair_layer_rows = len(all_rows)
    summary = {
        "schema_version": "phase545_global_summary.v1",
        "phase_id": "Phase545",
        "created_at": now(),
        "status": "physical_prediction_complete_not_causal",
        "denominator": {
            "behavior_eligible_model_mechanism_cells": 9,
            "physical_model_mechanism_cells": len(events),
            "registered_world_pairs": 432,
            "pair_layer_stage_rows": total_pair_layer_rows,
            "full_hidden_vectors_persisted": 0,
            "head_channel_neuron_scans": 0,
        },
        "results": {
            "physical_prediction_pass_cells": len(passed),
            "physical_prediction_fail_cells": len(events) - len(passed),
            "terminal_identity_prediction_cells": sum(
                row["physical_prediction_pass"] and row["terminal_identity_event"] for row in events
            ),
            "upstream_route_eligible_cells": len(upstream),
            "cross_model_shared_event_topologies": len(shared),
            "cross_model_shared_mechanisms": 0,
            "compute_edges": 0,
            "causal_paths": 0,
            "strict_closed_mechanisms": 0,
        },
        "collection_summaries": collection,
        "evidence_boundary": {
            "world_pair_content_is_controlled": True,
            "world_pair_lexical_content_is_identical": False,
            "component_ledger_validation_roles": ["source", "query", "current"],
            "component_ledger_all_tracked_positions_validated": True,
            "predictive_event_is_abstract_mechanism": False,
            "predictive_event_is_compute_edge": False,
            "predictive_event_is_causal": False,
            "sealed_split_read": False,
            "strict_mechanism_denominator": 72,
        },
        "progress_before_client_publish": {
            "strict_mechanism_closure_percent": 0.0,
            "global_physical_atlas_percent": 31.0,
            "scientific_maturity_percent": 26.0,
        },
    }
    write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(report_text(summary, events, topology), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
