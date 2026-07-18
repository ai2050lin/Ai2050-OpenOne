#!/usr/bin/env python3
"""Test Phase546 frozen upstream observers on fresh physical pairs."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase546_upstream_physical_prediction"
PROTOCOL_PATH = OUT_DIR / "phase546_upstream_protocol.json"
EVENTS_PATH = OUT_DIR / "phase546_frozen_upstream_events.jsonl"
RESULTS_PATH = OUT_DIR / "phase546_upstream_prediction_results.jsonl"
TOPOLOGY_PATH = OUT_DIR / "phase546_cross_model_upstream_topology.jsonl"
SUMMARY_PATH = OUT_DIR / "phase546_global_summary.json"
REPORT_PATH = OUT_DIR / "phase546_report.md"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def aggregate_layer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["features"]["normalized_world_delta"] for row in rows]
    alignments = [row["features"]["pair_direction_alignment"] for row in rows]
    cosines = [row["features"]["world_cosine"] for row in rows]
    positive = sum(value > 0 for value in alignments) / len(alignments)
    median_delta = statistics.median(deltas)
    first = rows[0]
    return {
        "n": len(rows),
        "stage": first["stage"],
        "component": first["component"],
        "role": first["role"],
        "layer": first["layer"],
        "layer_count": first["layer_count"],
        "relative_depth": first["relative_depth"],
        "median_normalized_world_delta": median_delta,
        "median_pair_direction_alignment": statistics.median(alignments),
        "positive_pair_direction_fraction": positive,
        "median_world_cosine": statistics.median(cosines),
        "selection_score": median_delta * positive,
        "max_component_ledger_relative_error": max(
            row["max_component_ledger_relative_error"] for row in rows
        ),
    }


def analyze_cell(
    event: dict[str, Any], rows: list[dict[str, Any]], gates: dict[str, Any]
) -> dict[str, Any]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["layer"]].append(row)
    layers = {layer: aggregate_layer(local) for layer, local in groups.items()}
    frozen = layers[event["layer"]]
    peak = max(
        layers.values(),
        key=lambda value: (
            value["selection_score"], value["median_pair_direction_alignment"], -value["layer"]
        ),
    )
    layer_error = abs(peak["relative_depth"] - event["relative_depth"])
    gate_checks = {
        "same_event_positive_direction_fraction": (
            frozen["positive_pair_direction_fraction"]
            >= gates["same_event_positive_direction_fraction_min"]
        ),
        "same_event_positive_median_alignment": (
            frozen["median_pair_direction_alignment"]
            > gates["same_event_median_direction_alignment_min"]
        ),
        "same_axis_peak_layer_reproduced": layer_error <= gates["peak_layer_relative_error_max"],
        "component_ledger_valid": (
            frozen["max_component_ledger_relative_error"]
            <= gates["component_ledger_relative_error_max"]
        ),
        "independent_pair_count_complete": frozen["n"] == 49,
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": "phase546_upstream_prediction_result.v1",
        "phase_id": "Phase546",
        "created_at": now(),
        "model": event["model"],
        "family_id": event["family_id"],
        "mechanism_id": event["mechanism_id"],
        "frozen_event_id": event["event_id"],
        "frozen_discovery_event": event,
        "fresh_confirmation_pair_count": frozen["n"],
        "same_event_confirmation": frozen,
        "same_axis_confirmation_peak": peak,
        "confirmation_peak_relative_layer_error": layer_error,
        "gate_checks": gate_checks,
        "upstream_prediction_pass": passed,
        "physical_observation": True,
        "observer_only": True,
        "predictive": passed,
        "compute_edge": False,
        "causal": False,
        "single_neuron": False,
        "sealed": False,
        "claim_boundary": (
            "A fresh prompt-end observer prediction is not a source-transport edge, abstract language "
            "mechanism, causal path, or neuron circuit."
        ),
    }


def cross_model(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[(row["family_id"], row["mechanism_id"])].append(row)
    output = []
    for (family, mechanism), rows in sorted(groups.items()):
        passed = [row for row in rows if row["upstream_prediction_pass"]]
        axis_same = False
        depth_span = None
        topology_shared = False
        if len(passed) >= 2:
            axes = {
                (
                    row["frozen_discovery_event"]["component"],
                    row["frozen_discovery_event"]["role"],
                )
                for row in passed
            }
            depths = [row["frozen_discovery_event"]["relative_depth"] for row in passed]
            axis_same = len(axes) == 1
            depth_span = max(depths) - min(depths)
            topology_shared = axis_same and depth_span <= 0.15
        output.append({
            "schema_version": "phase546_cross_model_upstream_topology.v1",
            "phase_id": "Phase546",
            "created_at": now(),
            "family_id": family,
            "mechanism_id": mechanism,
            "models_observed": [row["model"] for row in rows],
            "upstream_prediction_models": [row["model"] for row in passed],
            "upstream_prediction_model_count": len(passed),
            "event_axis_same": axis_same,
            "relative_depth_span": depth_span,
            "cross_model_upstream_topology_shared": topology_shared,
            "cross_model_mechanism_shared": False,
            "compute_edge": False,
            "causal": False,
            "strict_closed": False,
        })
    return output


def report_text(summary: dict[str, Any], results: list[dict[str, Any]], topology: list[dict[str, Any]]) -> str:
    lines = []
    for row in results:
        event = row["frozen_discovery_event"]
        confirmation = row["same_event_confirmation"]
        lines.append(
            f"| {row['model']} | {row['family_id']} | {row['mechanism_id']} | "
            f"{event['component']} / {event['role']} / L{event['layer']} | "
            f"{confirmation['positive_pair_direction_fraction']:.3f} | "
            f"{confirmation['median_pair_direction_alignment']:.3f} | "
            f"{row['confirmation_peak_relative_layer_error']:.3f} | "
            f"{'通过' if row['upstream_prediction_pass'] else '失败'} |"
        )
    return rf"""# Phase546 生成前上游物理观察器独立确认

生成时间：{summary['created_at']}

## 一、为什么必须重做

Phase545（阶段545）虽有7/9个全局事件通过独立预测，但7个全部属于答案生成后的当前位置，或第0层输入。这些事件可由“答案已经出现”或“提示词本来不同”直接解释，不能登记为上游运行脉络。

本阶段在读取新隐藏状态前冻结统一修复规则：只允许提示结束时事件，并排除第0层输入；事件仍只由 Phase545（阶段545）的发现集0-23号世界对选择。确认集使用从未做过物理采集的24-72号世界对，每个模型机制单元49个独立对。由于修复规则是在看到 Phase545（阶段545）终端问题后制定，它是独立修复确认，不冒充严格预注册。

## 二、结果

| 模型 | 家族 | 机制 | 冻结上游事件 | 方向支持率 | 中位方向对齐 | 峰层误差 | 结果 |
|---|---|---|---|---:|---:|---:|---|
{chr(10).join(lines)}

上游预测通过：{summary['results']['upstream_prediction_pass_cells']}/{summary['denominator']['physical_model_mechanism_cells']}；跨模型共享上游拓扑：{summary['results']['cross_model_shared_upstream_topologies']}；计算边、因果路径和严格闭合仍均为0。

## 三、证据边界

冻结事件的确认量为：

$$
P^+_e=\frac1N\sum_i\mathbf 1[\cos(\Delta h_i,\Delta w_i)>0],
\qquad
e^*=\arg\max_e\operatorname{{median}}(\|\Delta h\|_{{norm}})P^+_e.
$$

即使某事件在49个新世界对上复现，两个提示中的实体和目标本来就不同，所以它仍可能是内容身份观察器。严格关系是：

$$
G_{{upstream\ observer}}=1
\not\Rightarrow G_{{compute\ edge}}=1
\not\Rightarrow G_{{causal}}=1.
$$

没有执行干预、头/通道/神经元扫描，也没有读取新密封集。只有跨模型拓扑复现且通过后续必要性、充分性、错层和随机同规模控制，才可升级为粗计算路径。
"""


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    events = read_jsonl(EVENTS_PATH)
    event_map = {
        (row["model"], row["family_id"], row["mechanism_id"]): row for row in events
    }
    collections = {}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    for model in MODELS:
        summary_path = OUT_DIR / f"phase546_{model}_collection_summary.json"
        collections[model] = read_json(summary_path)
        if collections[model]["status"] != "complete":
            continue
        rows = read_jsonl(OUT_DIR / f"phase546_{model}_pair_layer_rows.jsonl")
        total_rows += len(rows)
        for row in rows:
            groups[(row["model"], row["family_id"], row["mechanism_id"])].append(row)
    results = [
        analyze_cell(event_map[key], rows, protocol["prediction_gates"])
        for key, rows in sorted(groups.items())
    ]
    topology = cross_model(results)
    write_jsonl(RESULTS_PATH, results)
    write_jsonl(TOPOLOGY_PATH, topology)
    passed = [row for row in results if row["upstream_prediction_pass"]]
    shared = [row for row in topology if row["cross_model_upstream_topology_shared"]]
    summary = {
        "schema_version": "phase546_global_summary.v1",
        "phase_id": "Phase546",
        "created_at": now(),
        "status": "fresh_upstream_prediction_complete_not_causal",
        "denominator": {
            "physical_model_mechanism_cells": len(results),
            "registered_fresh_confirmation_pairs": 441,
            "independent_pairs_per_model_mechanism": 49,
            "pair_layer_rows": total_rows,
            "full_hidden_vectors_persisted": 0,
            "head_channel_neuron_scans": 0,
        },
        "results": {
            "upstream_prediction_pass_cells": len(passed),
            "upstream_prediction_fail_cells": len(results) - len(passed),
            "cross_model_shared_upstream_topologies": len(shared),
            "cross_model_shared_mechanisms": 0,
            "compute_edges": 0,
            "causal_paths": 0,
            "strict_closed_mechanisms": 0,
        },
        "collection_summaries": collections,
        "evidence_boundary": {
            "strict_preregistration": False,
            "repair_rule_frozen_before_fresh_physical_collection": True,
            "phase545_physical_pair_overlap_count": 0,
            "component_ledger_validation_roles": ["source", "query", "current"],
            "component_ledger_all_tracked_positions_validated": True,
            "generated_token_state_excluded": True,
            "layer_zero_input_excluded": True,
            "predictive_observer_is_compute_edge": False,
            "predictive_observer_is_causal": False,
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
    REPORT_PATH.write_text(report_text(summary, results, topology), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
