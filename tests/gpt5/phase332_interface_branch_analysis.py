#!/usr/bin/env python3
"""Analyze Phase332 interface paths and exchanges with frozen basic gates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("raw_completion", "native_chat", "chat_no_think", "answer_aligned_chat")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [
        float(row[key]) for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return mean(values) if values else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def rounded(value: float) -> float:
    return round(float(value), 7)


def finite_metric_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(
        row.get(key) is not None and math.isfinite(float(row[key]))
        for row in rows
    )


def interface_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family_id"], row["mechanism_id"], row["interface"])].append(row)
    result = []
    for (model, family, mechanism, interface), values in sorted(grouped.items()):
        discovery = [row for row in values if row["split"] == "discovery"]
        heldout = [row for row in values if row["split"] == "heldout"]
        result.append({
            "schema_version": "10.0.0",
            "phase_id": "Phase332",
            "created_at": now(),
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "cohort": values[0]["cohort"],
            "interface": interface,
            "interface_equivalent_to": values[0]["interface_equivalent_to"],
            "answer_phase": values[0]["answer_phase"],
            "discovery_case_count": len(discovery),
            "heldout_case_count": len(heldout),
            "discovery_mean_target_margin": rounded(avg(discovery, "target_margin")),
            "heldout_mean_target_margin": rounded(avg(heldout, "target_margin")),
            "heldout_candidate_winner_rate": rounded(rate(heldout, "candidate_winner_is_target")),
            "heldout_target_in_top50_rate": rounded(rate(heldout, "target_in_top50")),
            "heldout_mean_phrase_logprob": rounded(avg(heldout, "target_phrase_logprob")),
            "heldout_answer_phase_reached_rate": rounded(rate(heldout, "answer_phase_reached")),
            "heldout_target_answer_match_rate": rounded(rate(heldout, "target_answer_segment_match")),
            "heldout_protocol_success_rate": rounded(rate(heldout, "protocol_success_answer_segment")),
            "heldout_behavior_success_rate": rounded(rate(heldout, "behavior_success")),
        })
    return result


def member_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["component_type"], int(row["component_layer"]), row["position_role"],
        int(row["component_index"]), int(row["component_start"]), int(row["component_end"]),
    )


def heldout_member_validation(member_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[tuple[str, str, str], dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    required_identities: dict[tuple[str, str, str], set[tuple[Any, ...]]] = defaultdict(set)
    for row in member_rows:
        if row["set_type"] == "shared_skeleton":
            key = (row["model"], row["family_id"], row["mechanism_id"])
            selected[key][member_identity(row)] = row
            for interface in INTERFACES:
                if row["model"] == "glm4" and interface == "chat_no_think":
                    continue
                required_identities[(row["model"], row["family_id"], row["mechanism_id"], interface)].add(member_identity(row))
        elif row["set_type"] == "interface_branch":
            key = (row["model"], row["family_id"], row["mechanism_id"], row["interface"])
            selected[key][member_identity(row)] = row
            for interface in INTERFACES:
                if row["model"] == "glm4" and interface == "chat_no_think":
                    continue
                required_identities[(row["model"], row["family_id"], row["mechanism_id"], interface)].add(member_identity(row))

    values: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    parquet_path = SOURCE / "phase332_natural_unit_rows.parquet"
    parquet = pq.ParquetFile(parquet_path)
    columns = [
        "model", "family_id", "mechanism_id", "interface", "split", "item_index",
        "component_type", "component_layer", "position_role", "component_index",
        "component_start", "component_end", "approx_target_readout_contribution",
    ]
    for batch in parquet.iter_batches(batch_size=65536, columns=columns):
        for row in batch.to_pylist():
            if row["split"] != "heldout":
                continue
            lookup_key = (row["model"], row["family_id"], row["mechanism_id"], row["interface"])
            identity = member_identity(row)
            if identity not in required_identities.get(lookup_key, set()):
                continue
            value_key = (*lookup_key, identity)
            values[value_key][int(row["item_index"])].append(float(row["approx_target_readout_contribution"]))

    validation_rows = []
    mechanism_keys = sorted({(row["model"], row["family_id"], row["mechanism_id"]) for row in member_rows})
    mechanism_summaries = []
    for model, family, mechanism in mechanism_keys:
        unique_interfaces = [value for value in INTERFACES if not (model == "glm4" and value == "chat_no_think")]
        shared_source = selected.get((model, family, mechanism), {})
        stable_shared = 0
        for identity, source in shared_source.items():
            interface_results = []
            for interface in unique_interfaces:
                item_values = values.get((model, family, mechanism, interface, identity), {})
                discovery_sign = float(source["mean_contribution"]) >= 0
                consistency = (
                    sum((mean(v) >= 0) == discovery_sign for v in item_values.values()) / len(item_values)
                    if item_values else 0.0
                )
                interface_results.append(consistency)
                validation_rows.append({
                    "schema_version": "10.0.0", "phase_id": "Phase332", "created_at": now(),
                    "model": model, "family_id": family, "mechanism_id": mechanism,
                    "set_type": "shared_skeleton", "owner_interface": "shared_all_unique_interfaces",
                    "validation_interface": interface, "member_identity": json.dumps(identity),
                    "heldout_item_count": len(item_values),
                    "heldout_item_sign_consistency": rounded(consistency),
                    "heldout_mean_abs_contribution": rounded(mean(
                        abs(mean(v)) for v in item_values.values()
                    ) if item_values else 0.0),
                })
            stable_shared += int(interface_results and min(interface_results) >= 0.75)
        stable_branch_counts = {}
        for owner in ("raw_completion", "answer_aligned_chat"):
            branch_source = selected.get((model, family, mechanism, owner), {})
            stable_count = 0
            for identity, source in branch_source.items():
                interface_means = {}
                owner_consistency = 0.0
                for interface in unique_interfaces:
                    item_values = values.get((model, family, mechanism, interface, identity), {})
                    interface_means[interface] = mean(
                        abs(mean(v)) for v in item_values.values()
                    ) if item_values else 0.0
                    if interface == owner and item_values:
                        discovery_sign = float(source["mean_contribution"]) >= 0
                        owner_consistency = sum(
                            (mean(v) >= 0) == discovery_sign for v in item_values.values()
                        ) / len(item_values)
                other_max = max((value for interface, value in interface_means.items() if interface != owner), default=0.0)
                specific = owner_consistency >= 0.75 and interface_means.get(owner, 0.0) > other_max
                stable_count += int(specific)
                validation_rows.append({
                    "schema_version": "10.0.0", "phase_id": "Phase332", "created_at": now(),
                    "model": model, "family_id": family, "mechanism_id": mechanism,
                    "set_type": "interface_branch", "owner_interface": owner,
                    "validation_interface": owner, "member_identity": json.dumps(identity),
                    "heldout_item_count": len(values.get((model, family, mechanism, owner, identity), {})),
                    "heldout_item_sign_consistency": rounded(owner_consistency),
                    "heldout_mean_abs_contribution": rounded(interface_means.get(owner, 0.0)),
                    "other_interface_max_abs_contribution": rounded(other_max),
                    "heldout_branch_specific": specific,
                })
            stable_branch_counts[owner] = stable_count
        mechanism_summaries.append({
            "schema_version": "10.0.0", "phase_id": "Phase332", "created_at": now(),
            "model": model, "family_id": family, "mechanism_id": mechanism,
            "shared_skeleton_member_count": len(shared_source),
            "heldout_stable_shared_member_count": stable_shared,
            "raw_branch_member_count": len(selected.get((model, family, mechanism, "raw_completion"), {})),
            "heldout_specific_raw_branch_member_count": stable_branch_counts["raw_completion"],
            "aligned_branch_member_count": len(selected.get((model, family, mechanism, "answer_aligned_chat"), {})),
            "heldout_specific_aligned_branch_member_count": stable_branch_counts["answer_aligned_chat"],
            "shared_skeleton_stable": stable_shared > 0,
            "interface_branch_specific": (
                stable_branch_counts["raw_completion"] > 0
                and stable_branch_counts["answer_aligned_chat"] > 0
            ),
        })
    return validation_rows, mechanism_summaries


def item_direction_consistency(rows: list[dict[str, Any]], key: str) -> float:
    by_item: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_item[int(row["item_index"])].append(float(row[key]))
    return sum(mean(values) > 0 for values in by_item.values()) / len(by_item) if by_item else 0.0


def exchange_local_summaries(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = protocol["thresholds"]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family_id"], row["mechanism_id"], row["exchange_direction"])].append(row)
    result = []
    for (model, family, mechanism, direction), values in sorted(grouped.items()):
        by_condition = {name: [row for row in values if row["condition"] == name] for name in {
            row["condition"] for row in values
        }}
        correct = by_condition["shared_plus_branch_correct"]
        wrong = by_condition["shared_plus_branch_wrong_item"]
        random = by_condition["matched_random_units_correct"]
        phrase_delta = avg(correct, "delta_phrase_logprob_vs_baseline")
        margin_delta = avg(correct, "delta_target_margin_vs_baseline")
        consistency = item_direction_consistency(correct, "delta_phrase_logprob_vs_baseline")
        behavior_gain = rate(correct, "behavior_gained_vs_baseline")
        behavior_loss = rate(correct, "behavior_lost_vs_baseline")
        metric_rows = [row for condition_rows in by_condition.values() for row in condition_rows]
        finite_phrase_count = finite_metric_count(metric_rows, "delta_phrase_logprob_vs_baseline")
        finite_margin_count = finite_metric_count(metric_rows, "delta_target_margin_vs_baseline")
        metrics_complete = (
            finite_phrase_count == len(metric_rows)
            and finite_margin_count == len(metric_rows)
        )
        correct_specific = (
            metrics_complete
            and phrase_delta >= thresholds["phrase_logprob_improvement_min"]
            and phrase_delta > avg(wrong, "delta_phrase_logprob_vs_baseline")
            and phrase_delta > avg(random, "delta_phrase_logprob_vs_baseline")
            and consistency >= thresholds["heldout_item_direction_consistency_min"]
        )
        result.append({
            "schema_version": "10.0.0", "phase_id": "Phase332", "created_at": now(),
            "model": model, "family_id": family, "mechanism_id": mechanism,
            "cohort": values[0]["cohort"], "exchange_direction": direction,
            "case_count": len(by_condition["baseline"]),
            "metric_row_count": len(metric_rows),
            "finite_phrase_metric_count": finite_phrase_count,
            "finite_margin_metric_count": finite_margin_count,
            "metrics_complete": metrics_complete,
            "mean_shared_phrase_delta": rounded(avg(by_condition["shared_skeleton_correct"], "delta_phrase_logprob_vs_baseline")),
            "mean_branch_phrase_delta": rounded(avg(by_condition["interface_branch_correct"], "delta_phrase_logprob_vs_baseline")),
            "mean_combined_phrase_delta": rounded(phrase_delta),
            "mean_combined_margin_delta": rounded(margin_delta),
            "wrong_item_phrase_delta": rounded(avg(wrong, "delta_phrase_logprob_vs_baseline")),
            "random_units_phrase_delta": rounded(avg(random, "delta_phrase_logprob_vs_baseline")),
            "combined_item_direction_consistency": rounded(consistency),
            "combined_behavior_gain_rate": rounded(behavior_gain),
            "combined_behavior_loss_rate": rounded(behavior_loss),
            "combined_protocol_loss_rate": rounded(rate(correct, "protocol_lost_vs_baseline")),
            "correct_exchange_specific": correct_specific,
            "full_string_improved": (
                metrics_complete and phrase_delta >= thresholds["phrase_logprob_improvement_min"]
            ),
            "free_generation_improved": (
                metrics_complete and behavior_gain >= thresholds["behavior_gain_min"]
            ),
            "interaction_phrase_delta": rounded(
                phrase_delta
                - avg(by_condition["shared_skeleton_correct"], "delta_phrase_logprob_vs_baseline")
                - avg(by_condition["interface_branch_correct"], "delta_phrase_logprob_vs_baseline")
            ),
        })
    return result


def cross_summaries(
    path_rows: list[dict[str, Any]], exchange_rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    path = {(row["model"], row["family_id"], row["mechanism_id"]): row for row in path_rows}
    exchange = {
        (row["model"], row["family_id"], row["mechanism_id"], row["exchange_direction"]): row
        for row in exchange_rows
    }
    pairs = (
        ("language_action", "summarize", "rewrite"),
        ("reasoning_constraint", "missing_condition_control", "two_hop_blocked"),
    )
    results = []
    for family, mechanism, control in pairs:
        positive_raw_to_aligned = [
            exchange[(model, family, mechanism, "raw_to_answer_aligned")] for model in MODELS
        ]
        positive_reverse = [
            exchange[(model, family, mechanism, "answer_aligned_to_raw")] for model in MODELS
        ]
        control_raw_to_aligned = [
            exchange[(model, family, control, "raw_to_answer_aligned")] for model in MODELS
        ]
        path_cells = [path[(model, family, mechanism)] for model in MODELS]
        shared = all(row["shared_skeleton_stable"] for row in path_cells)
        branch = all(row["interface_branch_specific"] for row in path_cells)
        exchange_specific = all(row["correct_exchange_specific"] for row in positive_raw_to_aligned)
        full_string = all(row["full_string_improved"] for row in positive_raw_to_aligned)
        generation = all(row["free_generation_improved"] for row in positive_raw_to_aligned)
        low_side_effect = all(
            row["metrics_complete"]
            and row["combined_behavior_gain_rate"] <= protocol["thresholds"]["behavior_side_effect_max"]
            and row["combined_protocol_loss_rate"] <= protocol["thresholds"]["protocol_side_effect_max"]
            and not row["correct_exchange_specific"]
            for row in control_raw_to_aligned
        )
        gate = {
            "shared_skeleton_stable": shared,
            "interface_branch_specific": branch,
            "path_exchange_effective": exchange_specific,
            "full_string_improved": full_string,
            "free_generation_improved": generation,
            "low_side_effect": low_side_effect,
            "cross_model": all(len([row for row in positive_raw_to_aligned if row["model"] == model]) == 1 for model in MODELS),
        }
        full_gate = all(gate.values())
        results.append({
            "schema_version": "10.0.0", "phase_id": "Phase332", "created_at": now(),
            "family_id": family, "mechanism_id": mechanism,
            "matched_negative_control_mechanism_id": control,
            "gate": gate, "full_gate_pass": full_gate,
            "mean_raw_to_aligned_phrase_delta": rounded(avg(positive_raw_to_aligned, "mean_combined_phrase_delta")),
            "mean_raw_to_aligned_behavior_gain_rate": rounded(avg(positive_raw_to_aligned, "combined_behavior_gain_rate")),
            "mean_reverse_phrase_delta": rounded(avg(positive_reverse, "mean_combined_phrase_delta")),
            "shared_member_counts": {
                row["model"]: row["heldout_stable_shared_member_count"] for row in path_cells
            },
            "raw_branch_counts": {
                row["model"]: row["heldout_specific_raw_branch_member_count"] for row in path_cells
            },
            "aligned_branch_counts": {
                row["model"]: row["heldout_specific_aligned_branch_member_count"] for row in path_cells
            },
            "evidence_level": "L5_distributed_interface_path_candidate" if full_gate else "L3_interface_path_map_not_causally_closed",
            "behavior_mechanism_closed": False,
            "single_unit_causal": False,
            "single_unit_intervention_gate_open": full_gate,
        })
    return results


def build_report(
    quality: dict[str, Any], interface_rows: list[dict[str, Any]], path_rows: list[dict[str, Any]],
    exchange_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase332 接口状态分叉、答案起点与路径交换图谱",
        "",
        "## 固定分母",
        "",
        f"- 注册接口案例：{quality['registered_interface_case_count']}。",
        f"- 基线自然生成：{quality['baseline_generation_count']}。",
        f"- 自然全层路径事件：{quality['natural_path_row_count']}。",
        f"- 自然细组件事件：{quality['natural_unit_row_count']}。",
        f"- 注册交换案例：{quality['registered_exchange_case_count']}；交换条件结果：{quality['exchange_condition_row_count']}。",
        "- 四机制各使用 8 个全新对象；对象 0-3 只做路径发现，对象 4-7 只做交换验证。",
        "",
        "## 跨模型门",
        "",
        "| 机制 | 稳定共享骨架 | 接口分支 | 正确交换 | 完整串 | 自由生成 | 低副作用 | 完整门 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cross_rows:
        gate = row["gate"]
        lines.append(
            f"| {row['family_id']}/{row['mechanism_id']} | {gate['shared_skeleton_stable']} | "
            f"{gate['interface_branch_specific']} | {gate['path_exchange_effective']} | "
            f"{gate['full_string_improved']} | {gate['free_generation_improved']} | "
            f"{gate['low_side_effect']} | {row['full_gate_pass']} |"
        )
    lines.extend([
        "",
        "## 证据边界",
        "",
        "本阶段不复用 Phase331 的固定四成员集合，而从新对象的自然接口响应中构造可变共享骨架与接口分支。",
        "GLM4 的 native_chat（原生对话）和 chat_no_think（关闭思考）提示完全相同，分析中只按一个独立接口计算。",
        "所有组件仍是注意力头或 MLP 组，路径交换阳性也不能直接改写成单神经元因果。",
        "非有限读出值按失败关闭处理；指标不完整的正机制或对照都不能通过对应门槛。",
    ])
    return "\n".join(lines) + "\n"


def analyze() -> dict[str, Any]:
    quality = read_json(SOURCE / "phase332_execution_quality.json")
    if not quality["valid"]:
        raise RuntimeError("Incomplete Phase332 execution denominator")
    protocol = read_json(SOURCE / "phase332_registered_protocol.json")
    baseline = read_jsonl(SOURCE / "phase332_baseline_rows.jsonl")
    members = read_jsonl(SOURCE / "phase332_member_sets.jsonl")
    exchanges = read_jsonl(SOURCE / "phase332_exchange_rows.jsonl")
    baseline_nonfinite_count = sum(
        row.get(key) is not None and not math.isfinite(float(row[key]))
        for row in baseline for key in ("target_margin", "target_phrase_logprob")
    )
    exchange_nonfinite_count = sum(
        row.get(key) is not None and not math.isfinite(float(row[key]))
        for row in exchanges for key in (
            "target_margin", "target_phrase_logprob",
            "delta_target_margin_vs_baseline", "delta_phrase_logprob_vs_baseline",
        )
    )
    interface_rows = interface_summaries(baseline)
    validation_rows, path_rows = heldout_member_validation(members)
    local_exchange = exchange_local_summaries(exchanges, protocol)
    cross = cross_summaries(path_rows, local_exchange, protocol)
    write_jsonl(SOURCE / "phase332_interface_summary.jsonl", interface_rows)
    write_jsonl(SOURCE / "phase332_member_validation.jsonl", validation_rows)
    write_jsonl(SOURCE / "phase332_path_summary.jsonl", path_rows)
    write_jsonl(SOURCE / "phase332_exchange_local_summary.jsonl", local_exchange)
    write_jsonl(SOURCE / "phase332_cross_model_summary.jsonl", cross)
    claims = [{
        "schema_version": "10.0.0", "phase_id": "Phase332", "created_at": now(),
        "family_id": row["family_id"], "mechanism_id": row["mechanism_id"],
        "claim": "interface-conditioned path map; mechanism closure not established",
        "evidence_level": row["evidence_level"], "full_gate_pass": row["full_gate_pass"],
        "behavior_mechanism_closed": False, "single_unit_causal": False,
        "evidence_boundary": (
            "Variable natural component sets were discovered on four new objects and exchanged on four disjoint "
            "objects. A set or path effect is not single-neuron causality."
        ),
    } for row in cross]
    write_jsonl(SOURCE / "phase332_claim_registry.jsonl", claims)
    summary = {
        "schema_version": "10.0.0", "phase_id": "Phase332", "created_at": now(),
        "denominator": quality,
        "results": {
            "positive_mechanism_count": 2,
            "cross_model_stable_shared_skeleton_count": sum(row["gate"]["shared_skeleton_stable"] for row in cross),
            "cross_model_specific_interface_branch_count": sum(row["gate"]["interface_branch_specific"] for row in cross),
            "cross_model_path_exchange_effective_count": sum(row["gate"]["path_exchange_effective"] for row in cross),
            "cross_model_free_generation_improved_count": sum(row["gate"]["free_generation_improved"] for row in cross),
            "full_gate_pass_count": sum(row["full_gate_pass"] for row in cross),
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
            "baseline_nonfinite_metric_count": baseline_nonfinite_count,
            "exchange_nonfinite_metric_count": exchange_nonfinite_count,
            "incomplete_exchange_cell_count": sum(not row["metrics_complete"] for row in local_exchange),
        },
        "progress_vector": {
            "nine_family_registered_coverage": "9/9",
            "mechanism_census_coverage": "72/72",
            "phase332_interface_path_execution": "2/2 positive mechanisms plus 2/2 controls",
            "behavior_mechanism_closure": "0/72",
            "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "single_unit_intervention_gate_open_count": sum(row["single_unit_intervention_gate_open"] for row in cross),
    }
    write_json(SOURCE / "phase332_global_summary.json", summary)
    (SOURCE / "phase332_report.md").write_text(
        build_report(quality, interface_rows, path_rows, local_exchange, cross), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
