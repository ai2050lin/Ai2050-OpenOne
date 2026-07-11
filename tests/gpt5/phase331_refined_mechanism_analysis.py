#!/usr/bin/env python3
"""Aggregate Phase331 without promoting collection coverage to causal closure."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE330 = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas"
SOURCE = ROOT / "tests/gpt5/result/phase331_refined_mechanism_audit/refined_mechanism_audit"


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


def rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else 0.0


def rounded(value: float) -> float:
    return round(float(value), 7)


def condition(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["condition"] == name]


def direction_consistency(rows: list[dict[str, Any]]) -> float:
    by_item: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_item[int(row["item_index"])].append(float(row["delta_target_margin_vs_baseline"]))
    if not by_item:
        return 0.0
    return sum(mean(values) < 0 for values in by_item.values()) / len(by_item)


def transition_rates(
    baseline: list[dict[str, Any]], changed: list[dict[str, Any]], key: str
) -> tuple[float, float, float]:
    base = {row["audit_case_id"]: row for row in baseline}
    losses = []
    gains = []
    changes = []
    for row in changed:
        before = base[row["audit_case_id"]].get(key)
        after = row.get(key)
        if before is None or after is None:
            continue
        before = bool(before)
        after = bool(after)
        losses.append(before and not after)
        gains.append(not before and after)
        changes.append(before != after)
    denominator = len(changes)
    if not denominator:
        return 0.0, 0.0, 0.0
    return sum(losses) / denominator, sum(gains) / denominator, sum(changes) / denominator


def build_local_summaries(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = protocol["thresholds"]
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["cohort"], row["model"], row["family_id"], row["mechanism_id"], row["interface"])].append(row)
    result = []
    for (cohort, model, family, mechanism, interface), values in sorted(grouped.items()):
        baseline = condition(values, "baseline")
        joint = condition(values, "joint_set_zero")
        random_joint = condition(values, "matched_random_joint_zero")
        wrong_layer = condition(values, "wrong_layer_joint_zero")
        paired_name = "paired_control_joint_zero" if cohort == "positive" else "paired_positive_joint_zero"
        paired = condition(values, paired_name)
        joint_delta = avg(joint, "delta_target_margin_vs_baseline")
        random_delta = avg(random_joint, "delta_target_margin_vs_baseline")
        wrong_layer_delta = avg(wrong_layer, "delta_target_margin_vs_baseline")
        paired_delta = avg(paired, "delta_target_margin_vs_baseline")
        consistency = direction_consistency(joint)
        readout_specific = (
            joint_delta <= thresholds["joint_margin_delta_max"]
            and joint_delta < random_delta
            and joint_delta < wrong_layer_delta
            and joint_delta < paired_delta
            and consistency >= thresholds["item_direction_consistency_min"]
        )
        members = []
        for index in range(4):
            single = condition(values, f"single_member_{index}_zero")
            if not single:
                continue
            member_delta = avg(single, "delta_target_margin_vs_baseline")
            member_consistency = direction_consistency(single)
            members.append({
                "member_index": index,
                "mean_margin_delta": rounded(member_delta),
                "item_direction_consistency": rounded(member_consistency),
                "localized": (
                    member_delta <= thresholds["member_margin_delta_max"]
                    and member_consistency >= thresholds["item_direction_consistency_min"]
                ),
                "leave_one_out_mean_margin_delta": rounded(avg(
                    condition(values, f"set_without_member_{index}_zero"),
                    "delta_target_margin_vs_baseline",
                )),
            })
        localized_count = sum(row["localized"] for row in members)
        behavior_loss, behavior_gain, generation_changed = transition_rates(
            baseline, joint, "behavior_success"
        )
        protocol_loss, protocol_gain, protocol_changed = transition_rates(
            baseline, joint, "protocol_success_answer_segment"
        )
        paired_behavior_loss, paired_behavior_gain, paired_behavior_changed = transition_rates(
            baseline, paired, "behavior_success"
        )
        paired_protocol_loss, paired_protocol_gain, paired_protocol_changed = transition_rates(
            baseline, paired, "protocol_success_answer_segment"
        )
        summary = {
            "schema_version": "9.0.0",
            "phase_id": "Phase331",
            "created_at": now(),
            "cohort": cohort,
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "interface": interface,
            "registered_item_count": len({row["item_index"] for row in baseline}),
            "registered_template_count": len({row["template_id"] for row in baseline}),
            "baseline_case_count": len(baseline),
            "baseline_candidate_winner_rate": rounded(rate(baseline, "candidate_winner_is_target")),
            "baseline_target_in_top50_rate": rounded(rate(baseline, "target_in_top50")),
            "baseline_behavior_success_rate": rounded(rate(baseline, "behavior_success")),
            "joint_mean_margin_delta": rounded(joint_delta),
            "joint_mean_phrase_logprob_delta": rounded(avg(joint, "delta_phrase_logprob_vs_baseline")),
            "random_joint_mean_margin_delta": rounded(random_delta),
            "wrong_layer_joint_mean_margin_delta": rounded(wrong_layer_delta),
            "paired_mechanism_mean_margin_delta": rounded(paired_delta),
            "paired_mechanism_behavior_changed_rate": rounded(paired_behavior_changed),
            "paired_mechanism_behavior_loss_rate": rounded(paired_behavior_loss),
            "paired_mechanism_behavior_gain_rate": rounded(paired_behavior_gain),
            "paired_mechanism_protocol_changed_rate": rounded(paired_protocol_changed),
            "paired_mechanism_protocol_loss_rate": rounded(paired_protocol_loss),
            "paired_mechanism_protocol_gain_rate": rounded(paired_protocol_gain),
            "joint_item_direction_consistency": rounded(consistency),
            "readout_specific": readout_specific,
            "expanded_heldout_pass": len(baseline) == 12 and consistency >= thresholds["item_direction_consistency_min"],
            "joint_generation_changed_rate": rounded(rate(joint, "generation_changed_vs_baseline")),
            "joint_behavior_changed_rate": rounded(generation_changed),
            "joint_behavior_loss_rate": rounded(behavior_loss),
            "joint_behavior_gain_rate": rounded(behavior_gain),
            "joint_protocol_changed_rate": rounded(protocol_changed),
            "joint_protocol_loss_rate": rounded(protocol_loss),
            "joint_protocol_gain_rate": rounded(protocol_gain),
            "full_generation_changed": behavior_loss >= thresholds["generation_behavior_change_min"],
            "member_results": members,
            "localized_member_count": localized_count,
            "member_localized": 1 <= localized_count <= 2,
            "single_unit_causal": False,
        }
        if cohort == "positive":
            correct = condition(values, "correct_donor_transplant")
            natural_controls = {
                name: avg(condition(values, name), "delta_target_margin_vs_baseline")
                for name in (
                    "wrong_donor_transplant", "same_target_donor_transplant",
                    "matched_random_donor_transplant", "wrong_layer_donor_transplant",
                )
            }
            correct_delta = avg(correct, "delta_target_margin_vs_baseline")
            summary.update({
                "correct_donor_mean_margin_delta": rounded(correct_delta),
                "natural_control_mean_margin_deltas": {
                    key: rounded(value) for key, value in natural_controls.items()
                },
                "natural_identity_specific": correct_delta > max(natural_controls.values()) + 0.05,
                "restoration_mean_margin_delta": rounded(avg(
                    condition(values, "correct_donor_restoration"), "delta_target_margin_vs_baseline"
                )),
            })
        result.append(summary)
    return result


def build_compensation_summaries(
    unit_rows: list[dict[str, Any]], path_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    unit_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        unit_by_case[(row["audit_case_id"], row["condition"])].append(row)
    path_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        path_by_case[(row["audit_case_id"], row["condition"])].append(row)
    identities: dict[str, dict[str, Any]] = {}
    for row in unit_rows:
        identities[row["audit_case_id"]] = row
    case_results = []
    for case_id, identity in identities.items():
        baseline_units = unit_by_case.get((case_id, "baseline"), [])
        joint_units = unit_by_case.get((case_id, "joint_set_zero"), [])
        baseline_unselected = sum(abs(float(row["approx_target_readout_contribution"])) for row in baseline_units if not row["selected_carrier_member"])
        joint_unselected = sum(abs(float(row["approx_target_readout_contribution"])) for row in joint_units if not row["selected_carrier_member"])
        compensation_ratio = joint_unselected / max(baseline_unselected, 1e-8)
        baseline_path = {
            (int(row["layer"]), row["position_role"]): float(row["projection"])
            for row in path_by_case.get((case_id, "baseline"), [])
            if row["component_type"] == "residual_cumulative_state"
        }
        joint_path = {
            (int(row["layer"]), row["position_role"]): float(row["projection"])
            for row in path_by_case.get((case_id, "joint_set_zero"), [])
            if row["component_type"] == "residual_cumulative_state"
        }
        losses = [baseline_path[key] - joint_path[key] for key in baseline_path if key in joint_path and key[1] == "last"]
        peak_loss = max((abs(value) for value in losses), default=0.0)
        final_loss = abs(losses[-1]) if losses else 0.0
        recovery_fraction = 1.0 - min(1.0, final_loss / peak_loss) if peak_loss > 1e-8 else 0.0
        case_results.append({
            "audit_case_id": case_id,
            "model": identity["model"],
            "cohort": identity["cohort"],
            "family_id": identity["family_id"],
            "mechanism_id": identity["mechanism_id"],
            "interface": identity["interface"],
            "unselected_component_compensation_ratio": compensation_ratio,
            "late_residual_recovery_fraction": recovery_fraction,
        })
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_results:
        grouped[(row["cohort"], row["model"], row["family_id"], row["mechanism_id"], row["interface"])].append(row)
    summaries = []
    for key, values in sorted(grouped.items()):
        cohort, model, family, mechanism, interface = key
        summaries.append({
            "schema_version": "9.0.0",
            "phase_id": "Phase331",
            "created_at": now(),
            "cohort": cohort,
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "interface": interface,
            "trace_case_count": len(values),
            "mean_unselected_component_compensation_ratio": rounded(avg(values, "unselected_component_compensation_ratio")),
            "max_unselected_component_compensation_ratio": rounded(max(
                row["unselected_component_compensation_ratio"] for row in values
            )),
            "mean_late_residual_recovery_fraction": rounded(avg(values, "late_residual_recovery_fraction")),
            "max_late_residual_recovery_fraction": rounded(max(
                row["late_residual_recovery_fraction"] for row in values
            )),
            "compensation_measurement_complete": len(values) == 4,
        })
    return summaries


def build_cross_summaries(
    locals_: list[dict[str, Any]], compensation: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    thresholds = protocol["thresholds"]
    comp = {
        (row["cohort"], row["model"], row["family_id"], row["mechanism_id"], row["interface"]): row
        for row in compensation
    }
    positive_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    controls = {}
    for row in locals_:
        if row["cohort"] == "positive":
            positive_groups[(row["family_id"], row["mechanism_id"])].append(row)
        else:
            controls[(row["family_id"], row["mechanism_id"], row["model"], row["interface"])] = row
    pair_lookup = {
        ("content_knowledge", "negated_attribute"): "attribute",
        ("language_action", "summarize"): "rewrite",
        ("language_action", "transform"): "extract",
        ("reasoning_constraint", "missing_condition_control"): "two_hop_blocked",
        ("syntax_structure", "singular_agreement"): "plural_agreement",
    }
    results = []
    for (family, mechanism), values in sorted(positive_groups.items()):
        control_mechanism = pair_lookup[(family, mechanism)]
        expected_cells = 6
        readout_specific = len(values) == expected_cells and all(row["readout_specific"] for row in values)
        expanded = len(values) == expected_cells and all(row["expanded_heldout_pass"] for row in values)
        member_localized = len(values) == expected_cells and all(row["member_localized"] for row in values)
        generation_changed = len(values) == expected_cells and all(row["full_generation_changed"] for row in values)
        natural_identity = len(values) == expected_cells and all(row["natural_identity_specific"] for row in values)
        control_cells = [
            controls[(family, control_mechanism, row["model"], row["interface"])]
            for row in values
        ]
        negative_control_failed = all(not row["readout_specific"] for row in control_cells)
        paired_behavior_change = max(row["paired_mechanism_behavior_changed_rate"] for row in control_cells)
        paired_protocol_change = max(row["paired_mechanism_protocol_changed_rate"] for row in control_cells)
        low_side_effect = (
            paired_behavior_change <= thresholds["paired_control_behavior_change_max"]
            and paired_protocol_change <= thresholds["paired_control_protocol_loss_max"]
        )
        compensation_cells = [
            comp[("positive", row["model"], family, mechanism, row["interface"])]
            for row in values
        ]
        compensation_accounted = len(compensation_cells) == expected_cells and all(
            row["compensation_measurement_complete"] for row in compensation_cells
        )
        compensation_below_gate = compensation_accounted and all(
            row["mean_unselected_component_compensation_ratio"] <= thresholds["nonselected_compensation_ratio_max"]
            and row["mean_late_residual_recovery_fraction"] <= thresholds["late_residual_recovery_fraction_max"]
            for row in compensation_cells
        )
        cross_interface = all(
            len([row for row in values if row["model"] == model and row["readout_specific"]]) == 2
            for model in ("qwen3", "glm4", "deepseek7b")
        )
        cross_model = all(
            any(row["model"] == model and row["readout_specific"] for row in values)
            for model in ("qwen3", "glm4", "deepseek7b")
        )
        gate = {
            "readout_specific": readout_specific,
            "expanded_heldout": expanded,
            "cross_interface": cross_interface,
            "cross_model": cross_model,
            "member_localized": member_localized,
            "compensation_accounted": compensation_accounted and compensation_below_gate,
            "full_generation_changed": generation_changed,
            "low_side_effect": low_side_effect and negative_control_failed,
        }
        full_gate = all(gate.values())
        if full_gate:
            evidence = "L5_registered_distributed_mechanism_candidate"
        elif readout_specific and expanded:
            evidence = "L4_expanded_set_readout"
        else:
            evidence = "L3_candidate_not_expanded_cross_interface"
        results.append({
            "schema_version": "9.0.0",
            "phase_id": "Phase331",
            "created_at": now(),
            "family_id": family,
            "mechanism_id": mechanism,
            "matched_negative_control_mechanism_id": control_mechanism,
            "model_interface_cell_count": len(values),
            "mean_joint_margin_delta": rounded(avg(values, "joint_mean_margin_delta")),
            "mean_joint_phrase_logprob_delta": rounded(avg(values, "joint_mean_phrase_logprob_delta")),
            "mean_joint_behavior_changed_rate": rounded(avg(values, "joint_behavior_changed_rate")),
            "mean_joint_behavior_loss_rate": rounded(avg(values, "joint_behavior_loss_rate")),
            "mean_joint_behavior_gain_rate": rounded(avg(values, "joint_behavior_gain_rate")),
            "natural_identity_specific": natural_identity,
            "negative_control_failed": negative_control_failed,
            "mean_compensation_ratio": rounded(avg(compensation_cells, "mean_unselected_component_compensation_ratio")),
            "mean_late_residual_recovery_fraction": rounded(avg(compensation_cells, "mean_late_residual_recovery_fraction")),
            "gate": gate,
            "full_gate_pass": full_gate,
            "evidence_level": evidence,
            "behavior_mechanism_closed": full_gate and generation_changed,
            "single_unit_causal": False,
            "single_unit_intervention_gate_open": full_gate,
        })
    return results


def build_report(
    execution: dict[str, Any], locals_: list[dict[str, Any]], cross: list[dict[str, Any]],
    compensation: list[dict[str, Any]], condition_rows: list[dict[str, Any]],
    path_count: int, unit_count: int,
) -> str:
    expanded = sum(row["gate"]["readout_specific"] and row["gate"]["expanded_heldout"] for row in cross)
    full = sum(row["full_gate_pass"] for row in cross)
    behavior = sum(row["behavior_mechanism_closed"] for row in cross)
    lines = [
        "# Phase331 五条候选链扩展留出、双接口与补偿审计",
        "",
        "## 客观分母",
        "",
        f"- 接口案例：{execution['interface_case_count']}（正机制 360，匹配负对照 360）。",
        f"- 条件结果：{execution['condition_row_count']}；执行自然生成：{execution['generation_row_count']}。",
        f"- 全层路径事件：{path_count}；组件响应事件：{unit_count}。",
        "- 留出对象固定为 19-22，覆盖三个模板、原始续写接口和模型对话模板接口。",
        "",
        "## 结果边界",
        "",
        f"- 五条 Phase330 候选中，扩展后仍通过跨模型、跨接口集合读出门槛：{expanded}/5。",
        f"- 通过完整八项 Phase331 门槛：{full}/5。",
        f"- 可宣称行为机制闭合：{behavior}/5；全 72 机制仍为 {behavior}/72。",
        "- 所有成员仍是注意力头或 MLP 乘积组候选，没有把组级效果改写成单神经元因果。",
        "",
        "## 五条候选链",
        "",
        "| 模式机制 | 集合读出扩展 | 成员定位 | 补偿受控 | 生成行为损失 | 负对照失败 | 证据等级 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in cross:
        gate = row["gate"]
        lines.append(
            f"| {row['family_id']}/{row['mechanism_id']} | "
            f"{gate['readout_specific'] and gate['expanded_heldout']} | {gate['member_localized']} | "
            f"{gate['compensation_accounted']} | {gate['full_generation_changed']} | "
            f"{row['negative_control_failed']} | {row['evidence_level']} |"
        )
    lines.extend([
        "",
        "## 关键校准",
        "",
        "1. 对话模板不是表面包装。模型会引入助手前缀或思考前缀，因此首词元读出必须和完整答案串概率、自然生成分开报告。",
        "2. 累计残差状态不能当作本层写入量。本轮同时保存累计状态、注意力增量、MLP 增量和层间残差增量。",
        "3. 组级联合干预可能被未干预头、其他 MLP 组或后层残差恢复补偿；没有补偿审计就不能把读出变化解释成行为必要性。",
        "4. 小模型之间的路径位置和思考协议差异较大；跨模型失败既可能是否定共享机制，也可能反映 4B-9B 模型的粗糙结构，二者尚不能区分。",
        "",
        "## 图谱进度向量",
        "",
        "- 九族注册与观察分母：9/9（100%）。",
        "- 72 机制三模型行为、读出和全层普查：72/72（100%）。",
        "- Phase330 五条候选的扩展留出与双接口审计：5/5（100% 已测试，不等于通过）。",
        f"- 通过 Phase331 全门槛：{full}/5。",
        f"- 语言机制行为闭合：{behavior}/72。",
        "- 单神经元因果闭合：0/72。",
        "",
        "因此不提供一个会混淆工程覆盖率与科学证据率的单一总百分比。",
    ])
    return "\n".join(lines) + "\n"


def analyze() -> dict[str, Any]:
    execution = read_json(SOURCE / "phase331_execution_quality.json")
    if not execution["valid"]:
        raise RuntimeError("Phase331 execution denominator is incomplete")
    protocol = read_json(SOURCE / "phase331_registered_protocol.json")
    condition_rows = read_jsonl(SOURCE / "phase331_condition_rows.jsonl")
    path_table = pq.read_table(SOURCE / "phase331_compensation_path_rows.parquet")
    unit_table = pq.read_table(SOURCE / "phase331_component_response_rows.parquet")
    path_rows = path_table.to_pylist()
    unit_rows = unit_table.to_pylist()
    locals_ = build_local_summaries(condition_rows, protocol)
    compensation = build_compensation_summaries(unit_rows, path_rows)
    cross = build_cross_summaries(locals_, compensation, protocol)
    write_jsonl(SOURCE / "phase331_local_summary.jsonl", locals_)
    write_jsonl(SOURCE / "phase331_compensation_summary.jsonl", compensation)
    write_jsonl(SOURCE / "phase331_cross_model_summary.jsonl", cross)
    claims = []
    for row in cross:
        claims.append({
            "schema_version": "9.0.0",
            "phase_id": "Phase331",
            "created_at": now(),
            "family_id": row["family_id"],
            "mechanism_id": row["mechanism_id"],
            "claim": (
                "expanded distributed mechanism candidate"
                if row["full_gate_pass"] else
                "frozen component-set candidate; closure not established"
            ),
            "evidence_level": row["evidence_level"],
            "full_gate_pass": row["full_gate_pass"],
            "behavior_mechanism_closed": row["behavior_mechanism_closed"],
            "single_unit_causal": False,
            "evidence_boundary": (
                "Phase331 tests a frozen four-member attention/MLP set on four untouched heldout objects, three "
                "templates, two interfaces, and three small models. It does not establish a single causal neuron."
            ),
        })
    write_jsonl(SOURCE / "phase331_claim_registry.jsonl", claims)
    global_summary = {
        "schema_version": "9.0.0",
        "phase_id": "Phase331",
        "created_at": now(),
        "denominator": {
            "registered_family_count": 9,
            "registered_mechanism_count": 72,
            "refined_positive_mechanism_count": 5,
            "matched_negative_control_mechanism_count": 5,
            "interface_case_count": execution["interface_case_count"],
            "condition_row_count": execution["condition_row_count"],
            "generation_row_count": execution["generation_row_count"],
            "compensation_path_row_count": execution["compensation_path_row_count"],
            "component_response_row_count": execution["component_response_row_count"],
        },
        "results": {
            "expanded_cross_model_cross_interface_readout_count": sum(
                row["gate"]["readout_specific"] and row["gate"]["expanded_heldout"] for row in cross
            ),
            "member_localized_count": sum(row["gate"]["member_localized"] for row in cross),
            "compensation_accounted_and_below_gate_count": sum(row["gate"]["compensation_accounted"] for row in cross),
            "full_generation_changed_count": sum(row["gate"]["full_generation_changed"] for row in cross),
            "full_gate_pass_count": sum(row["full_gate_pass"] for row in cross),
            "behavior_mechanism_closed_count": sum(row["behavior_mechanism_closed"] for row in cross),
            "single_unit_causal_count": 0,
        },
        "progress_vector": {
            "nine_family_registered_coverage": "9/9",
            "mechanism_census_coverage": "72/72",
            "phase331_refinement_execution": "5/5",
            "behavior_mechanism_closure": f"{sum(row['behavior_mechanism_closed'] for row in cross)}/72",
            "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "single_unit_intervention_gate_open_count": sum(row["single_unit_intervention_gate_open"] for row in cross),
        "small_model_deviation_warning": (
            "The three 4B-9B local models can use coarser or model-specific paths; cross-model differences cannot yet "
            "be projected directly onto larger language models."
        ),
    }
    write_json(SOURCE / "phase331_global_summary.json", global_summary)
    report = build_report(
        execution, locals_, cross, compensation, condition_rows, len(path_rows), len(unit_rows)
    )
    (SOURCE / "phase331_report.md").write_text(report, encoding="utf-8")
    return global_summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
