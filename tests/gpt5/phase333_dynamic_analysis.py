#!/usr/bin/env python3
"""Aggregate Phase333 dynamic sequences, block effects, and compensation candidates."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("raw_completion", "native_chat", "answer_aligned_chat")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = finite_values(rows, key)
    return mean(values) if values else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def rounded(value: float) -> float:
    return round(float(value), 7)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def heldout_peak_depths(event_rows: list[dict[str, Any]]) -> dict[str, float]:
    selected_steps = {
        row["case_id"]: (
            int(row["target_pressure_formation_step"])
            if int(row["target_pressure_formation_step"]) >= 0 else 0
        )
        for row in event_rows if row["split"] in {"calibration", "heldout"}
    }
    best: dict[str, tuple[float, float]] = {}
    parquet = pq.ParquetFile(SOURCE / "phase333_dynamic_path_rows.parquet")
    columns = ["case_id", "component_type", "generated_step", "relative_depth", "projection"]
    for batch in parquet.iter_batches(batch_size=65536, columns=columns):
        for row in batch.to_pylist():
            case_id = row["case_id"]
            if case_id not in selected_steps:
                continue
            if row["component_type"] != "residual_output":
                continue
            if int(row["generated_step"]) != selected_steps[case_id]:
                continue
            projection = finite_float(row["projection"])
            if projection is None:
                continue
            current = best.get(case_id)
            if current is None or projection > current[0]:
                best[case_id] = (projection, float(row["relative_depth"]))
    return {case_id: value[1] for case_id, value in best.items()}


def object_presence(rows: list[dict[str, Any]], key: str) -> float:
    by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[int(row["item_index"])].append(row)
    if not by_object:
        return 0.0
    passed = sum(
        sum(int(row[key]) >= 0 for row in values) / len(values) >= 2 / 3
        for values in by_object.values()
    )
    return passed / len(by_object)


def sequence_summaries(
    event_rows: list[dict[str, Any]], plans: list[dict[str, Any]], peak_depths: dict[str, float],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(row["model"], row["mechanism_id"], row["interface"])].append(row)
    plan_map = {(row["model"], row["mechanism_id"], row["interface"]): row for row in plans}
    results = []
    for key, values in sorted(grouped.items()):
        model, mechanism, interface = key
        discovery = [row for row in values if row["split"] == "discovery"]
        calibration = [row for row in values if row["split"] == "calibration"]
        heldout = [row for row in values if row["split"] == "heldout"]
        plan = plan_map[key]
        calibration_depths = [peak_depths[row["case_id"]] for row in calibration if row["case_id"] in peak_depths]
        heldout_depths = [peak_depths[row["case_id"]] for row in heldout if row["case_id"] in peak_depths]
        calibration_depth_error = abs(
            (median(calibration_depths) if calibration_depths else 1.0)
            - float(plan["median_relative_peak_depth"])
        )
        heldout_depth_error = abs(
            (median(heldout_depths) if heldout_depths else 1.0)
            - float(plan["median_relative_peak_depth"])
        )
        calibration_presence = object_presence(calibration, "target_pressure_formation_step")
        heldout_presence = object_presence(heldout, "target_pressure_formation_step")
        stable = (
            calibration_presence >= protocol["thresholds"]["object_event_presence_min"]
            and heldout_presence >= protocol["thresholds"]["object_event_presence_min"]
            and calibration_depth_error <= protocol["thresholds"]["relative_depth_tolerance"]
            and heldout_depth_error <= protocol["thresholds"]["relative_depth_tolerance"]
        )
        discovery_orders = Counter(row["event_order"] for row in discovery)
        results.append({
            "schema_version": "11.0.0", "phase_id": "Phase333", "created_at": now(),
            "model": model, "family_id": "reasoning_constraint", "mechanism_id": mechanism,
            "cohort": values[0]["cohort"], "interface": interface,
            "discovery_case_count": len(discovery), "calibration_case_count": len(calibration),
            "heldout_case_count": len(heldout),
            "discovery_modal_event_order": discovery_orders.most_common(1)[0][0] if discovery_orders else "[]",
            "discovery_modal_event_order_rate": rounded(
                discovery_orders.most_common(1)[0][1] / len(discovery) if discovery else 0.0
            ),
            "calibration_object_target_formation_rate": rounded(calibration_presence),
            "heldout_object_target_formation_rate": rounded(heldout_presence),
            "frozen_relative_peak_depth": plan["median_relative_peak_depth"],
            "calibration_median_relative_peak_depth": rounded(median(calibration_depths) if calibration_depths else 0.0),
            "heldout_median_relative_peak_depth": rounded(median(heldout_depths) if heldout_depths else 0.0),
            "calibration_relative_depth_error": rounded(calibration_depth_error),
            "heldout_relative_depth_error": rounded(heldout_depth_error),
            "dynamic_sequence_stable": stable,
        })
    return results


def interface_alignment(sequence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["model"], row["mechanism_id"], row["interface"]): row for row in sequence_rows}
    result = []
    for model in MODELS:
        for mechanism in ("missing_condition_control", "two_hop_blocked"):
            values = [lookup[(model, mechanism, interface)] for interface in INTERFACES]
            depths = [float(row["heldout_median_relative_peak_depth"]) for row in values]
            result.append({
                "schema_version": "11.0.0", "phase_id": "Phase333", "created_at": now(),
                "model": model, "family_id": "reasoning_constraint", "mechanism_id": mechanism,
                "unique_interface_count": 3,
                "stable_interface_count": sum(row["dynamic_sequence_stable"] for row in values),
                "heldout_peak_depth_span": rounded(max(depths) - min(depths)),
                "functional_interface_alignment": (
                    all(row["dynamic_sequence_stable"] for row in values)
                    and max(depths) - min(depths) <= 0.15
                ),
            })
    return result


def local_block_summaries(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["mechanism_id"], row["exchange_direction"])].append(row)
    controls = (
        "wrong_object_block_4", "wrong_interface_block_4", "wrong_time_block_4",
        "moment_matched_permutation_block_4", "matched_control_block_4",
    )
    result = []
    for (model, mechanism, direction), values in sorted(grouped.items()):
        by_condition = {
            condition: [row for row in values if row["condition"] == condition]
            for condition in {row["condition"] for row in values}
        }
        correct = by_condition["correct_block_4"]
        metric_rows = [row for condition_rows in by_condition.values() for row in condition_rows]
        metrics_complete = all(
            len(finite_values(metric_rows, key)) == len(metric_rows)
            for key in ("delta_phrase_logprob_vs_baseline", "delta_target_margin_vs_baseline")
        )
        phrase_delta = avg(correct, "delta_phrase_logprob_vs_baseline")
        rank_improvement = avg(correct, "target_rank_improvement_vs_baseline")
        control_max = max(avg(by_condition[name], "delta_phrase_logprob_vs_baseline") for name in controls)
        generation_patch_rate = rate(correct, "patch_reached_generation")
        phrase_patch_rate = rate(correct, "patch_reached_phrase")
        specific = (
            metrics_complete
            and generation_patch_rate == 1.0
            and phrase_patch_rate == 1.0
            and phrase_delta >= protocol["thresholds"]["phrase_logprob_improvement_min"]
            and rank_improvement >= protocol["thresholds"]["target_rank_improvement_min"]
            and phrase_delta >= control_max + protocol["thresholds"]["control_superiority_min"]
        )
        result.append({
            "schema_version": "11.0.0", "phase_id": "Phase333", "created_at": now(),
            "model": model, "family_id": "reasoning_constraint", "mechanism_id": mechanism,
            "cohort": values[0]["cohort"], "exchange_direction": direction,
            "case_count": len(by_condition["baseline"]), "metrics_complete": metrics_complete,
            "correct_block_generation_patch_rate": rounded(generation_patch_rate),
            "correct_block_phrase_patch_rate": rounded(phrase_patch_rate),
            "mean_correct_block_1_phrase_delta": rounded(avg(by_condition["correct_block_1"], "delta_phrase_logprob_vs_baseline")),
            "mean_correct_block_2_phrase_delta": rounded(avg(by_condition["correct_block_2"], "delta_phrase_logprob_vs_baseline")),
            "mean_correct_block_4_phrase_delta": rounded(phrase_delta),
            "mean_correct_block_4_margin_delta": rounded(avg(correct, "delta_target_margin_vs_baseline")),
            "mean_correct_block_4_rank_improvement": rounded(rank_improvement),
            "max_control_phrase_delta": rounded(control_max),
            "correct_block_specific": specific,
            "free_generation_gain_rate": rounded(rate(correct, "behavior_gained_vs_baseline")),
            "free_generation_loss_rate": rounded(rate(correct, "behavior_lost_vs_baseline")),
            "protocol_loss_rate": rounded(rate(correct, "protocol_lost_vs_baseline")),
            "block_scaling_monotonic": (
                avg(by_condition["correct_block_4"], "delta_phrase_logprob_vs_baseline")
                >= avg(by_condition["correct_block_2"], "delta_phrase_logprob_vs_baseline")
                >= avg(by_condition["correct_block_1"], "delta_phrase_logprob_vs_baseline")
            ),
            **{
                f"{name}_phrase_delta": rounded(avg(by_condition[name], "delta_phrase_logprob_vs_baseline"))
                for name in controls
            },
        })
    return result


def compensation_rows(response_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], float]]:
    residual = [row for row in response_rows if row["component_type"] == "residual_output"]
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in residual:
        if row["condition"] not in {"baseline", "correct_block_4"}:
            continue
        grouped[(row["model"], row["exchange_case_id"])][row["condition"]].append(row)
    result = []
    rates: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for (model, case_id), conditions in sorted(grouped.items()):
        if set(conditions) != {"baseline", "correct_block_4"}:
            continue
        baseline = {int(row["component_layer"]): row for row in conditions["baseline"]}
        correct = {int(row["component_layer"]): row for row in conditions["correct_block_4"]}
        sample = conditions["correct_block_4"][0]
        block_layers = json.loads(sample["block_layers"])
        block_end = max(block_layers)
        deltas = {}
        for layer in sorted(set(baseline) & set(correct)):
            correct_projection = finite_float(correct[layer]["projection"])
            baseline_projection = finite_float(baseline[layer]["projection"])
            if correct_projection is None or baseline_projection is None:
                continue
            deltas[layer] = correct_projection - baseline_projection
        immediate = deltas.get(block_end, 0.0)
        recovery_layer = -1
        if abs(immediate) > 1e-8:
            recovery_layer = next((
                layer for layer in sorted(deltas)
                if layer > block_end and abs(deltas[layer]) <= 0.5 * abs(immediate)
            ), -1)
        final_layer = max(deltas) if deltas else block_end
        downstream_layer_count = max(final_layer - block_end, 0)
        trace_complete = (
            downstream_layer_count > 0
            and all(layer in deltas for layer in range(block_end, final_layer + 1))
        )
        classification = (
            "recovered" if recovery_layer >= 0
            else "persisted" if trace_complete and abs(deltas.get(final_layer, 0.0)) >= 0.5 * abs(immediate)
            else "unresolved"
        )
        explained = classification in {"recovered", "persisted"} and abs(immediate) > 1e-8
        key = (model, sample["mechanism_id"], sample["exchange_direction"])
        rates[key].append(explained)
        result.append({
            "schema_version": "11.0.0", "phase_id": "Phase333", "created_at": now(),
            "model": model, "exchange_case_id": case_id,
            "family_id": sample["family_id"], "mechanism_id": sample["mechanism_id"],
            "cohort": sample["cohort"], "exchange_direction": sample["exchange_direction"],
            "block_start_layer": min(block_layers), "block_end_layer": block_end,
            "immediate_projection_delta": rounded(immediate),
            "final_projection_delta": rounded(deltas.get(final_layer, 0.0)),
            "recovery_layer": recovery_layer,
            "layer_lag": recovery_layer - block_end if recovery_layer >= 0 else -1,
            "downstream_layer_count": downstream_layer_count,
            "compensation_classification": classification,
            "trace_complete": trace_complete,
            "compensation_explained": explained,
            "causal_edge": False,
            "evidence_level": "L3_lagged_compensation_candidate",
        })
    return result, {key: sum(values) / len(values) for key, values in rates.items() if values}


def cross_summary(
    alignments: list[dict[str, Any]], local: list[dict[str, Any]], compensation_rates: dict[tuple[str, str, str], float],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    alignment = {(row["model"], row["mechanism_id"]): row for row in alignments}
    block = {(row["model"], row["mechanism_id"], row["exchange_direction"]): row for row in local}
    positive_cells = [
        block[(model, "missing_condition_control", direction)]
        for model in MODELS for direction in ("raw_to_answer_aligned", "answer_aligned_to_raw")
    ]
    control_cells = [
        block[(model, "two_hop_blocked", direction)]
        for model in MODELS for direction in ("raw_to_answer_aligned", "answer_aligned_to_raw")
    ]
    dynamic_stable = all(alignment[(model, "missing_condition_control")]["functional_interface_alignment"] for model in MODELS)
    block_effective = all(row["correct_block_specific"] for row in positive_cells)
    competition = all(
        row["mean_correct_block_4_phrase_delta"] >= protocol["thresholds"]["phrase_logprob_improvement_min"]
        and row["mean_correct_block_4_rank_improvement"] >= protocol["thresholds"]["target_rank_improvement_min"]
        for row in positive_cells
    )
    compensation = all(
        compensation_rates.get((row["model"], row["mechanism_id"], row["exchange_direction"]), 0.0)
        >= protocol["thresholds"]["compensation_explained_rate_min"]
        for row in positive_cells
    )
    generation = all(row["free_generation_gain_rate"] >= protocol["thresholds"]["behavior_gain_min"] for row in positive_cells)
    controls_clean = all(
        not row["correct_block_specific"]
        and row["free_generation_gain_rate"] <= protocol["thresholds"]["side_effect_max"]
        and row["protocol_loss_rate"] <= protocol["thresholds"]["side_effect_max"]
        for row in control_cells
    )
    gate = {
        "dynamic_sequence_stable": dynamic_stable,
        "state_block_effective": block_effective,
        "competition_consistent": competition,
        "compensation_explained": compensation,
        "free_generation_improved": generation,
        "matched_controls_clean": controls_clean,
        "cross_model": len({row["model"] for row in positive_cells}) == 3,
    }
    full = all(gate.values())
    return [{
        "schema_version": "11.0.0", "phase_id": "Phase333", "created_at": now(),
        "family_id": "reasoning_constraint", "mechanism_id": "missing_condition_control",
        "matched_negative_control_mechanism_id": "two_hop_blocked",
        "gate": gate, "full_gate_pass": full,
        "positive_model_direction_cell_count": len(positive_cells),
        "control_model_direction_cell_count": len(control_cells),
        "mean_correct_block_4_phrase_delta": rounded(avg(positive_cells, "mean_correct_block_4_phrase_delta")),
        "mean_correct_block_4_rank_improvement": rounded(avg(positive_cells, "mean_correct_block_4_rank_improvement")),
        "mean_free_generation_gain_rate": rounded(avg(positive_cells, "free_generation_gain_rate")),
        "evidence_level": "L5_dynamic_causal_candidate" if full else "L3_dynamic_path_not_causally_closed",
        "behavior_mechanism_closed": False,
        "single_unit_causal": False,
        "single_unit_intervention_gate_open": full,
    }]


def report(quality: dict[str, Any], cross: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    row = cross[0]
    gate = row["gate"]
    return "\n".join([
        "# Phase333 动态时序路径、连续残差块与补偿图谱",
        "",
        "## 固定执行",
        "",
        f"- 注册自然案例：{quality['registered_case_count']}。",
        f"- 逐词元读出：{quality['token_row_count']}。",
        f"- 五组件动态路径：{quality['dynamic_path_row_count']}。",
        f"- 注册留出交换：{quality['registered_exchange_case_count']}；条件生成：{quality['condition_row_count']}。",
        f"- 动态响应：{quality['dynamic_response_row_count']}。",
        "",
        "## 七门",
        "",
        *[f"- {key}: {value}" for key, value in gate.items()],
        f"- full_gate_pass: {row['full_gate_pass']}",
        "",
        "## 边界",
        "",
        "连续残差输出块是功能时间对齐的组件级干预，不是单神经元干预。",
        "补偿边只表示干预后按层滞后恢复或持续的候选关系，不是已闭合因果边。",
        (
            "无效条件指标按失败关闭："
            f"{summary['results']['invalid_condition_metric_count']}"
            f"（缺失 {summary['results']['missing_condition_metric_count']}，"
            f"非有限 {summary['results']['nonfinite_condition_metric_count']}）。"
        ),
    ]) + "\n"


def analyze() -> dict[str, Any]:
    quality = read_json(SOURCE / "phase333_execution_quality.json")
    if not quality["valid"]:
        raise RuntimeError("Incomplete Phase333 execution denominator")
    protocol = read_json(SOURCE / "phase333_registered_protocol.json")
    events = read_jsonl(SOURCE / "phase333_event_rows.jsonl")
    plans = read_jsonl(SOURCE / "phase333_block_plans.jsonl")
    conditions = read_jsonl(SOURCE / "phase333_condition_rows.jsonl")
    peaks = heldout_peak_depths(events)
    sequence = sequence_summaries(events, plans, peaks, protocol)
    alignments = interface_alignment(sequence)
    local = local_block_summaries(conditions, protocol)
    responses = pq.read_table(SOURCE / "phase333_dynamic_response_rows.parquet").to_pylist()
    compensation, compensation_rates = compensation_rows(responses)
    cross = cross_summary(alignments, local, compensation_rates, protocol)
    metric_keys = (
        "target_margin", "target_phrase_logprob", "delta_target_margin_vs_baseline",
        "delta_phrase_logprob_vs_baseline",
    )
    missing = sum(
        row.get(key) is None
        for row in conditions for key in metric_keys
    )
    nonfinite = sum(
        row.get(key) is not None and not math.isfinite(float(row[key]))
        for row in conditions for key in metric_keys
    )
    write_jsonl(SOURCE / "phase333_sequence_summary.jsonl", sequence)
    write_jsonl(SOURCE / "phase333_interface_alignment.jsonl", alignments)
    write_jsonl(SOURCE / "phase333_block_local_summary.jsonl", local)
    write_jsonl(SOURCE / "phase333_compensation_candidates.jsonl", compensation)
    write_jsonl(SOURCE / "phase333_cross_model_summary.jsonl", cross)
    claims = [{
        "schema_version": "11.0.0", "phase_id": "Phase333", "created_at": now(),
        "family_id": "reasoning_constraint", "mechanism_id": "missing_condition_control",
        "claim": "dynamic sequence and residual-block audit; mechanism closure not assumed",
        "evidence_level": cross[0]["evidence_level"],
        "full_gate_pass": cross[0]["full_gate_pass"],
        "behavior_mechanism_closed": False, "single_unit_causal": False,
        "evidence_boundary": (
            "Twelve new objects, three interfaces, three templates, and three models were used. "
            "Continuous residual-block effects and lagged recovery remain component-level evidence."
        ),
    }]
    write_jsonl(SOURCE / "phase333_claim_registry.jsonl", claims)
    summary = {
        "schema_version": "11.0.0", "phase_id": "Phase333", "created_at": now(),
        "denominator": quality,
        "results": {
            "positive_mechanism_count": 1,
            "cross_model_dynamic_sequence_stable_count": int(cross[0]["gate"]["dynamic_sequence_stable"]),
            "cross_model_state_block_effective_count": int(cross[0]["gate"]["state_block_effective"]),
            "cross_model_compensation_explained_count": int(cross[0]["gate"]["compensation_explained"]),
            "cross_model_free_generation_improved_count": int(cross[0]["gate"]["free_generation_improved"]),
            "full_gate_pass_count": int(cross[0]["full_gate_pass"]),
            "behavior_mechanism_closed_count": 0,
            "single_unit_causal_count": 0,
            "missing_condition_metric_count": missing,
            "nonfinite_condition_metric_count": nonfinite,
            "invalid_condition_metric_count": missing + nonfinite,
        },
        "progress_vector": {
            "nine_family_registered_coverage": "9/9",
            "mechanism_census_coverage": "72/72",
            "phase333_dynamic_execution": "1/1 positive plus 1/1 matched control",
            "behavior_mechanism_closure": "0/72",
            "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "single_unit_intervention_gate_open_count": int(cross[0]["single_unit_intervention_gate_open"]),
    }
    write_json(SOURCE / "phase333_global_summary.json", summary)
    (SOURCE / "phase333_report.md").write_text(report(quality, cross, summary), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))
