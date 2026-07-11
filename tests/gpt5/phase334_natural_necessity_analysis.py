#!/usr/bin/env python3
"""Collect and strictly analyze Phase334 natural-necessity evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("raw_completion", "native_chat", "answer_aligned_chat")
MECHANISMS = (
    "material", "attribute", "missing_condition_control", "two_hop_blocked",
    "past_tense", "plural_agreement",
)
CONTROL_CONDITIONS = (
    "wrong_time_delete", "wrong_object_increment", "matched_mechanism_increment",
    "moment_matched_permutation", "wrong_layer_delete",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_parquet(path: Path, tables: list[pa.Table]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.concat_tables(tables, promote_options="permissive"), path, compression="zstd"
    )


def finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def rounded(value: float) -> float:
    return round(float(value), 7)


def avg(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if finite(row.get(key))]
    return mean(values) if values else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def baseline_eligible(row: dict[str, Any]) -> bool:
    return bool(
        row["behavior_success"] and int(row["target_rank"]) <= 50
        and finite(row.get("target_phrase_logprob"))
    )


def collect() -> dict[str, Any]:
    baseline = []
    discovery_plans = []
    calibration_rows = []
    calibration_summaries = []
    frozen_plans = []
    registry = []
    heldout_rows = []
    contrast_tables = []
    response_tables = []
    survey_quality = []
    calibration_quality = []
    heldout_quality = []
    for model in MODELS:
        survey = SOURCE / "survey" / model
        calibration = SOURCE / "calibration" / model
        heldout = SOURCE / "heldout" / model
        survey_quality.append(read_json(survey / "complete.json"))
        calibration_quality.append(read_json(calibration / "complete.json"))
        heldout_quality.append(read_json(heldout / "complete.json"))
        baseline.extend(read_jsonl(survey / "baseline_rows.jsonl"))
        discovery_plans.extend(read_jsonl(survey / "discovery_candidate_plans.jsonl"))
        calibration_rows.extend(read_jsonl(calibration / "calibration_condition_rows.jsonl"))
        calibration_summaries.extend(read_jsonl(calibration / "calibration_candidate_summary.jsonl"))
        frozen_plans.extend(read_jsonl(calibration / "frozen_necessity_plans.jsonl"))
        registry.extend(read_jsonl(heldout / "registered_heldout_cases.jsonl"))
        heldout_rows.extend(read_jsonl(heldout / "heldout_condition_rows.jsonl"))
        contrast_tables.append(pq.read_table(survey / "natural_contrast_rows.parquet"))
        response_tables.append(pq.read_table(heldout / "downstream_response_rows.parquet"))
    write_jsonl(SOURCE / "phase334_baseline_rows.jsonl", baseline)
    write_jsonl(SOURCE / "phase334_discovery_candidate_plans.jsonl", discovery_plans)
    write_jsonl(SOURCE / "phase334_calibration_condition_rows.jsonl", calibration_rows)
    write_jsonl(SOURCE / "phase334_calibration_candidate_summary.jsonl", calibration_summaries)
    write_jsonl(SOURCE / "phase334_frozen_necessity_plans.jsonl", frozen_plans)
    write_jsonl(SOURCE / "phase334_registered_heldout_cases.jsonl", registry)
    write_jsonl(SOURCE / "phase334_heldout_condition_rows.jsonl", heldout_rows)
    write_parquet(SOURCE / "phase334_natural_contrast_rows.parquet", contrast_tables)
    write_parquet(SOURCE / "phase334_downstream_response_rows.parquet", response_tables)
    quality = {
        "schema_version": "12.0.0", "phase_id": "Phase334", "created_at": now(),
        "model_count": 3, "registered_case_count": len(baseline),
        "baseline_generation_count": len(baseline),
        "natural_contrast_row_count": sum(row["natural_contrast_row_count"] for row in survey_quality),
        "discovery_candidate_plan_count": len(discovery_plans),
        "calibration_case_count": sum(row["calibration_case_count"] for row in calibration_quality),
        "calibration_condition_row_count": len(calibration_rows),
        "calibration_candidate_summary_count": len(calibration_summaries),
        "frozen_necessity_plan_count": len(frozen_plans),
        "registered_heldout_case_count": len(registry),
        "heldout_condition_row_count": len(heldout_rows),
        "heldout_generation_count": len(heldout_rows),
        "downstream_response_row_count": sum(row["downstream_response_row_count"] for row in heldout_quality),
        "all_survey_valid": all(row["valid"] for row in survey_quality),
        "all_calibration_valid": all(row["valid"] for row in calibration_quality),
        "all_heldout_valid": all(row["valid"] for row in heldout_quality),
        "training_formation_track_available_count": 0,
        "selection_updates_allowed": False, "single_unit_intervention_gate_open": False,
    }
    quality["valid"] = bool(
        len(baseline) == 1944 and len(discovery_plans) == 162
        and len(calibration_rows) == 1458 and len(calibration_summaries) == 162
        and len(frozen_plans) == 54 and len(registry) == 486
        and len(heldout_rows) == 5346
        and quality["all_survey_valid"] and quality["all_calibration_valid"]
        and quality["all_heldout_valid"]
    )
    write_json(SOURCE / "phase334_execution_quality.json", quality)
    return quality


def propagation_candidates(
    response_rows: list[dict[str, Any]], condition_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    conditions = {
        (row["case_id"], row["condition"]): row for row in condition_rows
        if row["condition"] in {"baseline", "correct_selected_delete"}
    }
    traces: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        if row["component_type"] == "residual_output":
            traces[(row["case_id"], row["condition"])].append(row)
    result = []
    propagated: dict[str, bool] = {}
    case_ids = sorted({case_id for case_id, _condition in traces})
    for case_id in case_ids:
        baseline_rows = traces.get((case_id, "baseline"), [])
        patched_rows = traces.get((case_id, "correct_selected_delete"), [])
        baseline = {int(row["component_layer"]): row for row in baseline_rows}
        patched = {int(row["component_layer"]): row for row in patched_rows}
        condition = conditions.get((case_id, "correct_selected_delete"))
        if condition is None:
            continue
        selected_layer = int(condition["selected_layer"])
        deltas = {
            layer: float(patched[layer]["target_projection"]) - float(baseline[layer]["target_projection"])
            for layer in sorted(set(baseline) & set(patched))
            if finite(patched[layer].get("target_projection")) and finite(baseline[layer].get("target_projection"))
        }
        downstream = [value for layer, value in deltas.items() if layer > selected_layer]
        negative_count = sum(value < 0 for value in downstream)
        final_delta = deltas[max(deltas)] if deltas else 0.0
        is_propagated = bool(
            len(downstream) >= 2 and negative_count >= 2 and final_delta < 0
            and float(condition["phrase_logprob_loss_vs_baseline"]) > 0
            and float(condition["target_rank_loss_vs_baseline"]) > 0
        )
        propagated[case_id] = is_propagated
        result.append({
            "schema_version": "12.0.0", "phase_id": "Phase334", "created_at": now(),
            "model": condition["model"], "case_id": case_id,
            "semantic_case_id": condition["semantic_case_id"],
            "family_id": condition["family_id"], "mechanism_id": condition["mechanism_id"],
            "cohort": condition["cohort"], "item_index": condition["item_index"],
            "template_id": condition["template_id"], "interface": condition["interface"],
            "selected_component": condition["selected_component"],
            "selected_layer": selected_layer,
            "selected_position_role": condition["selected_position_role"],
            "downstream_layer_count": len(downstream),
            "negative_downstream_layer_count": negative_count,
            "final_target_projection_delta": rounded(final_delta),
            "phrase_logprob_loss": condition["phrase_logprob_loss_vs_baseline"],
            "target_rank_loss": condition["target_rank_loss_vs_baseline"],
            "propagation_candidate": is_propagated,
            "causal_edge": False,
            "evidence_level": "L3_directional_propagation_candidate",
        })
    return result, propagated


def local_summaries(
    rows: list[dict[str, Any]], propagated: dict[str, bool], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["family_id"], row["mechanism_id"], row["interface"])].append(row)
    threshold = protocol["thresholds"]
    result = []
    for (model, family, mechanism, interface), values in sorted(grouped.items()):
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            by_case[row["case_id"]].append(row)
        baseline_eligible_ids = {
            case_id for case_id, case_rows in by_case.items()
            if baseline_eligible(next(row for row in case_rows if row["condition"] == "baseline"))
        }
        common_valid_ids = set()
        for case_id in baseline_eligible_ids:
            case_rows = by_case[case_id]
            if len(case_rows) != 11:
                continue
            if any(not finite(row.get("target_phrase_logprob")) for row in case_rows):
                continue
            if any(
                not row["patch_reached_generation"] or not row["patch_reached_phrase"]
                for row in case_rows if row["condition"] != "baseline"
            ):
                continue
            common_valid_ids.add(case_id)
        by_condition = {
            condition: [
                row for row in values
                if row["condition"] == condition and row["case_id"] in common_valid_ids
            ]
            for condition in {row["condition"] for row in values}
        }
        correct = by_condition["correct_selected_delete"]
        control_phrase_max = max(
            (avg(by_condition[name], "phrase_logprob_loss_vs_baseline") for name in CONTROL_CONDITIONS),
            default=0.0,
        )
        propagation_rate = (
            sum(propagated.get(case_id, False) for case_id in common_valid_ids) / len(common_valid_ids)
            if common_valid_ids else 0.0
        )
        specific = bool(
            len(common_valid_ids) >= threshold["common_valid_case_count_min"]
            and avg(correct, "phrase_logprob_loss_vs_baseline") >= threshold["phrase_logprob_loss_min"]
            and avg(correct, "target_rank_loss_vs_baseline") >= threshold["target_rank_loss_min"]
            and rate(correct, "behavior_lost_vs_baseline") >= threshold["behavior_loss_rate_min"]
            and avg(correct, "phrase_logprob_loss_vs_baseline")
                >= control_phrase_max + threshold["control_superiority_min"]
            and rate(correct, "protocol_lost_vs_baseline") <= threshold["protocol_side_effect_max"]
        )
        propagation_pass = propagation_rate >= threshold["propagation_case_rate_min"]
        sample = values[0]
        result.append({
            "schema_version": "12.0.0", "phase_id": "Phase334", "created_at": now(),
            "model": model, "family_id": family, "mechanism_id": mechanism,
            "cohort": sample["cohort"], "interface": interface,
            "planned_case_count": len(by_case),
            "baseline_eligible_case_count": len(baseline_eligible_ids),
            "common_valid_case_count": len(common_valid_ids),
            "common_valid_case_rate": rounded(len(common_valid_ids) / len(by_case)) if by_case else 0.0,
            "selected_component": sample["selected_component"],
            "selected_layer": sample["selected_layer"],
            "selected_position_role": sample["selected_position_role"],
            "mean_correct_phrase_logprob_loss": rounded(avg(correct, "phrase_logprob_loss_vs_baseline")),
            "mean_correct_target_rank_loss": rounded(avg(correct, "target_rank_loss_vs_baseline")),
            "correct_behavior_loss_rate": rounded(rate(correct, "behavior_lost_vs_baseline")),
            "correct_protocol_loss_rate": rounded(rate(correct, "protocol_lost_vs_baseline")),
            "max_control_phrase_logprob_loss": rounded(control_phrase_max),
            "natural_necessity_specific": specific,
            "propagation_candidate_rate": rounded(propagation_rate),
            "propagation_pass": propagation_pass,
            "local_gate_pass": specific and propagation_pass,
            **{
                f"{name}_phrase_logprob_loss": rounded(avg(by_condition[name], "phrase_logprob_loss_vs_baseline"))
                for name in CONTROL_CONDITIONS
            },
            "attention_delete_phrase_loss": rounded(avg(by_condition["correct_attention_delete"], "phrase_logprob_loss_vs_baseline")),
            "mlp_delete_phrase_loss": rounded(avg(by_condition["correct_mlp_delete"], "phrase_logprob_loss_vs_baseline")),
            "residual_delete_phrase_loss": rounded(avg(by_condition["correct_residual_delete"], "phrase_logprob_loss_vs_baseline")),
            "joint_delete_phrase_loss": rounded(avg(by_condition["correct_joint_delete"], "phrase_logprob_loss_vs_baseline")),
            "evidence_level": "L4_controlled_natural_necessity_candidate" if specific else "L3_natural_necessity_not_confirmed",
            "single_unit_causal": False,
        })
    return result


def cross_model_summaries(local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in local:
        grouped[row["mechanism_id"]].append(row)
    result = []
    for mechanism, values in sorted(grouped.items()):
        local_pass = [row for row in values if row["local_gate_pass"]]
        models = {row["model"] for row in local_pass}
        interfaces_by_model = {
            model: {row["interface"] for row in local_pass if row["model"] == model}
            for model in MODELS
        }
        full = len(values) == 9 and models == set(MODELS) and all(
            interfaces_by_model[model] == set(INTERFACES) for model in MODELS
        )
        sample = values[0]
        result.append({
            "schema_version": "12.0.0", "phase_id": "Phase334", "created_at": now(),
            "family_id": sample["family_id"], "mechanism_id": mechanism,
            "cohort": sample["cohort"], "model_interface_cell_count": len(values),
            "baseline_eligible_cell_count": sum(row["baseline_eligible_case_count"] >= 6 for row in values),
            "natural_necessity_specific_cell_count": sum(row["natural_necessity_specific"] for row in values),
            "propagation_pass_cell_count": sum(row["propagation_pass"] for row in values),
            "local_gate_pass_cell_count": len(local_pass),
            "passing_model_count": len(models),
            "cross_model_natural_necessity_gate": full,
            "small_single_unit_scan_gate_open": full,
            "behavior_mechanism_closed": False,
            "single_unit_causal": False,
            "evidence_level": (
                "L4_cross_model_natural_necessity_candidate" if full
                else "L3_cross_model_natural_necessity_not_confirmed"
            ),
        })
    return result


def report(quality: dict[str, Any], summary: dict[str, Any], cross: list[dict[str, Any]]) -> str:
    return "\n".join([
        "# Phase334 three-family natural receiver-path necessity audit",
        "",
        "## Frozen execution",
        "",
        f"- Natural baselines: {quality['baseline_generation_count']}",
        f"- Natural component contrasts: {quality['natural_contrast_row_count']}",
        f"- Calibration conditions: {quality['calibration_condition_row_count']}",
        f"- Heldout conditions: {quality['heldout_condition_row_count']}",
        f"- Downstream responses: {quality['downstream_response_row_count']}",
        "",
        "## Strict results",
        "",
        f"- Baseline-eligible model/interface cells: {summary['results']['baseline_eligible_cell_count']}/54",
        f"- Local natural-necessity candidates: {summary['results']['local_natural_necessity_candidate_count']}/54",
        f"- Local propagation passes: {summary['results']['local_propagation_pass_count']}/54",
        f"- Cross-model mechanism gates: {summary['results']['cross_model_natural_necessity_gate_count']}/6",
        f"- Behavior mechanism closure: {summary['results']['behavior_mechanism_closed_count']}/72",
        f"- Single-unit causal closure: {summary['results']['single_unit_causal_count']}/72",
        "",
        "## Mechanisms",
        "",
        *[
            f"- {row['family_id']}/{row['mechanism_id']}: "
            f"local {row['local_gate_pass_cell_count']}/9, cross-model={row['cross_model_natural_necessity_gate']}"
            for row in cross
        ],
        "",
        "## Boundary",
        "",
        "Natural deletion evidence is component-level necessity evidence, not single-neuron causality.",
        "The explicit knowledge pair tests relation binding, not the training origin of parametric knowledge.",
        "No training-formation claim is made because no same-run checkpoint series is available.",
    ]) + "\n"


def analyze() -> dict[str, Any]:
    quality = read_json(SOURCE / "phase334_execution_quality.json")
    if not quality["valid"]:
        raise RuntimeError("Incomplete Phase334 execution denominator")
    protocol = read_json(SOURCE / "phase334_registered_protocol.json")
    conditions = read_jsonl(SOURCE / "phase334_heldout_condition_rows.jsonl")
    responses = pq.read_table(SOURCE / "phase334_downstream_response_rows.parquet").to_pylist()
    propagation, propagated = propagation_candidates(responses, conditions)
    local = local_summaries(conditions, propagated, protocol)
    cross = cross_model_summaries(local)
    invalid_metrics = sum(
        not finite(row.get(key))
        for row in conditions
        for key in ("target_margin", "target_phrase_logprob", "target_rank_loss_vs_baseline", "phrase_logprob_loss_vs_baseline")
    )
    results = {
        "baseline_eligible_cell_count": sum(row["baseline_eligible_case_count"] >= 6 for row in local),
        "local_natural_necessity_candidate_count": sum(row["natural_necessity_specific"] for row in local),
        "local_propagation_pass_count": sum(row["propagation_pass"] for row in local),
        "local_full_gate_pass_count": sum(row["local_gate_pass"] for row in local),
        "cross_model_natural_necessity_gate_count": sum(row["cross_model_natural_necessity_gate"] for row in cross),
        "small_single_unit_scan_gate_open_count": sum(row["small_single_unit_scan_gate_open"] for row in cross),
        "invalid_condition_metric_count": invalid_metrics,
        "behavior_mechanism_closed_count": 0,
        "single_unit_causal_count": 0,
    }
    summary = {
        "schema_version": "12.0.0", "phase_id": "Phase334", "created_at": now(),
        "denominator": quality, "results": results,
        "progress_vector": {
            "nine_family_registered_coverage": "9/9",
            "mechanism_census_coverage": "72/72",
            "natural_necessity_deep_audit_attempted": "6/72",
            "cross_model_natural_necessity_candidates": f"{results['cross_model_natural_necessity_gate_count']}/6",
            "behavior_mechanism_closure": "0/72",
            "single_unit_causal_closure": "0/72",
        },
        "language_encoding_mechanism_closed": False,
        "intelligent_theory_experimentally_closed": False,
    }
    claims = [
        {
            "schema_version": "12.0.0", "phase_id": "Phase334", "created_at": now(),
            "family_id": row["family_id"], "mechanism_id": row["mechanism_id"],
            "claim": "receiver-natural-path necessity audit",
            "evidence_level": row["evidence_level"],
            "cross_model_natural_necessity_gate": row["cross_model_natural_necessity_gate"],
            "behavior_mechanism_closed": False, "single_unit_causal": False,
            "evidence_boundary": (
                "Component-level deletion with common-valid heldout controls; no training-origin, "
                "single-neuron, or complete-mechanism claim."
            ),
        }
        for row in cross
    ]
    write_jsonl(SOURCE / "phase334_propagation_candidates.jsonl", propagation)
    write_jsonl(SOURCE / "phase334_local_necessity_summary.jsonl", local)
    write_jsonl(SOURCE / "phase334_cross_model_summary.jsonl", cross)
    write_jsonl(SOURCE / "phase334_claim_registry.jsonl", claims)
    write_json(SOURCE / "phase334_global_summary.json", summary)
    (SOURCE / "phase334_report.md").write_text(report(quality, summary, cross), encoding="utf-8")
    return summary


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    result = collect() if args.collect else analyze()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
