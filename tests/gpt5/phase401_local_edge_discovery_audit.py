#!/usr/bin/env python3
"""Audit Phase401 discovery local edges against the frozen gates.

The primary verdict is deliberately strict: every field in the execution
freeze must be defined and every one of the eight controls must pass.  A
separate, non-authorizing sensitivity view marks the same-target control's
semantic comparison as not applicable because that control has no distinct
donor target by construction.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase401_local_edge_graph"
DISCOVERY = OUT / "local_edges/discovery"
PRIVATE_OUT = DISCOVERY / "private/phase401_local_edge_layer_audit_rows.jsonl"
PUBLIC_OUT = OUT / "phase401_local_edge_discovery_audit.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling")
TRUE_CONTROL = "true_relation"
SAME_TARGET_CONTROL = "wrong_source_order_matched_same_target"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 9)


def numeric_median(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return clean(median(values)) if values else None


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return clean(left - right)


def ge(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def depth_zone(layer_index: int, layer_count: int) -> tuple[float, str]:
    relative = layer_index / max(layer_count - 1, 1)
    if relative <= 1.0 / 3.0:
        zone = "early"
    elif relative <= 2.0 / 3.0:
        zone = "middle"
    else:
        zone = "late"
    return clean(relative) or 0.0, zone


def control_audit(
    control: str,
    true_metrics: dict[str, float | None],
    control_rows: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    control_metrics = {
        "median_attention_state_recovery": numeric_median(
            control_rows, "median_attention_state_recovery"
        ),
        "median_pair_pass_rate": numeric_median(control_rows, "pair_pass_rate"),
        "median_semantic_competition_recovery": numeric_median(
            control_rows, "median_semantic_competition_recovery"
        ),
    }
    differences = {
        "attention_state_recovery": delta(
            true_metrics["median_attention_state_recovery"],
            control_metrics["median_attention_state_recovery"],
        ),
        "pair_pass_rate": delta(
            true_metrics["median_pair_pass_rate"],
            control_metrics["median_pair_pass_rate"],
        ),
        "semantic_competition_recovery": delta(
            true_metrics["median_semantic_competition_recovery"],
            control_metrics["median_semantic_competition_recovery"],
        ),
    }
    physical_recovery_pass = ge(
        differences["attention_state_recovery"],
        gates["true_minus_each_control_median_recovery_min"],
    )
    pair_pass = ge(
        differences["pair_pass_rate"],
        gates["true_minus_each_control_pair_pass_rate_min"],
    )
    semantic_defined = (
        differences["semantic_competition_recovery"] is not None
    )
    semantic_pass = ge(
        differences["semantic_competition_recovery"],
        gates["true_minus_each_control_semantic_recovery_min"],
    )
    semantic_not_applicable_in_sensitivity = (
        control == SAME_TARGET_CONTROL and not semantic_defined
    )
    strict_pass = physical_recovery_pass and pair_pass and semantic_pass
    sensitivity_pass = (
        physical_recovery_pass
        and pair_pass
        and (semantic_pass or semantic_not_applicable_in_sensitivity)
    )
    return {
        "control_name": control,
        "group_count": len(control_rows),
        "metrics": control_metrics,
        "true_minus_control": differences,
        "physical_recovery_pass": physical_recovery_pass,
        "pair_pass_rate_pass": pair_pass,
        "semantic_comparison_defined": semantic_defined,
        "semantic_recovery_pass": semantic_pass,
        "semantic_not_applicable_in_sensitivity": (
            semantic_not_applicable_in_sensitivity
        ),
        "strict_pass": strict_pass,
        "sensitivity_pass": sensitivity_pass,
    }


def layer_audit(
    model: str,
    surface: str,
    layer_index: int,
    rows: list[dict[str, Any]],
    controls: list[str],
    gates: dict[str, Any],
    group_gates: dict[str, Any],
    expected_groups: int,
) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_control[row["control_name"]].append(row)
    true_rows = by_control[TRUE_CONTROL]
    layer_count = int(true_rows[0]["layer_count"]) if true_rows else 0
    relative_depth, zone = depth_zone(layer_index, layer_count)
    true_metrics = {
        "median_attention_state_recovery": numeric_median(
            true_rows, "median_attention_state_recovery"
        ),
        "median_pair_pass_rate": numeric_median(true_rows, "pair_pass_rate"),
        "median_semantic_competition_recovery": numeric_median(
            true_rows, "median_semantic_competition_recovery"
        ),
        "median_semantic_informative_pair_rate": numeric_median(
            true_rows, "semantic_informative_pair_rate"
        ),
        "median_semantic_positive_shift_rate": numeric_median(
            true_rows, "semantic_positive_shift_rate"
        ),
    }
    qualified_count = sum(
        bool(row["true_group_layer_gate_pass"]) for row in true_rows
    )
    qualified_rate = qualified_count / expected_groups
    component_specs = {
        "attention_informative": (
            "informative_pair_rate",
            group_gates["informative_pair_rate_min"],
        ),
        "attention_pair_pass": (
            "pair_pass_rate",
            group_gates["direct_attention_pair_pass_rate_min"],
        ),
        "attention_median_recovery": (
            "median_attention_state_recovery",
            group_gates["direct_attention_median_recovery_min"],
        ),
        "semantic_informative": (
            "semantic_informative_pair_rate",
            group_gates["semantic_informative_pair_rate_min"],
        ),
        "semantic_positive_shift": (
            "semantic_positive_shift_rate",
            group_gates["semantic_positive_shift_rate_min"],
        ),
        "semantic_median_recovery": (
            "median_semantic_competition_recovery",
            group_gates["semantic_median_recovery_min"],
        ),
    }
    true_group_component_pass_rates = {}
    for name, (field, threshold) in component_specs.items():
        pass_count = sum(ge(row.get(field), threshold) for row in true_rows)
        true_group_component_pass_rates[name] = {
            "threshold": threshold,
            "pass_count": pass_count,
            "pass_rate": clean(pass_count / expected_groups),
        }
    denominator_complete = bool(
        len(true_rows) == expected_groups
        and all(len(by_control[name]) == expected_groups for name in controls)
    )
    control_rows = [
        control_audit(name, true_metrics, by_control[name], gates)
        for name in controls
    ]
    qualified_group_rate_pass = bool(
        qualified_rate >= gates["qualified_group_rate_min"]
    )
    all_controls_strict_pass = all(row["strict_pass"] for row in control_rows)
    all_controls_sensitivity_pass = all(
        row["sensitivity_pass"] for row in control_rows
    )
    strict_layer_pass = bool(
        denominator_complete
        and qualified_group_rate_pass
        and all_controls_strict_pass
    )
    sensitivity_layer_pass = bool(
        denominator_complete
        and qualified_group_rate_pass
        and all_controls_sensitivity_pass
    )
    failed_gates: list[str] = []
    if not denominator_complete:
        failed_gates.append("denominator_complete")
    if not qualified_group_rate_pass:
        failed_gates.append("qualified_group_rate")
    if any(not row["physical_recovery_pass"] for row in control_rows):
        failed_gates.append("control_attention_recovery")
    if any(not row["pair_pass_rate_pass"] for row in control_rows):
        failed_gates.append("control_pair_pass_rate")
    if any(not row["semantic_comparison_defined"] for row in control_rows):
        failed_gates.append("control_semantic_defined")
    if any(
        row["semantic_comparison_defined"] and not row["semantic_recovery_pass"]
        for row in control_rows
    ):
        failed_gates.append("control_semantic_recovery")
    return {
        "schema_version": "75.10.0",
        "phase_id": "Phase401-LocalEdgeLayerAudit",
        "model": model,
        "surface": surface,
        "edge": "source_KV_to_query_attention",
        "layer_index": layer_index,
        "layer_count": layer_count,
        "relative_depth": relative_depth,
        "relative_depth_zone": zone,
        "expected_group_count": expected_groups,
        "true_group_count": len(true_rows),
        "denominator_complete": denominator_complete,
        "qualified_true_group_count": qualified_count,
        "qualified_true_group_rate": clean(qualified_rate),
        "qualified_group_rate_pass": qualified_group_rate_pass,
        "true_metrics": true_metrics,
        "true_group_component_pass_rates": true_group_component_pass_rates,
        "controls": control_rows,
        "all_controls_strict_pass": all_controls_strict_pass,
        "all_controls_sensitivity_pass": all_controls_sensitivity_pass,
        "strict_layer_pass": strict_layer_pass,
        "sensitivity_layer_pass": sensitivity_layer_pass,
        "strict_failed_gates": failed_gates,
    }


def candidate_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    passing = sorted(
        (row for row in rows if row[field]), key=lambda row: row["layer_index"]
    )
    if not passing:
        return None
    row = passing[0]
    return {
        "layer_index": row["layer_index"],
        "layer_count": row["layer_count"],
        "relative_depth": row["relative_depth"],
        "relative_depth_zone": row["relative_depth_zone"],
        "qualified_true_group_rate": row["qualified_true_group_rate"],
    }


def diagnostic_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[float, int, int]:
        controls_passed = sum(item["sensitivity_pass"] for item in row["controls"])
        return (
            float(row["qualified_true_group_rate"]),
            controls_passed,
            -int(row["layer_index"]),
        )

    row = max(rows, key=score)
    return {
        "layer_index": row["layer_index"],
        "layer_count": row["layer_count"],
        "relative_depth_zone": row["relative_depth_zone"],
        "qualified_true_group_rate": row["qualified_true_group_rate"],
        "sensitivity_controls_passed": sum(
            item["sensitivity_pass"] for item in row["controls"]
        ),
        "control_count": len(row["controls"]),
        "failed_gates": row["strict_failed_gates"],
    }


def component_maxima(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = tuple(rows[0]["true_group_component_pass_rates"])
    result: dict[str, Any] = {}
    for name in names:
        best = max(
            rows,
            key=lambda row: (
                row["true_group_component_pass_rates"][name]["pass_rate"],
                -row["layer_index"],
            ),
        )
        component = best["true_group_component_pass_rates"][name]
        result[name] = {
            "maximum_pass_count": component["pass_count"],
            "maximum_pass_rate": component["pass_rate"],
            "threshold": component["threshold"],
            "earliest_layer_at_maximum": best["layer_index"],
        }
    return result


def crossmodel_candidates(
    summaries: dict[str, dict[str, dict[str, Any]]], candidate_key: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for surface in SURFACES:
        candidates = [summaries[model][surface][candidate_key] for model in MODELS]
        if any(candidate is None for candidate in candidates):
            continue
        zones = {candidate["relative_depth_zone"] for candidate in candidates}
        if len(zones) != 1:
            continue
        results.append(
            {
                "surface": surface,
                "edge": "source_KV_to_query_attention",
                "relative_depth_zone": next(iter(zones)),
                "models": {
                    model: candidate for model, candidate in zip(MODELS, candidates)
                },
            }
        )
    return results


def main() -> None:
    freeze = read_json(OUT / "phase401_local_edge_execution_freeze.json")
    controls = sorted(freeze["controls"])
    gates = freeze["model_surface_layer_gate"]
    group_gates = freeze["group_layer_gate"]
    expected_groups = int(freeze["discovery_denominator"]["groups_per_surface"])

    all_rows: list[dict[str, Any]] = []
    source_completeness: dict[str, Any] = {}
    for model in MODELS:
        complete = read_json(DISCOVERY / model / "complete.json")
        rows = read_jsonl(
            DISCOVERY / "private" / model / "group_layer_control_rows.jsonl"
        )
        source_completeness[model] = {
            "collection_valid": bool(complete["valid"]),
            "group_count": complete["group_count"],
            "case_count": complete["case_count"],
            "layer_count": complete["layer_count"],
            "aggregate_row_count": len(rows),
            "expected_aggregate_row_count": (
                len(SURFACES)
                * expected_groups
                * int(complete["layer_count"])
                * (len(controls) + 1)
            ),
        }
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["surface_private"], int(row["layer_index"]))].append(row)
        for surface in SURFACES:
            for layer_index in range(int(complete["layer_count"])):
                all_rows.append(
                    layer_audit(
                        model,
                        surface,
                        layer_index,
                        grouped[(surface, layer_index)],
                        controls,
                        gates,
                        group_gates,
                        expected_groups,
                    )
                )

    write_jsonl(PRIVATE_OUT, all_rows)

    by_model_surface: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_model_surface[(row["model"], row["surface"])].append(row)

    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        summaries[model] = {}
        for surface in SURFACES:
            rows = by_model_surface[(model, surface)]
            failures = Counter(
                failure for row in rows for failure in row["strict_failed_gates"]
            )
            undefined_by_control = {
                control: sum(
                    not next(
                        item
                        for item in row["controls"]
                        if item["control_name"] == control
                    )["semantic_comparison_defined"]
                    for row in rows
                )
                for control in controls
            }
            summaries[model][surface] = {
                "layer_count": len(rows),
                "strict_passing_layer_count": sum(
                    row["strict_layer_pass"] for row in rows
                ),
                "sensitivity_passing_layer_count": sum(
                    row["sensitivity_layer_pass"] for row in rows
                ),
                "strict_candidate": candidate_summary(rows, "strict_layer_pass"),
                "sensitivity_candidate_non_authorizing": candidate_summary(
                    rows, "sensitivity_layer_pass"
                ),
                "diagnostic_best_layer_non_authorizing": diagnostic_best(rows),
                "true_group_component_maxima_non_authorizing": component_maxima(
                    rows
                ),
                "strict_failure_layer_counts": dict(sorted(failures.items())),
                "undefined_semantic_layer_counts_by_control": undefined_by_control,
            }

    strict_crossmodel = crossmodel_candidates(summaries, "strict_candidate")
    sensitivity_crossmodel = crossmodel_candidates(
        summaries, "sensitivity_candidate_non_authorizing"
    )
    all_source_complete = all(
        item["collection_valid"]
        and item["aggregate_row_count"] == item["expected_aggregate_row_count"]
        for item in source_completeness.values()
    )
    payload = {
        "schema_version": "75.10.0",
        "phase_id": "Phase401-LocalEdgeDiscoveryAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "test_direct_source_KV_to_query_attention_edges_with_eight_controls"
        ),
        "source_completeness": source_completeness,
        "all_source_denominators_complete": all_source_complete,
        "frozen_gate_source": str(
            OUT / "phase401_local_edge_execution_freeze.json"
        ),
        "protocol_contradiction": {
            "present": True,
            "control": SAME_TARGET_CONTROL,
            "reason": (
                "control_requires_same_target_but_frozen_semantic_gate_requires_"
                "recipient_and_donor_target_ids_to_differ"
            ),
            "primary_treatment": (
                "undefined_semantic_comparison_fails_strict_all_fields_gate"
            ),
            "sensitivity_treatment": (
                "semantic_comparison_marked_not_applicable_for_this_control_only"
            ),
            "sensitivity_has_authorization_power": False,
        },
        "model_surface_summary": summaries,
        "strict_crossmodel_candidates": strict_crossmodel,
        "sensitivity_crossmodel_candidates_non_authorizing": sensitivity_crossmodel,
        "authorization": {
            "run_calibration": bool(strict_crossmodel) and all_source_complete,
            "run_physical_holdout": False,
            "run_head_channel_neuron_scan": False,
            "reason": (
                "strict_crossmodel_candidate_required_before_calibration;_"
                "physical_holdout_requires_independent_calibration_pass"
            ),
        },
        "claim_boundary": {
            "direct_local_response_is_language_mechanism": False,
            "logit_lens_competition_is_generated_behavior": False,
            "sensitivity_candidate_is_confirmed_candidate": False,
            "architecture_edge_and_language_specific_edge_are_separate": True,
        },
    }
    write_json(PUBLIC_OUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
