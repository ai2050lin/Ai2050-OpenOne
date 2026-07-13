#!/usr/bin/env python3
"""Build a group-first physical propagation profile from Phase401 pair rows."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase401_local_edge_graph"
DISCOVERY = OUT / "local_edges/discovery"
PRIVATE_OUT = DISCOVERY / "private/phase401_local_edge_stage_group_rows.jsonl"
PUBLIC_OUT = OUT / "phase401_local_edge_stage_profile.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("possession_relation", "role_filling")
STAGES = ("attention", "post_attention", "mlp", "layer_output")
TRUE_CONTROL = "true_relation"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return None if value is None else round(float(value), 9)


def med(values: list[float]) -> float | None:
    return clean(median(values)) if values else None


def numeric_median(rows: list[dict[str, Any]], field: str) -> float | None:
    return med([float(row[field]) for row in rows if row.get(field) is not None])


def ge(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return clean(left - right)


def new_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "informative": 0,
        "pair_pass": 0,
        "recoveries": [],
        "cosines": [],
        "positive_mass": [],
        "negative_mass": [],
        "net_mass": [],
        "absolute_mass": [],
    }


def group_stage_rows(
    model: str, gate: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    accumulators: dict[tuple[str, str, int, str, str], dict[str, Any]] = defaultdict(
        new_accumulator
    )
    pair_count = 0
    path = DISCOVERY / "private" / model / "pair_rows.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pair_count += 1
            row = json.loads(line)
            for stage_name in STAGES:
                stage = row["stages"][stage_name]
                key = (
                    row["surface_private"],
                    row["public_parallel_group_id"],
                    int(row["layer_index"]),
                    row["control_name"],
                    stage_name,
                )
                item = accumulators[key]
                item["count"] += 1
                item["informative"] += bool(stage["informative"])
                item["pair_pass"] += bool(stage["pair_pass"])
                if stage["informative"]:
                    item["recoveries"].append(float(stage["state_recovery"]))
                    item["cosines"].append(float(stage["direction_cosine"]))
                for field in (
                    "positive_mass",
                    "negative_mass",
                    "net_mass",
                    "absolute_mass",
                ):
                    item[field].append(float(stage[field]))

    rows: list[dict[str, Any]] = []
    for (surface, group_id, layer, control, stage_name), item in sorted(
        accumulators.items()
    ):
        count = item["count"]
        informative_rate = item["informative"] / count
        pair_pass_rate = item["pair_pass"] / count
        recovery = med(item["recoveries"])
        physical_group_pass = bool(
            informative_rate >= gate["informative_pair_rate_min"]
            and pair_pass_rate >= gate["direct_attention_pair_pass_rate_min"]
            and ge(recovery, gate["direct_attention_median_recovery_min"])
        )
        absolute_mass = med(item["absolute_mass"])
        net_mass = med([abs(value) for value in item["net_mass"]])
        cancellation_ratio = (
            clean(net_mass / absolute_mass)
            if net_mass is not None and absolute_mass not in (None, 0.0)
            else None
        )
        rows.append(
            {
                "schema_version": "75.11.0",
                "phase_id": "Phase401-LocalEdgeStageGroup",
                "model": model,
                "surface": surface,
                "public_parallel_group_id": group_id,
                "layer_index": layer,
                "control_name": control,
                "stage": stage_name,
                "pair_count": count,
                "informative_pair_rate": clean(informative_rate),
                "pair_pass_rate": clean(pair_pass_rate),
                "median_state_recovery": recovery,
                "median_direction_cosine": med(item["cosines"]),
                "median_positive_mass": med(item["positive_mass"]),
                "median_negative_mass": med(item["negative_mass"]),
                "median_absolute_net_mass": net_mass,
                "median_absolute_mass": absolute_mass,
                "median_cancellation_ratio": cancellation_ratio,
                "physical_group_pass": physical_group_pass,
            }
        )
    return rows, pair_count


def relative_zone(layer: int, layer_count: int) -> str:
    relative = layer / max(layer_count - 1, 1)
    if relative <= 1.0 / 3.0:
        return "early"
    if relative <= 2.0 / 3.0:
        return "middle"
    return "late"


def audit_layer(
    rows: list[dict[str, Any]],
    controls: list[str],
    model_gate: dict[str, Any],
    expected_groups: int,
) -> dict[str, Any]:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_control[row["control_name"]].append(row)
    true_rows = by_control[TRUE_CONTROL]
    true_recovery = numeric_median(true_rows, "median_state_recovery")
    true_pair_pass = numeric_median(true_rows, "pair_pass_rate")
    qualified_groups = sum(row["physical_group_pass"] for row in true_rows)
    control_rows = []
    for control in controls:
        comparison_rows = by_control[control]
        control_recovery = numeric_median(
            comparison_rows, "median_state_recovery"
        )
        control_pair_pass = numeric_median(comparison_rows, "pair_pass_rate")
        recovery_difference = subtract(true_recovery, control_recovery)
        pair_pass_difference = subtract(true_pair_pass, control_pair_pass)
        control_rows.append(
            {
                "control_name": control,
                "group_count": len(comparison_rows),
                "true_minus_control_recovery": recovery_difference,
                "true_minus_control_pair_pass_rate": pair_pass_difference,
                "recovery_separation_pass": ge(
                    recovery_difference,
                    model_gate["true_minus_each_control_median_recovery_min"],
                ),
                "pair_pass_separation_pass": ge(
                    pair_pass_difference,
                    model_gate["true_minus_each_control_pair_pass_rate_min"],
                ),
            }
        )
    denominator_complete = bool(
        len(true_rows) == expected_groups
        and all(len(by_control[control]) == expected_groups for control in controls)
    )
    group_rate = qualified_groups / expected_groups
    all_controls_pass = all(
        item["recovery_separation_pass"] and item["pair_pass_separation_pass"]
        for item in control_rows
    )
    physical_layer_pass = bool(
        denominator_complete
        and group_rate >= model_gate["qualified_group_rate_min"]
        and all_controls_pass
    )
    first = true_rows[0]
    return {
        "model": first["model"],
        "surface": first["surface"],
        "stage": first["stage"],
        "layer_index": first["layer_index"],
        "relative_depth_zone": relative_zone(
            first["layer_index"],
            read_json(DISCOVERY / first["model"] / "complete.json")["layer_count"],
        ),
        "denominator_complete": denominator_complete,
        "qualified_physical_group_count": qualified_groups,
        "qualified_physical_group_rate": clean(group_rate),
        "median_true_state_recovery": true_recovery,
        "median_true_pair_pass_rate": true_pair_pass,
        "median_true_cancellation_ratio": numeric_median(
            true_rows, "median_cancellation_ratio"
        ),
        "controls": control_rows,
        "all_physical_controls_pass": all_controls_pass,
        "physical_layer_pass": physical_layer_pass,
    }


def earliest_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = sorted(
        (row for row in rows if row["physical_layer_pass"]),
        key=lambda row: row["layer_index"],
    )
    if not passing:
        return None
    row = passing[0]
    return {
        "layer_index": row["layer_index"],
        "relative_depth_zone": row["relative_depth_zone"],
        "qualified_physical_group_rate": row["qualified_physical_group_rate"],
        "median_true_state_recovery": row["median_true_state_recovery"],
        "median_true_pair_pass_rate": row["median_true_pair_pass_rate"],
    }


def best_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple[float, int, int]:
        controls = sum(
            item["recovery_separation_pass"] and item["pair_pass_separation_pass"]
            for item in row["controls"]
        )
        return (row["qualified_physical_group_rate"], controls, -row["layer_index"])

    row = max(rows, key=score)
    return {
        "layer_index": row["layer_index"],
        "relative_depth_zone": row["relative_depth_zone"],
        "qualified_physical_group_rate": row["qualified_physical_group_rate"],
        "physical_controls_passed": sum(
            item["recovery_separation_pass"] and item["pair_pass_separation_pass"]
            for item in row["controls"]
        ),
        "control_count": len(row["controls"]),
        "median_true_cancellation_ratio": row["median_true_cancellation_ratio"],
    }


def main() -> None:
    freeze = read_json(OUT / "phase401_local_edge_execution_freeze.json")
    controls = sorted(freeze["controls"])
    group_gate = freeze["group_layer_gate"]
    model_gate = freeze["model_surface_layer_gate"]
    expected_groups = freeze["discovery_denominator"]["groups_per_surface"]

    group_rows: list[dict[str, Any]] = []
    pair_counts: dict[str, int] = {}
    for model in MODELS:
        rows, pair_count = group_stage_rows(model, group_gate)
        group_rows.extend(rows)
        pair_counts[model] = pair_count
    write_jsonl(PRIVATE_OUT, group_rows)

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in group_rows:
        grouped[(row["model"], row["surface"], row["stage"], row["layer_index"])].append(
            row
        )
    layer_rows = [
        audit_layer(rows, controls, model_gate, expected_groups)
        for _, rows in sorted(grouped.items())
    ]

    summary: dict[str, Any] = {}
    for model in MODELS:
        summary[model] = {}
        for surface in SURFACES:
            summary[model][surface] = {}
            for stage in STAGES:
                rows = [
                    row
                    for row in layer_rows
                    if row["model"] == model
                    and row["surface"] == surface
                    and row["stage"] == stage
                ]
                summary[model][surface][stage] = {
                    "layer_count": len(rows),
                    "physical_passing_layer_count": sum(
                        row["physical_layer_pass"] for row in rows
                    ),
                    "earliest_physical_candidate": earliest_candidate(rows),
                    "best_layer_non_authorizing": best_diagnostic(rows),
                    "eligible_for_local_edge_registration": stage == "attention",
                }

    attention_candidates = [
        {
            "model": model,
            "surface": surface,
            **summary[model][surface]["attention"]["earliest_physical_candidate"],
        }
        for model in MODELS
        for surface in SURFACES
        if summary[model][surface]["attention"]["earliest_physical_candidate"]
    ]
    payload = {
        "schema_version": "75.11.0",
        "phase_id": "Phase401-LocalEdgeStageProfile",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "separate_direct_child_physical_edges_from_downstream_propagation_shape",
        "pair_row_counts": pair_counts,
        "total_pair_row_count": sum(pair_counts.values()),
        "stage_count": len(STAGES),
        "group_stage_row_count": len(group_rows),
        "group_first_independent_denominator": True,
        "stage_summary": summary,
        "registered_direct_attention_local_physical_candidates": attention_candidates,
        "registered_direct_attention_local_physical_candidate_count": len(
            attention_candidates
        ),
        "authorization": {
            "promote_downstream_stage_profile_to_direct_edge": False,
            "promote_attention_candidate_to_language_path": False,
            "run_calibration": False,
            "run_physical_holdout": False,
        },
        "claim_boundary": {
            "attention_is_the_only_direct_child_in_this_intervention": True,
            "post_attention_mlp_and_layer_output_are_propagation_diagnostics": True,
            "architecture_response_is_language_specific": False,
            "physical_group_rows_are_independent_samples": True,
            "pair_rows_are_independent_samples": False,
        },
    }
    write_json(PUBLIC_OUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
