#!/usr/bin/env python3
"""Apply frozen Phase402 joint-parent and per-control discovery gates."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase402_multiparent_protocol import (  # noqa: E402
    CONTROL_NAMES,
    MODELS,
    OUT,
    PARENT_CATEGORIES,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase402 non-finite audit value: {value}")
    return round(value, 9)


def depth_zone(layer_index: int, layer_count: int) -> str:
    relative = (layer_index + 0.5) / layer_count
    if relative < 0.25:
        return "early"
    if relative < 0.50:
        return "middle_early"
    if relative < 0.75:
        return "middle_late"
    return "late"


def subset_id_for(categories: list[str]) -> str:
    mask = 0
    for category in categories:
        mask |= 1 << PARENT_CATEGORIES.index(category)
    return f"S{mask:04b}"


def main() -> None:
    freeze = read_json(OUT / "phase402_discovery_execution_freeze.json")
    gates = freeze["frozen_gates"]
    group_gate = gates["group_layer_subset_gate"]
    discovery_gate = gates["discovery_candidate_gate"]
    eligible_surfaces = freeze["discovery_denominator"]["eligible_surfaces"]
    groups_per_surface = freeze["discovery_denominator"]["groups_per_surface"]

    rows: list[dict[str, Any]] = []
    completes: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        complete = read_json(OUT / "multiparent/discovery" / model / "complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase402 discovery collection for {model}")
        completes[model] = complete
        rows.extend(
            read_jsonl(
                OUT
                / "multiparent/discovery/private"
                / model
                / "group_layer_subset_rows.jsonl"
            )
        )

    lookup = {
        (
            row["model"],
            row["public_parallel_group_id"],
            row["layer_index"],
            row["control_name"],
            row["subset_id"],
        ): row
        for row in rows
    }
    if len(lookup) != len(rows):
        raise RuntimeError("Phase402 duplicate group-layer-subset rows")

    candidate_rows: list[dict[str, Any]] = []
    base_true_pass_count = 0
    above_singleton_count = 0
    all_control_pass_count = 0
    control_failure_counts: Counter[str] = Counter()
    joint_denominator = 0
    for row in rows:
        if row["control_name"] != "true_relation" or row["subset_size"] < 2:
            continue
        joint_denominator += 1
        true_base = bool(
            row["control_applicable"]
            and row["informative_pair_rate"] is not None
            and row["informative_pair_rate"]
            >= group_gate["informative_pair_rate_min"]
            and row["pair_pass_rate"] is not None
            and row["pair_pass_rate"] >= group_gate["pair_pass_rate_min"]
            and row["median_state_recovery"] is not None
            and row["median_state_recovery"]
            >= group_gate["median_state_recovery_min"]
        )
        base_true_pass_count += int(true_base)

        singleton_rows = [
            lookup[
                (
                    row["model"],
                    row["public_parallel_group_id"],
                    row["layer_index"],
                    "true_relation",
                    subset_id_for([category]),
                )
            ]
            for category in row["subset_categories"]
        ]
        valid_singletons = [
            item
            for item in singleton_rows
            if item["median_state_recovery"] is not None
            and item["pair_pass_rate"] is not None
        ]
        best_singleton = (
            max(
                valid_singletons,
                key=lambda item: (
                    item["median_state_recovery"], item["pair_pass_rate"]
                ),
            )
            if valid_singletons
            else None
        )
        recovery_gain = (
            row["median_state_recovery"] - best_singleton["median_state_recovery"]
            if row["median_state_recovery"] is not None
            and best_singleton is not None
            else None
        )
        pass_rate_gain = (
            row["pair_pass_rate"] - best_singleton["pair_pass_rate"]
            if row["pair_pass_rate"] is not None and best_singleton is not None
            else None
        )
        above_singleton = bool(
            recovery_gain is not None
            and recovery_gain
            >= group_gate[
                "joint_minus_best_contained_singleton_median_recovery_min"
            ]
            and pass_rate_gain is not None
            and pass_rate_gain
            >= group_gate[
                "joint_minus_best_contained_singleton_pair_pass_rate_min"
            ]
        )
        above_singleton_count += int(above_singleton)

        control_results: dict[str, Any] = {}
        all_controls = True
        for control in CONTROL_NAMES:
            if control == "true_relation":
                continue
            control_row = lookup[
                (
                    row["model"],
                    row["public_parallel_group_id"],
                    row["layer_index"],
                    control,
                    row["subset_id"],
                )
            ]
            if not control_row["control_applicable"]:
                control_results[control] = {"applicable": False, "pass": None}
                continue
            recovery_margin = (
                row["median_state_recovery"]
                - control_row["median_state_recovery"]
                if row["median_state_recovery"] is not None
                and control_row["median_state_recovery"] is not None
                else None
            )
            pair_margin = (
                row["pair_pass_rate"] - control_row["pair_pass_rate"]
                if row["pair_pass_rate"] is not None
                and control_row["pair_pass_rate"] is not None
                else None
            )
            passed = bool(
                recovery_margin is not None
                and recovery_margin
                >= group_gate["true_minus_each_control_median_recovery_min"]
                and pair_margin is not None
                and pair_margin
                >= group_gate["true_minus_each_control_pair_pass_rate_min"]
            )
            if not passed:
                control_failure_counts[control] += 1
                all_controls = False
            control_results[control] = {
                "applicable": True,
                "median_recovery_margin": (
                    clean(recovery_margin) if recovery_margin is not None else None
                ),
                "pair_pass_rate_margin": (
                    clean(pair_margin) if pair_margin is not None else None
                ),
                "pass": passed,
            }
        all_control_pass_count += int(all_controls)
        strict = true_base and above_singleton and all_controls
        candidate_rows.append(
            {
                "schema_version": "76.8.0",
                "phase_id": "Phase402-DiscoveryGroupLayerCandidate",
                "model": row["model"],
                "surface": row["surface_private"],
                "public_parallel_group_id": row["public_parallel_group_id"],
                "layer_index": row["layer_index"],
                "layer_count": row["layer_count"],
                "relative_depth": clean((row["layer_index"] + 0.5) / row["layer_count"]),
                "depth_zone": depth_zone(row["layer_index"], row["layer_count"]),
                "subset_id": row["subset_id"],
                "subset_categories": row["subset_categories"],
                "subset_size": row["subset_size"],
                "true_informative_pair_rate": row["informative_pair_rate"],
                "true_pair_pass_rate": row["pair_pass_rate"],
                "true_median_state_recovery": row["median_state_recovery"],
                "true_base_gate_pass": true_base,
                "best_contained_singleton_id": (
                    best_singleton["subset_id"] if best_singleton else None
                ),
                "joint_minus_best_singleton_median_recovery": (
                    clean(recovery_gain) if recovery_gain is not None else None
                ),
                "joint_minus_best_singleton_pair_pass_rate": (
                    clean(pass_rate_gain) if pass_rate_gain is not None else None
                ),
                "joint_above_best_singleton_gate_pass": above_singleton,
                "control_results": control_results,
                "all_applicable_controls_gate_pass": all_controls,
                "strict_group_layer_candidate": strict,
            }
        )

    by_cell: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    groups_by_model_surface: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in candidate_rows:
        groups_by_model_surface[(row["model"], row["surface"])].add(
            row["public_parallel_group_id"]
        )
        by_cell[
            (row["model"], row["surface"], row["subset_id"], row["depth_zone"])
        ].append(row)

    model_cells: list[dict[str, Any]] = []
    for (model, surface, subset_id, zone), cell_rows in sorted(by_cell.items()):
        groups = groups_by_model_surface[(model, surface)]
        passing_groups = {
            row["public_parallel_group_id"]
            for row in cell_rows
            if row["strict_group_layer_candidate"]
        }
        rate = len(passing_groups) / len(groups)
        representative = max(
            cell_rows,
            key=lambda row: (
                row["strict_group_layer_candidate"],
                row["joint_minus_best_singleton_median_recovery"]
                if row["joint_minus_best_singleton_median_recovery"] is not None
                else -1e9,
                row["true_median_state_recovery"]
                if row["true_median_state_recovery"] is not None
                else -1e9,
            ),
        )
        candidate = rate >= discovery_gate["qualified_discovery_group_rate_min"]
        model_cells.append(
            {
                "schema_version": "76.8.0",
                "phase_id": "Phase402-DiscoveryModelCandidate",
                "model": model,
                "surface": surface,
                "subset_id": subset_id,
                "subset_categories": representative["subset_categories"],
                "depth_zone": zone,
                "discovery_group_count": len(groups),
                "passing_group_count": len(passing_groups),
                "passing_group_rate": clean(rate),
                "required_rate": discovery_gate[
                    "qualified_discovery_group_rate_min"
                ],
                "effective_minimum_passing_group_count": math.ceil(
                    discovery_gate["qualified_discovery_group_rate_min"] * len(groups)
                ),
                "representative_layer_index": representative["layer_index"],
                "model_candidate": candidate,
            }
        )

    model_cell_lookup = {
        (row["model"], row["surface"], row["subset_id"], row["depth_zone"]): row
        for row in model_cells
    }
    crossmodel_rows: list[dict[str, Any]] = []
    identities = sorted(
        {
            (row["surface"], row["subset_id"], row["depth_zone"])
            for row in model_cells
        }
    )
    for surface, subset_id, zone in identities:
        per_model = {
            model: model_cell_lookup[(model, surface, subset_id, zone)]
            for model in MODELS
        }
        passed_models = [
            model for model, row in per_model.items() if row["model_candidate"]
        ]
        crossmodel_rows.append(
            {
                "schema_version": "76.8.0",
                "phase_id": "Phase402-DiscoveryCrossModelCandidate",
                "surface": surface,
                "subset_id": subset_id,
                "subset_categories": per_model[MODELS[0]]["subset_categories"],
                "depth_zone": zone,
                "passed_models": passed_models,
                "model_pass_count": len(passed_models),
                "all_three_models_candidate": len(passed_models) == len(MODELS),
                "partial_two_model_replication": len(passed_models) == 2,
                "per_model": {
                    model: {
                        "passing_group_count": row["passing_group_count"],
                        "passing_group_rate": row["passing_group_rate"],
                        "representative_layer_index": row[
                            "representative_layer_index"
                        ],
                    }
                    for model, row in per_model.items()
                },
            }
        )

    strict_group_layers = sum(
        row["strict_group_layer_candidate"] for row in candidate_rows
    )
    model_candidate_rows = [row for row in model_cells if row["model_candidate"]]
    crossmodel_candidates = [
        row for row in crossmodel_rows if row["all_three_models_candidate"]
    ]
    partial_candidates = [
        row for row in crossmodel_rows if row["partial_two_model_replication"]
    ]
    payload = {
        "schema_version": "76.8.0",
        "phase_id": "Phase402-DiscoveryAudit",
        "created_at": now(),
        "denominator": {
            "models": list(MODELS),
            "surfaces": eligible_surfaces,
            "groups_per_surface_per_model": groups_per_surface,
            "collection_case_count": sum(
                row["case_count"] for row in completes.values()
            ),
            "pair_metric_count": sum(
                row["pair_row_count"] for row in completes.values()
            ),
            "group_layer_subset_metric_count": len(rows),
            "joint_group_layer_subset_count": joint_denominator,
        },
        "gate_flow": {
            "true_base_gate_pass_count": base_true_pass_count,
            "joint_above_best_singleton_count": above_singleton_count,
            "all_controls_pass_count_independent_of_other_gates": all_control_pass_count,
            "strict_group_layer_candidate_count": strict_group_layers,
            "model_candidate_count": len(model_candidate_rows),
            "crossmodel_candidate_count": len(crossmodel_candidates),
            "partial_two_model_candidate_count": len(partial_candidates),
        },
        "control_failure_counts_over_joint_group_layer_subsets": dict(
            control_failure_counts
        ),
        "protocol_precision_audit": {
            "stored_group_rate_threshold": discovery_gate[
                "qualified_discovery_group_rate_min"
            ],
            "groups_per_surface": groups_per_surface,
            "effective_required_count": math.ceil(
                discovery_gate["qualified_discovery_group_rate_min"]
                * groups_per_surface
            ),
            "note": (
                "decimal_0.666666667_requires_5_of_6_not_4_of_6_and_is_honored_"
                "without_posthoc_relaxation"
            ),
        },
        "crossmodel_candidates": crossmodel_candidates,
        "partial_two_model_candidates": partial_candidates,
        "authorization": {
            "freeze_and_run_calibration": bool(crossmodel_candidates),
            "run_physical_holdout": False,
            "run_propagation_terminal": False,
            "run_neuron_scan": False,
        },
        "stopping_decision": (
            "open_frozen_candidate_calibration"
            if crossmodel_candidates
            else "close_current_four_partition_direct_child_hypothesis"
        ),
        "claim_boundary": {
            "strict_group_layer_candidate_is_a_language_path": False,
            "model_candidate_is_crossmodel_replication": False,
            "negative_result_excludes_finer_parent_decompositions": False,
        },
    }
    analysis = OUT / "discovery_analysis"
    write_jsonl(
        analysis / "phase402_group_layer_candidate_rows.jsonl", candidate_rows
    )
    write_jsonl(analysis / "phase402_model_candidate_rows.jsonl", model_cells)
    write_jsonl(
        analysis / "phase402_crossmodel_candidate_rows.jsonl", crossmodel_rows
    )
    write_json(OUT / "phase402_discovery_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
