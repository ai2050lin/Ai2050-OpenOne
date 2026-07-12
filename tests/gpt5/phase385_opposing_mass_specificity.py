#!/usr/bin/env python3
"""Audit whether replicated Phase384 opposing mass is function-specific."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/gpt5/result/phase384_exact_subunit_mass_map"
OUT = ROOT / "tests/gpt5/result/phase385_opposing_mass_specificity"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def cell_key(row: dict[str, Any], model: str) -> tuple[Any, ...]:
    return (
        model,
        row["mechanism_id"],
        row["contrast_axis"],
        row["subunit_family"],
        row["receiver_role"],
        row["source_partition"],
        row["depth_bin"],
    )


def main() -> None:
    calibration = read_json(SOURCE / "phase384_calibration_summary.json")
    replicated = [
        row
        for row in read_jsonl(SOURCE / "phase384_calibration_replication_rows.jsonl")
        if row["calibration_level2_pass"]
        and row["upstream_cell"]
        and row["pattern_type"] == "opposing"
    ]
    if not calibration["results"]["parent_projection_conservation_pass"]:
        raise RuntimeError("Phase384 parent projection conservation failed")
    contract = {
        "schema_version": "59.0.0",
        "phase_id": "Phase385-OpposingMassSpecificity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_control_rule": {
            "functional_control": (
                "maximum mass among other mechanism or contrast axes at the same "
                "family, receiver, source partition, depth and model"
            ),
            "wrong_receiver_control": (
                "maximum mass at other receiver roles with mechanism, axis, family, "
                "source partition, depth and model fixed"
            ),
            "wrong_depth_control": (
                "maximum mass at other depth bins with mechanism, axis, family, "
                "receiver, source partition and model fixed"
            ),
            "specificity_gate": "candidate mass must strictly exceed all three maxima",
            "ratio_threshold_tuned": False,
            "composite_score_used": False,
        },
        "claim_boundary": {
            "strict_dominance_is_causal_specificity": False,
            "failure_means_no_language_encoding_anywhere": False,
            "success_would_only_authorize_relation_tracking": True,
        },
    }
    write_json(OUT / "phase385_specificity_contract.json", contract)

    split_cells = {
        split: read_jsonl(SOURCE / f"phase384_{split}_model_cells.jsonl")
        for split in ("discovery", "calibration")
    }
    audit_rows = []
    candidate_results = []
    for candidate in replicated:
        split_level2 = {}
        split_passing_models = {}
        for split, cells in split_cells.items():
            by_key = {
                (
                    row["model"],
                    row["mechanism_id"],
                    row["contrast_axis"],
                    row["subunit_family"],
                    row["receiver_role"],
                    row["source_partition"],
                    row["depth_bin"],
                ): row
                for row in cells
            }
            passing_models = []
            for model in MODELS:
                cell = by_key.get(cell_key(candidate, model))
                if cell is None:
                    continue
                mass = float(cell["median_absolute_projection_mass"])
                functional_controls = [
                    row["median_absolute_projection_mass"]
                    for row in cells
                    if row["model"] == model
                    and row["subunit_family"] == candidate["subunit_family"]
                    and row["receiver_role"] == candidate["receiver_role"]
                    and row["source_partition"] == candidate["source_partition"]
                    and row["depth_bin"] == candidate["depth_bin"]
                    and (
                        row["mechanism_id"] != candidate["mechanism_id"]
                        or row["contrast_axis"] != candidate["contrast_axis"]
                    )
                ]
                receiver_controls = [
                    row["median_absolute_projection_mass"]
                    for row in cells
                    if row["model"] == model
                    and row["mechanism_id"] == candidate["mechanism_id"]
                    and row["contrast_axis"] == candidate["contrast_axis"]
                    and row["subunit_family"] == candidate["subunit_family"]
                    and row["source_partition"] == candidate["source_partition"]
                    and row["depth_bin"] == candidate["depth_bin"]
                    and row["receiver_role"] != candidate["receiver_role"]
                ]
                depth_controls = [
                    row["median_absolute_projection_mass"]
                    for row in cells
                    if row["model"] == model
                    and row["mechanism_id"] == candidate["mechanism_id"]
                    and row["contrast_axis"] == candidate["contrast_axis"]
                    and row["subunit_family"] == candidate["subunit_family"]
                    and row["receiver_role"] == candidate["receiver_role"]
                    and row["source_partition"] == candidate["source_partition"]
                    and row["depth_bin"] != candidate["depth_bin"]
                ]
                functional_max = max(functional_controls, default=0.0)
                receiver_max = max(receiver_controls, default=0.0)
                depth_max = max(depth_controls, default=0.0)
                functional_pass = mass > functional_max
                receiver_pass = mass > receiver_max
                depth_pass = mass > depth_max
                specificity_pass = functional_pass and receiver_pass and depth_pass
                if specificity_pass:
                    passing_models.append(model)
                audit_rows.append(
                    {
                        "schema_version": "59.0.0",
                        "phase_id": "Phase385-OpposingMassSpecificity",
                        "split": split,
                        "model": model,
                        "mechanism_id": candidate["mechanism_id"],
                        "contrast_axis": candidate["contrast_axis"],
                        "subunit_family": candidate["subunit_family"],
                        "receiver_role": candidate["receiver_role"],
                        "source_partition": candidate["source_partition"],
                        "depth_bin": candidate["depth_bin"],
                        "candidate_absolute_projection_mass": mass,
                        "functional_control_max_mass": functional_max,
                        "wrong_receiver_control_max_mass": receiver_max,
                        "wrong_depth_control_max_mass": depth_max,
                        "functional_control_ratio": mass / max(functional_max, 1e-12),
                        "wrong_receiver_control_ratio": mass / max(receiver_max, 1e-12),
                        "wrong_depth_control_ratio": mass / max(depth_max, 1e-12),
                        "functional_control_pass": functional_pass,
                        "wrong_receiver_control_pass": receiver_pass,
                        "wrong_depth_control_pass": depth_pass,
                        "all_specificity_controls_pass": specificity_pass,
                    }
                )
            level2 = "glm4" in passing_models and bool(
                set(passing_models) & {"qwen3", "deepseek7b"}
            )
            split_level2[split] = level2
            split_passing_models[split] = passing_models
        candidate_results.append(
            {
                "schema_version": "59.0.0",
                "phase_id": "Phase385-OpposingMassSpecificity",
                "mechanism_id": candidate["mechanism_id"],
                "contrast_axis": candidate["contrast_axis"],
                "subunit_family": candidate["subunit_family"],
                "receiver_role": candidate["receiver_role"],
                "source_partition": candidate["source_partition"],
                "depth_bin": candidate["depth_bin"],
                "discovery_specificity_level2_pass": split_level2["discovery"],
                "calibration_specificity_level2_pass": split_level2["calibration"],
                "discovery_specificity_passing_models": split_passing_models[
                    "discovery"
                ],
                "calibration_specificity_passing_models": split_passing_models[
                    "calibration"
                ],
                "replicated_specificity_pass": (
                    split_level2["discovery"] and split_level2["calibration"]
                ),
                "language_path_established": False,
            }
        )
    write_jsonl(OUT / "phase385_specificity_control_rows.jsonl", audit_rows)
    write_jsonl(OUT / "phase385_candidate_specificity_rows.jsonl", candidate_results)
    replicated_specific = [
        row for row in candidate_results if row["replicated_specificity_pass"]
    ]
    summary = {
        "schema_version": "59.0.0",
        "phase_id": "Phase385-OpposingMassSpecificity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "replicated_upstream_opposing_candidate_count": len(replicated),
            "split_count": 2,
            "model_control_row_count": len(audit_rows),
        },
        "results": {
            "replicated_function_specific_candidate_count": len(replicated_specific),
            "candidate_results": candidate_results,
            "failed_control_counts": dict(
                Counter(
                    name
                    for row in audit_rows
                    for name, passed in (
                        ("functional", row["functional_control_pass"]),
                        ("wrong_receiver", row["wrong_receiver_control_pass"]),
                        ("wrong_depth", row["wrong_depth_control_pass"]),
                    )
                    if not passed
                )
            ),
            "opposing_mass_function_specificity_established": bool(
                replicated_specific
            ),
            "language_path_discovered": False,
        },
        "claim_boundary": {
            "replicated_opposing_mass_is_generic_architecture_proven": False,
            "current_resolution_separates_it_from_matched_controls": bool(
                replicated_specific
            ),
            "absence_of_specificity_rejects_all_dynamic_encoding": False,
        },
        "authorization": {
            "physical_holdout": False,
            "causal_intervention": False,
            "new_multi_time_relation_protocol": True,
        },
        "next_decision": (
            "design_new_multi_time_relation_protocol; do not reuse static mass as path"
        ),
    }
    write_json(OUT / "phase385_specificity_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
