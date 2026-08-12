#!/usr/bin/env python3
"""Finalize the expanded early-MLP replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1040_expanded_mlp_replication_protocol as protocol


def manifest(root: Path) -> dict[str, Any]:
    excluded = {
        root / "final" / "artifact_manifest.json",
        root / "final" / "audit.json",
    }
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path in excluded:
            continue
        data = path.read_bytes()
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    return {
        "phase": protocol.PHASE,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def row_passes(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    cross = row["cross_selected_shift"]
    return (
        cross["median"] is not None
        and cross["median"] > gate["cross_selected_shift_median_min"]
        and cross["positive_rate"]
        >= gate["cross_selected_positive_rate_min"]
        and row["selected_minus_unselected"]["median"] is not None
        and row["selected_minus_unselected"]["median"]
        > gate["selected_minus_unselected_median_min"]
        and row["selected_minus_wrong_target"]["median"] is not None
        and row["selected_minus_wrong_target"]["median"]
        > gate["selected_minus_wrong_target_median_min"]
        and row["cross_to_same_absolute_ratio"]
        >= gate["cross_to_same_absolute_ratio_min"]
        and row["whole_state_effect_retention"]
        >= gate["whole_state_effect_retention_min"]
    )


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {}
    metrics = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summaries[model] = protocol.read_json(atlas / "summary.json")
        metrics[model] = protocol.read_json(atlas / "metrics.json")

    gate = prereg["replication_gate"]
    required_groups = [
        f"template_{template}/{stratum}"
        for template in (0, 1)
        for stratum in protocol.SURFACE_STRATA
    ]
    model_results = {}
    passing_models = []
    for model in protocol.MODELS:
        rows = {
            row["group"]: row
            for row in metrics[model]["paired_rows"]
            if row["channel"] == "mlp_write"
        }
        group_pass = {
            group: row_passes(rows[group], gate)
            for group in required_groups
        }
        pair_rates = metrics[model][
            "ordered_pair_positive_median_rate"
        ]
        pair_gate = all(
            pair_rates[stratum]
            >= gate["ordered_pair_positive_median_rate_min"]
            for stratum in protocol.SURFACE_STRATA
        )
        passed = all(group_pass.values()) and pair_gate
        if passed:
            passing_models.append(model)
        model_results[model] = {
            "replication_passed": passed,
            "required_group_pass": group_pass,
            "ordered_pair_positive_median_rate": pair_rates,
            "behavior": metrics[model]["behavior"],
            "all_group_mlp_row": rows["all"],
            "single_token_mlp_row": rows["single_token"],
            "two_token_mlp_row": rows["two_token"],
        }

    cross_model_pass = (
        len(passing_models) >= gate["minimum_models"]
    )
    if cross_model_pass:
        route = (
            "Preserve the early-MLP family-routing contribution as a robust "
            "controlled-pattern result. Next move to a separately designed "
            "natural knowledge family task and a source-to-boundary path "
            "audit; do not yet fit a universal formula."
        )
    else:
        route = (
            "Do not promote the early-MLP candidate. Preserve Phase1039 as a "
            "task-specific local result and map multi-position/current-write "
            "alliances before any natural-task transfer."
        )
    aggregate = {
        "schema_version": "phase1040_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": {
            "all_models_present": len(summaries) == len(protocol.MODELS),
            "protocol_digest_consistent": all(
                row["protocol_digest"] == prereg["protocol_digest"]
                for row in summaries.values()
            ),
            "all_fp16_no_quantization": all(
                row["precision"]["has_fp16_parameters"]
                and not row["precision"]["has_bf16_parameters"]
                and not row["precision"]["has_quantized_modules"]
                for row in summaries.values()
            ),
            "array_finite_rate_at_least_0_99": all(
                all(
                    value["finite_value_rate"] >= 0.99
                    for value in row["array_finiteness"].values()
                )
                for row in summaries.values()
            ),
        },
        "model_results": model_results,
        "passing_models": passing_models,
        "cross_model_expanded_replication_passed": cross_model_pass,
        "automatic_next_decision": {
            "immediate_additional_execution_needed": False,
            "route": route,
            "reason": (
                "The preregistered expanded replication question is answered "
                "for all three models. Natural knowledge transfer is a new "
                "research block requiring a different protocol, not an "
                "unfinished audit of this result."
            ),
        },
        "model_summaries": summaries,
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)
    artifact = manifest(protocol.OUT_ROOT)
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "artifact_manifest.json",
        artifact,
    )
    audit = {
        "phase": protocol.PHASE,
        "checks": aggregate["checks"],
        "replication_passed_by_model": {
            model: row["replication_passed"]
            for model, row in model_results.items()
        },
        "passing_models": passing_models,
        "cross_model_expanded_replication_passed": cross_model_pass,
        "manifest_file_count": artifact["file_count"],
        "manifest_total_bytes": artifact["total_bytes"],
    }
    protocol.write_json(protocol.OUT_ROOT / "final" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(
        aggregate["automatic_next_decision"],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
