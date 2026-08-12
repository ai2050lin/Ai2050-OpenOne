#!/usr/bin/env python3
"""Finalize single source-write causal tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1039_source_channel_causal_protocol as protocol


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
    selected_unselected = row["selected_minus_unselected"]
    selected_wrong = row["selected_minus_wrong_target"]
    retention = row["whole_state_effect_retention"]
    ratio = row["cross_to_same_absolute_ratio"]
    return (
        cross["median"] is not None
        and cross["median"] > gate["cross_selected_shift_median_min"]
        and cross["positive_rate"]
        >= gate["cross_selected_positive_rate_min"]
        and selected_unselected["median"] is not None
        and selected_unselected["median"]
        > gate["selected_minus_unselected_median_min"]
        and selected_wrong["median"] is not None
        and selected_wrong["median"]
        > gate["selected_minus_wrong_target_median_min"]
        and ratio is not None
        and ratio >= gate["cross_to_same_absolute_ratio_min"]
        and retention is not None
        and retention >= gate["whole_state_effect_retention_min"]
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

    gate = prereg["single_channel_gate"]
    model_results: dict[str, Any] = {}
    conserved: dict[tuple[int, str], list[str]] = {}
    for model in protocol.MODELS:
        rows = metrics[model]["paired_channel_rows"]
        by_key = {
            (
                int(row["normalized_depth_slot"]),
                str(row["channel"]),
                str(row["group"]),
            ): row
            for row in rows
        }
        passing_cells = []
        for depth_slot in prereg["normalized_depth_slots"]:
            for channel in protocol.CHANNELS:
                template_rows = [
                    by_key[(int(depth_slot), channel, template)]
                    for template in ("template_2", "template_3")
                ]
                if all(row_passes(row, gate) for row in template_rows):
                    passing_cells.append({
                        "normalized_depth_slot": int(depth_slot),
                        "channel": channel,
                        "template_rows": template_rows,
                    })
                    conserved.setdefault(
                        (int(depth_slot), channel), []
                    ).append(model)
        model_results[model] = {
            "passing_single_channel_cells": passing_cells,
            "passing_cell_count": len(passing_cells),
            "all_group_rows": [
                row for row in rows if row["group"] == "all"
            ],
        }

    cross_model_cells = [
        {
            "normalized_depth_slot": key[0],
            "channel": key[1],
            "models": models,
        }
        for key, models in sorted(conserved.items())
        if len(models)
        >= gate["minimum_models_same_channel_and_depth_slot"]
    ]
    single_channel_gate = bool(cross_model_cells)
    if single_channel_gate:
        route = (
            "Do not combine channels yet. Replicate the passing source-write "
            "channel on new family pairs, multi-token concepts, and natural "
            "knowledge prompts before any mechanism formula."
        )
    else:
        route = (
            "Run Phase1040: preserve all Phase1039 negatives and test the "
            "current-layer Attention+MLP write alliance plus the upstream "
            "residual difference at the same frozen depths and controls. "
            "This tests distributed cooperation without ranking heads or "
            "neurons."
        )

    aggregate = {
        "schema_version": "phase1039_aggregate.v1",
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
            "all_arrays_finite": all(
                all(
                    value["finite_value_rate"] >= 0.99
                    for value in row["array_finiteness"].values()
                )
                for row in summaries.values()
            ),
            "zero_delta_identity_exact": all(
                row["zero_delta_identity_max_abs"] == 0.0
                for row in summaries.values()
            ),
        },
        "model_results": model_results,
        "cross_model_single_channel_cells": cross_model_cells,
        "single_channel_cross_model_gate_passed": single_channel_gate,
        "automatic_next_decision": {
            "immediate_followup_needed": not single_channel_gate,
            "route": route,
            "claim_limit": (
                (
                    "Passing in two models identifies a replication "
                    "candidate, not a universal MLP semantic transporter; "
                    "GLM4 remains a negative model."
                )
                if single_channel_gate
                else (
                    "A failed single-channel gate means neither isolated "
                    "write met the preregistered sufficiency and purity "
                    "criteria. It does not negate repeated write geometry."
                )
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
        "passing_cells_by_model": {
            model: row["passing_cell_count"]
            for model, row in model_results.items()
        },
        "cross_model_single_channel_cell_count": len(
            cross_model_cells
        ),
        "single_channel_cross_model_gate_passed": single_channel_gate,
        "immediate_followup_needed": not single_channel_gate,
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
