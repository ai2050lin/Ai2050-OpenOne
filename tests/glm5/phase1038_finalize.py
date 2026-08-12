#!/usr/bin/env python3
"""Finalize the source computation-channel atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1038_source_channel_protocol as protocol


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
    matched = row["same_pair_cross_split"]
    advantage = row["matched_minus_shuffled"]
    return (
        matched["median"] is not None
        and matched["median"]
        >= gate["same_pair_cross_split_median_min"]
        and matched["positive_rate"]
        >= gate["same_pair_positive_rate_min"]
        and advantage["median"] is not None
        and advantage["median"]
        >= gate["matched_minus_shuffled_median_min"]
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

    gate = prereg["descriptive_channel_gate"]
    model_results: dict[str, Any] = {}
    conserved: dict[tuple[int, str, str], list[str]] = {}
    for model in protocol.MODELS:
        rows = metrics[model]["channel_depth_rows"]
        passing = [row for row in rows if row_passes(row, gate)]
        for row in passing:
            key = (
                int(row["normalized_depth_slot"]),
                str(row["channel"]),
                str(row["role"]),
            )
            conserved.setdefault(key, []).append(model)
        model_results[model] = {
            "passing_row_count": len(passing),
            "passing_rows": passing,
            "passing_rows_by_channel": {
                channel: sum(
                    row["channel"] == channel for row in passing
                )
                for channel in protocol.CHANNELS
            },
            "instrumentation_closure": metrics[model][
                "instrumentation_closure"
            ],
        }

    conserved_rows = [
        {
            "normalized_depth_slot": key[0],
            "channel": key[1],
            "role": key[2],
            "models": models,
        }
        for key, models in sorted(conserved.items())
        if len(models) >= gate["minimum_models"]
    ]
    eligible_channels = set(gate["eligible_causal_channels"])
    causal_cells = [
        row
        for row in conserved_rows
        if row["channel"] in eligible_channels
    ]
    causal_channels = sorted({
        str(row["channel"]) for row in causal_cells
    })
    causal_followup = bool(causal_cells)

    if causal_followup:
        route = (
            "Run Phase1039 as a preregistered additive source-channel "
            "intervention. Test every conserved actual write channel at the "
            "three Phase1037 frozen depths, with same-family, cross-family, "
            "unselected, wrong-target, and full-output baselines. Preserve "
            "all negative rows and use scale-free paired specificity ratios "
            "alongside raw margins."
        )
    else:
        route = (
            "Do not infer a component transporter. Run a new multi-position "
            "source-span atlas covering nonce, concept, and joint spans while "
            "keeping the same family-pair and shuffled controls."
        )

    aggregate = {
        "schema_version": "phase1038_aggregate.v1",
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
            "uniform_depth_selection_only": True,
            "instrumentation_finite": all(
                row["array_finiteness"]["channel_closure"][
                    "finite_value_rate"
                ] >= 0.99
                for row in summaries.values()
            ),
        },
        "model_results": model_results,
        "conserved_channel_cells": conserved_rows,
        "causal_candidate_cells": causal_cells,
        "automatic_next_decision": {
            "causal_followup_needed": causal_followup,
            "eligible_channels": causal_channels,
            "route": route,
            "claim_limit": (
                "Eligibility means a repeated actual-write contrast deserves "
                "causal testing. It does not mean that the channel is a pure "
                "semantic transporter."
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
        "passing_rows_by_model_and_channel": {
            model: row["passing_rows_by_channel"]
            for model, row in model_results.items()
        },
        "conserved_channel_cell_count": len(conserved_rows),
        "causal_candidate_cell_count": len(causal_cells),
        "causal_candidate_channels": causal_channels,
        "causal_followup_needed": causal_followup,
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
