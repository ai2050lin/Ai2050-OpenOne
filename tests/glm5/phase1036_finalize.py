#!/usr/bin/env python3
"""Finalize family-pair specificity and decide on a causal follow-up."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1035_native_family_routing_protocol as source
import phase1036_family_contrast_protocol as protocol


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
        and matched["median"] >= gate["same_pair_cross_split_median_min"]
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
    source_aggregate = source.read_json(source.OUT_ROOT / "aggregate.json")
    source_metrics = {
        model: source.read_json(
            source.OUT_ROOT / "atlas" / model / "metrics.json"
        )
        for model in protocol.MODELS
    }
    summaries = {}
    metrics = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summaries[model] = protocol.read_json(atlas / "summary.json")
        metrics[model] = protocol.read_json(atlas / "metrics.json")

    evidence_gate = prereg["descriptive_evidence_gate"]
    causal_gate = prereg["causal_followup_gate"]
    model_results = {}
    shared_depth_cells: dict[tuple[int, str], list[str]] = {}
    for model in protocol.MODELS:
        rows = metrics[model]["contrast_depth_rows"]
        nonfinal = rows[:-2]
        passing = [row for row in nonfinal if row_passes(row, evidence_gate)]
        counts = {
            role: sum(row["role"] == role for row in passing)
            for role in protocol.ROLE_ANCHORS
        }
        family_gate = all(
            count >= evidence_gate["required_nonfinal_depths_per_role"]
            for count in counts.values()
        )
        depth_lookup = {
            depth: index
            for index, depth in enumerate(summaries[model]["selected_depths"])
        }
        for row in passing:
            depth_slot = depth_lookup[int(row["physical_depth"])]
            shared_depth_cells.setdefault(
                (depth_slot, str(row["role"])), []
            ).append(model)

        output = metrics[model]["output_candidate_factor_metrics"][
            "confirmation"
        ]
        output_invariance = output["bq_member_invariance"]
        behavior = source_metrics[model]["behavior"]["confirmation"]
        output_gate = (
            output_invariance["median"] is not None
            and output_invariance["median"]
            >= causal_gate[
                "confirmation_output_bq_member_invariance_median_min"
            ]
            and output_invariance["positive_rate"]
            >= causal_gate[
                "confirmation_output_bq_member_positive_rate_min"
            ]
        )
        behavior_gate = (
            behavior["candidate_set_accuracy"] is not None
            and behavior["candidate_set_accuracy"]
            >= causal_gate["confirmation_candidate_accuracy_min"]
            and behavior["candidate_logit_finite_row_rate"] >= 0.95
        )
        eligible = family_gate and output_gate and behavior_gate
        model_results[model] = {
            "family_contrast_gate_passed": family_gate,
            "passing_nonfinal_depths_by_role": counts,
            "passing_depth_rows": passing,
            "confirmation_output_factor_metrics": output,
            "output_bq_member_gate_passed": output_gate,
            "confirmation_behavior": behavior,
            "behavior_gate_passed": behavior_gate,
            "causal_followup_eligible": eligible,
            "heldout_internal_prototype_readout": source_aggregate[
                "model_events"
            ][model]["heldout_internal_prototype_readout"],
        }

    conserved_depth_cells = [
        {
            "normalized_depth_slot": key[0],
            "role": key[1],
            "models": models,
        }
        for key, models in sorted(shared_depth_cells.items())
        if len(models) >= evidence_gate["minimum_models"]
    ]
    eligible_models = [
        model
        for model, row in model_results.items()
        if row["causal_followup_eligible"]
    ]
    causal_followup = (
        len(eligible_models) >= causal_gate["minimum_models"]
        and bool(conserved_depth_cells)
    )
    if causal_followup:
        route = (
            "Run Phase1037 as a preregistered full-span source intervention. "
            "On confirmation units, compare same-family lexical-member donors "
            "with cross-family donors at the queried source, the unqueried "
            "source, self, and wrong-role controls. Report all rows and "
            "correct/error strata separately."
        )
    else:
        route = (
            "Do not patch. Keep the matched/shuffled contrast map as "
            "descriptive geometry and redesign the task or readout where "
            "either category-pair specificity, output BxQ invariance, or "
            "behavior did not repeat in two models."
        )

    aggregate = {
        "schema_version": "phase1036_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_protocol_digest": prereg["source_protocol_digest"],
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
        },
        "model_results": model_results,
        "conserved_family_direction_depth_cells": conserved_depth_cells,
        "automatic_next_decision": {
            "causal_followup_needed": causal_followup,
            "eligible_models": eligible_models,
            "route": route,
            "claim_limit": (
                "Passing this gate means a causal comparison is worth doing; "
                "it does not prove the family direction transports knowledge."
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
        "family_contrast_gate_by_model": {
            model: row["family_contrast_gate_passed"]
            for model, row in model_results.items()
        },
        "output_bq_member_gate_by_model": {
            model: row["output_bq_member_gate_passed"]
            for model, row in model_results.items()
        },
        "behavior_gate_by_model": {
            model: row["behavior_gate_passed"]
            for model, row in model_results.items()
        },
        "conserved_family_direction_cell_count": len(
            conserved_depth_cells
        ),
        "causal_followup_needed": causal_followup,
        "eligible_models": eligible_models,
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
