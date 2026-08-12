#!/usr/bin/env python3
"""Finalize Phase1037 with paired causal controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import phase1035_native_family_routing_protocol as source
import phase1037_family_source_causal_protocol as protocol


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


def scalar_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    clean = values[np.isfinite(values)]
    return {
        "count": int(len(values)),
        "finite_count": int(len(clean)),
        "finite_rate": float(len(clean) / max(1, len(values))),
        "mean": float(np.mean(clean)) if len(clean) else None,
        "median": float(np.median(clean)) if len(clean) else None,
        "positive_rate": (
            float(np.mean(clean > 0)) if len(clean) else None
        ),
    }


def paired_controls(
    model: str,
    targets: list[dict[str, Any]],
    depths: list[int],
) -> list[dict[str, Any]]:
    logits = np.load(
        protocol.OUT_ROOT
        / "atlas"
        / model
        / "patched_candidate_logits.fp32.npy",
        mmap_mode="r",
    )
    source_logits = np.load(
        source.OUT_ROOT
        / "atlas"
        / model
        / "candidate_logits.fp32.npy",
        mmap_mode="r",
    )
    target_family = np.asarray([
        int(row["target_family_index"]) for row in targets
    ], dtype=np.int64)
    cross_family = np.asarray([
        int(row["cross_family_index"]) for row in targets
    ], dtype=np.int64)
    target_cases = np.asarray([
        int(row["target_case_index"]) for row in targets
    ], dtype=np.int64)
    clean = np.asarray(source_logits[target_cases], dtype=np.float32)
    clean_margin = (
        clean[np.arange(len(targets)), cross_family]
        - clean[np.arange(len(targets)), target_family]
    )
    condition_index = {
        condition: index
        for index, condition in enumerate(protocol.CONDITIONS)
    }
    groups = {
        "all": np.arange(len(targets)),
        "template_2": np.asarray([
            index for index, row in enumerate(targets)
            if int(row["template_index"]) == 2
        ], dtype=np.int64),
        "template_3": np.asarray([
            index for index, row in enumerate(targets)
            if int(row["template_index"]) == 3
        ], dtype=np.int64),
        "selected_role_concept_a": np.asarray([
            index for index, row in enumerate(targets)
            if row["selected_role"] == "concept_a"
        ], dtype=np.int64),
        "selected_role_concept_b": np.asarray([
            index for index, row in enumerate(targets)
            if row["selected_role"] == "concept_b"
        ], dtype=np.int64),
        "query_0": np.asarray([
            index for index, row in enumerate(targets)
            if int(row["query"]) == 0
        ], dtype=np.int64),
        "query_1": np.asarray([
            index for index, row in enumerate(targets)
            if int(row["query"]) == 1
        ], dtype=np.int64),
    }
    rows = []
    for depth_slot, physical_depth in enumerate(depths):
        shifts = {}
        for condition, index in condition_index.items():
            current = np.asarray(
                logits[:, depth_slot, index], dtype=np.float32
            )
            margin = (
                current[np.arange(len(targets)), cross_family]
                - current[np.arange(len(targets)), target_family]
            )
            shifts[condition] = margin - clean_margin
        for group, indices in groups.items():
            rows.append({
                "physical_depth": int(physical_depth),
                "depth_slot": depth_slot,
                "group": group,
                "cross_selected_shift": scalar_summary(
                    shifts["cross_family_selected"][indices]
                ),
                "same_family_absolute_shift": scalar_summary(
                    np.abs(
                        shifts["same_family_selected"][indices]
                    )
                ),
                "self_absolute_shift": scalar_summary(
                    np.abs(shifts["self_selected"][indices])
                ),
                "selected_minus_unselected": scalar_summary(
                    (
                        shifts["cross_family_selected"]
                        - shifts["cross_family_unselected"]
                    )[indices]
                ),
                "selected_minus_wrong_target": scalar_summary(
                    (
                        shifts["cross_family_selected"]
                        - shifts["cross_family_wrong_target"]
                    )[indices]
                ),
                "unselected_shift": scalar_summary(
                    shifts["cross_family_unselected"][indices]
                ),
                "wrong_target_shift": scalar_summary(
                    shifts["cross_family_wrong_target"][indices]
                ),
            })
    return rows


def gate_passes(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        row["cross_selected_shift"]["finite_rate"] >= 0.95
        and row["cross_selected_shift"]["median"]
        >= gate["cross_selected_margin_shift_median_min"]
        and row["selected_minus_unselected"]["median"]
        >= gate["selected_minus_unselected_median_min"]
        and row["selected_minus_wrong_target"]["median"]
        >= gate["selected_minus_wrong_target_median_min"]
        and row["self_absolute_shift"]["median"]
        <= gate["self_absolute_margin_shift_median_max"]
        and row["same_family_absolute_shift"]["median"]
        <= gate["same_family_absolute_margin_shift_median_max"]
    )


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    summaries = {}
    metrics = {}
    model_results = {}
    gate = prereg["causal_evidence_gate"]
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summaries[model] = protocol.read_json(atlas / "summary.json")
        metrics[model] = protocol.read_json(atlas / "metrics.json")
        paired = paired_controls(
            model, targets, summaries[model]["patch_depths"]
        )
        by_key = {
            (int(row["depth_slot"]), str(row["group"])): row
            for row in paired
        }
        passing_depths = []
        for depth_slot, physical_depth in enumerate(
            summaries[model]["patch_depths"]
        ):
            template_rows = [
                by_key[(depth_slot, f"template_{template}")]
                for template in (2, 3)
            ]
            if all(gate_passes(row, gate) for row in template_rows):
                passing_depths.append({
                    "depth_slot": depth_slot,
                    "physical_depth": int(physical_depth),
                    "template_rows": template_rows,
                    "all_rows": by_key[(depth_slot, "all")],
                })
        model_results[model] = {
            "causal_gate_passed": (
                len(passing_depths)
                >= gate["minimum_depths_per_model"]
            ),
            "passing_depths": passing_depths,
            "paired_control_rows": paired,
            "stratified_causal_summary": metrics[model]["causal_summary"],
            "array_finiteness": summaries[model]["array_finiteness"],
        }

    successful_models = [
        model
        for model, row in model_results.items()
        if row["causal_gate_passed"]
    ]
    cross_model_success = (
        len(successful_models) >= gate["minimum_models"]
    )
    if cross_model_success:
        route = (
            "Preserve this as a local causal mechanism for the controlled "
            "definition-to-family retrieval pattern. The next large task "
            "should independently replicate it with new semantic families "
            "and then test hierarchical relations; do not expand the claim "
            "to translation, punctuation, or contrast yet."
        )
    else:
        route = (
            "Do not continue whole-vector source patching. Preserve the "
            "category-pair geometry, mark causal transport as unresolved, "
            "and next decompose the source state into distributed channels "
            "or multi-position spans without deleting negative controls."
        )

    aggregate = {
        "schema_version": "phase1037_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_protocol_digest": prereg["source_protocol_digest"],
        "evidence_protocol_digest": prereg[
            "evidence_protocol_digest"
        ],
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
            "all_targets_retained": all(
                row["sample_counts"]["targets"] == len(targets)
                for row in summaries.values()
            ),
        },
        "model_results": model_results,
        "cross_model_causal_result": {
            "passed": cross_model_success,
            "successful_models": successful_models,
            "minimum_models": gate["minimum_models"],
        },
        "automatic_next_decision": {
            "immediate_additional_execution_needed": False,
            "route": route,
            "reason": (
                "The preregistered local causal question is now answered "
                "either positively or negatively across all three models. "
                "A further run would be a new independent research block, "
                "not an unfinished audit of this result."
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
        "causal_gate_by_model": {
            model: row["causal_gate_passed"]
            for model, row in model_results.items()
        },
        "passing_depth_count_by_model": {
            model: len(row["passing_depths"])
            for model, row in model_results.items()
        },
        "cross_model_causal_passed": cross_model_success,
        "successful_models": successful_models,
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
