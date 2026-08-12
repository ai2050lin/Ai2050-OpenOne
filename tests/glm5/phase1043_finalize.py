#!/usr/bin/env python3
"""Finalize Phase1043 late query-write causal confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1043_late_readout_causal_protocol as protocol


def stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def artifact_manifest(root: Path) -> dict[str, Any]:
    excluded = {
        root / "final" / "artifact_manifest.json",
        root / "final" / "audit.json",
    }
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path in excluded:
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": stream_sha256(path),
        })
    return {
        "phase": protocol.PHASE,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def candidate_pass(
    row: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    groups = row["condition_groups"]
    cross = groups["candidate/cross_matched"]["all"]
    group_medians = {
        group: values["median"]
        for group, values in groups[
            "candidate/cross_matched"
        ].items()
        if group != "all"
    }
    checks = {
        "cross_median_positive": (
            cross["median"] is not None and cross["median"] > 0
        ),
        "cross_positive_rate": (
            cross["positive_rate"]
            >= gate["cross_positive_rate_min"]
        ),
        "all_group_medians_positive": all(
            value is not None and value > 0
            for value in group_medians.values()
        ),
        "matched_to_same_ratio": (
            row["matched_to_same_absolute_ratio"] is not None
            and row["matched_to_same_absolute_ratio"]
            >= gate["matched_to_same_absolute_ratio_min"]
        ),
        "matched_to_shuffled_ratio": (
            row["matched_to_shuffled_absolute_ratio"] is not None
            and row["matched_to_shuffled_absolute_ratio"]
            >= gate["matched_to_shuffled_absolute_ratio_min"]
        ),
        "full_state_retention": (
            row["full_state_retention"] is not None
            and row["full_state_retention"]
            >= gate["full_state_retention_min"]
        ),
        "finite_support": (
            cross["finite_rate"] >= gate["minimum_finite_rate"]
        ),
    }
    return all(checks.values()), {
        "checks": checks,
        "cross_matched": cross,
        "same_lexical": groups["candidate/same_lexical"]["all"],
        "cross_shuffled": groups[
            "candidate/cross_shuffled"
        ]["all"],
        "full_state_cross": groups[
            "full_state/cross_matched"
        ]["all"],
        "group_cross_medians": group_medians,
        "matched_to_same_absolute_ratio": row[
            "matched_to_same_absolute_ratio"
        ],
        "matched_to_shuffled_absolute_ratio": row[
            "matched_to_shuffled_absolute_ratio"
        ],
        "full_state_retention": row["full_state_retention"],
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    summaries = {}
    metrics = {}
    for model in protocol.MODELS:
        atlas = protocol.OUT_ROOT / "atlas" / model
        summaries[model] = protocol.read_json(atlas / "summary.json")
        metrics[model] = protocol.read_json(atlas / "metrics.json")

    gate = prereg["confirmation_gate"]
    model_rows = {}
    candidate_models: dict[int, list[str]] = {}
    for model in protocol.MODELS:
        rows = []
        for row in metrics[model]["candidate_rows"]:
            passed, evidence = candidate_pass(row, gate)
            current = {
                "candidate_index": int(row["candidate_index"]),
                "normalized_depth_slot": int(
                    row["normalized_depth_slot"]
                ),
                "physical_depth": int(
                    row["physical_depths"][model]
                ),
                "channel": row["channel"],
                "site": row["site"],
                "eligible": passed,
                **evidence,
            }
            rows.append(current)
            if passed:
                candidate_models.setdefault(
                    current["candidate_index"], []
                ).append(model)
        model_rows[model] = rows

    repeated = []
    for candidate_index, models in candidate_models.items():
        if len(models) < int(gate["minimum_models"]):
            continue
        source = next(
            row for row in prereg["candidates"]
            if int(row["candidate_index"]) == candidate_index
        )
        repeated.append({
            "candidate_index": candidate_index,
            "normalized_depth_slot": int(
                source["normalized_depth_slot"]
            ),
            "channel": source["channel"],
            "site": source["site"],
            "models": models,
            "model_evidence": {
                model: next(
                    row for row in model_rows[model]
                    if row["candidate_index"] == candidate_index
                )
                for model in models
            },
        })

    if repeated:
        decision = {
            "automatic_followup_needed": True,
            "route": (
                "Preserve the independently confirmed local contributors "
                "and preregister a small receiver/mediation test without "
                "searching new layers."
            ),
            "claim_limit": gate["claim_limit"],
        }
    else:
        decision = {
            "automatic_followup_needed": False,
            "route": (
                "No frozen single-query component is family-specific in "
                "at least two models. Stop this controlled-family "
                "single-position route. Preserve the response atlas and "
                "move future work to naturally recomputed multi-position "
                "and cross-depth state transitions, under a new protocol."
            ),
            "claim_limit": (
                "This negative result excludes only the three frozen "
                "single-position additive write interventions. It does "
                "not exclude distributed, redundant, nonlinear, or "
                "cross-depth language mechanisms."
            ),
        }

    aggregate = {
        "schema_version": "phase1043_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": {
            "all_models_present": set(summaries) == set(protocol.MODELS),
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
                evidence["all_finite"]
                for row in summaries.values()
                for evidence in row["array_finiteness"].values()
            ),
            "instrumentation_closure_exact": all(
                row["instrumentation_closure"]["median"] == 0
                and row["instrumentation_closure"]["finite_rate"] == 1
                for row in summaries.values()
            ),
            "zero_delta_identity_exact": all(
                row["zero_delta_identity"]["exact"]
                and row["zero_delta_identity"][
                    "max_absolute_logit_difference"
                ] == 0
                for row in summaries.values()
            ),
            "protocol_audit_passed": all(
                protocol_audit["checks"].values()
            ),
        },
        "model_candidate_rows": model_rows,
        "cross_model_confirmed_candidates": repeated,
        "automatic_next_decision": decision,
        "model_summaries": summaries,
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)

    manifest = artifact_manifest(protocol.OUT_ROOT)
    audit = {
        "phase": protocol.PHASE,
        "checks": aggregate["checks"],
        "model_pass_counts": {
            model: sum(row["eligible"] for row in rows)
            for model, rows in model_rows.items()
        },
        "cross_model_confirmed_candidate_count": len(repeated),
        "automatic_followup_needed": decision[
            "automatic_followup_needed"
        ],
        "manifest_file_count": manifest["file_count"],
        "manifest_total_bytes": manifest["total_bytes"],
    }
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "artifact_manifest.json",
        manifest,
    )
    protocol.write_json(
        protocol.OUT_ROOT / "final" / "audit.json",
        audit,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
