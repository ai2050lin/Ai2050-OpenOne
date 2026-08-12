#!/usr/bin/env python3
"""Finalize Phase1042 and freeze repeated late readout-write cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1042_role_depth_write_atlas_protocol as protocol


MIN_VALID_RATE = 0.80
LATE_SITES = {"query_nonce", "pre_output"}


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


def candidate_evidence(
    row: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    group = row["groups"]["all"]
    same = group["same_pair_cosine"]
    advantage = group["matched_minus_shuffled"]
    ratio = group["family_to_lexical_norm_ratio"]
    checks = {
        "actual_write_channel": (
            row["channel"] in gate["actual_write_channels"]
        ),
        "same_pair_cosine": (
            same["median"] is not None
            and same["median"] >= gate["same_pair_cosine_median_min"]
        ),
        "matched_minus_shuffled": (
            advantage["median"] is not None
            and advantage["median"]
            >= gate["matched_minus_shuffled_median_min"]
        ),
        "advantage_positive_rate": (
            advantage["positive_rate"]
            >= gate["advantage_positive_rate_min"]
        ),
        "family_to_lexical_ratio": (
            ratio["median"] is not None
            and ratio["median"]
            >= gate["family_to_lexical_norm_ratio_min"]
        ),
        "sufficient_finite_support": (
            min(
                same["finite_rate"],
                advantage["finite_rate"],
                ratio["finite_rate"],
            )
            >= MIN_VALID_RATE
        ),
    }
    return all(checks.values()), {
        "checks": checks,
        "same_pair_cosine": same,
        "shuffled_pair_cosine": group["shuffled_pair_cosine"],
        "matched_minus_shuffled": advantage,
        "family_contrast_norm": group["family_contrast_norm"],
        "same_family_lexical_norm": group[
            "same_family_lexical_norm"
        ],
        "family_to_lexical_norm_ratio": ratio,
    }


def cell_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["normalized_depth_slot"]),
        str(row["channel"]),
        str(row["site"]),
    )


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

    gate = prereg["descriptive_gate"]
    model_rows: dict[str, list[dict[str, Any]]] = {}
    candidate_models: dict[
        tuple[int, str, str], list[str]
    ] = {}
    for model in protocol.MODELS:
        current = []
        for row in metrics[model]["role_depth_rows"]:
            passed, evidence = candidate_evidence(row, gate)
            item = {
                "normalized_depth_slot": int(
                    row["normalized_depth_slot"]
                ),
                "physical_depth": int(row["physical_depth"]),
                "channel": row["channel"],
                "site": row["site"],
                "eligible": passed,
                **evidence,
            }
            current.append(item)
            if passed:
                candidate_models.setdefault(cell_key(row), []).append(
                    model
                )
        model_rows[model] = current

    repeated = []
    for key, models in candidate_models.items():
        if len(models) < int(gate["minimum_models"]):
            continue
        slot, channel, site = key
        evidence = {
            model: next(
                row
                for row in model_rows[model]
                if (
                    row["normalized_depth_slot"],
                    row["channel"],
                    row["site"],
                )
                == key
            )
            for model in models
        }
        repeated.append({
            "normalized_depth_slot": slot,
            "channel": channel,
            "site": site,
            "models": models,
            "physical_depths": {
                model: evidence[model]["physical_depth"]
                for model in models
            },
            "minimum_same_pair_cosine": min(
                float(row["same_pair_cosine"]["median"])
                for row in evidence.values()
            ),
            "minimum_advantage_median": min(
                float(row["matched_minus_shuffled"]["median"])
                for row in evidence.values()
            ),
            "minimum_advantage_positive_rate": min(
                float(
                    row["matched_minus_shuffled"]["positive_rate"]
                )
                for row in evidence.values()
            ),
            "minimum_family_to_lexical_ratio": min(
                float(
                    row["family_to_lexical_norm_ratio"]["median"]
                )
                for row in evidence.values()
            ),
            "model_evidence": evidence,
        })
    repeated.sort(
        key=lambda row: (
            len(row["models"]),
            row["minimum_advantage_median"],
            row["minimum_same_pair_cosine"],
        ),
        reverse=True,
    )

    late = [
        row for row in repeated
        if row["normalized_depth_slot"]
        in prereg["automatic_followup"]["late_depth_slots"]
        and row["site"] in LATE_SITES
        and row["channel"] in gate["actual_write_channels"]
    ]
    frozen = [
        {
            "rank": rank,
            "normalized_depth_slot": row[
                "normalized_depth_slot"
            ],
            "channel": row["channel"],
            "site": row["site"],
            "models": row["models"],
            "physical_depths": row["physical_depths"],
        }
        for rank, row in enumerate(late[:3], 1)
    ]
    if frozen:
        decision = {
            "causal_confirmation_needed": True,
            "route": (
                "Freeze the repeated late query/boundary actual-write "
                "cells and test each cell separately on the untouched "
                "surface-1 targets with matched, same-family lexical, "
                "strict shuffled-family, and zero-delta controls."
            ),
            "claim_limit": (
                "The atlas reports repeated response geometry only; it "
                "does not establish transport, sufficiency, a minimum "
                "circuit, or a complete language mechanism."
            ),
        }
    else:
        decision = {
            "causal_confirmation_needed": False,
            "route": (
                "No late query/boundary actual-write cell repeats in at "
                "least two models. Preserve the atlas and stop this "
                "controlled-family route."
            ),
            "claim_limit": (
                "Absence under this gate would not exclude distributed "
                "cross-depth or naturally recomputed mechanisms."
            ),
        }

    nonfinite = {
        model: {
            name: row["array_finiteness"][name]
            for name in ("family_contrasts", "candidate_logits")
        }
        for model, row in summaries.items()
    }
    aggregate = {
        "schema_version": "phase1042_aggregate.v1",
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
            "instrumentation_closure_exact": all(
                row["instrumentation_closure"]["median"] == 0
                and row["instrumentation_closure"]["finite_rate"] == 1
                for row in summaries.values()
            ),
            "protocol_audit_passed": all(
                protocol_audit["checks"].values()
            ),
            "glm4_nonfinite_is_bounded_and_disclosed": (
                summaries["glm4"]["array_finiteness"][
                    "family_contrasts"
                ]["finite_value_rate"]
                >= 0.99
                and summaries["glm4"]["array_finiteness"][
                    "candidate_logits"
                ]["finite_value_rate"]
                >= 0.98
            ),
            "other_models_all_finite": all(
                summaries[model]["array_finiteness"][
                    "family_contrasts"
                ]["all_finite"]
                and summaries[model]["array_finiteness"][
                    "candidate_logits"
                ]["all_finite"]
                for model in ("qwen3", "deepseek7b")
            ),
        },
        "instrument_warning": {
            "glm4": (
                "Six template-1/two-token material-family targets become "
                "non-finite from normalized depth slot 4 onward in the "
                "b1l1 world. The pattern reproduced with GLM4 target "
                "batch sizes 2 and 1 under FP16 CPU/GPU offload. Metrics "
                "use finite observations only; no values were imputed."
            ),
            "minimum_finite_rate_for_candidate": MIN_VALID_RATE,
            "array_finiteness": nonfinite,
        },
        "model_candidate_rows": model_rows,
        "cross_model_repeated_cells": repeated,
        "late_query_boundary_cells": late,
        "frozen_causal_candidates": frozen,
        "automatic_next_decision": decision,
        "model_summaries": summaries,
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)

    manifest = artifact_manifest(protocol.OUT_ROOT)
    audit = {
        "phase": protocol.PHASE,
        "checks": aggregate["checks"],
        "model_candidate_counts": {
            model: sum(row["eligible"] for row in rows)
            for model, rows in model_rows.items()
        },
        "cross_model_repeated_cell_count": len(repeated),
        "late_query_boundary_cell_count": len(late),
        "frozen_causal_candidate_count": len(frozen),
        "causal_confirmation_needed": decision[
            "causal_confirmation_needed"
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
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
