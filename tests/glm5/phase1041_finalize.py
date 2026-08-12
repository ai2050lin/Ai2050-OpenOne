#!/usr/bin/env python3
"""Finalize Phase1041 and freeze any cross-model alliance candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import phase1041_position_write_alliance_protocol as protocol


def artifact_manifest(root: Path) -> dict[str, Any]:
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


def model_candidate(
    position_row: dict[str, Any],
    alliance_row: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    all_group = position_row["groups"]["all"]
    cross = all_group["cross_shift"]
    purity_gain = alliance_row[
        "purity_gain_over_selected_concept"
    ]
    cross_gain = alliance_row[
        "cross_gain_over_selected_concept"
    ]["median"]
    best_gain = alliance_row["best_constituent_gain"]["median"]
    group_medians = {
        group: values["cross_shift"]["median"]
        for group, values in position_row["groups"].items()
        if group != "all"
    }
    reasons = {
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
        "purity_gain_positive": (
            purity_gain is not None
            and purity_gain
            > gate["purity_gain_over_selected_concept_min"]
        ),
        "matched_to_shuffled_ratio": (
            all_group["matched_to_shuffled_ratio"] is not None
            and all_group["matched_to_shuffled_ratio"]
            >= gate["matched_to_shuffled_ratio_min"]
        ),
        "cross_gain_positive": (
            cross_gain is not None
            and cross_gain
            > gate[
                "cross_median_gain_over_selected_concept_min"
            ]
        ),
        "best_constituent_gain_positive": (
            best_gain is not None
            and best_gain
            > gate["best_constituent_gain_median_min"]
        ),
        "full_state_retention": (
            position_row["full_state_retention"] is not None
            and position_row["full_state_retention"]
            >= gate["full_state_retention_min"]
        ),
    }
    return all(reasons.values()), {
        "checks": reasons,
        "cross_shift": cross,
        "purity_ratio": all_group["purity_ratio"],
        "matched_to_shuffled_ratio": all_group[
            "matched_to_shuffled_ratio"
        ],
        "full_state_retention": position_row[
            "full_state_retention"
        ],
        "purity_gain_over_selected_concept": purity_gain,
        "cross_gain_over_selected_concept": alliance_row[
            "cross_gain_over_selected_concept"
        ],
        "best_constituent_gain": alliance_row[
            "best_constituent_gain"
        ],
        "additivity_residual": alliance_row[
            "additivity_residual"
        ],
        "group_cross_medians": group_medians,
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

    gate = prereg["discovery_candidate_rule"]
    model_rows = {}
    candidate_models: dict[tuple[str, str], list[str]] = {}
    for model in protocol.MODELS:
        positions = {
            (row["mask"], row["mode"]): row
            for row in metrics[model]["position_mode_rows"]
        }
        alliances = {
            (row["mask"], row["mode"]): row
            for row in metrics[model]["alliance_rows"]
        }
        current_rows = []
        for mask in protocol.CANDIDATE_MASKS:
            for mode in ("mlp_write", "current_write"):
                key = (mask, mode)
                passed, evidence = model_candidate(
                    positions[key], alliances[key], gate
                )
                current_rows.append({
                    "mask": mask,
                    "mode": mode,
                    "eligible": passed,
                    **evidence,
                })
                if passed:
                    candidate_models.setdefault(key, []).append(model)
        model_rows[model] = current_rows

    repeated = []
    for (mask, mode), models in candidate_models.items():
        if len(models) < int(gate["minimum_models"]):
            continue
        evidence_rows = {
            model: next(
                row for row in model_rows[model]
                if row["mask"] == mask and row["mode"] == mode
            )
            for model in models
        }
        repeated.append({
            "mask": mask,
            "mode": mode,
            "models": models,
            "model_evidence": evidence_rows,
            "minimum_purity_gain": min(
                float(row["purity_gain_over_selected_concept"])
                for row in evidence_rows.values()
            ),
            "minimum_best_constituent_gain": min(
                float(row["best_constituent_gain"]["median"])
                for row in evidence_rows.values()
            ),
            "minimum_full_state_retention": min(
                float(row["full_state_retention"])
                for row in evidence_rows.values()
            ),
        })
    repeated.sort(
        key=lambda row: (
            len(row["models"]),
            row["minimum_purity_gain"],
            row["minimum_best_constituent_gain"],
            row["minimum_full_state_retention"],
        ),
        reverse=True,
    )
    frozen = [
        {
            "rank": rank,
            "mask": row["mask"],
            "mode": row["mode"],
            "models": row["models"],
        }
        for rank, row in enumerate(repeated[:3], 1)
    ]

    if frozen:
        decision = {
            "confirmation_needed": True,
            "route": (
                "Freeze the repeated discovery alliances and run them on "
                "the reserved surface-1 confirmation targets with matched, "
                "same-lexical, shuffled-pair, wrong-position, singleton, "
                "and full-state controls."
            ),
            "claim_limit": (
                "Discovery eligibility does not establish a minimum causal "
                "subgraph or a natural-language mechanism."
            ),
        }
    else:
        decision = {
            "confirmation_needed": False,
            "route": (
                "No same-depth position alliance improves over its best "
                "constituents in at least two models. Preserve the map and "
                "move to a separately preregistered role-by-depth atlas; do "
                "not tune coalition thresholds."
            ),
            "claim_limit": (
                "A negative same-depth cached-write coalition test does not "
                "exclude cross-depth or naturally recomputed alliances."
            ),
        }

    aggregate = {
        "schema_version": "phase1041_aggregate.v1",
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
            "instrumentation_closure_finite": all(
                row["array_finiteness"]["channel_closure"][
                    "all_finite"
                ]
                for row in summaries.values()
            ),
            "zero_delta_identity_exact": all(
                evidence["exact"]
                for row in summaries.values()
                for evidence in row["zero_delta_identity"].values()
            ),
            "protocol_audit_passed": all(
                protocol_audit["checks"].values()
            ),
        },
        "model_candidate_rows": model_rows,
        "cross_model_repeated_alliances": repeated,
        "frozen_confirmation_candidates": frozen,
        "automatic_next_decision": decision,
        "model_summaries": summaries,
    }
    protocol.write_json(protocol.OUT_ROOT / "aggregate.json", aggregate)

    manifest = artifact_manifest(protocol.OUT_ROOT)
    audit = {
        "phase": protocol.PHASE,
        "checks": aggregate["checks"],
        "cross_model_repeated_alliance_count": len(repeated),
        "frozen_confirmation_candidate_count": len(frozen),
        "confirmation_needed": decision["confirmation_needed"],
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
    if frozen:
        print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
