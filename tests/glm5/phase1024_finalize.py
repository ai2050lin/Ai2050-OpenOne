#!/usr/bin/env python3
"""Aggregate and audit Phase1024 without turning metrics into a theory."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1024_lexical_semantic_protocol as protocol


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def median(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    return float(np.median(values)) if values else None


def selected_track_row(
    atlas: dict[str, Any],
    track: str,
) -> dict[str, Any]:
    rows = atlas["representative_residual_metrics"][track]
    return rows[0]


def head_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    focus = [row for row in rows if row["role"] == "focus_end"]
    definitions = {
        "nonce": (
            lambda row: (
                row["metrics"]["nonce"]["discovery"][
                    "concept_cross_surface_top1"
                ],
                row["alignment"]["discovery"]["definition_query_top1"],
            ),
            lambda row: (
                row["metrics"]["nonce"]["confirmation"][
                    "concept_cross_surface_top1"
                ] >= 0.50
                and row["alignment"]["confirmation"][
                    "definition_query_top1"
                ] >= 0.50
            ),
        ),
        "polysemy": (
            lambda row: (
                row["metrics"]["polysemy"]["discovery"]["sense_top1"],
                row["metrics"]["polysemy"]["discovery"][
                    "difference_direction_cosine"
                ],
            ),
            lambda row: (
                row["metrics"]["polysemy"]["confirmation"][
                    "sense_top1"
                ] >= 0.75
            ),
        ),
        "synonym": (
            lambda row: (
                row["metrics"]["synonym"]["discovery"]["group_top1"],
                row["metrics"]["synonym"]["discovery"][
                    "same_vs_shifted_margin"
                ],
            ),
            lambda row: (
                row["metrics"]["synonym"]["confirmation"][
                    "group_top1"
                ] >= 0.50
            ),
        ),
    }
    result = {}
    for track, (rank_key, confirm) in definitions.items():
        frozen = sorted(focus, key=rank_key, reverse=True)[:10]
        result[track] = {
            "discovery_frozen_count": len(frozen),
            "confirmation_repeat_count": sum(confirm(row) for row in frozen),
            "representatives": [
                {
                    "depth": row["depth"],
                    "head": row["head"],
                    "discovery": (
                        row["metrics"][track]["discovery"]
                        if track != "nonce"
                        else {
                            **row["metrics"]["nonce"]["discovery"],
                            "alignment": row["alignment"]["discovery"],
                        }
                    ),
                    "confirmation": (
                        row["metrics"][track]["confirmation"]
                        if track != "nonce"
                        else {
                            **row["metrics"]["nonce"]["confirmation"],
                            "alignment": row["alignment"]["confirmation"],
                        }
                    ),
                }
                for row in frozen[:5]
            ],
        }
    return result


def mlp_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = {
        "nonce": (
            "nonce_confirmation_concept_corr",
            lambda row: (
                row["metrics"]["nonce_confirmation_concept_corr"] > 0
                and row["metrics"]["nonce_cross_split_family_corr"] > 0
            ),
        ),
        "polysemy": (
            "polysemy_confirmation_corr",
            lambda row: row["metrics"]["polysemy_confirmation_corr"] > 0,
        ),
        "synonym": (
            "synonym_confirmation_corr",
            lambda row: row["metrics"]["synonym_confirmation_corr"] > 0,
        ),
    }
    result = {}
    for track, (metric_key, confirm) in definitions.items():
        selected = [
            row for row in rows
            if row["candidate_type"] == "selected"
            and track in row["selected_tracks"]
        ]
        controls = [
            row for row in rows
            if row["candidate_type"] == "random_control"
        ]
        result[track] = {
            "discovery_selected_count": len(selected),
            "confirmation_positive_count": sum(
                confirm(row) for row in selected
            ),
            "confirmation_positive_rate": (
                sum(confirm(row) for row in selected) / len(selected)
                if selected else None
            ),
            "selected_confirmation_median": median([
                float(row["metrics"][metric_key]) for row in selected
            ]),
            "random_control_confirmation_median": median([
                float(row["metrics"][metric_key]) for row in controls
            ]),
            "representatives": [
                {
                    "depth": row["depth"],
                    "role": row["role"],
                    "coordinate": row["coordinate"],
                    "metrics": row["metrics"],
                }
                for row in sorted(
                    selected,
                    key=lambda value: value["metrics"][metric_key],
                    reverse=True,
                )[:5]
            ],
        }
    return result


def trajectory_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    focus = sorted(
        [row for row in rows if row["role"] == "focus_end"],
        key=lambda row: int(row["depth"]),
    )

    def first_depth(predicate) -> int | None:
        for row in focus:
            if int(row["depth"]) >= 1 and predicate(row):
                return int(row["depth"])
        return None

    return {
        "nonce_concept_half_accuracy_depth_discovery": first_depth(
            lambda row: row["metrics"]["nonce"]["discovery"][
                "concept_cross_surface_top1"
            ] >= 0.50
        ),
        "nonce_definition_query_half_accuracy_depth_discovery": first_depth(
            lambda row: row["alignment"]["discovery"][
                "definition_query_top1"
            ] >= 0.50
        ),
        "polysemy_three_quarter_depth_discovery": first_depth(
            lambda row: row["metrics"]["polysemy"]["discovery"][
                "sense_top1"
            ] >= 0.75
        ),
        "synonym_half_accuracy_depth_discovery": first_depth(
            lambda row: row["metrics"]["synonym"]["discovery"][
                "group_top1"
            ] >= 0.50
        ),
        "surface_identity_minimum_discovery": min(
            row["metrics"]["nonce"]["discovery"][
                "surface_cross_concept_top1"
            ]
            for row in focus
        ),
        "surface_identity_final_discovery": focus[-1]["metrics"]["nonce"][
            "discovery"
        ]["surface_cross_concept_top1"],
    }


def model_summary(model_name: str) -> dict[str, Any]:
    behavior = read_json(
        protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    atlas_dir = protocol.OUT_ROOT / "atlas" / model_name
    atlas = read_json(atlas_dir / "summary.json")
    residual_rows = read_jsonl(atlas_dir / "residual_metrics.jsonl")
    head_rows = read_jsonl(atlas_dir / "attention_head_metrics.jsonl")
    mlp_rows = read_jsonl(atlas_dir / "mlp_coordinate_metrics.jsonl")
    representative = {
        track: selected_track_row(atlas, track)
        for track in ("nonce", "polysemy", "synonym")
    }
    nonce_row = representative["nonce"]
    observational_gates = {
        "nonce_repeated": (
            nonce_row["metrics"]["nonce"]["confirmation"][
                "concept_cross_surface_top1"
            ] >= 0.50
            and nonce_row["alignment"]["confirmation"][
                "definition_query_top1"
            ] >= 0.45
        ),
        "polysemy_repeated": (
            representative["polysemy"]["metrics"]["polysemy"][
                "confirmation"
            ]["sense_top1"] >= 0.75
        ),
        "synonym_repeated": (
            representative["synonym"]["metrics"]["synonym"][
                "confirmation"
            ]["group_top1"] >= 0.50
        ),
        "all_internal_tensors_finite": all(
            value["all_finite"]
            for value in atlas["tensor_finiteness"].values()
        ),
        "candidate_logits_finite": behavior["numerical_audit"][
            "all_logits_finite"
        ],
        "behavior_claim_qualified": behavior["behavior_claim_qualified"],
    }
    return {
        "model": model_name,
        "behavior": behavior,
        "atlas": atlas,
        "representative": representative,
        "trajectory": trajectory_summary(residual_rows),
        "head": head_summary(head_rows),
        "mlp": mlp_summary(mlp_rows),
        "observational_gates": observational_gates,
    }


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def manifest() -> dict[str, Any]:
    excluded = {
        protocol.OUT_ROOT / "final" / "artifact_manifest.json",
    }
    rows = []
    for path in sorted(
        item for item in protocol.OUT_ROOT.rglob("*")
        if item.is_file() and item not in excluded
    ):
        rows.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        })
    return {
        "schema_version": "phase1024_artifact_manifest.v1",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def main() -> None:
    prereg = read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    models = {
        model_name: model_summary(model_name)
        for model_name in protocol.MODELS
    }
    repeated_counts = {
        key: sum(
            row["observational_gates"][key] for row in models.values()
        )
        for key in (
            "nonce_repeated",
            "polysemy_repeated",
            "synonym_repeated",
            "all_internal_tensors_finite",
            "candidate_logits_finite",
            "behavior_claim_qualified",
        )
    }
    summary = {
        "schema_version": "phase1024_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "protocol_digest": prereg["protocol_digest"],
        "models": models,
        "cross_model": {
            "repeat_counts": repeated_counts,
            "fully_finite_model_count": repeated_counts[
                "all_internal_tensors_finite"
            ],
            "finite_logit_model_count": repeated_counts[
                "candidate_logits_finite"
            ],
            "core_observation": (
                "surface identity persists while context-assigned concept "
                "and sense structure becomes readable; representation and "
                "correct execution remain separated"
            ),
        },
        "hypothesis_status": {
            "brain_plasticity_near_optimality": (
                "not tested; no brain data, training trajectory, energy, or "
                "optimality comparison was measured"
            ),
            "reuse_as_efficiency_optimum": (
                "physical reuse is observed; efficiency and optimality are "
                "not identified"
            ),
            "relative_encoding": (
                "strengthened observationally by balanced surface x meaning "
                "and same-surface different-sense controls; not yet causal"
            ),
            "language_as_pattern_collection": (
                "compatible with nonce binding, polysemy, synonym and prior "
                "pattern-family results; not an exhaustive language theory"
            ),
            "unique_lexical_ecological_niche": (
                "partly supported, but the same surface changes with context "
                "and semantic state coexists with surface identity; niches "
                "are conditional rather than fixed word slots"
            ),
            "small_model_roughness": (
                "FP16 numerical fragility is directly observed in GLM4 and "
                "DeepSeek7B; model size is not isolated as the cause"
            ),
        },
        "claim_limits": {
            "not_proved": (
                "closed token mechanism, full language family map, causal "
                "necessity/sufficiency, brain homology, energy optimum, "
                "global mathematical theory"
            ),
            "measurement_formulas_only": True,
        },
    }
    final_dir = protocol.OUT_ROOT / "final"
    protocol.write_json(final_dir / "summary.json", summary)

    auto_next = {
        "schema_version": "phase1024_automatic_next_action.v1",
        "authorized": (
            repeated_counts["nonce_repeated"] >= 2
            and repeated_counts["polysemy_repeated"] >= 2
            and repeated_counts["synonym_repeated"] >= 2
        ),
        "action": "phase1025_binding_specificity_controls",
        "reason": (
            "the repeated nonce signal may still be caused by prompt-wide "
            "concept co-occurrence rather than the binding relation; compare "
            "bound, unbound co-occurrence, reversed relation, and distractor "
            "conditions before causal intervention"
        ),
        "causal_output_patch_authorized": False,
        "causal_output_patch_blocker": (
            "only Qwen3 has fully finite internal tensors and candidate "
            "logits; nonce output classification is at chance"
        ),
    }
    protocol.write_json(
        final_dir / "automatic_next_action.json",
        auto_next,
    )

    checks = {
        "protocol_audits_passed": (
            read_json(
                protocol.OUT_ROOT / "protocol" / "audit.common.json"
            )["all_checks_passed"]
            and read_json(
                protocol.OUT_ROOT / "protocol" / "audit.models.json"
            )["all_checks_passed"]
        ),
        "all_model_behavior_present": all(
            (
                protocol.OUT_ROOT / "behavior" / model / "summary.json"
            ).exists()
            for model in protocol.MODELS
        ),
        "all_model_atlas_present": all(
            (
                protocol.OUT_ROOT / "atlas" / model / "summary.json"
            ).exists()
            for model in protocol.MODELS
        ),
        "fp16_no_quantization": all(
            row["behavior"]["precision"] == "fp16"
            and row["behavior"]["quantization"] == "none"
            and not row["behavior"]["runtime_precision_audit"][
                "has_quantized_modules"
            ]
            for row in models.values()
        ),
        "discovery_only_selection": all(
            row["atlas"]["selection"]["selection_source"]
            == "discovery_only"
            for row in models.values()
        ),
        "claim_limits_present": bool(summary["claim_limits"]),
    }
    audit = {
        "schema_version": "phase1024_final_audit.v1",
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    protocol.write_json(final_dir / "audit.json", audit)
    artifact_manifest = manifest()
    protocol.write_json(
        final_dir / "artifact_manifest.json",
        artifact_manifest,
    )
    print(json.dumps({
        "cross_model": summary["cross_model"],
        "automatic_next_action": auto_next,
        "audit": audit,
        "manifest": {
            "file_count": artifact_manifest["file_count"],
            "total_bytes": artifact_manifest["total_bytes"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
