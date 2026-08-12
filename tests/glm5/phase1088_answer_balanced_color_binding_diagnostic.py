#!/usr/bin/env python3
"""Posthoc physical map for Phase1088 pair-conditioned binding fingerprints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1088_answer_balanced_color_binding_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_finalize as engine


def physical_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for event_index, event in enumerate(data["summary"]["events"]):
        for role in protocol.CAPTURE_ROLES:
            replicates = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                content_source = engine.operation_bank(
                    data, protocol.WORLDS, "discovery", "content", replicate,
                    centered=True, role=role, event_index=event_index,
                )
                content_target = engine.operation_bank(
                    data, protocol.WORLDS, "confirmation", "content", replicate,
                    centered=True, role=role, event_index=event_index,
                )
                null_source = engine.operation_bank(
                    data, protocol.WORLDS, "discovery", "field_null", replicate,
                    centered=True, role=role, event_index=event_index,
                )
                null_target = engine.operation_bank(
                    data, protocol.WORLDS, "confirmation", "field_null", replicate,
                    centered=True, role=role, event_index=event_index,
                )
                content = engine.exact_assignment(
                    content_source @ content_target.T
                )
                null = engine.exact_assignment(null_source @ null_target.T)
                replicates.append({
                    "replicate": replicate,
                    "content_top1": content["top1_correct"],
                    "field_null_top1": null["top1_correct"],
                    "content_identity_mean": content["identity_mean_score"],
                    "field_null_identity_mean": null["identity_mean_score"],
                    "content_over_null_identity_advantage": (
                        content["identity_mean_score"]
                        - null["identity_mean_score"]
                    ),
                })
            rows.append({
                **event,
                "role": role,
                "replicates": replicates,
                "minimum_content_top1": min(
                    row["content_top1"] for row in replicates
                ),
                "minimum_content_over_null_identity_advantage": min(
                    row["content_over_null_identity_advantage"]
                    for row in replicates
                ),
                "mean_content_identity": float(np.mean([
                    row["content_identity_mean"] for row in replicates
                ])),
                "mean_field_null_identity": float(np.mean([
                    row["field_null_identity_mean"] for row in replicates
                ])),
            })
    return rows


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    phase1087 = protocol.read_json(
        protocol.SOURCE_ROOT / "analysis" / "final_summary.json"
    )
    phase1088 = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json"
    )
    models = {name: engine.load_model(name) for name in protocol.MODELS}
    by_model = {}
    for name, data in models.items():
        rows = physical_rows(data)
        old_fraction = phase1087["shared_relation_summary"][name][
            "median_signed_shared_fraction"
        ]["content"]
        new_fraction = phase1088["shared_binding_summary"][name][
            "median_signed_shared_fraction"
        ]["content"]
        by_model[name] = {
            "phase1087_truth_aligned_shared_fraction": old_fraction,
            "phase1088_answer_balanced_shared_fraction": new_fraction,
            "shared_fraction_retained_ratio": (
                new_fraction / old_fraction if old_fraction else None
            ),
            "physical_rows": rows,
            "top_pair_conditioned_fingerprint": sorted(
                rows,
                key=lambda row: (
                    row["minimum_content_top1"],
                    row["minimum_content_over_null_identity_advantage"],
                    row["mean_content_identity"],
                ),
                reverse=True,
            )[:16],
        }
    result = {
        "schema_version": "phase1088_pair_fingerprint_posthoc_map.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "evidence_status": (
            "posthoc_non_upgrading; P3/P4/P5/P8/P10 remain failed"
        ),
        "by_model": by_model,
        "interpretation": (
            "A high pair-retrieval score localizes a color-token-pair-conditioned "
            "binding fingerprint. It does not distinguish lexical carrier "
            "identity from color semantics and cannot authorize intervention."
        ),
    }
    result["pair_fingerprint_map_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "pair_fingerprint_posthoc_map.json",
        result,
    )
    print({
        "phase": protocol.PHASE,
        "retained_shared_fraction": {
            name: row["shared_fraction_retained_ratio"]
            for name, row in by_model.items()
        },
        "top_locations": {
            name: [
                {
                    "component": row["component"],
                    "relative_depth": row["relative_depth"],
                    "role": row["role"],
                    "minimum_top1": row["minimum_content_top1"],
                    "minimum_advantage": row[
                        "minimum_content_over_null_identity_advantage"
                    ],
                }
                for row in value["top_pair_conditioned_fingerprint"][:3]
            ]
            for name, value in by_model.items()
        },
        "digest": result["pair_fingerprint_map_digest"],
    })


if __name__ == "__main__":
    main()
