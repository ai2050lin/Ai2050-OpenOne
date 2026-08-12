#!/usr/bin/env python3
"""Posthoc, non-upgrading diagnostics for Phase1087 failed P6 and P8."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1087_color_relation_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_finalize as engine


def mean_other_pair_direction(
    data: dict[str, Any],
    heldout: str,
    world: str,
    split: str,
    field: str,
    replicate: int,
    template: int,
) -> np.ndarray:
    values = [
        engine.unit_vector(engine.profile(
            data, operation, (world,), split, field, replicate,
            template=template,
        ))
        for operation in protocol.OPERATIONS
        if operation != heldout
    ]
    return engine.unit_vector(np.mean(np.stack(values), axis=0))


def analyze_model(model_name: str, data: dict[str, Any]) -> dict[str, Any]:
    cosine_min = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_shared_split_cosine"]
    )
    advantage_min = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_shared_content_over_null_advantage"
        ]
    )
    template_rows = []
    cross_template_rows = []
    cross_template_geometry_rows = []
    heldout_rows = []
    residual_rows = []
    for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
        for template in (0, 1):
            for world in protocol.WORLDS:
                content_source, _ = engine.shared_centroid(
                    data, (world,), "discovery", "content", replicate,
                    template=template,
                )
                content_target, _ = engine.shared_centroid(
                    data, (world,), "confirmation", "content", replicate,
                    template=template,
                )
                null_source, _ = engine.shared_centroid(
                    data, (world,), "discovery", "field_null", replicate,
                    template=template,
                )
                null_target, _ = engine.shared_centroid(
                    data, (world,), "confirmation", "field_null", replicate,
                    template=template,
                )
                content_cosine = engine.cosine(content_source, content_target)
                null_cosine = engine.cosine(null_source, null_target)
                advantage = content_cosine - null_cosine
                template_rows.append({
                    "replicate": replicate,
                    "template": template,
                    "world": world,
                    "content_cosine": content_cosine,
                    "field_null_cosine": null_cosine,
                    "content_over_null_advantage": advantage,
                    "passed_descriptive_threshold": (
                        content_cosine >= cosine_min
                        and advantage >= advantage_min
                    ),
                })

                for heldout in protocol.OPERATIONS:
                    content_source = mean_other_pair_direction(
                        data, heldout, world, "discovery", "content",
                        replicate, template,
                    )
                    content_target = engine.unit_vector(engine.profile(
                        data, heldout, (world,), "confirmation", "content",
                        replicate, template=template,
                    ))
                    null_source = mean_other_pair_direction(
                        data, heldout, world, "discovery", "field_null",
                        replicate, template,
                    )
                    null_target = engine.unit_vector(engine.profile(
                        data, heldout, (world,), "confirmation", "field_null",
                        replicate, template=template,
                    ))
                    content_cosine = engine.cosine(
                        content_source, content_target
                    )
                    null_cosine = engine.cosine(null_source, null_target)
                    advantage = content_cosine - null_cosine
                    heldout_rows.append({
                        "replicate": replicate,
                        "template": template,
                        "world": world,
                        "heldout_color_pair": heldout,
                        "content_cosine": content_cosine,
                        "field_null_cosine": null_cosine,
                        "content_over_null_advantage": advantage,
                        "passed_descriptive_threshold": (
                            content_cosine >= cosine_min
                            and advantage >= advantage_min
                        ),
                    })

        for source_template, target_template in ((0, 1), (1, 0)):
            for world in protocol.WORLDS:
                content_source, _ = engine.shared_centroid(
                    data, (world,), "discovery", "content", replicate,
                    template=source_template,
                )
                content_target, _ = engine.shared_centroid(
                    data, (world,), "confirmation", "content", replicate,
                    template=target_template,
                )
                null_source, _ = engine.shared_centroid(
                    data, (world,), "discovery", "field_null", replicate,
                    template=source_template,
                )
                null_target, _ = engine.shared_centroid(
                    data, (world,), "confirmation", "field_null", replicate,
                    template=target_template,
                )
                content_cosine = engine.cosine(content_source, content_target)
                null_cosine = engine.cosine(null_source, null_target)
                advantage = content_cosine - null_cosine
                cross_template_rows.append({
                    "replicate": replicate,
                    "source_template": source_template,
                    "target_template": target_template,
                    "world": world,
                    "content_cosine": content_cosine,
                    "field_null_cosine": null_cosine,
                    "content_over_null_advantage": advantage,
                    "passed_descriptive_threshold": (
                        content_cosine >= cosine_min
                        and advantage >= advantage_min
                    ),
                })

            content_source_bank = engine.operation_bank(
                data, protocol.WORLDS, "discovery", "content", replicate,
                centered=True, template=source_template,
            )
            content_target_bank = engine.operation_bank(
                data, protocol.WORLDS, "confirmation", "content", replicate,
                centered=True, template=target_template,
            )
            null_source_bank = engine.operation_bank(
                data, protocol.WORLDS, "discovery", "field_null", replicate,
                centered=True, template=source_template,
            )
            null_target_bank = engine.operation_bank(
                data, protocol.WORLDS, "confirmation", "field_null", replicate,
                centered=True, template=target_template,
            )
            content_geometry = engine.cosine(
                engine.relation_vector(content_source_bank),
                engine.relation_vector(content_target_bank),
            )
            null_geometry = engine.cosine(
                engine.relation_vector(null_source_bank),
                engine.relation_vector(null_target_bank),
            )
            cross_template_geometry_rows.append({
                "replicate": replicate,
                "source_template": source_template,
                "target_template": target_template,
                "content_geometry_cosine": content_geometry,
                "field_null_geometry_cosine": null_geometry,
                "content_over_null_geometry_advantage": (
                    content_geometry - null_geometry
                ),
                "passed_descriptive_threshold": (
                    content_geometry >= cosine_min
                    and content_geometry - null_geometry >= advantage_min
                ),
            })

        content_source = engine.operation_bank(
            data, protocol.WORLDS, "discovery", "content", replicate,
            centered=True,
        )
        content_target = engine.operation_bank(
            data, protocol.WORLDS, "confirmation", "content", replicate,
            centered=True,
        )
        null_source = engine.operation_bank(
            data, protocol.WORLDS, "discovery", "field_null", replicate,
            centered=True,
        )
        null_target = engine.operation_bank(
            data, protocol.WORLDS, "confirmation", "field_null", replicate,
            centered=True,
        )
        content_assignment = engine.exact_assignment(
            content_source @ content_target.T
        )
        null_assignment = engine.exact_assignment(null_source @ null_target.T)
        residual_rows.append({
            "replicate": replicate,
            "content_top1": content_assignment["top1_correct"],
            "field_null_top1": null_assignment["top1_correct"],
            "content_identity_mean": content_assignment["identity_mean_score"],
            "field_null_identity_mean": null_assignment["identity_mean_score"],
            "content_over_null_identity_advantage": (
                content_assignment["identity_mean_score"]
                - null_assignment["identity_mean_score"]
            ),
        })

    def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row[name] for name in keys)
            groups.setdefault(key, []).append(row)
        output = []
        for key, values in sorted(groups.items()):
            output.append({
                **dict(zip(keys, key)),
                "count": len(values),
                "passing_count": sum(
                    int(row["passed_descriptive_threshold"]) for row in values
                ),
                "mean_content_cosine": float(np.mean([
                    row["content_cosine"] for row in values
                ])),
                "mean_field_null_cosine": float(np.mean([
                    row["field_null_cosine"] for row in values
                ])),
                "mean_content_over_null_advantage": float(np.mean([
                    row["content_over_null_advantage"] for row in values
                ])),
            })
        return output

    null_saturated = all(
        row["field_null_top1"] >= 6 for row in residual_rows
    )
    return {
        "template_split_rows": template_rows,
        "template_split_summary": summarize(
            template_rows, ("replicate", "template")
        ),
        "cross_template_rows": cross_template_rows,
        "cross_template_summary": summarize(
            cross_template_rows,
            ("replicate", "source_template", "target_template"),
        ),
        "cross_template_relation_geometry": cross_template_geometry_rows,
        "template_conditioned_heldout_pair_rows": heldout_rows,
        "template_conditioned_heldout_pair_summary": summarize(
            heldout_rows, ("replicate", "template")
        ),
        "pair_identity_rows": residual_rows,
        "diagnostic_flags": {
            "same_template_relation_repeats": all(
                row["passing_count"] >= 3
                for row in summarize(
                    template_rows, ("replicate", "template")
                )
            ),
            "cross_template_relation_repeats": all(
                row["passing_count"] >= 3
                for row in summarize(
                    cross_template_rows,
                    ("replicate", "source_template", "target_template"),
                )
            ),
            "cross_template_relation_geometry_repeats": all(
                row["passed_descriptive_threshold"]
                for row in cross_template_geometry_rows
            ),
            "field_null_pair_identity_saturated": null_saturated,
        },
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    models = {name: engine.load_model(name) for name in protocol.MODELS}
    result = {
        "schema_version": "phase1087_posthoc_surface_pair_diagnostic.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "evidence_status": (
            "posthoc_non_upgrading; frozen P6 and P8 remain failed"
        ),
        "by_model": {
            name: analyze_model(name, data) for name, data in models.items()
        },
        "interpretation_limits": [
            "These diagnostics cannot change any preregistered prediction.",
            "Same-template repetition does not establish paraphrase invariance.",
            "Null pair retrieval indicates lexical/binding carrier identity, not semantic pair coding.",
        ],
    }
    result["posthoc_diagnostic_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis"
        / "posthoc_surface_pair_diagnostic.json",
        result,
    )
    print({
        "phase": protocol.PHASE,
        "flags": {
            name: row["diagnostic_flags"]
            for name, row in result["by_model"].items()
        },
        "posthoc_diagnostic_digest": result["posthoc_diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
