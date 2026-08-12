#!/usr/bin/env python3
"""Finalize Phase1091 cross-surface color-pair geometry."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1091_cross_surface_color_signed_protocol as protocol

sys.modules["phase1086_signed_shared_field_protocol"] = protocol
import phase1086_signed_shared_field_finalize as engine


def route_worlds(route: str) -> tuple[str, ...]:
    return tuple(f"{world}@{route}" for world in protocol.BASE_WORLDS)


def bank(
    data: dict[str, Any],
    route: str,
    split: str,
    field: str,
    replicate: int,
    *,
    role: str = "answer_boundary",
    event_index: int | None = None,
    centered: bool = True,
) -> np.ndarray:
    return engine.operation_bank(
        data,
        route_worlds(route),
        split,
        field,
        replicate,
        centered=centered,
        role=role,
        event_index=event_index,
    )


def assignment_pass(
    content: dict[str, Any], null: dict[str, Any]
) -> tuple[bool, float]:
    advantage = (
        float(content["identity_mean_score"])
        - float(null["identity_mean_score"])
    )
    passed = (
        int(content["top1_correct"])
        >= int(protocol.EVIDENCE_THRESHOLDS["minimum_route_split_top1"])
        and float(content["exact_upper_tail_p"])
        <= float(protocol.EVIDENCE_THRESHOLDS["permutation_p_max"])
        and int(content["top1_correct"]) > int(null["top1_correct"])
        and advantage
        >= float(protocol.EVIDENCE_THRESHOLDS[
            "minimum_cross_surface_content_advantage"
        ])
    )
    return passed, advantage


def within_route_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for model_name, data in models.items():
        replicate_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            rows = []
            for route in protocol.SURFACE_ROUTES:
                content = engine.assignment_record(
                    bank(data, route, "discovery", "content", replicate),
                    bank(data, route, "confirmation", "content", replicate),
                    comparison="within_route_split",
                    model=model_name,
                    route=route,
                    replicate=replicate,
                    field="content",
                )
                null = engine.assignment_record(
                    bank(data, route, "discovery", "field_null", replicate),
                    bank(data, route, "confirmation", "field_null", replicate),
                    comparison="within_route_split",
                    model=model_name,
                    route=route,
                    replicate=replicate,
                    field="field_null",
                )
                passed, advantage = assignment_pass(content, null)
                rows.append({
                    "route": route,
                    "content": content,
                    "field_null": null,
                    "content_identity_advantage": advantage,
                    "passed": passed,
                })
            replicate_rows.append({
                "replicate": replicate,
                "passing_routes": sum(int(row["passed"]) for row in rows),
                "passed": all(row["passed"] for row in rows),
                "rows": rows,
            })
        by_model[model_name] = {
            "replicates": replicate_rows,
            "passed": all(row["passed"] for row in replicate_rows),
        }
    return {"by_model": by_model}


def cross_surface_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    minimum_directions = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_surface_pair_directions"]
    )
    gram_min = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_surface_pair_gram_cosine"]
    )
    advantage_min = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_surface_content_advantage"]
    )
    by_model = {}
    for model_name, data in models.items():
        replicate_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            rows = []
            for source_route in protocol.SURFACE_ROUTES:
                for target_route in protocol.SURFACE_ROUTES:
                    if source_route == target_route:
                        continue
                    content_source = bank(
                        data, source_route, "discovery", "content", replicate
                    )
                    content_target = bank(
                        data, target_route, "confirmation", "content", replicate
                    )
                    null_source = bank(
                        data, source_route, "discovery", "field_null", replicate
                    )
                    null_target = bank(
                        data, target_route, "confirmation", "field_null", replicate
                    )
                    content_assignment = engine.assignment_record(
                        content_source,
                        content_target,
                        comparison="cross_surface_route",
                        model=model_name,
                        source_route=source_route,
                        target_route=target_route,
                        replicate=replicate,
                        field="content",
                    )
                    null_assignment = engine.assignment_record(
                        null_source,
                        null_target,
                        comparison="cross_surface_route",
                        model=model_name,
                        source_route=source_route,
                        target_route=target_route,
                        replicate=replicate,
                        field="field_null",
                    )
                    identity_passed, identity_advantage = assignment_pass(
                        content_assignment, null_assignment
                    )
                    content_gram = engine.cosine(
                        engine.relation_vector(content_source),
                        engine.relation_vector(content_target),
                    )
                    null_gram = engine.cosine(
                        engine.relation_vector(null_source),
                        engine.relation_vector(null_target),
                    )
                    gram_advantage = content_gram - null_gram
                    gram_passed = (
                        content_gram >= gram_min
                        and gram_advantage >= advantage_min
                    )
                    rows.append({
                        "source_route": source_route,
                        "target_route": target_route,
                        "content_assignment": content_assignment,
                        "field_null_assignment": null_assignment,
                        "content_identity_advantage": identity_advantage,
                        "identity_passed": identity_passed,
                        "content_pair_gram_cosine": content_gram,
                        "field_null_pair_gram_cosine": null_gram,
                        "content_over_null_gram_advantage": gram_advantage,
                        "gram_passed": gram_passed,
                    })
            identity_count = sum(int(row["identity_passed"]) for row in rows)
            gram_count = sum(int(row["gram_passed"]) for row in rows)
            replicate_rows.append({
                "replicate": replicate,
                "identity_passing_directions": identity_count,
                "identity_gate_passed": identity_count >= minimum_directions,
                "gram_passing_directions": gram_count,
                "gram_gate_passed": gram_count >= minimum_directions,
                "rows": rows,
            })
        by_model[model_name] = {
            "replicates": replicate_rows,
            "identity_gate_passed": all(
                row["identity_gate_passed"] for row in replicate_rows
            ),
            "gram_gate_passed": all(
                row["gram_gate_passed"] for row in replicate_rows
            ),
        }
    return {"by_model": by_model}


def common_field_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cosine_min = float(
        protocol.EVIDENCE_THRESHOLDS["minimum_shared_split_cosine"]
    )
    advantage_min = float(
        protocol.EVIDENCE_THRESHOLDS[
            "minimum_shared_content_over_null_advantage"
        ]
    )
    minimum_directions = int(
        protocol.EVIDENCE_THRESHOLDS["minimum_cross_surface_pair_directions"]
    )
    by_model = {}
    for model_name, data in models.items():
        replicate_rows = []
        for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
            rows = []
            for source_route in protocol.SURFACE_ROUTES:
                for target_route in protocol.SURFACE_ROUTES:
                    if source_route == target_route:
                        continue
                    content_source = engine.unit_vector(np.mean(bank(
                        data, source_route, "discovery", "content", replicate,
                        centered=False,
                    ), axis=0))
                    content_target = engine.unit_vector(np.mean(bank(
                        data, target_route, "confirmation", "content", replicate,
                        centered=False,
                    ), axis=0))
                    null_source = engine.unit_vector(np.mean(bank(
                        data, source_route, "discovery", "field_null", replicate,
                        centered=False,
                    ), axis=0))
                    null_target = engine.unit_vector(np.mean(bank(
                        data, target_route, "confirmation", "field_null", replicate,
                        centered=False,
                    ), axis=0))
                    content_cosine = engine.cosine(content_source, content_target)
                    null_cosine = engine.cosine(null_source, null_target)
                    advantage = content_cosine - null_cosine
                    rows.append({
                        "source_route": source_route,
                        "target_route": target_route,
                        "content_cosine": content_cosine,
                        "field_null_cosine": null_cosine,
                        "content_over_null_advantage": advantage,
                        "passed": (
                            content_cosine >= cosine_min
                            and advantage >= advantage_min
                        ),
                    })
            count = sum(int(row["passed"]) for row in rows)
            replicate_rows.append({
                "replicate": replicate,
                "passing_directions": count,
                "passed": count >= minimum_directions,
                "rows": rows,
            })
        by_model[model_name] = {
            "replicates": replicate_rows,
            "passed": all(row["passed"] for row in replicate_rows),
        }
    return {"by_model": by_model}


def cross_model_analysis(
    models: dict[str, dict[str, Any]], healthy: set[str]
) -> dict[str, Any]:
    rows = []
    for source_model in protocol.MODELS:
        for target_model in protocol.MODELS:
            if source_model == target_model:
                continue
            route_rows = []
            for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                for route in protocol.SURFACE_ROUTES:
                    content = engine.cosine(
                        engine.relation_vector(bank(
                            models[source_model], route, "discovery", "content", replicate
                        )),
                        engine.relation_vector(bank(
                            models[target_model], route, "confirmation", "content", replicate
                        )),
                    )
                    null = engine.cosine(
                        engine.relation_vector(bank(
                            models[source_model], route, "discovery", "field_null", replicate
                        )),
                        engine.relation_vector(bank(
                            models[target_model], route, "confirmation", "field_null", replicate
                        )),
                    )
                    route_rows.append({
                        "replicate": replicate,
                        "route": route,
                        "content_pair_gram_cosine": content,
                        "field_null_pair_gram_cosine": null,
                        "advantage": content - null,
                        "passed": (
                            content >= protocol.EVIDENCE_THRESHOLDS[
                                "minimum_cross_surface_pair_gram_cosine"
                            ]
                            and content - null >= protocol.EVIDENCE_THRESHOLDS[
                                "minimum_cross_surface_content_advantage"
                            ]
                        ),
                    })
            passing_by_rep = [
                sum(int(row["passed"]) for row in route_rows
                    if row["replicate"] == replicate)
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES)
            ]
            rows.append({
                "source_model": source_model,
                "target_model": target_model,
                "healthy_pair": source_model in healthy and target_model in healthy,
                "passing_routes_by_replicate": passing_by_rep,
                "passed": all(value >= 3 for value in passing_by_rep),
                "routes": route_rows,
            })
    return {"rows": rows}


def physical_band_analysis(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for model_name, data in models.items():
        events = data["summary"]["events"]
        rows = []
        for event_index, event in enumerate(events):
            for role in ("query_end", "answer_boundary"):
                passing_by_rep = []
                mean_advantages = []
                for replicate in range(protocol.SIGNED_PROJECTION_REPLICATES):
                    passes = 0
                    advantages = []
                    for source_route in protocol.SURFACE_ROUTES:
                        for target_route in protocol.SURFACE_ROUTES:
                            if source_route == target_route:
                                continue
                            content_source = bank(
                                data, source_route, "discovery", "content", replicate,
                                role=role, event_index=event_index,
                            )
                            content_target = bank(
                                data, target_route, "confirmation", "content", replicate,
                                role=role, event_index=event_index,
                            )
                            null_source = bank(
                                data, source_route, "discovery", "field_null", replicate,
                                role=role, event_index=event_index,
                            )
                            null_target = bank(
                                data, target_route, "confirmation", "field_null", replicate,
                                role=role, event_index=event_index,
                            )
                            content_result = engine.assignment_record(
                                content_source, content_target
                            )
                            null_result = engine.assignment_record(
                                null_source, null_target
                            )
                            passed, advantage = assignment_pass(
                                content_result, null_result
                            )
                            passes += int(passed)
                            advantages.append(advantage)
                    passing_by_rep.append(passes)
                    mean_advantages.append(float(np.mean(advantages)))
                rows.append({
                    "event_index": event_index,
                    "event_id": event["event_id"],
                    "component": event["component"],
                    "depth": event["depth"],
                    "relative_depth": event["relative_depth"],
                    "role": role,
                    "passing_directions_by_replicate": passing_by_rep,
                    "mean_identity_advantage_by_replicate": mean_advantages,
                    "passed": all(value >= 8 for value in passing_by_rep),
                })
        ranked = sorted(
            rows,
            key=lambda row: (
                min(row["passing_directions_by_replicate"]),
                min(row["mean_identity_advantage_by_replicate"]),
            ),
            reverse=True,
        )
        by_model[model_name] = {
            "rows": rows,
            "top_rows": ranked[:12],
            "passed": any(row["passed"] for row in rows),
        }
    return {
        "by_model": by_model,
        "scope": "preregistered_relative_depth_0.30_to_0.45_only",
        "causal_selection_authorized": False,
    }


def write_output(
    root: Path, filename: str, schema: str, payload: dict[str, Any],
    digest_key: str, protocol_digest: str,
) -> None:
    row = {
        "schema_version": schema,
        "phase": protocol.PHASE,
        "protocol_digest": protocol_digest,
        **payload,
    }
    engine.write_output(root / filename, row, digest_key)


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {name: engine.load_model(name) for name in protocol.MODELS}
    root = protocol.OUT_ROOT / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    behavior_healthy = set(prereg["behavior_healthy_models"])

    projection = engine.projection_gate(models)
    numeric = engine.numeric_gate(models, authorization)
    healthy = behavior_healthy.intersection(numeric["healthy_models"])
    within = within_route_analysis(models)
    cross_surface = cross_surface_analysis(models)
    common = common_field_analysis(models)
    cross_model = cross_model_analysis(models, healthy)
    physical = physical_band_analysis(models)

    minimum_models = int(
        prereg["evidence_thresholds"]["minimum_cross_surface_models"]
    )
    p2_models = [
        name for name in healthy if projection["by_model"][name]["passed"]
    ]
    p3_models = [
        name for name in healthy if within["by_model"][name]["passed"]
    ]
    p4_models = [
        name for name in healthy
        if cross_surface["by_model"][name]["identity_gate_passed"]
    ]
    p5_models = [
        name for name in healthy
        if cross_surface["by_model"][name]["gram_gate_passed"]
    ]
    p6_models = [
        name for name in healthy if common["by_model"][name]["passed"]
    ]
    p7_rows = [
        row for row in cross_model["rows"]
        if row["healthy_pair"] and row["passed"]
    ]
    p8_models = [
        name for name in healthy if physical["by_model"][name]["passed"]
    ]
    predictions = {
        "P1": {
            "passed": bool(authorization["hidden_scan_authorized"]),
        },
        "P2": {
            "passed": len(p2_models) >= minimum_models,
            "passing_models": p2_models,
        },
        "P3": {
            "passed": len(p3_models) >= minimum_models,
            "passing_models": p3_models,
        },
        "P4": {
            "passed": len(p4_models) >= minimum_models,
            "passing_models": p4_models,
        },
        "P5": {
            "passed": len(p5_models) >= minimum_models,
            "passing_models": p5_models,
        },
        "P6": {
            "passed": len(p6_models) >= minimum_models,
            "passing_models": p6_models,
        },
        "P7": {
            "passed": len(p7_rows) >= 2,
            "passing_directed_pairs": len(p7_rows),
            "rows": p7_rows,
        },
        "P8": {
            "passed": len(p8_models) >= minimum_models,
            "passing_models": p8_models,
        },
    }
    passed = [name for name, row in predictions.items() if row["passed"]]
    failed = [name for name, row in predictions.items() if not row["passed"]]
    semantic_surface_evidence = all(
        predictions[name]["passed"] for name in ("P1", "P2", "P3", "P4", "P5")
    )
    if semantic_surface_evidence:
        decision = (
            "retain_first_cross_surface_color_pair_geometry; "
            "next_require_natural_context_and_nontranslation_controls"
        )
    else:
        decision = (
            "cross_surface_pair_geometry_not_confirmed; retain_lexical_conditional_map"
        )

    outputs = (
        ("within_route_pair_audit.json", "phase1091_within_route_pair_audit.v1", within, "within_route_digest"),
        ("cross_surface_pair_audit.json", "phase1091_cross_surface_pair_audit.v1", cross_surface, "cross_surface_pair_digest"),
        ("cross_surface_common_field.json", "phase1091_cross_surface_common_field.v1", common, "common_field_digest"),
        ("cross_model_geometry.json", "phase1091_cross_model_geometry.v1", cross_model, "cross_model_digest"),
        ("physical_band_map.json", "phase1091_physical_band_map.v1", physical, "physical_band_digest"),
        ("projection_audit.json", "phase1091_projection_audit.v1", projection, "projection_digest"),
        ("numeric_audit.json", "phase1091_numeric_audit.v1", numeric, "numeric_digest"),
    )
    for filename, schema, payload, digest_key in outputs:
        write_output(
            root, filename, schema, payload, digest_key,
            prereg["protocol_digest"],
        )

    automatic = {
        "schema_version": "phase1091_automatic_next.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "decision": decision,
        "semantic_surface_evidence": semantic_surface_evidence,
        "local_causal_authorized": False,
        "automatic_hidden_extension_authorized": False,
        "reason": (
            "Even a passing cross-surface map must be separated from generic "
            "equality, translation, and mixed-prompt routing before component "
            "or neuron causality. The next protocol is a major redesign, not "
            "an automatic scope extension."
        ),
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)
    protocol.write_json(root / "automatic_next.json", automatic)

    summary = {
        "schema_version": "phase1091_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "behavior_healthy_models": sorted(behavior_healthy),
        "numeric_healthy_models": sorted(healthy),
        "predictions": predictions,
        "passed_predictions": passed,
        "failed_predictions": failed,
        "decision": decision,
        "models": {
            name: {
                "summary_digest": data["summary"]["summary_digest"],
                "npz_sha256": data["npz_sha256"],
                "candidate_accuracy": data["summary"]["candidate_accuracy"],
                "candidate_finite_fraction": data["summary"]["candidate_finite_fraction"],
                "hidden_finite_fraction": data["summary"]["hidden_finite_fraction_lower_bound"],
                "event_count": data["summary"]["event_count"],
                "confirmatory": name in healthy,
            }
            for name, data in models.items()
        },
        "interpretation": [
            "Passing direct pair retrieval means canonical color-pair identity survived a change of lexical surface within the controlled dossier task.",
            "Passing pair Gram means the relations among the eight pair fingerprints, not necessarily their raw directions, repeated across surfaces.",
            "Neither result alone separates color semantics from generic equality, translation, or prompt routing.",
            "GLM4 remains exploratory because its Phase1090 and Phase1091 candidate finite fractions failed the numeric gate.",
            "No causal component or neuron localization is authorized.",
        ],
        "automatic_next_digest": automatic["automatic_next_digest"],
    }
    summary["summary_digest"] = protocol.digest(summary)
    protocol.write_json(root / "final_summary.json", summary)
    print({
        "phase": protocol.PHASE,
        "healthy_models": sorted(healthy),
        "passed": passed,
        "failed": failed,
        "decision": decision,
        "summary_digest": summary["summary_digest"],
    })


if __name__ == "__main__":
    main()
