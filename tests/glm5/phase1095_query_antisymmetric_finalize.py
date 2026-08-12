#!/usr/bin/env python3
"""Finalize Phase1095 query-antisymmetric relation-transport analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1094_semantic_topology_finalize as engine
import phase1095_query_antisymmetric_protocol as protocol


engine.protocol = protocol


def interaction_strength(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Describe relative magnitudes after query-antisymmetric cancellation."""
    by_model: dict[str, Any] = {}
    for model_name, data in models.items():
        values = data["relative_mean"]
        role_indices = [
            protocol.CAPTURE_ROLES.index("query_end"),
            protocol.CAPTURE_ROLES.index("answer_boundary"),
        ]
        fields = {}
        for field in protocol.SIGNED_FIELDS:
            field_index = protocol.SIGNED_FIELDS.index(field)
            selected = values[:, :, :, role_indices, field_index, :, :]
            finite = selected[np.isfinite(selected)]
            fields[field] = {
                "mean_relative_magnitude": float(np.mean(finite)),
                "median_relative_magnitude": float(np.median(finite)),
                "p95_relative_magnitude": float(np.quantile(finite, 0.95)),
                "maximum_relative_magnitude": float(np.max(finite)),
            }
        active = fields["active_binding"]["median_relative_magnitude"]
        null = fields["field_null"]["median_relative_magnitude"]
        content = fields["content"]["median_relative_magnitude"]
        by_model[model_name] = {
            "fields": fields,
            "null_to_active_median_ratio": float(null / max(active, 1e-12)),
            "content_to_active_median_ratio": float(content / max(active, 1e-12)),
        }
    return {
        "models": by_model,
        "interpretation": (
            "Descriptive magnitude audit only. A small null magnitude supports cancellation, "
            "but graph-specific matched-null advantages remain the formal test."
        ),
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    static_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    if not static_audit["all_checks_passed"] or not behavior["hidden_scan_authorized"]:
        raise RuntimeError("Phase1095 protocol or inherited behavior gate failed")

    models = engine.load_models()
    hidden = engine.hidden_audit(models, behavior)
    identity = engine.edge_identity_analysis(models, behavior)
    semantic = engine.semantic_topology_analysis(models, behavior)
    cross_model = engine.cross_model_analysis(models, behavior)
    physical = engine.physical_map(models)
    strength = interaction_strength(models)

    authorized = set(behavior["authorized_models"])
    attribute_rows = {
        model: semantic["models"][model]["attributes"][protocol.PRIMARY_ATTRIBUTE]
        for model in authorized
    }
    p5_models = sorted(
        model for model, row in attribute_rows.items() if row["P5_passed"]
    )
    p6_models = sorted(
        model for model, row in attribute_rows.items() if row["P6_passed"]
    )
    p7_models = sorted(set(p5_models) & set(p6_models))
    minimum_models = int(protocol.EVIDENCE_THRESHOLDS["minimum_query_interaction_models"])
    predictions = {
        "P1": {
            "passed": bool(static_audit["all_checks_passed"]),
            "criterion": "source provenance and query-antisymmetric algebra audits",
        },
        "P2": {
            "passed": bool(behavior["hidden_scan_authorized"]),
            "authorized_models": sorted(authorized),
        },
        "P3": {
            "passed": bool(hidden["passed"]),
            "passing_models": hidden["passing_models"],
        },
        "P4": {
            "passed": bool(identity["passed"]),
            "passing_models": identity["passing_models"],
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
            "passed": len(p7_models) >= minimum_models,
            "passing_models": p7_models,
        },
        "P8": {
            "passed": bool(cross_model["passed"]),
            "passing_directed_pairs": cross_model["passing_directed_pairs"],
        },
    }
    predictions["P9"] = {
        "passed": all(predictions[f"P{index}"]["passed"] for index in range(1, 9)),
        "criterion": "all prior gates pass before physical promotion",
    }

    query_conditioned_candidate = bool(
        predictions["P5"]["passed"]
        or predictions["P6"]["passed"]
        or predictions["P4"]["passed"]
    )
    if query_conditioned_candidate:
        decision = "independent_new_lexical_replication_required_before_causal_localization"
        automatic_next_required = True
        automatic_next_reason = (
            "A query-conditioned relation candidate survived cancellation and requires "
            "new lexical material before any causal localization."
        )
    else:
        decision = "phase1094_semantic_graph_is_not_query_conditioned_retain_only_descriptive_representation_geometry"
        automatic_next_required = False
        automatic_next_reason = (
            "The focused cancellation gate is decisive for this candidate; another nearby "
            "variant would repeat the same loop. The next work should be a separately designed "
            "language-family map, not an automatic micro-phase."
        )

    source_final = protocol.read_json(
        protocol.SOURCE_ROOT / "analysis" / "final_summary.json"
    )
    result = {
        "schema_version": "phase1095_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "behavior_digest": behavior["summary_digest"],
        "source_phase1094_summary_digest": source_final["summary_digest"],
        "predictions": predictions,
        "hidden_audit": hidden,
        "query_antisymmetric_edge_identity": identity,
        "query_antisymmetric_semantic_topology": semantic,
        "semantic_topology_compact": engine.compact_semantic(semantic),
        "query_interaction_strength": strength,
        "cross_model": cross_model,
        "physical_map": physical,
        "source_phase1094_compact": source_final["semantic_topology_compact"],
        "decision": decision,
        "automatic_next_required": automatic_next_required,
        "automatic_next_reason": automatic_next_reason,
        "causal_authorized": False,
        "theory_status": {
            "generic_directed_binding_skeleton": "retained",
            "phase1094_size_semantic_graph": (
                "query-conditioned candidate" if query_conditioned_candidate
                else "descriptive lexical/semantic representation geometry only"
            ),
            "specific_relative_concept_edge_code": "not established",
            "complete_language_code": "not established",
            "new_mathematics_required": False,
        },
        "hard_limits": [
            "The contrast isolates a query-by-binding interaction but remains observational.",
            "The same Phase1094 aliases are reused intentionally; independent lexical replication has not occurred here.",
            "Researcher-defined synonyms are approximate and can preserve distributional rather than conceptual relations.",
            "Only one semantic family, two graph topologies, two languages, and binary judgments are tested.",
            "No result closes rare-word meaning, punctuation, translation, contrast, grammar, reasoning, or the global knowledge network.",
        ],
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", result)
    print({
        "phase": protocol.PHASE,
        "predictions": {key: value["passed"] for key, value in predictions.items()},
        "semantic_compact": result["semantic_topology_compact"],
        "decision": decision,
        "automatic_next_required": automatic_next_required,
        "summary_digest": result["summary_digest"],
    })


if __name__ == "__main__":
    main()
