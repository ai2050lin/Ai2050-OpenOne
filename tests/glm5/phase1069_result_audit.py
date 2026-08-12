#!/usr/bin/env python3
"""Strictly audit Phase1069 protocol, model outputs, and decisions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1069_local_coordinate_protocol as protocol


def strict_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def read_json(path: Path) -> Any:
    return strict_loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = strict_loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not an object")
        result.append(value)
    return result


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def close(left: float, right: float) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = read_json(root / "protocol" / "preregistration.json")
    protocol_payload = dict(prereg)
    recorded_digest = protocol_payload.pop("protocol_digest")
    recomputed_digest = protocol.digest(protocol_payload)
    protocol_audit = read_json(root / "protocol" / "audit.json")
    aggregate = read_json(root / "aggregate.json")
    automatic_next = read_json(
        root / "analysis" / "automatic_next.json"
    )
    model_gates = read_jsonl(
        root / "analysis" / "model_operation_gates.jsonl"
    )
    fingerprints = read_json(
        root / "analysis" / "relation_fingerprints.json"
    )
    cross_profiles = read_jsonl(
        root
        / "analysis"
        / "cross_model_operation_profiles.jsonl"
    )
    relation_evidence = read_jsonl(
        root / "analysis" / "relation_evidence.jsonl"
    )
    posthoc_controls = read_json(
        root
        / "analysis"
        / "posthoc_task_and_selection_controls.json"
    )

    model_checks = {}
    model_details = {}
    for model in protocol.MODELS:
        atlas = root / "atlas" / model
        summary = read_json(atlas / "summary.json")
        candidates = read_jsonl(
            atlas / "candidate_behavior.jsonl"
        )
        natural = read_jsonl(
            atlas / "natural_generation_audit.jsonl"
        )
        responses = read_jsonl(
            atlas / "response_metrics.jsonl"
        )
        readouts = read_jsonl(
            atlas / "local_readout_metrics.jsonl"
        )
        expected_events = int(summary["event_count"])
        expected_response_rows = (
            len(protocol.RELATION_NAMES)
            * len(protocol.SPLITS)
            * 2
            * expected_events
            * len(protocol.CAPTURE_ROLES)
            * 2
        )
        expected_readout_rows = (
            len(protocol.RELATION_NAMES)
            * len(protocol.SPLITS)
            * 2
            * len(protocol.QUERY_TYPES)
            * expected_events
            * 2
        )
        relation_details = {}
        relation_consistent = True
        for relation in protocol.RELATION_NAMES:
            relation_candidates = [
                row for row in candidates
                if row["relation"] == relation
            ]
            relation_natural = [
                row for row in natural
                if row["relation"] == relation
            ]
            candidate_rate = (
                sum(
                    bool(row["candidate_hit"])
                    for row in relation_candidates
                ) / len(relation_candidates)
            )
            semantic_first_rate = (
                sum(
                    bool(row["semantic_first"])
                    for row in relation_natural
                ) / len(relation_natural)
            )
            strict_rate = (
                sum(
                    bool(row["strict_name_only"])
                    for row in relation_natural
                ) / len(relation_natural)
            )
            recorded = summary["relations"][relation]
            consistent = (
                close(
                    candidate_rate,
                    recorded["candidate_first_token_accuracy"],
                )
                and close(
                    semantic_first_rate,
                    recorded["semantic_first_natural_rate"],
                )
                and close(
                    strict_rate,
                    recorded["strict_name_only_rate"],
                )
            )
            relation_consistent = relation_consistent and consistent
            relation_details[relation] = {
                "candidate_case_count": len(relation_candidates),
                "candidate_accuracy_recomputed": candidate_rate,
                "natural_case_count": len(relation_natural),
                "semantic_first_rate_recomputed": semantic_first_rate,
                "strict_rate_recomputed": strict_rate,
                "summary_consistent": consistent,
                "strong_behavior_gate_passed": bool(
                    recorded["strong_behavior_gate_passed"]
                ),
            }
        checks = {
            "summary_protocol_digest_matches": (
                summary["protocol_digest"] == recorded_digest
            ),
            "case_count_is_2400": (
                len(candidates)
                == int(prereg["case_count_per_model"])
                == int(summary["case_count"])
            ),
            "natural_case_count_is_500": (
                len(natural)
                == len(protocol.RELATION_NAMES)
                * int(prereg["natural_audit_per_relation"])
            ),
            "response_row_count_matches_schema": (
                len(responses) == expected_response_rows
            ),
            "readout_row_count_matches_schema": (
                len(readouts) == expected_readout_rows
            ),
            "relation_summary_recomputed": relation_consistent,
            "fp16_parameters_only": (
                summary["precision"]["has_fp16_parameters"]
                and not summary["precision"]["has_bf16_parameters"]
                and not summary["precision"][
                    "has_quantized_modules"
                ]
            ),
            "all_model_json_values_finite_or_null": finite_tree({
                "summary": summary,
                "candidates": candidates,
                "natural": natural,
                "responses": responses,
                "readouts": readouts,
            }),
        }
        model_checks[model] = checks
        model_details[model] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "relations": relation_details,
            "row_counts": {
                "candidate_behavior": len(candidates),
                "natural_generation_audit": len(natural),
                "response_metrics": len(responses),
                "local_readout_metrics": len(readouts),
            },
            "elapsed_seconds": summary["elapsed_seconds"],
            "nonfinite_candidate_count": summary[
                "nonfinite_candidate_count"
            ],
        }

    passed_gate_models = [
        row["model"] for row in model_gates
        if row["shared_order_operation_gate_passed"]
    ]
    expected_continue = (
        len(passed_gate_models)
        >= int(prereg["gates"]["minimum_repeated_models"])
    )
    global_checks = {
        "protocol_digest_recomputed": (
            recomputed_digest == recorded_digest
        ),
        "protocol_audit_passed": (
            protocol_audit["all_checks_passed"]
        ),
        "aggregate_protocol_digest_matches": (
            aggregate["protocol_digest"] == recorded_digest
        ),
        "all_model_checks_passed": all(
            all(checks.values())
            for checks in model_checks.values()
        ),
        "relation_evidence_row_count_is_15": (
            len(relation_evidence)
            == len(protocol.MODELS)
            * len(protocol.RELATION_NAMES)
        ),
        "model_gate_row_count_is_3": (
            len(model_gates) == len(protocol.MODELS)
        ),
        "cross_model_profile_row_count_is_3": (
            len(cross_profiles) == 3
        ),
        "fingerprint_models_complete": (
            set(fingerprints["models"]) == set(protocol.MODELS)
        ),
        "posthoc_control_models_complete": (
            set(posthoc_controls["models"]) == set(protocol.MODELS)
        ),
        "automatic_next_recomputed": (
            bool(automatic_next["should_continue_automatically"])
            == expected_continue
            and set(automatic_next["selected_models"])
            == set(passed_gate_models)
        ),
        "all_analysis_json_values_finite_or_null": finite_tree({
            "aggregate": aggregate,
            "automatic_next": automatic_next,
            "model_gates": model_gates,
            "fingerprints": fingerprints,
            "cross_profiles": cross_profiles,
            "relation_evidence": relation_evidence,
            "posthoc_controls": posthoc_controls,
        }),
    }
    result = {
        "schema_version": "phase1069_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": recorded_digest,
        "recomputed_protocol_digest": recomputed_digest,
        "all_integrity_checks_passed": (
            all(global_checks.values())
            and all(
                detail["all_checks_passed"]
                for detail in model_details.values()
            )
        ),
        "global_checks": global_checks,
        "model_checks": model_checks,
        "models": model_details,
        "automatic_next": automatic_next,
    }
    protocol.write_json(
        root / "analysis" / "integrity_audit.json",
        result,
    )
    if not result["all_integrity_checks_passed"]:
        raise RuntimeError(
            f"Phase1069 integrity audit failed: {result}"
        )
    print(json.dumps({
        "phase": protocol.PHASE,
        "all_integrity_checks_passed": True,
        "automatic_next": automatic_next,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
