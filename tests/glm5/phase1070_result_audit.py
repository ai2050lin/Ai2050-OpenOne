#!/usr/bin/env python3
"""Strictly audit every formal Phase1070 JSON result."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1070_process_answer_protocol as protocol


def strict_loads(text: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(text, parse_constant=reject)


def read_json(path: Path) -> Any:
    return strict_loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = strict_loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not an object"
                )
            rows.append(value)
    return rows


def finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = read_json(root / "protocol" / "preregistration.json")
    protocol_copy = dict(prereg)
    recorded_digest = protocol_copy.pop("protocol_digest")
    recomputed_digest = protocol.digest(protocol_copy)
    protocol_audit = read_json(root / "protocol" / "audit.json")
    relation_evidence = read_jsonl(
        root / "analysis" / "relation_evidence.jsonl"
    )
    model_gates = read_jsonl(
        root / "analysis" / "model_gates.jsonl"
    )
    cross_model = read_jsonl(
        root
        / "analysis"
        / "cross_model_process_profiles.jsonl"
    )
    automatic_next = read_json(
        root / "analysis" / "automatic_next.json"
    )
    atlas_summary = read_json(
        root / "analysis" / "atlas_summary.json"
    )
    posthoc = read_json(
        root / "analysis" / "posthoc_diagnostics.json"
    )

    model_checks = {}
    model_payloads = {}
    for model in protocol.MODELS:
        atlas = root / "atlas" / model
        summary = read_json(atlas / "summary.json")
        behavior = read_jsonl(atlas / "candidate_behavior.jsonl")
        natural = read_jsonl(
            atlas / "natural_generation_audit.jsonl"
        )
        responses = read_jsonl(atlas / "response_metrics.jsonl")
        readouts = read_jsonl(
            atlas / "local_readout_metrics.jsonl"
        )
        event_count = int(summary["event_count"])
        expected_response_count = (
            len(protocol.RELATION_NAMES)
            * len(protocol.SPLITS)
            * len(protocol.QUERY_TYPES)
            * event_count
            * len(protocol.CAPTURE_ROLES)
            * 2
        )
        expected_readout_count = (
            len(protocol.RELATION_NAMES)
            * len(protocol.SPLITS)
            * len(protocol.QUERY_TYPES)
            * event_count
            * 2
        )
        relation_recomputed = {}
        relation_consistent = True
        for relation in protocol.RELATION_NAMES:
            relation_behavior = [
                row for row in behavior
                if row["relation"] == relation
            ]
            relation_natural = [
                row for row in natural
                if row["relation"] == relation
            ]
            candidate_accuracy = (
                sum(
                    int(row["candidate_hit"])
                    for row in relation_behavior
                ) / len(relation_behavior)
            )
            semantic_rate = (
                sum(
                    int(row["semantic_first"])
                    for row in relation_natural
                ) / len(relation_natural)
            )
            strict_rate = (
                sum(
                    int(row["strict_name_only"])
                    for row in relation_natural
                ) / len(relation_natural)
            )
            recorded = summary["relations"][relation]
            consistent = (
                len(relation_behavior) == 768
                and len(relation_natural) == 96
                and close(
                    candidate_accuracy,
                    recorded["candidate_first_token_accuracy"],
                )
                and close(
                    semantic_rate,
                    recorded["semantic_first_natural_rate"],
                )
                and close(
                    strict_rate,
                    recorded["strict_name_only_rate"],
                )
            )
            relation_consistent = (
                relation_consistent and consistent
            )
            relation_recomputed[relation] = {
                "candidate_case_count": len(relation_behavior),
                "candidate_accuracy": candidate_accuracy,
                "natural_case_count": len(relation_natural),
                "semantic_first_rate": semantic_rate,
                "strict_rate": strict_rate,
                "summary_consistent": consistent,
            }

        checks = {
            "summary_protocol_digest_matches": (
                summary["protocol_digest"] == recorded_digest
            ),
            "case_count_is_3840": (
                len(behavior)
                == int(prereg["case_count_per_model"])
                == int(summary["case_count"])
            ),
            "natural_case_count_is_480": (
                len(natural)
                == int(prereg["natural_audit_per_model"])
            ),
            "response_row_count_matches_schema": (
                len(responses) == expected_response_count
            ),
            "readout_row_count_matches_schema": (
                len(readouts) == expected_readout_count
            ),
            "relation_summary_recomputed": relation_consistent,
            "fp16_parameters_only": (
                summary["precision"]["has_fp16_parameters"]
                and not summary["precision"][
                    "has_bf16_parameters"
                ]
                and not summary["precision"][
                    "has_quantized_modules"
                ]
            ),
            "finite_rates_in_unit_interval": all(
                0.0 <= float(summary[key]) <= 1.0
                for key in (
                    "candidate_finite_rate",
                    "residual_metric_finite_rate",
                    "internal_readout_finite_rate",
                )
            ),
            "all_model_json_values_finite_or_null": all(
                finite_tree(value)
                for value in (
                    summary,
                    behavior,
                    natural,
                    responses,
                    readouts,
                )
            ),
        }
        model_checks[model] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "relations": relation_recomputed,
            "row_counts": {
                "candidate_behavior": len(behavior),
                "natural_generation_audit": len(natural),
                "response_metrics": len(responses),
                "local_readout_metrics": len(readouts),
            },
            "elapsed_seconds": summary["elapsed_seconds"],
            "nonfinite_candidate_count": summary[
                "nonfinite_candidate_count"
            ],
            "nonfinite_residual_metric_count": summary[
                "nonfinite_residual_metric_count"
            ],
            "nonfinite_internal_readout_count": summary[
                "nonfinite_internal_readout_count"
            ],
        }
        model_payloads[model] = (
            summary,
            behavior,
            natural,
            responses,
            readouts,
        )

    recomputed_selected = [
        row["model"]
        for row in model_gates
        if row["process_model_gate_passed"]
    ]
    recomputed_continue = (
        len(recomputed_selected)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    global_checks = {
        "protocol_digest_recomputed": (
            recorded_digest == recomputed_digest
        ),
        "protocol_audit_passed": (
            protocol_audit["all_checks_passed"]
        ),
        "all_model_checks_passed": all(
            row["all_checks_passed"]
            for row in model_checks.values()
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
            len(cross_model)
            == len(list(itertools_combinations_count(
                len(protocol.MODELS), 2
            )))
        ),
        "automatic_next_recomputed": (
            bool(automatic_next["should_continue_automatically"])
            == recomputed_continue
            and automatic_next["selected_models"]
            == recomputed_selected
        ),
        "atlas_summary_protocol_digest_matches": (
            atlas_summary["protocol_digest"] == recorded_digest
        ),
        "embedding_control_recorded_for_all_relations": all(
            row["discovery"]["embedding_process_did_max"]
            is not None
            and row["confirmation"]["embedding_process_did_max"]
            is not None
            for row in relation_evidence
        ),
        "all_analysis_json_values_finite_or_null": all(
            finite_tree(value)
            for value in (
                relation_evidence,
                model_gates,
                cross_model,
                automatic_next,
                atlas_summary,
                posthoc,
            )
        ),
    }
    payload = {
        "schema_version": "phase1070_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": recorded_digest,
        "recomputed_protocol_digest": recomputed_digest,
        "all_integrity_checks_passed": (
            all(global_checks.values())
            and all(
                row["all_checks_passed"]
                for row in model_checks.values()
            )
        ),
        "global_checks": global_checks,
        "model_checks": model_checks,
        "automatic_next": automatic_next,
    }
    protocol.write_json(
        root / "analysis" / "integrity_audit.json",
        payload,
    )
    print({
        "phase": protocol.PHASE,
        "all_integrity_checks_passed": payload[
            "all_integrity_checks_passed"
        ],
        "automatic_next": automatic_next,
    })
    if not payload["all_integrity_checks_passed"]:
        raise RuntimeError(f"Phase1070 integrity audit failed: {payload}")


def itertools_combinations_count(n: int, r: int):
    # Keep the expected-row expression explicit without importing a full
    # result iterator into the audit payload.
    import itertools

    return itertools.combinations(range(n), r)


if __name__ == "__main__":
    main()
