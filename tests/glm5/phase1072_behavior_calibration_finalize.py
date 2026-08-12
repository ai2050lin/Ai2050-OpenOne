#!/usr/bin/env python3
"""Finalize held-out exact-prompt behavior calibration for Phase1072."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1072_behavior_calibration_protocol as protocol
import phase1072_bidirectional_pattern_protocol as formal


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    prereg = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "protocol"
        / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.CALIBRATION_ROOT
        / "protocol"
        / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1072 calibration audit failed")

    rows: list[dict[str, Any]] = []
    model_rows = []
    for model in protocol.MODELS:
        summary = protocol.read_json(
            protocol.CALIBRATION_ROOT
            / "atlas"
            / model
            / "summary.json"
        )
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(
                f"Phase1072 calibration digest drift: {model}"
            )
        condition_metrics = summary["styles"]["0"]["relations"]
        target_pass_by_relation: dict[str, list[bool]] = defaultdict(
            list
        )
        for condition in protocol.RELATION_NAMES:
            parsed = formal.parse_condition(condition)
            metrics = condition_metrics[condition]
            candidate = float(metrics["candidate_accuracy"])
            semantic = float(metrics["semantic_first_rate"])
            passed = bool(
                candidate
                >= formal.GATES[
                    "calibration_candidate_accuracy_min"
                ]
                and semantic
                >= formal.GATES[
                    "calibration_semantic_first_rate_min"
                ]
            )
            row = {
                "schema_version": (
                    "phase1072_calibration_condition.v1"
                ),
                "phase": formal.PHASE,
                "model": model,
                "condition": condition,
                **parsed,
                "candidate_count": int(
                    metrics["candidate_count"]
                ),
                "candidate_accuracy": candidate,
                "semantic_first_rate": semantic,
                "condition_behavior_gate_passed": passed,
            }
            rows.append(row)
            if parsed["task_family"] == "transitive":
                target_pass_by_relation[
                    parsed["base_relation"]
                ].append(passed)
        fully_calibrated_relations = [
            relation
            for relation, values in target_pass_by_relation.items()
            if len(values) == (
                len(formal.PROMPT_BRANCHES)
                * len(formal.EVIDENCE_ORDERS)
            )
            and all(values)
        ]
        finite_passed = bool(
            float(summary["candidate_finite_rate"])
            >= formal.GATES["candidate_finite_rate_min"]
        )
        model_passed = bool(
            finite_passed
            and len(fully_calibrated_relations)
            >= formal.GATES[
                "minimum_strong_relations_per_model"
            ]
        )
        model_rows.append({
            "schema_version": (
                "phase1072_calibration_model_gate.v1"
            ),
            "phase": formal.PHASE,
            "model": model,
            "candidate_finite_rate": float(
                summary["candidate_finite_rate"]
            ),
            "numerical_gate_passed": finite_passed,
            "fully_calibrated_target_relations": (
                fully_calibrated_relations
            ),
            "fully_calibrated_target_relation_count": len(
                fully_calibrated_relations
            ),
            "calibration_model_gate_passed": model_passed,
        })

    repeated_models = [
        row["model"]
        for row in model_rows
        if row["calibration_model_gate_passed"]
    ]
    result = {
        "schema_version": "phase1072_calibration_summary.v1",
        "phase": formal.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "condition_rows": rows,
        "model_gates": model_rows,
        "repeated_model_count": len(repeated_models),
        "selected_models": repeated_models,
        "calibration_gate_passed": (
            len(repeated_models)
            >= formal.GATES["minimum_repeated_models"]
        ),
        "pooled_target_candidate_accuracy": mean([
            float(row["candidate_accuracy"])
            for row in rows
            if row["task_family"] == "transitive"
        ]),
        "pooled_target_semantic_first_rate": mean([
            float(row["semantic_first_rate"])
            for row in rows
            if row["task_family"] == "transitive"
        ]),
        "interpretation": (
            "This is behavior-only instrument calibration. It reads no "
            "hidden state and does not select a mechanism or delete any "
            "frozen prompt branch."
        ),
    }
    analysis = protocol.CALIBRATION_ROOT / "analysis"
    protocol.write_jsonl(
        analysis / "condition_behavior.jsonl", rows
    )
    protocol.write_jsonl(
        analysis / "model_gates.jsonl", model_rows
    )
    protocol.write_json(
        analysis / "calibration_summary.json", result
    )
    print(
        f"Phase1072 calibration finalized: "
        f"models={repeated_models} "
        f"passed={result['calibration_gate_passed']}"
    )


if __name__ == "__main__":
    main()
