#!/usr/bin/env python3
"""Select the common Phase1071 prompt from frozen behavior-only metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1071_behavior_calibration_protocol as protocol


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
        raise RuntimeError("Phase1071 calibration protocol audit failed")
    summaries = {
        model: protocol.read_json(
            protocol.CALIBRATION_ROOT
            / "atlas"
            / model
            / "summary.json"
        )
        for model in protocol.MODELS
    }
    if any(
        row["protocol_digest"] != prereg["protocol_digest"]
        for row in summaries.values()
    ):
        raise RuntimeError("Phase1071 calibration digest drift")

    rule = prereg["selection_rule"]
    candidates = []
    for style in protocol.PROMPT_STYLES:
        model_rows = []
        for model in protocol.MODELS:
            summary = summaries[model]
            row = summary["styles"][str(style)]
            minimum_relation = min(
                value["candidate_accuracy"]
                for value in row["relations"].values()
            )
            minimum_path_semantic = min(
                value["semantic_first_rate"]
                for value in row["paths"].values()
            )
            checks = {
                "candidate_accuracy": (
                    row["candidate_accuracy"]
                    >= rule["model_candidate_accuracy_min"]
                ),
                "semantic_first_rate": (
                    row["semantic_first_rate"]
                    >= rule["model_semantic_first_rate_min"]
                ),
                "minimum_relation_candidate": (
                    minimum_relation
                    >= rule["relation_candidate_accuracy_min"]
                ),
                "minimum_path_semantic": (
                    minimum_path_semantic
                    >= rule["path_semantic_first_rate_min"]
                ),
                "candidate_finite": (
                    summary["candidate_finite_rate"]
                    >= rule["candidate_finite_rate_min"]
                ),
            }
            model_rows.append({
                "model": model,
                "candidate_accuracy": row["candidate_accuracy"],
                "semantic_first_rate": row["semantic_first_rate"],
                "strict_name_only_rate": (
                    row["strict_name_only_rate"]
                ),
                "minimum_relation_candidate_accuracy": (
                    minimum_relation
                ),
                "minimum_path_semantic_first_rate": (
                    minimum_path_semantic
                ),
                "candidate_finite_rate": (
                    summary["candidate_finite_rate"]
                ),
                "checks": checks,
                "eligible": all(checks.values()),
            })
        eligible_count = sum(
            int(row["eligible"]) for row in model_rows
        )
        worst_semantic = min(
            row["semantic_first_rate"] for row in model_rows
        )
        worst_candidate = min(
            row["candidate_accuracy"] for row in model_rows
        )
        macro_semantic = sum(
            row["semantic_first_rate"] for row in model_rows
        ) / len(model_rows)
        macro_candidate = sum(
            row["candidate_accuracy"] for row in model_rows
        ) / len(model_rows)
        selection_key = (
            eligible_count,
            worst_semantic,
            worst_candidate,
            macro_semantic,
            macro_candidate,
            -style,
        )
        candidates.append({
            "prompt_style": style,
            "prompt_style_label": protocol.STYLE_LABELS[style],
            "model_evidence": model_rows,
            "eligible_model_count": eligible_count,
            "worst_model_semantic_first_rate": worst_semantic,
            "worst_model_candidate_accuracy": worst_candidate,
            "macro_semantic_first_rate": macro_semantic,
            "macro_candidate_accuracy": macro_candidate,
            "selection_key": list(selection_key),
        })

    selected = max(
        candidates,
        key=lambda row: tuple(row["selection_key"]),
    )
    calibration_gate = (
        selected["eligible_model_count"]
        >= rule["minimum_eligible_models"]
    )
    result = {
        "schema_version": "phase1071_calibration_selection.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selection_rule": rule,
        "candidate_styles": candidates,
        "selected_prompt_style": selected["prompt_style"],
        "selected_prompt_style_label": selected[
            "prompt_style_label"
        ],
        "selected_style_evidence": selected,
        "calibration_gate_passed": calibration_gate,
        "interpretation": (
            "The selected style maximizes the preregistered lexicographic "
            "behavior criterion on held-out names. It is an instrument "
            "choice and contains no hidden-state evidence."
        ),
    }
    protocol.write_json(
        protocol.CALIBRATION_ROOT
        / "analysis"
        / "prompt_selection.json",
        result,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "selected_prompt_style": result["selected_prompt_style"],
        "selected_prompt_style_label": (
            result["selected_prompt_style_label"]
        ),
        "eligible_model_count": selected["eligible_model_count"],
        "calibration_gate_passed": calibration_gate,
    }), flush=True)


if __name__ == "__main__":
    main()
