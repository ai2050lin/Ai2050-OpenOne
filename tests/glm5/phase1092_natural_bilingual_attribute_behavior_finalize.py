#!/usr/bin/env python3
"""Freeze Phase1092 behavior authorization before hidden-state collection."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    static = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    threshold = prereg["evidence_thresholds"]
    minimum_accuracy = float(threshold["minimum_candidate_accuracy"])
    minimum_finite = float(threshold["minimum_candidate_finite_fraction"])
    minimum_generation = float(threshold["minimum_generation_accuracy"])
    minimum_worlds = int(threshold["minimum_behavior_worlds_per_pair"])
    minimum_pairs = int(threshold["minimum_behavior_pairs_per_attribute"])
    minimum_attributes = int(threshold["minimum_behavior_attributes"])
    minimum_models = int(threshold["minimum_behavior_models"])

    models = {}
    authorized_models = []
    for model_name in protocol.MODELS:
        raw = protocol.read_json(protocol.OUT_ROOT / "pilot" / f"{model_name}.json")
        attributes = {}
        passing_attributes = []
        for attribute in protocol.ATTRIBUTES:
            operations = [
                value for value in protocol.OPERATIONS
                if value.startswith(f"{attribute}_")
            ]
            surfaces = {}
            for surface in protocol.SURFACES:
                pair_rows = {}
                passing_pairs = []
                for operation in operations:
                    panel_worlds = {}
                    for panel in protocol.PANELS:
                        passing_worlds = []
                        for world in protocol.BASE_WORLDS:
                            cell = raw["per_cell"][
                                f"{attribute}__{operation}__{surface}__{world}__{panel}"
                            ]
                            if (
                                float(cell["accuracy"]) >= minimum_accuracy
                                and float(cell["finite_fraction"]) >= minimum_finite
                            ):
                                passing_worlds.append(world)
                        panel_worlds[panel] = passing_worlds
                    pair_passed = all(
                        len(panel_worlds[panel]) >= minimum_worlds
                        for panel in protocol.PANELS
                    )
                    if pair_passed:
                        passing_pairs.append(operation)
                    pair_rows[operation] = {
                        "passing_worlds_by_panel": panel_worlds,
                        "passed": pair_passed,
                    }
                generation = raw["generation_by_attribute_surface"][
                    f"{attribute}__{surface}"
                ]
                generation_passed = all(
                    float(generation[panel]["target_before_distractor_accuracy"])
                    >= minimum_generation
                    for panel in protocol.PANELS
                )
                surface_passed = (
                    len(passing_pairs) >= minimum_pairs and generation_passed
                )
                surfaces[surface] = {
                    "passing_pairs": passing_pairs,
                    "passing_pair_count": len(passing_pairs),
                    "pairs": pair_rows,
                    "generation": generation,
                    "generation_passed": generation_passed,
                    "passed": surface_passed,
                }
            attribute_passed = all(
                surfaces[surface]["passed"] for surface in protocol.SURFACES
            )
            if attribute_passed:
                passing_attributes.append(attribute)
            attributes[attribute] = {
                "surfaces": surfaces,
                "passed": attribute_passed,
            }
        precision_ok = (
            raw["precision"]["has_fp16_parameters"]
            and not raw["precision"]["has_bf16_parameters"]
            and not raw["precision"]["has_quantized_modules"]
        )
        numerical_ok = float(raw["candidate_finite_fraction"]) >= minimum_finite
        model_authorized = (
            precision_ok
            and numerical_ok
            and len(passing_attributes) >= minimum_attributes
        )
        if model_authorized:
            authorized_models.append(model_name)
        models[model_name] = {
            "precision_ok": precision_ok,
            "numerical_ok": numerical_ok,
            "candidate_finite_fraction": raw["candidate_finite_fraction"],
            "passing_attributes": passing_attributes,
            "passing_attribute_count": len(passing_attributes),
            "attributes": attributes,
            "model_authorized": model_authorized,
            "elapsed_seconds": raw["elapsed_seconds"],
            "result_digest": raw["result_digest"],
        }

    predictions = {
        "P1": {"passed": bool(static["all_checks_passed"])},
        "P2": {
            "passed": all(row["precision_ok"] for row in models.values()),
            "criterion": "all three models loaded as FP16 without quantization",
        },
        "P3": {
            "passed": len(authorized_models) >= minimum_models,
            "authorized_models": authorized_models,
            "minimum_models": minimum_models,
        },
    }
    hidden_scan_authorized = all(row["passed"] for row in predictions.values())
    result = {
        "schema_version": "phase1092_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "predictions": predictions,
        "models": models,
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_scan_authorized,
        "decision": (
            "run_phase1092_signed_hidden_scan"
            if hidden_scan_authorized
            else "stop_before_hidden_state_collection"
        ),
        "causal_authorized": False,
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result
    )
    print({
        "phase": protocol.PHASE,
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_scan_authorized,
        "decision": result["decision"],
        "summary_digest": result["summary_digest"],
    })


if __name__ == "__main__":
    main()
