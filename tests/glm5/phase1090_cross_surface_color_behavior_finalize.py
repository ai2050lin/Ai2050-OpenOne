#!/usr/bin/env python3
"""Finalize Phase1090 behavior feasibility and automatic-next decision."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1090_cross_surface_color_behavior_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    static = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    threshold_accuracy = float(
        prereg["evidence_thresholds"]["minimum_candidate_accuracy"]
    )
    threshold_finite = float(
        prereg["evidence_thresholds"]["minimum_candidate_finite_fraction"]
    )
    threshold_generation = float(
        prereg["evidence_thresholds"]["minimum_generation_accuracy"]
    )
    minimum_worlds = int(
        prereg["evidence_thresholds"]["minimum_worlds_per_operation"]
    )
    minimum_operations = int(
        prereg["evidence_thresholds"]["minimum_operations_per_route"]
    )
    minimum_models = int(
        prereg["evidence_thresholds"]["minimum_models_per_route"]
    )
    models = {}
    route_models = {route: [] for route in protocol.SURFACE_ROUTES}
    for model_name in protocol.MODELS:
        row = protocol.read_json(
            protocol.OUT_ROOT / "pilot" / f"{model_name}.json"
        )
        route_rows = {}
        for route in protocol.SURFACE_ROUTES:
            passing_operations = []
            operation_rows = {}
            for operation in protocol.OPERATIONS:
                passing_by_panel = {}
                for panel in protocol.PANELS:
                    passing = []
                    for world in protocol.BASE_WORLDS:
                        cell = row["per_cell"][
                            f"{route}__{operation}__{world}__{panel}"
                        ]
                        if (
                            cell["accuracy"] >= threshold_accuracy
                            and cell["finite_fraction"] >= threshold_finite
                        ):
                            passing.append(world)
                    passing_by_panel[panel] = passing
                passed = all(
                    len(passing_by_panel[panel]) >= minimum_worlds
                    for panel in protocol.PANELS
                )
                if passed:
                    passing_operations.append(operation)
                operation_rows[operation] = {
                    "passing_worlds_by_panel": passing_by_panel,
                    "passed": passed,
                }
            generation_rows = row["generation_by_route"][route]
            generation_passed = all(
                generation_rows[panel]["target_before_distractor_accuracy"]
                >= threshold_generation
                for panel in protocol.PANELS
            )
            viable = (
                len(passing_operations) >= minimum_operations
                and generation_passed
                and row["candidate_finite_fraction"] >= threshold_finite
            )
            if viable:
                route_models[route].append(model_name)
            route_rows[route] = {
                "passing_operations": passing_operations,
                "passing_operation_count": len(passing_operations),
                "generation": generation_rows,
                "generation_passed": generation_passed,
                "viable": viable,
                "operations": operation_rows,
            }
        precision_ok = (
            row["precision"]["has_fp16_parameters"]
            and not row["precision"]["has_bf16_parameters"]
            and not row["precision"]["has_quantized_modules"]
        )
        models[model_name] = {
            "precision_ok": precision_ok,
            "candidate_finite_fraction": row["candidate_finite_fraction"],
            "routes": route_rows,
            "elapsed_seconds": row["elapsed_seconds"],
            "result_digest": row["result_digest"],
        }

    viable_routes = [
        route for route, names in route_models.items()
        if len(names) >= minimum_models
    ]
    viable_same = [
        route for route in viable_routes
        if route not in protocol.MIXED_SURFACE_ROUTES
    ]
    viable_mixed = [
        route for route in viable_routes
        if route in protocol.MIXED_SURFACE_ROUTES
    ]
    p1 = bool(static["all_checks_passed"])
    p2 = all(row["precision_ok"] for row in models.values())
    p3 = len(viable_same) >= 2
    p4 = len(viable_mixed) >= int(
        prereg["evidence_thresholds"]["minimum_viable_mixed_routes"]
    )
    admitted = viable_same + viable_mixed
    p5 = all(
        all(
            models[model]["routes"][route]["generation_passed"]
            for model in route_models[route]
        )
        for route in admitted
    )
    predictions = {
        "P1": {"passed": p1},
        "P2": {"passed": p2},
        "P3": {"passed": p3, "viable_same_surface_routes": viable_same},
        "P4": {"passed": p4, "viable_mixed_surface_routes": viable_mixed},
        "P5": {"passed": p5, "admitted_routes": admitted},
    }
    hidden_protocol_authorized = all(
        row["passed"] for row in predictions.values()
    )
    result = {
        "schema_version": "phase1090_behavior_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "predictions": predictions,
        "route_models": route_models,
        "models": models,
        "hidden_protocol_authorized": hidden_protocol_authorized,
        "selected_routes_for_phase1091": (
            ["en_en", "zh_zh", *viable_mixed]
            if hidden_protocol_authorized else []
        ),
        "decision": (
            "authorize_phase1091_cross_surface_signed_map"
            if hidden_protocol_authorized
            else "stop_before_cross_surface_hidden_state_collection"
        ),
        "causal_authorized": False,
    }
    result["summary_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json", result
    )
    print({
        "phase": protocol.PHASE,
        "viable_same": viable_same,
        "viable_mixed": viable_mixed,
        "hidden_protocol_authorized": hidden_protocol_authorized,
        "decision": result["decision"],
        "summary_digest": result["summary_digest"],
    })


if __name__ == "__main__":
    main()
