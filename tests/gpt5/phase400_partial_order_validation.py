#!/usr/bin/env python3
"""Validate the frozen Phase400 partial-order graph on untouched splits."""

from __future__ import annotations

import argparse
import json

from phase400_partial_order_common import (
    MODELS,
    OUT,
    assess_frozen_cell,
    crossmodel_surfaces,
    load_stage,
    now,
    prediction_assessment,
    read_json,
    write_json,
)


STAGES = ("calibration", "physical_holdout")


def main(stage: str) -> None:
    protocol = read_json(OUT / "phase400_partial_order_protocol.json")
    frozen = read_json(OUT / "phase400_partial_order_candidate_freeze.json")
    if stage == "calibration" and not frozen["authorization"]["run_calibration_trace"]:
        raise RuntimeError("Phase400 calibration is not authorized")
    calibration = None
    if stage == "physical_holdout":
        calibration = read_json(OUT / "phase400_partial_order_calibration.json")
        if not calibration["authorization"]["open_physical_holdout"]:
            raise RuntimeError("Phase400 physical holdout is not authorized")
    events, predictions, denominator = load_stage(stage)
    cells = []
    for frozen_cell in frozen["cells"]:
        model = frozen_cell["model"]
        surface = frozen_cell["surface"]
        cell_events = [
            row
            for row in events
            if row["model"] == model and row["surface_private"] == surface
        ]
        cell_predictions = [
            row
            for row in predictions
            if row["model"] == model and row["surface_private"] == surface
        ]
        cell = assess_frozen_cell(frozen_cell, cell_events, protocol)
        cell["prediction"] = prediction_assessment(
            cell,
            cell_events,
            cell_predictions,
            protocol,
            frozen_best_single_layer=frozen_cell["prediction"]["best_single_layer"][
                "frozen_layer_index"
            ],
        )
        cells.append(cell)
    crossmodel = crossmodel_surfaces(cells, protocol)
    discovery_by_surface = {
        row["surface"]: row for row in frozen["crossmodel_surfaces"]
    }
    calibration_by_surface = (
        {row["surface"]: row for row in calibration["crossmodel_surfaces"]}
        if calibration
        else {}
    )
    for row in crossmodel:
        row["discovery_crossmodel_pass"] = discovery_by_surface[row["surface"]][
            "crossmodel_functional_isomorphism_pass"
        ]
        row["calibration_crossmodel_pass"] = (
            row["crossmodel_functional_isomorphism_pass"]
            if stage == "calibration"
            else calibration_by_surface[row["surface"]][
                "crossmodel_functional_isomorphism_pass"
            ]
        )
        row["all_prior_stage_crossmodel_pass"] = bool(
            row["discovery_crossmodel_pass"]
            and row["calibration_crossmodel_pass"]
        )
    crossmodel_count = sum(
        row["crossmodel_functional_isomorphism_pass"] for row in crossmodel
    )
    prior_and_current = [
        row
        for row in crossmodel
        if row["all_prior_stage_crossmodel_pass"]
        and row["crossmodel_functional_isomorphism_pass"]
    ]
    causal_surfaces = [
        row
        for row in prior_and_current
        if row["all_three_prediction_pass"]
        and all(
            frozen_cell["prediction"]["prediction_pass"]
            for frozen_cell in frozen["cells"]
            if frozen_cell["surface"] == row["surface"]
        )
        and (
            stage != "physical_holdout"
            or all(
                calibration_cell["prediction"]["prediction_pass"]
                for calibration_cell in calibration["cells"]
                if calibration_cell["surface"] == row["surface"]
            )
        )
    ]
    result = {
        "schema_version": "74.7.0",
        "phase_id": f"Phase400-PartialOrder-{stage}-Validation",
        "created_at": now(),
        "stage": stage,
        "denominator": denominator,
        "cells": cells,
        "crossmodel_surfaces": crossmodel,
        "results": {
            "partial_order_graph_cell_count": sum(
                cell["partial_order_graph_pass"] for cell in cells
            ),
            "prediction_pass_cell_count": sum(
                cell["prediction"]["prediction_pass"] for cell in cells
            ),
            "model_surface_cell_count": len(cells),
            "crossmodel_isomorphism_surface_count": crossmodel_count,
            "all_stage_joint_gate_surface_count": len(causal_surfaces)
            if stage == "physical_holdout"
            else 0,
        },
        "authorization": {
            "open_physical_holdout": stage == "calibration"
            and denominator["all_collection_quality_gates_pass"]
            and bool(prior_and_current),
            "run_joint_causal_intervention": stage == "physical_holdout"
            and denominator["all_collection_quality_gates_pass"]
            and bool(causal_surfaces),
            "head_channel_or_neuron_scan": False,
        },
        "causal_candidate_surfaces": [row["surface"] for row in causal_surfaces],
        "freeze_audit": {
            "event_ids_reselected": False,
            "best_single_layers_reselected": False,
            "thresholds_changed": False,
        },
        "claim_boundary": {
            "validated_graph_is_causal": False,
            "prediction_is_natural_necessity": False,
            "failed_graph_means_no_dynamic_process": False,
        },
    }
    output = (
        "phase400_partial_order_calibration.json"
        if stage == "calibration"
        else "phase400_partial_order_physical.json"
    )
    write_json(OUT / output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    main(args.stage)

