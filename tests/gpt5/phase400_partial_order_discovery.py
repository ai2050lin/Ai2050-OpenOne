#!/usr/bin/env python3
"""Discover and freeze Phase400 event types and graph-readout baselines."""

from __future__ import annotations

import json

from phase400_partial_order_common import (
    MODELS,
    OUT,
    crossmodel_surfaces,
    load_stage,
    now,
    prediction_assessment,
    read_json,
    select_discovery_cell,
    write_json,
)


def main() -> None:
    protocol = read_json(OUT / "phase400_partial_order_protocol.json")
    instrument = read_json(OUT / "phase400_instrument_audit.json")
    behavior = read_json(OUT / "phase400_behavior_freeze_summary.json")
    if not instrument["authorization"]["run_discovery_trace"]:
        raise RuntimeError("Phase400 discovery was not authorized by the instrument audit")
    events, predictions, denominator = load_stage("discovery")
    cells = []
    for model in MODELS:
        for surface in behavior["eligible_surfaces"]:
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
            cell = select_discovery_cell(model, surface, cell_events, protocol)
            cell["prediction"] = prediction_assessment(
                cell, cell_events, cell_predictions, protocol
            )
            cells.append(cell)
    crossmodel = crossmodel_surfaces(cells, protocol)
    crossmodel_count = sum(
        row["crossmodel_functional_isomorphism_pass"] for row in crossmodel
    )
    result = {
        "schema_version": "74.6.0",
        "phase_id": "Phase400-PartialOrderDiscovery",
        "created_at": now(),
        "stage": "discovery",
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
        },
        "authorization": {
            "run_calibration_trace": denominator[
                "all_collection_quality_gates_pass"
            ]
            and any(cell["partial_order_graph_pass"] for cell in cells),
            "open_physical_holdout": False,
            "run_joint_causal_intervention": False,
            "head_channel_or_neuron_scan": False,
        },
        "freeze_contract": {
            "event_ids_frozen_now": True,
            "best_single_layers_frozen_now": True,
            "thresholds_changed_after_discovery": False,
            "calibration_or_holdout_used_for_selection": False,
        },
        "claim_boundary": {
            "discovery_graph_is_confirmed": False,
            "prediction_is_causal": False,
            "crossmodel_isomorphism_is_identical_neurons": False,
        },
    }
    write_json(OUT / "phase400_partial_order_discovery.json", result)
    write_json(OUT / "phase400_partial_order_candidate_freeze.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

