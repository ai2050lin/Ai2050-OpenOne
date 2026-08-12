#!/usr/bin/env python3
"""Offline descriptive phase audit for the frozen Phase1097 atlas."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1097_conditional_transition_protocol as protocol
import phase1097_conditional_transition_finalize as finalize


def finite_mean(value: np.ndarray, axes) -> np.ndarray:
    return np.nanmean(value, axis=axes)


def first_crossing(curve: np.ndarray, threshold: float) -> dict | None:
    for index, value in enumerate(curve):
        if math.isfinite(float(value)) and float(value) >= threshold:
            return {
                "anchor_index": index,
                "relative_depth": protocol.DEPTH_ANCHORS[index],
                "value": float(value),
            }
    return None


def curve_records(values: np.ndarray) -> list[dict]:
    return [
        {
            "anchor_index": index,
            "relative_depth": protocol.DEPTH_ANCHORS[index],
            "value": float(value) if math.isfinite(float(value)) else None,
        }
        for index, value in enumerate(values)
    ]


def model_diagnostic(model_name: str, data: dict) -> dict:
    role_index = {name: index for index, name in enumerate(protocol.CAPTURE_ROLES)}
    field_index = {name: index for index, name in enumerate(protocol.FIELDS)}
    roles = {}
    for role_name in ("task_cue", "query_end", "answer_boundary"):
        role = role_index[role_name]
        amplitude_curves = {
            field_name: finite_mean(data["amplitude"][:, :, :, field_index[field_name], role, :], (0, 1, 2))
            for field_name in (
                "relational_representation", "relational_control",
                "relational_execution", "relational_carrier",
                "lookup_execution", "lookup_carrier",
            )
        }
        local_margin_curves = {
            field_name: finite_mean(data["local_margin"][:, :, :, field_index[field_name], role, :], (0, 1, 2))
            for field_name in (
                "relational_control", "relational_execution", "relational_carrier",
                "lookup_execution", "lookup_carrier",
            )
        }
        panel_execution = finite_mean(data["panel_alignment"][:, :, :, 0, role, :], (0, 1, 2))
        panel_carrier = finite_mean(data["panel_alignment"][:, :, :, 1, role, :], (0, 1, 2))
        ledger_curves = {
            name: finite_mean(data["ledger_alignment"][:, :, :, index, role, :], (0, 1, 2))
            for index, name in enumerate(data["summary"]["ledger_alignment_kinds"])
        }
        roles[role_name] = {
            "relative_amplitude": {key: curve_records(value) for key, value in amplitude_curves.items()},
            "local_candidate_margin_interaction": {key: curve_records(value) for key, value in local_margin_curves.items()},
            "panel_execution_alignment": curve_records(panel_execution),
            "panel_carrier_alignment": curve_records(panel_carrier),
            "ledger_alignment": {key: curve_records(value) for key, value in ledger_curves.items()},
            "landmarks": {
                "panel_execution_alignment_first_0_5": first_crossing(panel_execution, 0.5),
                "panel_execution_alignment_first_0_8": first_crossing(panel_execution, 0.8),
                "execution_control_alignment_first_0_5": first_crossing(
                    ledger_curves["relational_execution_control"], 0.5
                ),
                "execution_representation_alignment_first_0_5": first_crossing(
                    ledger_curves["relational_execution_representation"], 0.5
                ),
                "execution_amplitude_peak": {
                    "anchor_index": int(np.nanargmax(amplitude_curves["relational_execution"])),
                    "relative_depth": protocol.DEPTH_ANCHORS[int(np.nanargmax(amplitude_curves["relational_execution"]))],
                    "value": float(np.nanmax(amplitude_curves["relational_execution"])),
                },
            },
        }
    return {
        "model": model_name,
        "behavior_formal": data["summary"]["behavior_formal"],
        "roles": roles,
        "physical_hotspots": finalize.physical_hotspots(data, limit=20),
    }


def main() -> None:
    final_summary = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    models = {
        model_name: model_diagnostic(model_name, finalize.load_model(model_name))
        for model_name in protocol.MODELS
    }
    result = {
        "schema_version": "phase1097_transition_diagnostic.v1",
        "phase": protocol.PHASE,
        "protocol_digest": final_summary["protocol_digest"],
        "final_summary_digest": final_summary["summary_digest"],
        "status": "posthoc_descriptive_only_no_gate_upgrade",
        "models": models,
    }
    result["diagnostic_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "transition_diagnostic.json", result)
    print({
        "phase": protocol.PHASE,
        "status": result["status"],
        "answer_boundary_landmarks": {
            model: models[model]["roles"]["answer_boundary"]["landmarks"]
            for model in protocol.MODELS
        },
        "diagnostic_digest": result["diagnostic_digest"],
    })


if __name__ == "__main__":
    main()
