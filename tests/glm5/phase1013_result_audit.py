#!/usr/bin/env python3
"""Audit remote-receiver and singleton head-morphology artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1011_native_semantic_protocol import (
    ANALYSIS_OPERATIONS,
    FAMILIES,
    MODELS,
    OUTPUT_MODES,
    read_json,
    read_jsonl,
    write_json,
)
from phase1013_head_response_morphology import (
    HEAD_DIRECTION_AXES,
    OP_INDEX,
    OUT_ROOT,
    PHASE,
    PHASE1012_ROOT,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_phase1012() -> dict[str, Any]:
    summary = read_json(PHASE1012_ROOT / "summary.json")
    events = read_jsonl(PHASE1012_ROOT / "receiver_events.jsonl")
    curves = read_jsonl(PHASE1012_ROOT / "receiver_curves.jsonl")
    selection = read_json(
        PHASE1012_ROOT / "discovery_selection.json"
    )
    require(
        float(summary["receiver_input_delta_at_residual_depth0"])
        == 0.0,
        "receiver depth-zero input is not invariant",
    )
    require(
        len(events) == int(summary["event_count"]),
        "Phase1012 event count drift",
    )
    require(
        len(curves) == int(summary["curve_count"]),
        "Phase1012 curve count drift",
    )
    require(
        selection["selection_used_confirmation"] is False,
        "Phase1012 confirmation leakage",
    )
    require(
        all(
            row["selection_used_confirmation"] is False
            for row in selection["selections"]
        ),
        "Phase1012 region selection leakage",
    )
    return {
        "event_count": len(events),
        "both_split_pass_count": int(
            summary["both_split_pass_count"]
        ),
        "curve_count": len(curves),
        "frozen_region_count": len(selection["selections"]),
        "selection_leakage": False,
    }


def audit_scans() -> list[dict[str, Any]]:
    results = []
    for model in MODELS:
        root = OUT_ROOT / "scan" / model
        summary = read_json(root / "summary.json")
        events = read_jsonl(root / "events.jsonl")
        require(
            summary["state_forward_mode"] == "singleton_8bit",
            f"{model}: not singleton",
        )
        require(
            summary["direction_axes"] == list(HEAD_DIRECTION_AXES),
            f"{model}: direction axes drift",
        )
        require(
            int(summary["model_forward_count"]) == 432 * 9,
            f"{model}: forward count drift",
        )
        require(
            float(summary["identity_maximum"]) == 0.0,
            f"{model}: identity drift",
        )
        scalar_count = 0
        for family in FAMILIES:
            for output_mode in OUTPUT_MODES:
                panel = root / family / output_mode
                panel_summary = read_json(panel / "summary.json")
                units = read_jsonl(panel / "units.jsonl")
                scalar = np.load(panel / "response_scalars.npz")
                direction = np.load(
                    panel / "direction_consistency.npz"
                )
                values = scalar["normalized_magnitude"]
                require(
                    values.shape == (
                        48,
                        len(ANALYSIS_OPERATIONS),
                        len(events),
                    ),
                    f"{model}/{family}/{output_mode}: scalar shape",
                )
                require(
                    np.all(np.isfinite(values)),
                    f"{model}/{family}/{output_mode}: nonfinite",
                )
                require(
                    np.all(
                        values[:, OP_INDEX["I"], :] == 0
                    ),
                    f"{model}/{family}/{output_mode}: identity",
                )
                require(
                    np.all(scalar["all_units_qualified"]),
                    f"{model}/{family}/{output_mode}: all-unit axis",
                )
                require(
                    direction["direction_consistency"].shape == (
                        len(HEAD_DIRECTION_AXES),
                        len(ANALYSIS_OPERATIONS),
                        2,
                        len(events),
                    ),
                    f"{model}/{family}/{output_mode}: direction shape",
                )
                require(
                    len(units) == 48,
                    f"{model}/{family}/{output_mode}: units",
                )
                require(
                    int(panel_summary["model_forward_count"]) == 48 * 9,
                    f"{model}/{family}/{output_mode}: forwards",
                )
                scalar_count += int(values.size)
                scalar.close()
                direction.close()
        require(
            scalar_count == int(summary["scalar_measurement_count"]),
            f"{model}: scalar count drift",
        )
        results.append({
            "model": model,
            "event_count": len(events),
            "model_forward_count": int(
                summary["model_forward_count"]
            ),
            "scalar_measurement_count": scalar_count,
            "identity_maximum": 0.0,
        })
    forbidden = [
        path for path in (OUT_ROOT / "scan").rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".safetensors"}
    ]
    require(not forbidden, f"raw hidden tensor files: {forbidden[:3]}")
    return results


def audit_final() -> dict[str, Any]:
    summary = read_json(OUT_ROOT / "summary.json")
    profiles = read_jsonl(OUT_ROOT / "head_profiles.jsonl")
    selections = read_jsonl(
        OUT_ROOT / "discovery_frozen_heads.jsonl"
    )
    require(
        len(profiles) == int(summary["profile_count"]),
        "Phase1013 profile count drift",
    )
    require(
        len(selections)
        == int(summary["discovery_frozen_head_count"]),
        "Phase1013 selection count drift",
    )
    require(
        summary["selection_used_confirmation"] is False,
        "Phase1013 summary leakage",
    )
    require(
        all(
            row["selection_used_confirmation"] is False
            for row in selections
        ),
        "Phase1013 head selection leakage",
    )
    profile_coordinates = {
        (
            row["model"],
            row["operation"],
            int(row["depth"]),
            int(row["head"]),
        )
        for row in profiles
    }
    require(
        all(
            (
                row["model"],
                row["operation"],
                int(row["depth"]),
                int(row["head"]),
            ) in profile_coordinates
            for row in selections
        ),
        "selected head missing profile",
    )
    return {
        "profile_count": len(profiles),
        "both_split_pass_count": int(
            summary["both_split_pass_count"]
        ),
        "frozen_head_count": len(selections),
        "confirming_head_count": int(
            summary["frozen_heads_confirming_any_panel"]
        ),
        "selection_leakage": False,
    }


def main() -> None:
    phase1012 = audit_phase1012()
    scans = audit_scans()
    final = audit_final()
    result = {
        "schema_version": "phase1013_result_audit.v1",
        "phase": PHASE,
        "status": "PASS",
        "remote_receiver": phase1012,
        "singleton_head_scans": scans,
        "head_final": final,
        "claim_limit": (
            "integrity and leakage audit; it does not certify a language "
            "mechanism"
        ),
    }
    audit_root = OUT_ROOT / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    write_json(audit_root / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
