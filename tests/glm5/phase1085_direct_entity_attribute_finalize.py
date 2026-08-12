#!/usr/bin/env python3
"""Finalize Phase1085 targeted descriptive evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1085_direct_entity_attribute_protocol as protocol

sys.modules["phase1082_semantic_output_operation_world_protocol"] = protocol
import phase1082_semantic_output_operation_world_finalize as engine


def main() -> None:
    engine.protocol = protocol
    engine.PRIMARY_ROLES = tuple(protocol.PRIMARY_PROFILE_ROLES)
    engine.base.protocol = protocol
    engine.base.DEPTH_GRID = np.linspace(
        protocol.TARGET_RELATIVE_DEPTH_MIN,
        protocol.TARGET_RELATIVE_DEPTH_MAX,
        7,
    )
    engine.main()

    analysis_root = protocol.OUT_ROOT / "analysis"
    final = protocol.read_json(analysis_root / "final_summary.json")
    predictions = final["predictions"]["predictions"]
    purity_names = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9")
    purity_passed = all(bool(predictions[name]["passed"]) for name in purity_names)
    cross_model_passed = bool(predictions["P8"]["passed"])
    failed = [name for name in purity_names if not predictions[name]["passed"]]
    if purity_passed and cross_model_passed:
        decision = "continue_to_full_role_depth_descriptive_atlas"
    elif purity_passed:
        decision = "continue_full_atlas_then_functional_cross_model_alignment"
    else:
        decision = "stop_hidden_escalation_and_diagnose_failed_purity_gates"
    automatic = {
        "schema_version": "phase1085_automatic_next.v1",
        "phase": protocol.PHASE,
        "decision": decision,
        "targeted_scope_complete": True,
        "full_atlas_authorized": purity_passed,
        "cross_model_direct_alignment_passed": cross_model_passed,
        "local_causal_authorized": False,
        "failed_purity_gates": failed,
        "reason": (
            "Full mapping requires protocol, behavior, repeatability, transfer, "
            "carrier advantage, control ratio, and numerical integrity together."
        ),
    }
    automatic["automatic_next_digest"] = protocol.digest(automatic)
    protocol.write_json(analysis_root / "automatic_next.json", automatic)
    final["schema_version"] = "phase1085_final_summary.v1"
    final["automatic_next"] = automatic
    final["target_scope"] = {
        "roles": list(protocol.CAPTURE_ROLES),
        "primary_roles": list(protocol.PRIMARY_PROFILE_ROLES),
        "relative_depth_min": protocol.TARGET_RELATIVE_DEPTH_MIN,
        "relative_depth_max": protocol.TARGET_RELATIVE_DEPTH_MAX,
    }
    final["summary_digest"] = protocol.digest({
        key: value for key, value in final.items() if key != "summary_digest"
    })
    protocol.write_json(analysis_root / "final_summary.json", final)
    print({
        "phase": protocol.PHASE,
        "purity_passed": purity_passed,
        "failed_purity_gates": failed,
        "decision": decision,
        "summary_digest": final["summary_digest"],
    })


if __name__ == "__main__":
    main()
