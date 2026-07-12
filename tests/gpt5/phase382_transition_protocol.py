#!/usr/bin/env python3
"""Freeze an offline transition-event identifiability audit after Phase381."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
P381 = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
OUT = ROOT / "tests/gpt5/result/phase382_transition_event_audit"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    freeze_path = P381 / "phase381_joint_scan_freeze.json"
    freeze = read_json(freeze_path)
    splits = {}
    for mechanism, groups in freeze["selected_replay_qualified_groups"].items():
        ordered = sorted(groups)
        if len(ordered) < 7:
            raise RuntimeError(f"Need at least seven groups for {mechanism}")
        splits[mechanism] = {
            "offline_discovery": ordered[:4],
            "offline_validation": ordered[4:],
        }
    protocol = {
        "schema_version": "55.0.0",
        "phase_id": "Phase382-TransitionProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_whether_component_transition_updates_are_more_identifiable_than_static_states",
        "input_phase": "Phase381",
        "fresh_cuda_execution": False,
        "frozen_group_splits": splits,
        "basic_formulas": {
            "layer_update": "U_l_r(condition)=layer_output_l_r(condition)-layer_input_l_r(condition)",
            "content_effect": "E_content=0.5*((C-A)+(D-B))",
            "operation_effect": "E_operation=0.5*((A-B)+(C-D))",
            "interaction_effect": "E_interaction=0.5*((A-B)-(C-D))",
            "signed_alignment": "min(1,norm(local_effect)/norm(terminal_effect))*cos(local_effect,terminal_effect)",
            "common_backbone": "per_model_per_split_per_effect_mean_profile_across_three_mechanisms",
            "function_residual": "mechanism_profile-common_backbone",
        },
        "profile_grid": {
            "relative_depth_bins": 5,
            "position_roles": ["source", "query", "current"],
            "profile_width": 15,
            "effect_axes": ["content", "operation", "interaction"],
            "top_k_used": False,
        },
        "parameter_free_identifiability_gate": {
            "own_validation_profile_must_exceed_every_wrong_mechanism_profile": True,
            "transition_own_win_count_must_exceed_static_own_win_count": True,
            "transition_within_mechanism_median_must_exceed_static_median": True,
            "transition_crossmodel_median_must_exceed_static_median": True,
            "threshold_fitting_allowed": False,
        },
        "claim_boundary": {
            "offline_identifiability_is_causal_path": False,
            "layer_update_is_full_internal_operator": False,
            "positive_result_authorizes_immediate_neuron_scan": False,
            "phase381_groups_can_be_reused_for_confirmatory_intervention": False,
            "language_encoding_mechanism_closed": False,
        },
        "input_sha256": {"phase381_joint_scan_freeze": sha256(freeze_path)},
    }
    write_json(OUT / "phase382_transition_protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
