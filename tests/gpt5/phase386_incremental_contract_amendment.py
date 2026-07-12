#!/usr/bin/env python3
"""Freeze the Phase386 incremental-cache correction before relation analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    probe = read_json(OUT / "phase386_cache_path_probe.json")
    if not probe["all_actual_incremental_transitions_match"]:
        raise RuntimeError("Phase386 actual incremental cache probe did not pass")
    pilot_root = OUT / "collection_teacher_forced_pilot/discovery/models"
    pilot = {
        model: read_json(pilot_root / model / "manifest.json")
        for model in ("qwen3", "glm4", "deepseek7b")
    }
    payload = {
        "schema_version": "60.4.2",
        "phase_id": "Phase386-IncrementalContractAmendment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trigger": {
            "teacher_forced_discovery_case_count": 288,
            "qwen3_required_transition_pass_count": pilot["qwen3"][
                "required_transition_pass_count"
            ],
            "glm4_required_transition_pass_count": pilot["glm4"][
                "required_transition_pass_count"
            ],
            "deepseek7b_required_transition_pass_count": pilot["deepseek7b"][
                "required_transition_pass_count"
            ],
            "deepseek7b_target_encoded_mismatch_count": 2,
            "component_conservation_failure_count": 0,
            "relation_analysis_started": False,
            "calibration_opened": False,
            "physical_holdout_opened": False,
        },
        "probe": {
            "case_count": probe["case_count"],
            "all_actual_incremental_transitions_match": probe[
                "all_actual_incremental_transitions_match"
            ],
            "max_actual_vs_full_logit_abs_difference": max(
                row["actual_vs_full_max_logit_abs_difference"]
                for row in probe["rows"]
            ),
        },
        "retired_contract": {
            "three_full_sequence_teacher_forced_replays": True,
            "status": "engineering_pilot_only",
            "raw_files_preserved": True,
            "may_support_language_path_claim": False,
        },
        "replacement_contract": {
            "generation_path": "actual_incremental_kv_cache",
            "model_call_count_formula": (
                "3 + target_decision_step: prompt + every pre-target prefix token "
                "+ target token + post-decision token"
            ),
            "semantic_coordinate_count": 5,
            "semantic_coordinates_are_independent_times": False,
            "all_three_models_must_be_rerun": True,
            "instrument_audit_must_be_rerun": True,
            "no_failed_case_replacement": True,
        },
        "authorization": {
            "incremental_instrument_collection": True,
            "incremental_discovery_before_instrument_audit": False,
            "calibration_collection": False,
            "physical_holdout_collection": False,
        },
    }
    write_json(OUT / "phase386_incremental_contract_amendment.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
