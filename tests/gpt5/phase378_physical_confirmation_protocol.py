#!/usr/bin/env python3
"""Freeze narrow Phase378 physical confirmation before opening sealed cases."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PHASE377 = ROOT / "tests/gpt5/result/phase377_decision_aligned_calibration"
OUT = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
PHYSICAL_CASES = (
    PHASE371
    / "phase371c_case_bank/sealed/private/phase371c_physical_execution_cases.jsonl"
)
CALIBRATION = PHASE377 / "phase377_intervention/phase377_calibration_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if not calibration["authorization"]["freeze_physical_confirmation_protocol"]:
        raise RuntimeError("Phase377 did not authorize physical protocol")
    candidates = [
        row
        for row in calibration["cross_model_rows"]
        if row["heterogeneous_level2_calibration_pass"]
    ]
    payload = {
        "schema_version": "51.0.0",
        "phase_id": "Phase378-PhysicalProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "narrow_physical_confirmation_of_decision_aligned_late_residual_content_transfer",
        "scope": {
            "models": ["qwen3", "glm4", "deepseek7b"],
            "execution_order": ["qwen3", "glm4", "deepseek7b"],
            "mechanisms": ["relation_binding", "entity_recency"],
            "registered_physical_groups_per_mechanism": 4,
            "registered_case_count": 96,
            "templates": ["residual_current", "residual_source_query_current"],
            "relative_depth": "late",
            "other_mechanisms_opened": False,
            "single_neuron_scan": False,
        },
        "behavior_gate": {
            "max_new_tokens": 24,
            "equal_prompt_length_bucketed": True,
            "target_present_and_no_distractor": True,
            "all_three_models_all_four_conditions_required": True,
            "minimum_common_groups_per_mechanism": 3,
            "failed_groups_replaced": False,
        },
        "intervention_gate": {
            "decision_alignment": "observed_target_completion_token",
            "transfer_pairs": ["A_to_C", "C_to_A"],
            "controls": ["wrong_depth", "wrong_role", "wrong_time"],
            "minimum_correct_transfer_gain": 0.10,
            "minimum_control_margin": 0.05,
            "minimum_common_groups_per_model_mechanism_template": 3,
            "both_treatment_directions_required": True,
        },
        "candidate_contract": candidates,
        "claim_boundary": {
            "physical_transfer_confirms_terminal_content_carrier_only": True,
            "upstream_encoding_rule_claimed": False,
            "natural_necessity_claimed": False,
            "full_generation_sufficiency_claimed": False,
            "language_mechanism_claimed": False,
        },
        "input_hashes": {
            "calibration_summary": sha256(CALIBRATION),
            "sealed_physical_cases": sha256(PHYSICAL_CASES),
        },
        "authorization": {
            "open_two_mechanism_physical_behavior_cases": True,
            "run_physical_intervention_before_behavior_gate": False,
            "open_other_mechanisms": False,
            "single_neuron_scan": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "phase378_physical_protocol.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
