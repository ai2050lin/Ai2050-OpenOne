#!/usr/bin/env python3
"""Freeze Phase397 causal factor-separation scenarios and gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
CASES = OUT / "factor_trace/protocol/private/phase397_discovery_trace_cases.jsonl"
SCENARIOS = (
    "no_intervention",
    "identity_relation_candidate",
    "donor_relation_candidate",
    "donor_content_candidate",
    "donor_order_candidate",
    "donor_syntax_candidate",
    "donor_query_source_candidate",
    "donor_entities_candidate",
    "donor_random_candidate",
    "donor_relation_wrong_depth",
    "donor_full_source_candidate",
)
GATES = {
    "minimum_median_relation_normalized_margin_mediation": 0.1,
    "minimum_relation_advantage_over_each_local_control": 0.05,
    "minimum_positive_relation_direction_rate": 0.75,
    "minimum_relation_answer_switch_rate": 0.5,
    "minimum_candidate_advantage_over_wrong_depth": 0.05,
    "all_three_models_and_three_surfaces_required": True,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    discovery = read_json(OUT / "phase397_factor_discovery_analysis.json")
    physical = read_json(OUT / "phase397_factor_physical_analysis.json")
    if not discovery["authorization"]["causal_discovery_intervention"]:
        raise RuntimeError("Phase397 causal discovery is not authorized")
    if not physical["results"]["crossmodel_crosssurface_physical_observational_gate_pass"]:
        raise RuntimeError("Phase397 physical observational replication did not pass")
    case_count = sum(bool(line.strip()) for line in CASES.read_text(encoding="utf-8").splitlines())
    if case_count != 720:
        raise RuntimeError(f"Expected 720 Phase397 discovery cases, got {case_count}")
    protocol = {
        "schema_version": "71.9.0",
        "phase_id": "Phase397-CausalProtocol",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_relation_context_sufficiency_against_content_order_syntax_query_entity_random_depth_and_full_source_controls",
        "denominator": {
            "models": ["qwen3", "glm4", "deepseek7b"],
            "task_surfaces": ["possession_relation", "role_filling", "coreference_resolution"],
            "groups_per_surface_model": 8,
            "conditions_per_group": 10,
            "directions_per_group": 2,
            "direction_count": 144,
            "scenario_count_per_direction": len(SCENARIOS),
            "generation_scenario_count_per_direction": 1,
        },
        "directions": {
            "axis_x": "A_recipient_from_B_relation_donor",
            "axis_y": "F_recipient_from_G_relation_donor",
        },
        "scenario_contract": {
            "common_evaluation_margin": "relation_donor_target_logit_minus_recipient_target_logit",
            "relation": "same literal values and same value positions; context donor differs only in preceding entity slots",
            "content": "different literal values at the same value positions",
            "order": "same literal binding with whole-clause order changed; mapped by literal identity",
            "syntax": "same literal binding under paraphrase; mapped by literal identity",
            "query_source": "identical source with later query changed; source value states must be invariant",
            "entity": "source entity states mapped by entity identity",
            "random": "same number of non-source causal prefix positions",
            "wrong_depth": "same relation donor at externally frozen shallow layer",
            "full_source": "entire relation donor source prefix",
        },
        "scenarios": list(SCENARIOS),
        "frozen_gates_inherited_and_extended_from_phase395": GATES,
        "authorization": {
            "run_discovery_causal_three_models_sequentially": True,
            "run_calibration_causal": False,
            "open_physical_causal_holdout": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "causal_discovery_pass_is_calibrated_rule": False,
            "aggregate_value_patch_is_single_neuron": False,
            "sufficiency_is_natural_necessity": False,
        },
    }
    path = OUT / "phase397_causal_protocol.json"
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
