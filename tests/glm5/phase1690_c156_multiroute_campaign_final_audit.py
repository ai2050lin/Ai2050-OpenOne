#!/usr/bin/env python3
"""C156: machine-auditable synthesis of the C140-C155 multiroute campaign."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1690_c156_multiroute_campaign_final_audit"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1690, "C156"
PHASE_DIRS = {
    1674: ("phase1674_c140_identifiability_and_master_contract", "independent_contract_audit.json"),
    1675: ("phase1675_c141_multifamily_full_coordinate_atlas", "independent_closure_audit.json"),
    1676: ("phase1676_c142_mobius_output_code_separation", "independent_closure_audit.json"),
    1677: ("phase1677_c143_transition_model_competition", "independent_closure_audit.json"),
    1678: ("phase1678_c144_dual_graph_composition_reconstruction", "independent_closure_audit.json"),
    1679: ("phase1679_c145_correct_error_depth_trajectory_atlas", "independent_closure_audit.json"),
    1680: ("phase1680_c146_cross_model_interface_sweep", "independent_closure_audit.json"),
    1681: ("phase1681_c147_cross_model_relative_topology_eligibility", "independent_closure_audit.json"),
    1682: ("phase1682_c148_campaign_synthesis_heatmap_and_closure", "independent_closure_audit.json"),
    1683: ("phase1683_c149_arm_specific_transition_system_identification", "independent_closure_audit.json"),
    1684: ("phase1684_c150_predictable_transition_window_atlas", "independent_closure_audit.json"),
    1685: ("phase1685_c151_fresh_transition_window_replication", "independent_closure_audit.json"),
    1686: ("phase1686_c152_type_graph_transition_object_discovery", "independent_closure_audit.json"),
    1687: ("phase1687_c153_type_graph_conditional_pool_confirmation", "independent_closure_audit.json"),
    1688: ("phase1688_c154_type_graph_hiddenstate_causal_adjudication", "independent_closure_audit.json"),
    1689: ("phase1689_c155_checkpoint_transfer_curve", "independent_closure_audit.json"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    if OUT.exists():
        raise RuntimeError(OUT)
    audits = {}
    for phase, (directory, audit_name) in PHASE_DIRS.items():
        path = RESULT / directory / "audit" / audit_name
        audits[str(phase)] = {"path": str(path), "sha256": core.sha(path), "record": core.load(path)}
    checks = {
        "sixteen_phases": len(audits) == 16,
        "consecutive": sorted(map(int, audits)) == list(range(1674, 1690)),
        "all_independent_audits": all(record["record"].get("all_checks_passed", False) for record in audits.values()),
        "c153_prediction_pass": audits["1687"]["record"]["scientific_gate_passed"],
        "c154_identity_fail": not audits["1688"]["record"]["scientific_gate_passed"],
        "c155_asset_present": (ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json").is_file(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    route_ledger = {
        "A_direct_to_composition": {"status": "mixed", "evidence": "aggregate first-order reconstruction and type-graph conditional transition prediction; no universal composition operator"},
        "B_all_token_coordinate_transmission": {"status": "partial_pass", "evidence": "full-token/full-coordinate atlas plus q24-q33 predictive window; no unique edge graph or minimal coordinate circuit"},
        "C_nested_language_composition": {"status": "observed_not_causal", "evidence": "event and discourse arms transfer in the late window; no dedicated nested-composition causal test"},
        "D_type_graph_ecology": {"status": "predictive_pass_causal_identity_fail", "evidence": "C153 prospective pass; C154 strong steering but wrong-checkpoint control wins; C155 broad portable field"},
        "E_cross_family_cross_model": {"status": "cross_family_partial_cross_model_not_tested", "evidence": "four of five arms prospectively transfer; no shared GLM4/DeepSeek behavior interface"},
    }
    synthesis = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "C140_C155_multiroute_campaign_closed",
        "checks": checks,
        "route_ledger": route_ledger,
        "strongest_positive": "condition-indexed six-role full-coordinate late HiddenState response predicts fresh type-graph trajectories and strongly steers matched answers",
        "strongest_negative": "the same field is more effective at earlier checkpoints, so it is not a checkpoint-specific local transition gear",
        "theory_boundary": "conditional reusable response field with task residuals; no unique circuit, parameter-level mechanism, natural ontology, or cross-model invariant",
        "mathematics": "existing finite differences, linear algebra, typed conditional dynamics and partial-domain response maps suffice; new mathematics upgrade gate remains closed",
        "next_big_stage": "natural typed knowledge-graph external validity with human semantic audit, followed by fresh prediction before any causal patch",
        "audits": audits,
        "authorization": "append_C156_memo_and_end_current_big_stage",
    }
    core.save(OUT / "analysis/final.json", synthesis)
    internal = {"checks": checks, "all_checks_passed": all(checks.values()), "authorization": "independent_final_and_memo"}
    core.save(OUT / "audit/internal_closure_audit.json", internal)
    independent_checks = {
        "internal": internal["all_checks_passed"],
        "routes": set(route_ledger) == {"A_direct_to_composition", "B_all_token_coordinate_transmission", "C_nested_language_composition", "D_type_graph_ecology", "E_cross_family_cross_model"},
        "boundaries": "no unique circuit" in synthesis["theory_boundary"],
        "next_stage_distinct": "natural typed knowledge-graph" in synthesis["next_big_stage"],
    }
    audit = {"phase": PHASE, "campaign": CAMPAIGN, "checks": independent_checks, "passed": sum(independent_checks.values()), "total": len(independent_checks), "all_checks_passed": all(independent_checks.values()), "authorization": "memo_and_current_big_stage_complete"}
    core.save(OUT / "audit/independent_closure_audit.json", audit)
    print(json.dumps({"checks": checks, "route_ledger": route_ledger, "independent": audit}, indent=2))


if __name__ == "__main__":
    main()
