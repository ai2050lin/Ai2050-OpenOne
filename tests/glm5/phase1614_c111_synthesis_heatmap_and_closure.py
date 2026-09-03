#!/usr/bin/env python3
"""Phase1614 / C111: synthesize the read-only observations and extend the atlas."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1612_c111_value_identity_role_coalition_observation"
C110 = TESTS / "result/phase1607_c110_fresh_readout_control_separation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def family_rollup(pair_summary: list[dict], family: str) -> dict:
    selected = [row for row in pair_summary if row["family"] == family]
    return {
        "target_vs_permuted_movement_cosine_range": [min(row["median_target_vs_permuted_movement_cosine"] for row in selected), max(row["median_target_vs_permuted_movement_cosine"] for row in selected)],
        "target_movement_field_cosine_range": [min(row["median_target_movement_field_cosine"] for row in selected), max(row["median_target_movement_field_cosine"] for row in selected)],
        "permuted_movement_field_cosine_range": [min(row["median_permuted_movement_field_cosine"] for row in selected), max(row["median_permuted_movement_field_cosine"] for row in selected)],
        "permuted_to_target_l2_ratio_range": [min(row["median_permuted_to_target_l2_ratio"] for row in selected), max(row["median_permuted_to_target_l2_ratio"] for row in selected)],
        "target_output_gain_gt_permuted_pairs": int(sum(row["target_output_gain_gt_permuted_pairs"] for row in selected)),
        "positive_focus_record_increment_pairs": int(sum(row["positive_focus_record_increment_pairs"] for row in selected)),
        "focus_record_increment_median_range": [min(row["median_focus_record_increment"] for row in selected), max(row["median_focus_record_increment"] for row in selected)],
        "additional_truth_flips": int(sum(row["additional_truth_flips"] for row in selected)),
    }


def main() -> None:
    observation = core.load(OUT / "analysis/observation_report.json")
    source_audit = core.load(OUT / "audit/independent_observation_audit.json")
    if observation["authorization"] != "run_phase1614_c111_synthesis_heatmap_and_closure" or not source_audit["all_checks_passed"]:
        raise RuntimeError("C111 closure authorization missing")
    pair_summary = core.rows(OUT / "analysis/pair_value_role_geometry_summary.jsonl")
    trajectory = core.rows(OUT / "analysis/cross_archive_role_state_trajectory.jsonl")
    role_matrix = core.rows(OUT / "analysis/state19_role_cosine_matrix.jsonl")
    locators = core.rows(OUT / "analysis/role_state_descriptive_locators.jsonl")
    rollup = {family: family_rollup(pair_summary, family) for family in ("attribute_binding", "agent_patient")}

    payload = core.load(PUBLIC)
    payload.update({
        "phase": 1614,
        "campaign": "C109-C111",
        "title": "C109-C111 Prospective Readout / Control / Role-Field Atlas",
        "c111_observation": {
            "pair_summary": pair_summary,
            "family_rollup": rollup,
            "trajectory_rows": trajectory,
            "role_matrix_state19": role_matrix,
            "locators": locators,
            "planned_missingness": observation["planned_missingness"],
        },
        "claim_boundary": "C109-C111 show prospectively stable full-coordinate truth fields and relation-specific output leverage in Qwen3 controlled English. C111 is a read-only archive observation: its role clocks and value-geometry summaries are descriptive, and the existing coordinate-permutation transport is L2-confounded. No exact coordinate-value identity, minimal role coalition, natural necessity, weight, or universal neuron claim is established.",
        "created_at_utc": now(),
    })
    canonical = OUT / "visualization/c109_c111_role_field_atlas.json"
    core.save(canonical, payload)
    shutil.copyfile(canonical, PUBLIC)

    locator_map = {(row["family"], row["role"]): row for row in locators}
    closure = {
        "phase": 1614,
        "campaign": "C111",
        "created_at_utc": now(),
        "status": "read_only_value_identity_role_coalition_major_stage_complete",
        "headline": {
            "family_rollup": rollup,
            "attribute_role_clock": {role: locator_map[("attribute_binding", role)]["earliest_c109_c110_stable_high_amplitude_state"] for role in ("focus_record", "query_anchor", "code_instruction", "boundary")},
            "agent_role_clock": {role: locator_map[("agent_patient", role)]["earliest_c109_c110_stable_high_amplitude_state"] for role in ("focus_record", "query_anchor", "code_instruction", "boundary")},
        },
        "new_puzzles": {
            "K289-CONTROL": "the existing coordinate-permuted donor-value control is not an energy-matched value-identity test: movement direction is weakly aligned to the target movement and its L2 is 1.46-2.31 times larger at the cell medians",
            "K290-OBS": "cross-archive role clocks differ by relation: the attribute high-amplitude stable truth field first localizes at query_anchor S16, boundary S21, and code S23, while agent truth is already high-amplitude stable at focus_record S3 and query_anchor S6 before boundary S21",
            "K291-R1": "focus_record addition has relation-conditioned continuous leverage: it increases raw truth margin in 84/96 agent pairs but only 28/96 attribute pairs; no pair gains an additional truth flip over whole-query transport",
        },
        "theory_update": "RDC should treat the field as a typed role-state family R[f,r,s], not one global semantic vector. Value assignment V and role context r can alter transport leverage independently of field readability; current observations do not yet supply the clean interventions needed to identify either object minimally.",
        "claim_boundary": payload["claim_boundary"],
        "heatmap": {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "bytes": PUBLIC.stat().st_size, "sha256": core.sha(PUBLIC)},
        "next_authorization": "C112 frozen batch intervention: multiple within-support movement-vector permutations with exact L2 preservation, plus single-role and predeclared role-coalition transports on C110; continue raw truth readout only and do not inspect attention or MLP",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {
        "source": source_audit["all_checks_passed"],
        "pair_summary": len(pair_summary) == 8,
        "trajectory": len(trajectory) == 518,
        "role_matrix": len(role_matrix) == 98,
        "rollup": rollup["agent_patient"]["positive_focus_record_increment_pairs"] == 84 and rollup["attribute_binding"]["positive_focus_record_increment_pairs"] == 28,
        "no_flips": sum(value["additional_truth_flips"] for value in rollup.values()) == 0,
        "locators": closure["headline"]["attribute_role_clock"] == {"focus_record": None, "query_anchor": 16, "code_instruction": 23, "boundary": 21} and closure["headline"]["agent_role_clock"] == {"focus_record": 3, "query_anchor": 6, "code_instruction": 20, "boundary": 21},
        "identity": core.sha(canonical) == core.sha(PUBLIC),
        "boundary": "read-only archive observation" in payload["claim_boundary"] and "L2-confounded" in payload["claim_boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    report = {"phase": 1614, "campaign": "C111", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "asset_sha256": core.sha(canonical), "authorization": "audit_frontend_append_c111_memo_and_authorize_c112"}
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"checks": checks, "headline": closure["headline"], "new_puzzles": closure["new_puzzles"], "heatmap": closure["heatmap"]}, indent=2))


if __name__ == "__main__":
    main()
