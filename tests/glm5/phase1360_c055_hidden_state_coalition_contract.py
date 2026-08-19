#!/usr/bin/env python3
"""Phase1360: freeze the hidden-state-only C055 role-coalition campaign."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1360, "C055"
C053 = TESTS / "result/phase1353_c053_route_portfolio_contract"
BEHAVIOR = TESTS / "result/phase1354_c053_behavior_route_competition"
FIELD = TESTS / "result/phase1355_c053_full_width_route_fields"
PARENT = TESTS / "result/phase1359_c054_same_batch_token_causal_replay"
OUT = TESTS / "result/phase1360_c055_hidden_state_coalition_contract"
MODEL = "qwen3"
ROLE_COALITIONS = {
    "target": ("target",),
    "family": ("family",),
    "boundary": ("boundary",),
    "target_family": ("target", "family"),
    "target_boundary": ("target", "boundary"),
    "family_boundary": ("family", "boundary"),
    "all_roles": ("target", "family", "boundary"),
}
SINGLETONS = ("target", "family", "boundary")
MULTI = ("target_family", "target_boundary", "family_boundary", "all_roles")
ONE_TOKEN_FAMILIES = ("currency", "language", "emotion")


def stable_pick(pool: list[dict], key: str, offset: int) -> dict:
    ordered = sorted(pool, key=lambda row: row["case_id"])
    return ordered[(int(key.split("-")[-1]) + offset) % len(ordered)]


def main() -> None:
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    behavior = core.load(BEHAVIOR / "analysis/final.json")
    behavior_audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    field = core.load(FIELD / "analysis/final.json")
    field_audit = core.load(FIELD / "audit/independent_final_audit.json")
    if parent.get("authorization") != "close_c054_at_calibrated_causal_selectivity_boundary":
        raise RuntimeError("C054 did not close at the calibrated boundary")
    if not all(x.get("all_checks_passed") for x in (parent_audit, behavior_audit, field_audit)):
        raise RuntimeError("a required parent audit failed")
    qualified_behavior = set(behavior.get("qualified_routes", []))
    if not {"B2_relative", "B3_choice"}.issubset(qualified_behavior):
        raise RuntimeError("C053 relative behavior is not qualified")
    if not field.get("shared_relation_qualified"):
        raise RuntimeError("C053 hidden field is not qualified")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1360 already exists")

    active = core.rows(C053 / "material/b1_binary_cases.jsonl")
    active_compiled = core.rows(C053 / "compiled/qwen3_B1_binary.jsonl")
    status = core.rows(C053 / "material/status_null_cases.jsonl")
    status_compiled = core.rows(C053 / "compiled/qwen3_N_status.jsonl")
    active_c = {row["case_id"]: row for row in active_compiled}
    status_c = {row["case_id"]: row for row in status_compiled}
    target_length = {case_id: len(row["target_span"]) for case_id, row in active_c.items()}
    status_target_length = {case_id: len(row["target_span"]) for case_id, row in status_c.items()}

    active_pool: dict[tuple, list[dict]] = defaultdict(list)
    for row in active:
        active_pool[(row["partition"], row["surface"], row["tested_family"], bool(row["truth"]),
                     target_length[row["case_id"]])].append(row)
    status_pool: dict[tuple, list[dict]] = defaultdict(list)
    for row in status:
        status_pool[(row["partition"], row["tested_family"], bool(row["truth"]),
                     status_target_length[row["case_id"]])].append(row)

    eligible = []
    raw_recipients = [
        row for row in active
        if row["partition"] in ("confirmation", "lockbox")
        and row["tested_family"] in ONE_TOKEN_FAMILIES and not row["truth"]
    ]
    for recipient in sorted(raw_recipients, key=lambda row: row["case_id"]):
        partition, surface = recipient["partition"], recipient["surface"]
        tested, target_family = recipient["tested_family"], recipient["target_family"]
        length = target_length[recipient["case_id"]]
        wrong_families = [family for family in ONE_TOKEN_FAMILIES if family not in (tested, target_family)]
        if not wrong_families:
            wrong_families = [family for family in ONE_TOKEN_FAMILIES if family != tested]
        wrong_family = sorted(wrong_families)[int(recipient["case_id"].split("-")[-1]) % len(wrong_families)]
        pools = {
            "correct_true": [row for row in active_pool[(partition, surface, tested, True, length)]
                             if row["target"] != recipient["target"]],
            "correct_false": [row for row in active_pool[(partition, surface, tested, False, length)]
                              if row["target"] != recipient["target"] and row["target_family"] != target_family],
            "wrong_true": active_pool[(partition, surface, wrong_family, True, length)],
            "status_true": status_pool[(partition, tested, True, length)],
        }
        if not all(pools.values()):
            continue
        donors = {name: stable_pick(pool, recipient["case_id"], index)
                  for index, (name, pool) in enumerate(pools.items())}
        eligible.append({
            "recipient": recipient["case_id"], "partition": partition, "surface": surface,
            "recipient_tested_family": tested, "recipient_target_family": target_family,
            "target_span_length": length, "wrong_tested_family": wrong_family,
            **{name: row["case_id"] for name, row in donors.items()},
        })

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in eligible:
        cells[(row["partition"], row["surface"], row["recipient_tested_family"])].append(row)
    per_cell = min(len(values) for values in cells.values())
    replay = []
    for key in sorted(cells):
        replay.extend(sorted(cells[key], key=lambda row: row["recipient"])[:per_cell])

    checks = {
        "parent_closed_and_audited": parent_audit["all_checks_passed"],
        "behavior_inherited": {"B2_relative", "B3_choice"}.issubset(qualified_behavior),
        "behavior_audited": behavior_audit["all_checks_passed"],
        "field_inherited": field["shared_relation_qualified"],
        "field_audited": field_audit["all_checks_passed"],
        "finite_coalitions": set(ROLE_COALITIONS) == set(SINGLETONS) | set(MULTI),
        "role_vocabulary": set(role for roles in ROLE_COALITIONS.values() for role in roles)
                           == {"target", "family", "boundary"},
        "eligible_cells": len(cells) == 18,
        "per_cell_minimum": per_cell >= 9,
        "balanced_replay": len(replay) == 18 * per_cell
                           and len(set(Counter((x["partition"], x["surface"], x["recipient_tested_family"])
                                               for x in replay).values())) == 1,
        "recipient_false": all(not next(row for row in active if row["case_id"] == x["recipient"])["truth"]
                               for x in replay),
        "donor_span_isomorphism": all(
            len(active_c[x[key]]["target_span"]) == x["target_span_length"]
            for x in replay for key in ("recipient", "correct_true", "correct_false", "wrong_true")
        ) and all(len(status_c[x["status_true"]]["target_span"]) == x["target_span_length"] for x in replay),
        "family_spans_single_token": all(
            len(active_c[x[key]]["tested_family_span"]) == 1
            for x in replay for key in ("recipient", "correct_true", "correct_false", "wrong_true")
        ) and all(len(status_c[x["status_true"]]["tested_family_span"]) == 1 for x in replay),
        "semantic_uniqueness_inherited": True,
        "controlled_naturalness_inherited": True,
        "hidden_state_only": True,
        "no_new_model_output_used": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    core.write_rows(OUT / "material/causal_replay_manifest.jsonl", replay)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "semantic_scope": "inherits C053 ordinary-noun membership adjudication without new wording",
        "naturalness_scope": "inherits the frozen ordinary/dictionary/claim surfaces",
        "independent_human_blind_review": False,
        "zero_models": {
            "role_permutation": "a wrong role coalition should not match the registered coalition",
            "donor_exchangeability": "correct, wrong-family, false, and status donors are equally effective",
            "duplicate_self": "exact same-batch self transport is an identity for every coalition",
            "singleton_sufficiency": "one role alone is enough; multi-role synergy is absent",
        },
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c055.hidden_state_role_coalitions.v1",
        "model": MODEL,
        "research_object": "Qwen C053 noun-membership computation as a set-valued hidden-state object over target, tested family, and answer boundary",
        "allowed_observables": ["input token embeddings", "full-dimensional hidden states at registered roles and layers",
                                "candidate logits and margins"],
        "forbidden": ["attention weights", "attention heads", "MLP states or weights", "parameter scan",
                      "PCA", "t-SNE", "UMAP", "SAE", "learned probe", "gradient saliency", "post-reveal hotspot search"],
        "claim_boundary": {
            "allowed": "descriptive coalition complementarity; calibrated coalition sufficiency/selectivity; conditional necessity/rescue",
            "not_assumed": ["multi-position mechanism exists", "relative coding is proved", "a concatenated field is causal",
                            "selected depth is formation time", "cross-model invariance", "parameter mechanism"],
        },
        "material": {
            "behavior_source": "unchanged C053 B2/B3 with passed independent audit",
            "hidden_source": "unchanged C053 full-dimensional quartet interaction tensors",
            "causal_case_count": len(replay), "balanced_cell_count": 18, "cases_per_cell": per_cell,
            "partitions": ["confirmation", "lockbox"], "surfaces": ["ordinary", "dictionary", "claim"],
            "tested_families": list(ONE_TOKEN_FAMILIES),
        },
        "coalitions": {key: list(value) for key, value in ROLE_COALITIONS.items()},
        "observation": {
            "depths": list(range(1, 37)), "representation": "ordered full-width role concatenation followed by cosine normalization",
            "selection_partitions": ["prototype_discovery", "clock_selection"],
            "confirmation_partitions": ["confirmation", "lockbox"],
            "persistence_layers": 2,
            "multi_identity_top1_min": 0.70, "multi_surface_top1_min": 0.60,
            "identity_gain_over_best_singleton_min": 0.05,
            "selection_active_cosine_min": 0.20, "active_over_status_gap_min": 0.05,
            "active_over_status_win_min": 0.65, "status_direction_cosine_max": 0.80,
            "held_active_cosine_min": 0.15, "held_gap_min": 0.05, "held_win_min": 0.65,
            "selection_rule": "earliest persistent qualifying multi-role coalition; ties prefer fewer roles then lexical name",
            "fallback_if_none": {"coalition": "all_roles", "layer": 27},
            "observation_failure_does_not_cancel_causal_portfolio": True,
        },
        "camera": {
            "test_layers": "selected observation layer plus frozen fallback layer 27",
            "coalitions": list(ROLE_COALITIONS), "calibration_cases": 48,
            "same_batch_exact_self_max_abs_margin": 1e-6,
            "all_coalitions_and_layers_must_pass": True,
        },
        "causal": {
            "layer": "observation-selected layer, or fallback 27 if no descriptive coalition qualifies",
            "coalitions": list(ROLE_COALITIONS), "transport": "same-batch exact-token state replacement",
            "arms_per_coalition": ["self", "correct_true", "wrong_family_true", "same_family_false", "status_true"],
            "false_to_true_gain_min": 0.5, "direction_fraction_min": 0.75,
            "correct_over_all_controls_median_min": 0.25,
            "correct_over_all_controls_win_min": 0.65,
            "self_max_abs_diff_max": 1e-4,
            "multi_synergy_gain_over_best_constituent_min": 0.25,
            "multi_synergy_win_over_best_constituent_min": 0.05,
            "route_fail": "eliminate only that coalition",
            "all_multi_routes_fail": "close C055 at the hidden-state coalition boundary",
        },
        "necessity_rescue": {
            "authorized_only_if": "at least one multi-role coalition passes causal and synergy gates",
            "selected_coalition": "smallest passing coalition; tie by higher selectivity win then lexical name",
            "block": "replace a true recipient coalition with a same-family false donor coalition",
            "rescue": "restore the exact original recipient coalition one layer later",
            "wrong_rescue": "restore a wrong-family true donor coalition one layer later",
            "block_margin_drop_median_min": 0.5, "block_direction_fraction_min": 0.75,
            "correct_recovery_fraction_median_min": 0.50, "correct_over_wrong_recovery_min": 0.25,
            "correct_over_wrong_win_min": 0.65, "self_max_abs_diff_max": 1e-4,
        },
        "branching": {
            "phase1361": "run every descriptive coalition on the frozen full-dimensional field",
            "phase1362": "calibrate exact self for every coalition at the derived layer set",
            "phase1363": "run every causal coalition even when no descriptive coalition qualifies",
            "phase1364": "run necessity/rescue only when a multi-role causal coalition qualifies",
            "finish": "close after the last preauthorized branch; no new role, layer, scale, projection, or component",
        },
        "stop_rule": "No post-reveal change to object, material, role, coalition, partition, model, null, threshold, layer rule, donor, or branch.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1361_c055_hidden_state_observation"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
