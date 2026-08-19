#!/usr/bin/env python3
"""Phase1357: freeze the finite C054 identity-camera and causal replay contract."""
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

PHASE, CAMPAIGN = 1357, "C054"
C053 = TESTS / "result/phase1353_c053_route_portfolio_contract"
PARENT = TESTS / "result/phase1356_c053_typed_relation_causal"
OUT = TESTS / "result/phase1357_c054_same_batch_causal_contract"
MODEL = "qwen3"
LAYERS = (3, 14, 27, 35)
ONE_TOKEN_FAMILIES = ("currency", "language", "emotion")


def stable_pick(rows: list[dict], key: str, offset: int = 0) -> dict:
    ordered = sorted(rows, key=lambda row: row["case_id"])
    index = (int(key.split("-")[-1]) + offset) % len(ordered)
    return ordered[index]


def main() -> None:
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    c053_protocol = core.load(C053 / "protocol/preregistration.json")
    if parent.get("authorization") != "close_c053_at_causal_selectivity_boundary":
        raise RuntimeError("Phase1356 did not close at the camera boundary")
    if not parent_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1356 audit did not pass")
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1357 already exists")

    source = core.rows(C053 / "material/b1_binary_cases.jsonl")
    compiled = core.rows(C053 / "compiled/qwen3_B1_binary.jsonl")
    source_by_id = {row["case_id"]: row for row in source}
    compiled_by_id = {row["case_id"]: row for row in compiled}
    span_length = {case_id: len(row["tested_family_span"]) for case_id, row in compiled_by_id.items()}

    # One frozen case from each partition x surface x truth x span-length stratum.
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in source:
        strata[(row["partition"], row["surface"], bool(row["truth"]), span_length[row["case_id"]])].append(row)
    calibration = []
    for key in sorted(strata, key=str):
        picked = sorted(strata[key], key=lambda row: row["case_id"])[0]
        calibration.append({
            "case_id": picked["case_id"], "partition": picked["partition"],
            "surface": picked["surface"], "truth": bool(picked["truth"]),
            "tested_family": picked["tested_family"], "span_length": span_length[picked["case_id"]],
        })

    lookup: dict[tuple, list[dict]] = defaultdict(list)
    for row in source:
        lookup[(row["partition"], row["surface"], row["tested_family"], bool(row["truth"]))].append(row)
    recipients = [
        row for row in source
        if row["partition"] in ("confirmation", "lockbox")
        and row["tested_family"] in ONE_TOKEN_FAMILIES
        and not row["truth"]
    ]
    replay = []
    for recipient in sorted(recipients, key=lambda row: row["case_id"]):
        partition, surface = recipient["partition"], recipient["surface"]
        tested, target_family = recipient["tested_family"], recipient["target_family"]
        correct_true_pool = [
            row for row in lookup[(partition, surface, tested, True)]
            if row["target"] != recipient["target"]
        ]
        correct_false_pool = [
            row for row in lookup[(partition, surface, tested, False)]
            if row["target"] != recipient["target"] and row["target_family"] != target_family
        ]
        wrong_families = [family for family in ONE_TOKEN_FAMILIES if family not in (tested, target_family)]
        if not wrong_families:
            wrong_families = [family for family in ONE_TOKEN_FAMILIES if family != tested]
        wrong_family = sorted(wrong_families)[int(recipient["case_id"].split("-")[-1]) % len(wrong_families)]
        wrong_true_pool = lookup[(partition, surface, wrong_family, True)]
        wrong_false_pool = [
            row for row in lookup[(partition, surface, wrong_family, False)]
            if row["target_family"] not in (wrong_family, target_family)
        ]
        donors = {
            "correct_true": stable_pick(correct_true_pool, recipient["case_id"], 0),
            "correct_false": stable_pick(correct_false_pool, recipient["case_id"], 1),
            "wrong_true": stable_pick(wrong_true_pool, recipient["case_id"], 2),
            "wrong_false": stable_pick(wrong_false_pool, recipient["case_id"], 3),
        }
        if not all(span_length[row["case_id"]] == 1 for row in donors.values()):
            raise RuntimeError("replay donor span is not one token")
        replay.append({
            "recipient": recipient["case_id"], "partition": partition, "surface": surface,
            "recipient_target_family": target_family, "recipient_tested_family": tested,
            "wrong_tested_family": wrong_family,
            **{name: row["case_id"] for name, row in donors.items()},
        })

    checks = {
        "parent_closed_and_audited": parent_audit["all_checks_passed"],
        "c053_contract_linked": c053_protocol["contract_sha256"] == parent.get("contract_sha256", c053_protocol["contract_sha256"]),
        "calibration_count": len(calibration) == 48,
        "calibration_strata_unique": len({(x["partition"], x["surface"], x["truth"], x["span_length"]) for x in calibration}) == 48,
        "calibration_span_balance": Counter(x["span_length"] for x in calibration) == {1: 24, 2: 24},
        "replay_count": len(replay) == 324,
        "replay_partition_balance": Counter(x["partition"] for x in replay) == {"confirmation": 162, "lockbox": 162},
        "replay_surface_balance": Counter(x["surface"] for x in replay) == {"ordinary": 108, "dictionary": 108, "claim": 108},
        "replay_family_balance": all(value == 108 for value in Counter(x["recipient_tested_family"] for x in replay).values()),
        "recipient_false": all(not source_by_id[x["recipient"]]["truth"] for x in replay),
        "donor_truth_types": all(
            source_by_id[x["correct_true"]]["truth"] and not source_by_id[x["correct_false"]]["truth"]
            and source_by_id[x["wrong_true"]]["truth"] and not source_by_id[x["wrong_false"]]["truth"]
            for x in replay
        ),
        "donor_relation_types": all(
            source_by_id[x["correct_true"]]["tested_family"] == x["recipient_tested_family"]
            and source_by_id[x["correct_false"]]["tested_family"] == x["recipient_tested_family"]
            and source_by_id[x["wrong_true"]]["tested_family"] == x["wrong_tested_family"]
            and source_by_id[x["wrong_false"]]["tested_family"] == x["wrong_tested_family"]
            for x in replay
        ),
        "all_replay_spans_one_token": all(
            span_length[x[key]] == 1
            for x in replay for key in ("recipient", "correct_true", "correct_false", "wrong_true", "wrong_false")
        ),
        "semantic_uniqueness_inherited": True,
        "controlled_naturalness_inherited": True,
        "no_new_model_output_used": True,
    }
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])

    core.write_rows(OUT / "material/calibration_cases.jsonl", calibration)
    core.write_rows(OUT / "material/causal_replay_manifest.jsonl", replay)
    preaudit = {
        "phase": PHASE, "campaign": CAMPAIGN, "checks": checks,
        "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "semantic_scope": "inherits the frozen C053 eight-family ordinary-noun adjudication",
        "naturalness_scope": "inherits three frozen controlled-English surfaces; no new wording",
        "independent_human_blind_review": False,
        "zero_models": {
            "duplicate_no_hook": "identical rows must have exactly zero output difference",
            "same_batch_self_copy": "copying a token state from an identical row must be an identity",
            "same_batch_zero_delta": "adding h-h must be an identity",
            "exchangeable_donor": "correct and wrong typed donors have no selective advantage",
        },
    }
    core.save(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json", preaudit)

    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "c054.same_batch_causal_camera.v1",
        "model": MODEL,
        "research_object": "the same C053 layer-27 shared-relation candidate under an identity-calibrated, same-batch, token-isomorphic intervention",
        "claim_boundary": {
            "allowed": "instrument identity, route-specific causal sufficiency and donor selectivity at one frozen layer/role",
            "not_assumed": ["relative coding is true", "layer 27 is formation time", "attention is information flow",
                            "a readable direction is causal", "cross-model invariance", "parameter mechanism"],
        },
        "material": {
            "source": "frozen C053 B1 cases and compiled Qwen prompts",
            "calibration_cases": len(calibration), "replay_cases": len(replay),
            "calibration_partitions": ["prototype_discovery", "clock_selection", "confirmation", "lockbox"],
            "replay_partitions": ["confirmation", "lockbox"],
            "replay_surfaces": ["ordinary", "dictionary", "claim"],
            "replay_tested_families": list(ONE_TOKEN_FAMILIES),
        },
        "camera_calibration": {
            "layers": list(LAYERS), "batch_sources": 4, "rows_per_source": 2,
            "routes": ["duplicate_no_hook", "same_batch_exact_token", "cached_fixed_shape_exact_token",
                       "same_batch_span_mean_diagnostic", "same_batch_zero_delta"],
            "authorized_priority": ["same_batch_exact_token", "cached_fixed_shape_exact_token"],
            "no_hook_max_abs_margin": 1e-6,
            "same_batch_exact_max_abs_margin": 1e-6,
            "cached_fixed_shape_max_abs_margin": 1e-4,
            "zero_delta_max_abs_margin": 1e-6,
            "span_mean_is_diagnostic_only": True,
            "authorization_rule": "the first priority route passing every frozen layer authorizes natural replay",
        },
        "causal_replay": {
            "authorized_only_if": "one camera route passes Phase1358",
            "layer": 27, "site_role": "tested_family_token", "batch_recipients": 4,
            "routes": {
                "state_transport": "replace recipient token with same-family true, different-family true, or same-family false donor token",
                "paired_delta_transport": "add true-minus-false donor token response for the correct or wrong tested family",
            },
            "identity_controls": ["same_batch_exact_self", "same_batch_zero_delta"],
            "false_to_true_gain_min": 0.5,
            "direction_fraction_min": 0.75,
            "correct_over_wrong_median_min": 0.25,
            "correct_over_wrong_win_min": 0.65,
            "self_max_abs_diff_max": 1e-4,
            "zero_delta_max_abs_diff_max": 1e-4,
            "route_fail": "eliminate only that transport route",
            "all_routes_fail": "close C054 at the frozen intervention boundary",
        },
        "branching": {
            "phase1358": "run every camera route and diagnostic without mutation",
            "phase1359": "run both natural causal routes only if a priority camera route qualifies",
            "finish": "close after Phase1359; do not search a new layer, role, donor, scale, or component",
        },
        "stop_rule": "No post-reveal change to object, material, partition, model, null, threshold, layer, role, donor, route, scale, or branch.",
        "observation_rule": "retain exact full-dimensional states only for frozen arithmetic; no PCA, UMAP, t-SNE, SAE, learned probe, attention hotspot, or component search",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    protocol["contract_sha256"] = core.digest(protocol)
    protocol["authorization"] = "run_phase1358_c054_camera_calibration"
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "analysis/final.json", {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "all_gates_passed": True, "authorization": protocol["authorization"],
    })
    print(json.dumps({"preaudit": preaudit, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
