#!/usr/bin/env python3
"""Freeze an independent causal test for Phase1065 transitive reasoning."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1065_multimode_response_atlas_protocol as source


PHASE = 1066
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SELECTED_FAMILY = "transitive_reasoning"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1066_reasoning_role_causal"
)
SOURCE_AGGREGATE = source.OUT_ROOT / "aggregate.json"
SOURCE_PHASE1064_PREREG = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1064_cross_panel_transport"
    / "protocol"
    / "preregistration.json"
)
PAIR_LIMIT = 120
GATES = {
    "clean_candidate_replay_min": 0.99,
    "bidirectional_flip_rate_min": 0.40,
    "maximum_role_control_flip_rate": 0.10,
    "source_minus_control_min": 0.30,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def quarter_layers(n_layers: int) -> dict[str, list[int]]:
    result = {"q1": [], "q2": [], "q3": [], "q4": []}
    for depth in range(1, n_layers + 1):
        quarter = min(3, int((depth - 1) * 4 / n_layers))
        result[f"q{quarter + 1}"].append(depth)
    result["q2_q4"] = (
        result["q2"] + result["q3"] + result["q4"]
    )
    result["q3_q4"] = result["q3"] + result["q4"]
    result["all"] = list(range(1, n_layers + 1))
    return result


def conditions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    all_groups = [int(value) for value in plan["all_groups"]]
    depth_sets = plan["depth_sets"]
    rows = [
        {
            "condition": "premise1_all_kv",
            "sites": ["source_primary"],
            "channels": ["k", "v"],
            "depths": depth_sets["all"],
            "groups": all_groups,
            "kind": "source",
        },
        {
            "condition": "premise2_all_kv",
            "sites": ["source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["all"],
            "groups": all_groups,
            "kind": "source",
        },
        {
            "condition": "both_all_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["all"],
            "groups": all_groups,
            "kind": "source",
        },
        {
            "condition": "both_q1_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["q1"],
            "groups": all_groups,
            "kind": "depth",
        },
        {
            "condition": "both_q2_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["q2"],
            "groups": all_groups,
            "kind": "depth",
        },
        {
            "condition": "both_q3_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["q3"],
            "groups": all_groups,
            "kind": "depth",
        },
        {
            "condition": "both_q4_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["q4"],
            "groups": all_groups,
            "kind": "depth",
        },
        {
            "condition": "both_q2_q4_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["q2_q4"],
            "groups": all_groups,
            "kind": "depth",
        },
        {
            "condition": "both_q3_q4_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["q3_q4"],
            "groups": all_groups,
            "kind": "depth",
        },
        {
            "condition": "both_all_k_only",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k"],
            "depths": depth_sets["all"],
            "groups": all_groups,
            "kind": "channel",
        },
        {
            "condition": "both_all_v_only",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["v"],
            "depths": depth_sets["all"],
            "groups": all_groups,
            "kind": "channel",
        },
        {
            "condition": "operator_all_kv",
            "sites": ["operator"],
            "channels": ["k", "v"],
            "depths": depth_sets["all"],
            "groups": all_groups,
            "kind": "role_control",
        },
        {
            "condition": "query_all_kv",
            "sites": ["query"],
            "channels": ["k", "v"],
            "depths": depth_sets["all"],
            "groups": all_groups,
            "kind": "role_control",
        },
    ]
    for group in all_groups:
        rows.append({
            "condition": f"both_all_group{group}_kv",
            "sites": ["source_primary", "source_secondary"],
            "channels": ["k", "v"],
            "depths": depth_sets["all"],
            "groups": [group],
            "kind": "group",
        })
    return rows


def build_protocol() -> dict[str, Any]:
    source_aggregate = read_json(SOURCE_AGGREGATE)
    automatic = source_aggregate["automatic_next_decision"]
    if (
        not automatic["should_continue_automatically"]
        or automatic["selected_family"] != SELECTED_FAMILY
    ):
        raise RuntimeError("Phase1065 did not authorize Phase1066")
    source_prereg = read_json(
        source.OUT_ROOT / "protocol" / "preregistration.json"
    )
    phase1064 = read_json(SOURCE_PHASE1064_PREREG)
    model_plans = {}
    model_audits = {}
    for model in MODELS:
        cases = source.read_jsonl(
            source.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
        behavior_rows = source.read_jsonl(
            source.OUT_ROOT
            / "atlas"
            / model
            / "candidate_behavior.jsonl"
        )
        hit = {
            int(row["semantic_case_index"]): bool(row["candidate_hit"])
            for row in behavior_rows
        }
        by_unit: dict[str, dict[str, dict[str, Any]]] = {}
        for row in cases:
            if row["family"] != SELECTED_FAMILY:
                continue
            by_unit.setdefault(str(row["unit_id"]), {})[
                str(row["state"])
            ] = row
        pairs = []
        for unit_id, by_state in sorted(by_unit.items()):
            for lexical in (0, 1):
                left = by_state[f"b0_l{lexical}"]
                right = by_state[f"b1_l{lexical}"]
                if not (
                    hit[int(left["semantic_case_index"])]
                    and hit[int(right["semantic_case_index"])]
                ):
                    continue
                pairs.append({
                    "schema_version": "phase1066_reasoning_pair.v1",
                    "phase": PHASE,
                    "model": model,
                    "pair_index": len(pairs),
                    "unit_id": unit_id,
                    "split": left["split"],
                    "lexical_branch": lexical,
                    "left_case_index": int(
                        left["semantic_case_index"]
                    ),
                    "right_case_index": int(
                        right["semantic_case_index"]
                    ),
                    "left_expected_class": "b0",
                    "right_expected_class": "b1",
                })
        pairs = pairs[:PAIR_LIMIT]
        source_plan = phase1064["model_plans"][model]
        n_layers = len(source_plan["all_layers"])
        plan = {
            "n_layers": n_layers,
            "n_kv_heads": int(source_plan["n_kv_heads"]),
            "all_groups": [
                int(value) for value in source_plan["all_groups"]
            ],
            "depth_sets": quarter_layers(n_layers),
        }
        plan["conditions"] = conditions(plan)
        checks = {
            "pair_count": len(pairs) == PAIR_LIMIT,
            "discovery_pair_count": sum(
                row["split"] == "discovery" for row in pairs
            ) == PAIR_LIMIT // 2,
            "confirmation_pair_count": sum(
                row["split"] == "confirmation" for row in pairs
            ) == PAIR_LIMIT // 2,
            "both_lexical_branches_present": set(
                int(row["lexical_branch"]) for row in pairs
            ) == {0, 1},
            "quarter_partition_complete": sorted(
                plan["depth_sets"]["q1"]
                + plan["depth_sets"]["q2"]
                + plan["depth_sets"]["q3"]
                + plan["depth_sets"]["q4"]
            ) == plan["depth_sets"]["all"],
            "channels_are_explicit": all(
                set(row["channels"]) <= {"k", "v"}
                and bool(row["channels"])
                for row in plan["conditions"]
            ),
        }
        audit = {
            "schema_version": "phase1066_protocol_model_audit.v1",
            "phase": PHASE,
            "model": model,
            "pair_count": len(pairs),
            "condition_count": len(plan["conditions"]),
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"Phase1066 audit failed: {audit}")
        write_jsonl(
            OUT_ROOT / "protocol" / f"pairs.{model}.jsonl",
            pairs,
        )
        write_json(
            OUT_ROOT / "protocol" / f"audit.{model}.json",
            audit,
        )
        model_plans[model] = plan
        model_audits[model] = audit

    payload = {
        "schema_version": "phase1066_reasoning_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "selected_family": SELECTED_FAMILY,
        "pair_limit_per_model": PAIR_LIMIT,
        "source_phase1065_digest": source_prereg["protocol_digest"],
        "source_phase1065_automatic_decision": automatic,
        "model_plans": model_plans,
        "gates": dict(GATES),
        "primary_outcomes": [
            "bidirectional donor-class flip rate",
            "individual donor-class flip rate",
            "mean change toward the donor candidate margin",
        ],
        "measurement_scope": (
            "K/V at the final token of each marked role span; this is an "
            "endpoint intervention, not full-premise replacement."
        ),
        "interpretation_limits": [
            "A successful donor swap establishes local conditional sufficiency, not natural necessity.",
            "A failed endpoint swap does not show that a premise is unused.",
            "K and V are physical projection channels, not preassigned semantic address and content.",
            "All-layer replacement may create conflict or out-of-distribution states.",
            "The reasoning task is controlled three-item transitivity, not general reasoning.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "At least two models pass a source condition with low "
                "role-control effects."
            ),
            "next_phase": (
                "separate natural-necessity and component-localization "
                "protocol, not automatic formula fitting"
            ),
        },
        "model_audits": model_audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1066_protocol_audit.v1",
            "phase": PHASE,
            "protocol_digest": payload["protocol_digest"],
            "model_audits": model_audits,
            "all_checks_passed": all(
                row["all_checks_passed"]
                for row in model_audits.values()
            ),
        },
    )
    return payload


def main() -> None:
    payload = build_protocol()
    print(
        f"Phase{PHASE} protocol {payload['protocol_digest']} "
        f"pairs={payload['pair_limit_per_model']}/model"
    )


if __name__ == "__main__":
    main()
