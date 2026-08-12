#!/usr/bin/env python3
"""Freeze reasoning neutralization and K/V-group coalition tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1065_multimode_response_atlas_protocol as atlas
import phase1066_reasoning_role_causal_protocol as source


PHASE = 1067
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1067_reasoning_necessity_coalition"
)
SOURCE_AGGREGATE = source.OUT_ROOT / "aggregate.json"
PAIR_LIMIT = 120
GATES = {
    "clean_candidate_replay_min": 0.99,
    "neutralization_accuracy_drop_min": 0.20,
    "semantic_preserving_control_drop_max": 0.10,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def unique_group_sets(groups: list[int]) -> list[tuple[str, list[int]]]:
    rows: list[tuple[str, list[int]]] = []
    seen = set()

    def add(name: str, values: list[int]) -> None:
        key = tuple(sorted(values))
        if not key or key in seen:
            return
        seen.add(key)
        rows.append((name, list(key)))

    midpoint = max(1, len(groups) // 2)
    add("first_half", groups[:midpoint])
    add("second_half", groups[midpoint:])
    add("even_groups", [value for value in groups if value % 2 == 0])
    add("odd_groups", [value for value in groups if value % 2 == 1])
    for removed in groups:
        add(
            f"leave_group{removed}_out",
            [value for value in groups if value != removed],
        )
    return rows


def conditions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    groups = [int(value) for value in plan["all_groups"]]
    depths = [int(value) for value in plan["depths"]]
    sites = ["source_primary", "source_secondary"]
    rows = [
        {
            "condition": "semantic_swap_kv",
            "pair_set": "semantic",
            "mode": "swap",
            "sites": sites,
            "channels": ["k", "v"],
            "groups": groups,
            "depths": depths,
            "kind": "semantic_swap",
        },
        {
            "condition": "semantic_swap_k_only",
            "pair_set": "semantic",
            "mode": "swap",
            "sites": sites,
            "channels": ["k"],
            "groups": groups,
            "depths": depths,
            "kind": "channel",
        },
        {
            "condition": "semantic_swap_v_only",
            "pair_set": "semantic",
            "mode": "swap",
            "sites": sites,
            "channels": ["v"],
            "groups": groups,
            "depths": depths,
            "kind": "channel",
        },
        {
            "condition": "semantic_mean_kv",
            "pair_set": "semantic",
            "mode": "mean",
            "sites": sites,
            "channels": ["k", "v"],
            "groups": groups,
            "depths": depths,
            "kind": "neutralization",
        },
        {
            "condition": "semantic_mean_k_only",
            "pair_set": "semantic",
            "mode": "mean",
            "sites": sites,
            "channels": ["k"],
            "groups": groups,
            "depths": depths,
            "kind": "neutralization",
        },
        {
            "condition": "semantic_mean_v_only",
            "pair_set": "semantic",
            "mode": "mean",
            "sites": sites,
            "channels": ["v"],
            "groups": groups,
            "depths": depths,
            "kind": "neutralization",
        },
        {
            "condition": "semantic_mean_premise1_kv",
            "pair_set": "semantic",
            "mode": "mean",
            "sites": ["source_primary"],
            "channels": ["k", "v"],
            "groups": groups,
            "depths": depths,
            "kind": "neutralization_role",
        },
        {
            "condition": "semantic_mean_premise2_kv",
            "pair_set": "semantic",
            "mode": "mean",
            "sites": ["source_secondary"],
            "channels": ["k", "v"],
            "groups": groups,
            "depths": depths,
            "kind": "neutralization_role",
        },
        {
            "condition": "surface_preserving_swap_kv",
            "pair_set": "surface",
            "mode": "swap",
            "sites": sites,
            "channels": ["k", "v"],
            "groups": groups,
            "depths": depths,
            "kind": "semantic_preserving_control",
        },
    ]
    for name, selected in unique_group_sets(groups):
        rows.append({
            "condition": f"semantic_swap_{name}_kv",
            "pair_set": "semantic",
            "mode": "swap",
            "sites": sites,
            "channels": ["k", "v"],
            "groups": selected,
            "depths": depths,
            "kind": "group_coalition",
        })
    return rows


def build_protocol() -> dict[str, Any]:
    source_aggregate = read_json(SOURCE_AGGREGATE)
    automatic = source_aggregate["automatic_next_decision"]
    if not automatic["should_continue_automatically"]:
        raise RuntimeError("Phase1066 did not authorize Phase1067")
    source_prereg = read_json(
        source.OUT_ROOT / "protocol" / "preregistration.json"
    )
    model_plans = {}
    audits = {}
    for model in MODELS:
        semantic_pairs = read_jsonl(
            source.OUT_ROOT / "protocol" / f"pairs.{model}.jsonl"
        )
        cases = atlas.read_jsonl(
            atlas.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
        by_unit: dict[str, dict[str, dict[str, Any]]] = {}
        for row in cases:
            if row["family"] != source.SELECTED_FAMILY:
                continue
            by_unit.setdefault(str(row["unit_id"]), {})[
                str(row["state"])
            ] = row
        surface_pairs = []
        for unit_id, by_state in sorted(by_unit.items()):
            for branch in (0, 1):
                left = by_state[f"b{branch}_l0"]
                right = by_state[f"b{branch}_l1"]
                surface_pairs.append({
                    "schema_version": (
                        "phase1067_surface_control_pair.v1"
                    ),
                    "phase": PHASE,
                    "model": model,
                    "pair_index": len(surface_pairs),
                    "unit_id": unit_id,
                    "split": left["split"],
                    "semantic_branch": branch,
                    "left_case_index": int(
                        left["semantic_case_index"]
                    ),
                    "right_case_index": int(
                        right["semantic_case_index"]
                    ),
                    "left_expected_class": f"b{branch}",
                    "right_expected_class": f"b{branch}",
                })
        semantic_pairs = semantic_pairs[:PAIR_LIMIT]
        surface_pairs = surface_pairs[:PAIR_LIMIT]
        source_plan = source_prereg["model_plans"][model]
        plan = {
            "n_layers": int(source_plan["n_layers"]),
            "n_kv_heads": int(source_plan["n_kv_heads"]),
            "all_groups": [
                int(value) for value in source_plan["all_groups"]
            ],
            "depths": [
                int(value)
                for value in source_plan["depth_sets"]["q2_q4"]
            ],
        }
        plan["conditions"] = conditions(plan)
        checks = {
            "semantic_pair_count": len(semantic_pairs) == PAIR_LIMIT,
            "surface_pair_count": len(surface_pairs) == PAIR_LIMIT,
            "surface_pairs_preserve_expected_class": all(
                row["left_expected_class"]
                == row["right_expected_class"]
                for row in surface_pairs
            ),
            "semantic_pairs_change_expected_class": all(
                row["left_expected_class"]
                != row["right_expected_class"]
                for row in semantic_pairs
            ),
            "depths_are_q2_q4": plan["depths"]
            == source_plan["depth_sets"]["q2_q4"],
            "conditions_nonempty": bool(plan["conditions"]),
        }
        audit = {
            "schema_version": "phase1067_protocol_model_audit.v1",
            "phase": PHASE,
            "model": model,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"Phase1067 audit failed: {audit}")
        write_jsonl(
            OUT_ROOT / "protocol" / f"semantic_pairs.{model}.jsonl",
            semantic_pairs,
        )
        write_jsonl(
            OUT_ROOT / "protocol" / f"surface_pairs.{model}.jsonl",
            surface_pairs,
        )
        write_json(
            OUT_ROOT / "protocol" / f"audit.{model}.json",
            audit,
        )
        model_plans[model] = plan
        audits[model] = audit

    payload = {
        "schema_version": "phase1067_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "selected_family": source.SELECTED_FAMILY,
        "pair_limit_per_set_model": PAIR_LIMIT,
        "source_phase1066_digest": source_prereg["protocol_digest"],
        "source_phase1066_automatic_decision": automatic,
        "common_functional_region": (
            "both premise endpoints over normalized quarters Q2-Q4"
        ),
        "model_plans": model_plans,
        "gates": dict(GATES),
        "primary_outcomes": [
            "own-class accuracy loss under paired-mean neutralization",
            "own-class margin suppression under paired-mean neutralization",
            "semantic-preserving surface-swap accuracy loss",
            "donor-flip retention under group coalitions",
        ],
        "interpretation_limits": [
            "Pair-mean neutralization is a local necessity stress test, not a natural lesion.",
            "A group coalition is a GQA K/V projection group, not a single attention head in all architectures.",
            "The common Q2-Q4 region is a functional alignment choice, not a shared physical coordinate.",
            "The controlled task always asks for the tallest of three named people.",
            "No result proves a complete reasoning algorithm or biological optimality.",
        ],
        "automatic_next": {
            "should_not_continue_with_more_compression": True,
            "next_large_task": (
                "expand reasoning relations, chain lengths, query forms, "
                "and answer positions under a separately frozen protocol"
            ),
        },
        "model_audits": audits,
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json", payload
    )
    write_json(
        OUT_ROOT / "protocol" / "audit.json",
        {
            "schema_version": "phase1067_protocol_audit.v1",
            "phase": PHASE,
            "protocol_digest": payload["protocol_digest"],
            "model_audits": audits,
            "all_checks_passed": all(
                row["all_checks_passed"] for row in audits.values()
            ),
        },
    )
    return payload


def main() -> None:
    payload = build_protocol()
    print(
        f"Phase{PHASE} protocol {payload['protocol_digest']} "
        f"pairs={payload['pair_limit_per_set_model']}/set/model"
    )


if __name__ == "__main__":
    main()
