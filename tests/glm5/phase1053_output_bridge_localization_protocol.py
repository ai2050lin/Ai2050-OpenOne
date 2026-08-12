#!/usr/bin/env python3
"""Freeze greedy K/V group-depth coalition localization."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1052_full_vocab_kv_bridge_protocol as bridge


PHASE = 1053
PROTOCOL_REVISION = 1
MODELS = behavior.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
BEHAVIOR_ROOT = behavior.OUT_ROOT
BRIDGE_ROOT = bridge.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1053_output_bridge_localization"
)
DISCOVERY_PAIR_COUNT = 80
DEPTH_SLOT_COUNT = 6
RETENTION_FRACTION = 0.90
DISCOVERY_ABSOLUTE_RATE_MIN = 0.20
ROLLOUT_STEPS = 8
ROLLOUT_PAIR_LIMIT = 32
CONFIRMATION_CONDITIONS = (
    "selected_full",
    "selected_joint_coalition",
    "unselected_joint_coalition",
    "query_joint_coalition",
)
GATES = {
    "confirmation_clean_pair_count_min": 100,
    "confirmation_family_coverage_min": 8,
    "joint_both_counterfactual_rate_min": 0.50,
    "joint_retained_fraction_min": 0.80,
    "selected_minus_control_rate_min": 0.20,
    "rollout_pair_count_min": 20,
    "rollout_both_match_other_clean_rate_min": 0.50,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def depth_slots(depths: list[int], count: int) -> list[list[int]]:
    quotient, remainder = divmod(len(depths), count)
    slots = []
    cursor = 0
    for index in range(count):
        width = quotient + int(index < remainder)
        slots.append(depths[cursor:cursor + width])
        cursor += width
    if cursor != len(depths) or any(not slot for slot in slots):
        raise RuntimeError("depth slot construction drift")
    return slots


def evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError("sample count exceeds population")
    indices = [
        (index * len(rows)) // count for index in range(count)
    ]
    if len(set(indices)) != count:
        raise RuntimeError("even sampling collision")
    return [dict(rows[index]) for index in indices]


def main() -> None:
    bridge_aggregate = read_json(BRIDGE_ROOT / "aggregate.json")
    next_decision = bridge_aggregate["automatic_next_decision"]
    if (
        not next_decision["should_continue_automatically"]
        or next_decision["route"]
        != "phase1053_output_bridge_localization"
    ):
        raise RuntimeError(
            f"Phase1052 did not authorize Phase1053: {next_decision}"
        )
    bridge_prereg = read_json(
        BRIDGE_ROOT / "protocol" / "preregistration.json"
    )
    source_discovery = read_jsonl(
        BRIDGE_ROOT / "protocol" / "targets.jsonl"
    )
    discovery_targets = evenly_spaced(
        source_discovery, DISCOVERY_PAIR_COUNT
    )
    confirmation_targets = read_jsonl(
        BEHAVIOR_ROOT / "protocol" / "confirmation_targets.jsonl"
    )
    if len(confirmation_targets) != 400:
        raise RuntimeError("confirmation target count drift")
    write_jsonl(
        OUT_ROOT / "protocol" / "discovery_targets.jsonl",
        discovery_targets,
    )
    write_jsonl(
        OUT_ROOT / "protocol" / "confirmation_targets.jsonl",
        confirmation_targets,
    )

    model_plans: dict[str, Any] = {}
    model_audits: dict[str, Any] = {}
    behavior_aggregate = read_json(BEHAVIOR_ROOT / "aggregate.json")
    eligible = set(behavior_aggregate["passing_models"])
    for model_name in MODELS:
        plan = bridge_prereg["model_plans"][model_name]
        all_depths = [
            int(value)
            for value in plan["conditions"][
                "selected_all_groups_all_postsource"
            ]["depths"]
        ]
        slots = depth_slots(all_depths, DEPTH_SLOT_COUNT)
        all_groups = list(range(int(plan["n_kv_heads"])))
        model_plans[model_name] = {
            "behavior_eligible": model_name in eligible,
            "frozen_variant": plan["frozen_variant"],
            "n_kv_heads": int(plan["n_kv_heads"]),
            "all_groups": all_groups,
            "all_postsource_depths": all_depths,
            "depth_slots": slots,
        }

        bridge_cases = read_jsonl(
            BRIDGE_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        discovery_indices = {
            int(target[key])
            for target in discovery_targets
            for key in ("target_case_index", "cross_case_index")
        }
        discovery_cases = [
            row for row in bridge_cases
            if int(row["semantic_case_index"]) in discovery_indices
        ]
        behavior_cases = read_jsonl(
            BEHAVIOR_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        confirmation_indices = {
            int(target[key])
            for target in confirmation_targets
            for key in ("target_case_index", "cross_case_index")
        }
        confirmation_cases = [
            row for row in behavior_cases
            if row["variant"] == plan["frozen_variant"]
            and int(row["semantic_case_index"]) in confirmation_indices
        ]
        if (
            len(discovery_cases) != 2 * DISCOVERY_PAIR_COUNT
            or len(confirmation_cases) != 800
        ):
            raise RuntimeError(
                f"{model_name} case reserve drift: "
                f"{len(discovery_cases)}/{len(confirmation_cases)}"
            )
        combined = discovery_cases + confirmation_cases
        combined.sort(key=lambda row: int(row["semantic_case_index"]))
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            combined,
        )
        model_audits[model_name] = {
            "discovery_case_count": len(discovery_cases),
            "confirmation_case_count": len(confirmation_cases),
            "case_index_overlap": bool(
                {
                    int(row["semantic_case_index"])
                    for row in discovery_cases
                }.intersection(
                    int(row["semantic_case_index"])
                    for row in confirmation_cases
                )
            ),
            "depth_slot_sizes": [len(slot) for slot in slots],
            "depth_slot_flattened": [
                depth for slot in slots for depth in slot
            ],
        }

    discovery_family_counts = Counter(
        (
            row["target_expected_label"],
            row["cross_expected_label"],
        )
        for row in discovery_targets
    )
    audit = {
        "schema_version": "phase1053_protocol_audit.v1",
        "phase": PHASE,
        "discovery_pair_count": len(discovery_targets),
        "confirmation_pair_count": len(confirmation_targets),
        "discovery_family_pair_count": len(discovery_family_counts),
        "model_audits": model_audits,
    }
    audit["all_checks_passed"] = (
        len(discovery_targets) == DISCOVERY_PAIR_COUNT
        and len(confirmation_targets) == 400
        and len(discovery_family_counts) >= 10
        and all(
            not row["case_index_overlap"]
            and row["depth_slot_flattened"]
            == model_plans[model]["all_postsource_depths"]
            for model, row in model_audits.items()
        )
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1053 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1053_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_bridge_digest": bridge_prereg["protocol_digest"],
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "discovery_pair_count": len(discovery_targets),
        "confirmation_pair_count": len(confirmation_targets),
        "model_plans": model_plans,
        "greedy_group_rule": (
            "Start with all groups. At each step test every one-group "
            "removal and accept the removal with the highest discovery "
            "both-direction top-1 rate only if it retains at least 90% "
            "of the full rate and remains >= 0.20."
        ),
        "greedy_depth_rule": (
            "Apply the same deletion rule to six contiguous normalized "
            "post-source depth slots while using all groups."
        ),
        "retention_fraction": RETENTION_FRACTION,
        "discovery_absolute_rate_min": DISCOVERY_ABSOLUTE_RATE_MIN,
        "confirmation_conditions": list(CONFIRMATION_CONDITIONS),
        "gates": GATES,
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "automatic_next": {
            "joint_coalition_repeats": (
                "phase1054_pattern_family_extension"
            ),
            "otherwise": "stop_with_distributed_coalition_map",
        },
        "interpretation_limits": [
            "Greedy deletion does not prove a globally minimal coalition.",
            "Discovery reuses a fixed subset of Phase1052 material.",
            "Confirmation units were used only for clean behavior before.",
            "Physical groups and slots are model-specific coordinates.",
            "A coalition is a functional intervention set, not a module.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", payload)
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"discovery={len(discovery_targets)} "
        f"confirmation={len(confirmation_targets)}"
    )


if __name__ == "__main__":
    main()
