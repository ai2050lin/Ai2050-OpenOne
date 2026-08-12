#!/usr/bin/env python3
"""Freeze joint K/V rectangle search and EOS-aware rollout auditing."""

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
import phase1053_output_bridge_localization_protocol as localization


PHASE = 1054
PROTOCOL_REVISION = 1
MODELS = behavior.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
BEHAVIOR_ROOT = behavior.OUT_ROOT
BRIDGE_ROOT = bridge.OUT_ROOT
LOCALIZATION_ROOT = localization.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1054_joint_kv_rollout_audit"
)
DISCOVERY_PAIR_COUNT = 120
BEAM_WIDTH = 3
SEARCH_RETENTION_FRACTION = 0.85
SEARCH_ABSOLUTE_RATE_MIN = 0.20
SEARCH_MAX_EVALUATIONS = {
    "qwen3": 180,
    "glm4": 80,
    "deepseek7b": 120,
}
ROLLOUT_STEPS = 8
ROLLOUT_PAIR_LIMIT = 64
CONFIRMATION_CONDITIONS = (
    "selected_full",
    "selected_joint_rectangle",
    "unselected_joint_rectangle",
    "query_joint_rectangle",
)
GATES = {
    "confirmation_clean_pair_count_min": 100,
    "confirmation_family_coverage_min": 8,
    "joint_both_counterfactual_rate_min": 0.50,
    "joint_retained_fraction_min": 0.80,
    "selected_minus_control_rate_min": 0.20,
    "maximum_block_fraction": 0.75,
    "rollout_pair_count_min": 30,
    "eos_censored_both_match_rate_min": 0.80,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def evenly_spaced(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError("sample count exceeds population")
    indices = [(index * len(rows)) // count for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("even sampling collision")
    return [dict(rows[index]) for index in indices]


def relabel_targets(
    rows: list[dict[str, Any]],
    partition: str,
    offset: int,
) -> list[dict[str, Any]]:
    result = []
    for local_index, source in enumerate(rows):
        row = dict(source)
        row["source_target_index"] = int(row["target_index"])
        row["target_index"] = offset + local_index
        row["phase1054_partition"] = partition
        result.append(row)
    return result


def main() -> None:
    behavior_prereg = read_json(
        BEHAVIOR_ROOT / "protocol" / "preregistration.json"
    )
    bridge_prereg = read_json(
        BRIDGE_ROOT / "protocol" / "preregistration.json"
    )
    localization_prereg = read_json(
        LOCALIZATION_ROOT / "protocol" / "preregistration.json"
    )
    localization_aggregate = read_json(
        LOCALIZATION_ROOT / "aggregate.json"
    )
    if localization_aggregate["automatic_next_decision"][
        "should_continue_automatically"
    ]:
        raise RuntimeError(
            "Phase1054 is a user-authorized redesign after a stop gate"
        )

    bridge_targets = read_jsonl(
        BRIDGE_ROOT / "protocol" / "targets.jsonl"
    )
    prior_discovery = read_jsonl(
        LOCALIZATION_ROOT / "protocol" / "discovery_targets.jsonl"
    )
    prior_indices = {
        int(row["source_target_index"])
        if "source_target_index" in row
        else int(row["target_index"])
        for row in prior_discovery
    }
    remaining = [
        row for row in bridge_targets
        if int(row["target_index"]) not in prior_indices
    ]
    if len(remaining) != 360:
        raise RuntimeError(
            f"remaining bridge target drift: {len(remaining)}"
        )
    discovery = relabel_targets(
        evenly_spaced(remaining, DISCOVERY_PAIR_COUNT),
        "discovery",
        0,
    )
    confirmation_source = read_jsonl(
        BEHAVIOR_ROOT / "protocol" / "discovery_targets.jsonl"
    )
    confirmation = relabel_targets(
        confirmation_source,
        "causal_confirmation",
        len(discovery),
    )
    if len(confirmation) != 120:
        raise RuntimeError("causal confirmation reserve drift")
    write_jsonl(
        OUT_ROOT / "protocol" / "discovery_targets.jsonl",
        discovery,
    )
    write_jsonl(
        OUT_ROOT / "protocol" / "confirmation_targets.jsonl",
        confirmation,
    )

    behavior_aggregate = read_json(BEHAVIOR_ROOT / "aggregate.json")
    eligible = set(behavior_aggregate["passing_models"])
    model_plans: dict[str, Any] = {}
    model_audits: dict[str, Any] = {}
    for model_name in MODELS:
        prior_plan = localization_prereg["model_plans"][model_name]
        model_plans[model_name] = {
            "behavior_eligible": model_name in eligible,
            "frozen_variant": prior_plan["frozen_variant"],
            "n_kv_heads": int(prior_plan["n_kv_heads"]),
            "all_groups": [
                int(value) for value in prior_plan["all_groups"]
            ],
            "all_postsource_depths": [
                int(value)
                for value in prior_plan["all_postsource_depths"]
            ],
            "depth_slots": [
                [int(value) for value in slot]
                for slot in prior_plan["depth_slots"]
            ],
            "search_max_evaluations": SEARCH_MAX_EVALUATIONS[
                model_name
            ],
        }

        bridge_cases = read_jsonl(
            BRIDGE_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        discovery_indices = {
            int(row[key])
            for row in discovery
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
            int(row[key])
            for row in confirmation
            for key in ("target_case_index", "cross_case_index")
        }
        confirmation_cases = [
            row for row in behavior_cases
            if row["variant"] == prior_plan["frozen_variant"]
            and int(row["semantic_case_index"]) in confirmation_indices
        ]
        if (
            len(discovery_cases) != 2 * len(discovery)
            or len(confirmation_cases) != 2 * len(confirmation)
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
        discovery_case_indices = {
            int(row["semantic_case_index"]) for row in discovery_cases
        }
        confirmation_case_indices = {
            int(row["semantic_case_index"]) for row in confirmation_cases
        }
        model_audits[model_name] = {
            "discovery_case_count": len(discovery_cases),
            "confirmation_case_count": len(confirmation_cases),
            "case_index_overlap": bool(
                discovery_case_indices.intersection(
                    confirmation_case_indices
                )
            ),
            "maximum_role_span": max(
                end - start + 1
                for row in combined
                for start, end in row["role_spans"].values()
            ),
        }

    discovery_units = {int(row["unit_index"]) for row in discovery}
    confirmation_units = {
        int(row["unit_index"]) for row in confirmation
    }
    discovery_family_pairs = Counter(
        (
            str(row["target_expected_label"]),
            str(row["cross_expected_label"]),
        )
        for row in discovery
    )
    confirmation_family_pairs = Counter(
        (
            str(row["target_expected_label"]),
            str(row["cross_expected_label"]),
        )
        for row in confirmation
    )
    audit = {
        "schema_version": "phase1054_protocol_audit.v1",
        "phase": PHASE,
        "discovery_pair_count": len(discovery),
        "confirmation_pair_count": len(confirmation),
        "unit_overlap": bool(
            discovery_units.intersection(confirmation_units)
        ),
        "discovery_family_pair_count": len(discovery_family_pairs),
        "confirmation_family_pair_count": len(
            confirmation_family_pairs
        ),
        "model_audits": model_audits,
    }
    audit["all_checks_passed"] = (
        len(discovery) == DISCOVERY_PAIR_COUNT
        and len(confirmation) == 120
        and not audit["unit_overlap"]
        and len(discovery_family_pairs) >= 10
        and len(confirmation_family_pairs) >= 10
        and all(
            not row["case_index_overlap"]
            and row["maximum_role_span"] <= bridge.MAX_ROLE_SPAN
            for row in model_audits.values()
        )
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1054 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1054_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_behavior_digest": behavior_prereg["protocol_digest"],
        "source_bridge_digest": bridge_prereg["protocol_digest"],
        "source_localization_digest": localization_prereg[
            "protocol_digest"
        ],
        "authorization": (
            "User-authorized method redesign after the Phase1053 "
            "automatic stop gate."
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "discovery_pair_count": len(discovery),
        "confirmation_pair_count": len(confirmation),
        "confirmation_exposure": (
            "The confirmation units selected the frozen output format "
            "in Phase1051 but have never received a K/V intervention."
        ),
        "model_plans": model_plans,
        "beam_width": BEAM_WIDTH,
        "search_retention_fraction": SEARCH_RETENTION_FRACTION,
        "search_absolute_rate_min": SEARCH_ABSOLUTE_RATE_MIN,
        "search_rule": (
            "Starting from all KV groups and all six depth slots, jointly "
            "delete either one group or one slot. Retain the three "
            "feasible states with the fewest Cartesian group-depth "
            "blocks, breaking ties by higher discovery top-1 flip rate. "
            "Continue until no feasible child or the frozen evaluation "
            "budget is reached."
        ),
        "confirmation_conditions": list(CONFIRMATION_CONDITIONS),
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "rollout_rule": (
            "Generate eight greedy tokens for comparability, then include "
            "the first model-configured EOS token and censor every token "
            "after it before trajectory comparison."
        ),
        "gates": GATES,
        "automatic_next": {
            "two_models_repeat_and_rollout_is_valid": (
                "phase1055_pattern_family_transfer"
            ),
            "broad_cut_repeats_but_rectangle_does_not": (
                "phase1055_nonrectangular_block_atlas"
            ),
            "otherwise": "stop_and_reassess_output_bridge",
        },
        "interpretation_limits": [
            "The search returns a compact rectangle, not a global minimum.",
            "A rectangle cannot represent an arbitrary sparse cell set.",
            "Confirmation behavior was exposed to format selection only.",
            "EOS-censored matching audits a one-label output protocol.",
            "It does not establish free-form multi-token language closure.",
            "Coordinates are model-specific and only roles are comparable.",
            "No result establishes brain homology or optimality.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", payload)
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"discovery={len(discovery)} confirmation={len(confirmation)}"
    )


if __name__ == "__main__":
    main()
