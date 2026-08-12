#!/usr/bin/env python3
"""Freeze natural K/V bridging on untouched full-vocabulary behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1049_qkv_read_path_protocol as route
import phase1050_head_group_natural_validation_protocol as groups
import phase1051_natural_behavior_protocol as behavior


PHASE = 1052
PROTOCOL_REVISION = 1
MODELS = behavior.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
BEHAVIOR_ROOT = behavior.OUT_ROOT
ROUTE_ROOT = route.OUT_ROOT
GROUP_ROOT = groups.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1052_full_vocab_kv_bridge"
)
MAX_ROLE_SPAN = 4
ROLLOUT_STEPS = 8
ROLLOUT_PAIR_LIMIT = 32
CONDITION_ORDER = (
    "selected_frozen_groups_frozen_depths",
    "unselected_frozen_groups_frozen_depths",
    "query_frozen_groups_frozen_depths",
    "selected_weak_group_frozen_depths",
    "selected_all_groups_frozen_depths",
    "selected_frozen_groups_all_postsource",
    "selected_all_groups_all_postsource",
)
ROLLOUT_CONDITION_PRIORITY = (
    "selected_frozen_groups_frozen_depths",
    "selected_frozen_groups_all_postsource",
    "selected_all_groups_frozen_depths",
    "selected_all_groups_all_postsource",
)
GATES = {
    "clean_correct_pair_count_min": 100,
    "clean_family_coverage_min": 8,
    "local_both_counterfactual_pair_count_min": 20,
    "local_both_counterfactual_rate_min": 0.10,
    "local_selected_minus_control_rate_min": 0.05,
    "broad_both_counterfactual_rate_min": 0.50,
    "rollout_pair_count_min": 20,
    "rollout_both_match_other_clean_rate_min": 0.50,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def main() -> None:
    behavior_aggregate = read_json(BEHAVIOR_ROOT / "aggregate.json")
    next_decision = behavior_aggregate["automatic_next_decision"]
    if (
        not next_decision["should_continue_automatically"]
        or next_decision["route"] != "phase1052_full_vocab_kv_bridge"
    ):
        raise RuntimeError(
            f"Phase1051 did not authorize Phase1052: {next_decision}"
        )
    behavior_prereg = read_json(
        BEHAVIOR_ROOT / "protocol" / "preregistration.json"
    )
    route_prereg = read_json(
        ROUTE_ROOT / "protocol" / "preregistration.json"
    )
    targets = read_jsonl(
        BEHAVIOR_ROOT / "protocol" / "causal_holdout_targets.jsonl"
    )
    if len(targets) != 440:
        raise RuntimeError(f"causal holdout target drift: {len(targets)}")
    write_jsonl(OUT_ROOT / "protocol" / "targets.jsonl", targets)

    model_plans: dict[str, Any] = {}
    model_audits: dict[str, Any] = {}
    passing = set(behavior_aggregate["passing_models"])
    for model_name in MODELS:
        summary = read_json(
            GROUP_ROOT / "atlas" / model_name / "summary.json"
        )
        behavior_summary = read_json(
            BEHAVIOR_ROOT / "atlas" / model_name / "summary.json"
        )
        frozen_groups = [
            int(value) for value in summary["frozen_kv_groups"]
        ]
        ranking = [
            int(row["kv_group"]) for row in summary["discovery_ranking"]
        ]
        n_kv_heads = int(summary["model_info"]["n_kv_heads"])
        all_groups = list(range(n_kv_heads))
        complement = [
            value for value in ranking if value not in frozen_groups
        ]
        weak_group = (
            complement[-1:] if complement else ranking[-1:]
        )
        route_info = route_prereg["model_info"][model_name]
        conditions = {
            "selected_frozen_groups_frozen_depths": {
                "site": "selected_concept",
                "groups": frozen_groups,
                "depths": route_info["frozen_union_depths"],
            },
            "unselected_frozen_groups_frozen_depths": {
                "site": "unselected_concept",
                "groups": frozen_groups,
                "depths": route_info["frozen_union_depths"],
            },
            "query_frozen_groups_frozen_depths": {
                "site": "query_nonce",
                "groups": frozen_groups,
                "depths": route_info["frozen_union_depths"],
            },
            "selected_weak_group_frozen_depths": {
                "site": "selected_concept",
                "groups": weak_group,
                "depths": route_info["frozen_union_depths"],
            },
            "selected_all_groups_frozen_depths": {
                "site": "selected_concept",
                "groups": all_groups,
                "depths": route_info["frozen_union_depths"],
            },
            "selected_frozen_groups_all_postsource": {
                "site": "selected_concept",
                "groups": frozen_groups,
                "depths": route_info["all_postsource_depths"],
            },
            "selected_all_groups_all_postsource": {
                "site": "selected_concept",
                "groups": all_groups,
                "depths": route_info["all_postsource_depths"],
            },
        }
        if tuple(conditions) != CONDITION_ORDER:
            raise RuntimeError("condition order drift")
        model_plans[model_name] = {
            "behavior_eligible": model_name in passing,
            "frozen_variant": behavior_summary["frozen_variant"],
            "n_layers": int(route_info["n_layers"]),
            "n_kv_heads": n_kv_heads,
            "source_depth": int(route_info["source_depth"]),
            "frozen_groups": frozen_groups,
            "weak_group_control": weak_group,
            "conditions": conditions,
        }

        model_cases = read_jsonl(
            BEHAVIOR_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        frozen_variant = behavior_summary["frozen_variant"]
        causal_indices = {
            int(target[key])
            for target in targets
            for key in ("target_case_index", "cross_case_index")
        }
        filtered = [
            row for row in model_cases
            if row["variant"] == frozen_variant
            and int(row["semantic_case_index"]) in causal_indices
        ]
        filtered.sort(key=lambda row: int(row["semantic_case_index"]))
        if len(filtered) != 880:
            raise RuntimeError(
                f"{model_name} causal case count {len(filtered)}"
            )
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            filtered,
        )
        lookup = {
            int(row["semantic_case_index"]): row for row in filtered
        }
        aligned = []
        for target in targets:
            left = lookup[int(target["target_case_index"])]
            right = lookup[int(target["cross_case_index"])]
            aligned.append(
                len(left["input_ids"]) == len(right["input_ids"])
                and left["role_spans"]["selected_concept"]
                == right["role_spans"]["selected_concept"]
                and left["role_spans"]["unselected_concept"]
                == right["role_spans"]["unselected_concept"]
                and left["role_spans"]["query_nonce"]
                == right["role_spans"]["query_nonce"]
            )
        model_audits[model_name] = {
            "case_count": len(filtered),
            "all_pairs_aligned": all(aligned),
            "maximum_role_span": max(
                end - start + 1
                for row in filtered
                for start, end in row["role_spans"].values()
            ),
        }
    audit = {
        "schema_version": "phase1052_protocol_audit.v1",
        "phase": PHASE,
        "target_count": len(targets),
        "unique_unit_count": len({
            int(row["unit_index"]) for row in targets
        }),
        "model_audits": model_audits,
    }
    audit["all_checks_passed"] = (
        audit["unique_unit_count"] == 110
        and all(
            row["all_pairs_aligned"]
            and row["maximum_role_span"] <= MAX_ROLE_SPAN
            for row in model_audits.values()
        )
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1052 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1052_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_behavior_digest": behavior_prereg["protocol_digest"],
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "causal_holdout_pair_count": len(targets),
        "model_plans": model_plans,
        "condition_order": list(CONDITION_ORDER),
        "rollout_condition_priority": list(
            ROLLOUT_CONDITION_PRIORITY
        ),
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "gates": GATES,
        "metric_priority": [
            "full_vocabulary_both_direction_counterfactual_top1",
            "multi_token_matches_other_clean_trajectory",
            "candidate_margin_only_as_diagnostic",
        ],
        "automatic_next": {
            "local_repeated_and_rollout": (
                "phase1053_pattern_family_extension"
            ),
            "broad_repeated_only": (
                "phase1053_output_bridge_localization"
            ),
            "otherwise": "stop_and_reassess_transport",
        },
        "interpretation_limits": [
            "Only Phase1051 passing models count toward repetition gates.",
            "DS7B is still run as an exploratory model.",
            "All-postsource/all-group swap is a graph cut, not localization.",
            "A selected-position swap transfers a complete natural K/V slice.",
            "No threshold is a language-law formula.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", payload)
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"targets={len(targets)}"
    )


if __name__ == "__main__":
    main()
