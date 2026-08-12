#!/usr/bin/env python3
"""Freeze an independent receiver reset/replay test for Phase1044."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1041_position_write_alliance_protocol as alliance
import phase1044_natural_recompute_trajectory_protocol as source


PHASE = 1045
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = source.OUT_ROOT
MATERIAL_ROOT = material.OUT_ROOT
ALLIANCE_ROOT = alliance.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1045_receiver_mediation_confirmation"
)

SOURCE_CONDITIONS = (
    "cross_selected",
    "cross_unselected",
    "cross_shuffled",
    "same_family_lexical",
)
OPERATIONS_BY_CONDITION = {
    "cross_selected": ("none", "query_swap", "wrong_site_swap"),
    "cross_unselected": ("none",),
    "cross_shuffled": ("none",),
    "same_family_lexical": ("none",),
}
RECEIVER_SITE = "query_nonce"
WRONG_RECEIVER_SITE = "unselected_concept"
MAX_SOURCE_SPAN = 2
MAX_RECEIVER_SPAN = 3


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def semantic_role(site: str, target: dict[str, Any]) -> str:
    if site == "selected_concept":
        return f"concept_{target['selected_slot']}"
    if site == "unselected_concept":
        return f"concept_{target['unselected_slot']}"
    if site == "query_nonce":
        return "query_nonce"
    raise ValueError(site)


def donor_spec(
    target: dict[str, Any],
    condition: str,
    targets_by_index: dict[int, dict[str, Any]],
) -> tuple[int, dict[str, Any], str]:
    if condition == "cross_selected":
        return (
            int(target["cross_family_case_index"]),
            target,
            "selected_concept",
        )
    if condition == "cross_unselected":
        return (
            int(target["cross_family_case_index"]),
            target,
            "unselected_concept",
        )
    if condition == "cross_shuffled":
        donor_target = targets_by_index[
            int(target["shuffled_target_index"])
        ]
        return (
            int(target["shuffled_cross_case_index"]),
            donor_target,
            "selected_concept",
        )
    if condition == "same_family_lexical":
        return (
            int(target["same_family_case_index"]),
            target,
            "selected_concept",
        )
    raise ValueError(condition)


def used_case_indices(targets: list[dict[str, Any]]) -> set[int]:
    values = set()
    for row in targets:
        values.update({
            int(row["target_case_index"]),
            int(row["cross_family_case_index"]),
            int(row["same_family_case_index"]),
            int(row["shuffled_cross_case_index"]),
        })
    return values


def main() -> None:
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    decision = source_aggregate["automatic_next_decision"]
    if not decision["confirmation_needed"]:
        raise RuntimeError("Phase1044 did not authorize confirmation")
    candidate = decision["candidates"][0]
    expected = {
        "source_mode": "full_state",
        "depth_slot": 3,
        "channel": "layer_output",
        "receiver_site": "query_nonce",
    }
    if any(candidate[key] != value for key, value in expected.items()):
        raise RuntimeError("Phase1044 frozen candidate drift")

    reserved = read_jsonl(
        ALLIANCE_ROOT / "protocol" / "reserved_holdout_targets.jsonl"
    )
    targets = [
        row for row in reserved if int(row["surface_index"]) == 2
    ]
    targets = alliance.add_shuffled_donors(targets)
    discovery_target_indices = {
        int(row["target_index"])
        for row in read_jsonl(
            SOURCE_ROOT / "protocol" / "targets.jsonl"
        )
    }
    for index, row in enumerate(targets):
        row["confirmation_index"] = index
    if len(targets) != 240:
        raise RuntimeError("surface-2 confirmation count drift")
    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)

    used = used_case_indices(targets)
    model_audits = {}
    for model_name in MODELS:
        rows = [
            row for row in read_jsonl(
                MATERIAL_ROOT
                / "protocol"
                / f"cases.{model_name}.jsonl"
            )
            if int(row["case_index"]) in used
        ]
        rows.sort(key=lambda row: int(row["case_index"]))
        if {int(row["case_index"]) for row in rows} != used:
            raise RuntimeError(f"{model_name} confirmation case drift")
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", rows
        )
        cases = {int(row["case_index"]): row for row in rows}
        span_failures = []
        for target in targets:
            for condition in SOURCE_CONDITIONS:
                donor_case, donor_target, site = donor_spec(
                    target,
                    condition,
                    {int(row["target_index"]): row for row in targets},
                )
                target_role = semantic_role(site, target)
                donor_role = semantic_role(site, donor_target)
                target_span = cases[int(target["target_case_index"])][
                    "anchor_spans"
                ][target_role]
                donor_span = cases[donor_case]["anchor_spans"][donor_role]
                lengths = (
                    int(target_span[1]) - int(target_span[0]) + 1,
                    int(donor_span[1]) - int(donor_span[0]) + 1,
                )
                if lengths[0] != lengths[1] or lengths[0] > 2:
                    span_failures.append({
                        "target_index": int(target["target_index"]),
                        "condition": condition,
                        "lengths": lengths,
                    })
        checks = {
            "target_count_240": len(targets) == 240,
            "all_cases_present": len(rows) == len(used),
            "all_source_spans_aligned": not span_failures,
            "surface_2_only": all(
                int(row["surface_index"]) == 2 for row in targets
            ),
            "disjoint_from_phase1044_discovery": not set(
                int(row["target_index"]) for row in targets
            ).intersection(discovery_target_indices),
        }
        model_audits[model_name] = {
            "model": model_name,
            "case_count": len(rows),
            "span_failures": span_failures,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }

    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    model_depths = {
        model_name: {
            "source_depth": int(
                source_prereg["model_depths"][model_name][
                    "source_depth"
                ]
            ),
            "receiver_depth": int(
                source_prereg["model_depths"][model_name][
                    "receiver_depths"
                ][2]
            ),
            "receiver_relative_slot": 3,
        }
        for model_name in MODELS
    }
    payload = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": source.PHASE,
        "source_protocol_digest": source_aggregate[
            "protocol_digest"
        ],
        "models": MODELS,
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "source_conditions": SOURCE_CONDITIONS,
        "operations_by_condition": OPERATIONS_BY_CONDITION,
        "receiver_site": RECEIVER_SITE,
        "wrong_receiver_site": WRONG_RECEIVER_SITE,
        "model_depths": model_depths,
        "target_indices": [
            int(row["target_index"]) for row in targets
        ],
    }
    prereg = {
        "schema_version": "phase1045_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "frozen_candidate": candidate,
        "sample_plan": {
            "targets": len(targets),
            "source_conditions": len(SOURCE_CONDITIONS),
            "forward_operations_per_target": sum(
                len(values)
                for values in OPERATIONS_BY_CONDITION.values()
            ),
            "paired_forward_rows_per_model": (
                len(targets)
                * sum(
                    len(values)
                    for values in OPERATIONS_BY_CONDITION.values()
                )
                * 2
            ),
        },
        "operation_semantics": {
            "none": (
                "Row 0 receives the early selected source edit; row 1 is "
                "the identical zero-payload baseline."
            ),
            "query_swap": (
                "At the frozen receiver layer output, swap query-position "
                "states between the source-edited and zero rows. Row 0 is "
                "a receiver reset; row 1 is a receiver replay."
            ),
            "wrong_site_swap": (
                "Apply the same swap at the unselected concept position as "
                "a location control."
            ),
        },
        "mediation_gate": {
            "source_shift_median_min": 0.0,
            "source_positive_rate_min": 0.8,
            "query_blocked_amount_median_min": 0.0,
            "query_blocked_positive_rate_min": 0.6,
            "query_mediation_fraction_median_min": 0.2,
            "query_minus_wrong_blocked_median_min": 0.0,
            "query_replay_shift_median_min": 0.0,
            "query_replay_positive_rate_min": 0.6,
            "query_replay_recovery_median_min": 0.1,
            "primary_models": ["qwen3", "glm4"],
            "deepseek7b_role": "preregistered negative/generalization model",
        },
        "claim_limits": [
            "Passing supports a partial query-state mediator for this "
            "controlled family-routing task, not a complete mechanism.",
            "Resetting one query position cannot exclude parallel routes.",
            "Replay sufficiency is local to the frozen layer and prompt.",
            "Failure does not refute distributed query routing.",
            "No biological optimality or universal language structure is "
            "inferred.",
        ],
        "model_audits": model_audits,
        "all_model_audits_passed": all(
            row["all_checks_passed"]
            for row in model_audits.values()
        ),
    }
    if not prereg["all_model_audits_passed"]:
        raise RuntimeError("Phase1045 protocol audit failed")
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.json", {
        "schema_version": "phase1045_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "all_checks_passed": True,
    })
    print(
        f"Phase{PHASE} protocol frozen: {prereg['protocol_digest']}"
    )


if __name__ == "__main__":
    main()
