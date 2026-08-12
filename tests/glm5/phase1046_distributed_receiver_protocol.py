#!/usr/bin/env python3
"""Freeze a distributed receiver-coalition atlas after Phase1045."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1041_position_write_alliance_protocol as alliance
import phase1044_natural_recompute_trajectory_protocol as trajectory
import phase1045_receiver_mediation_protocol as source


PHASE = 1046
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = source.OUT_ROOT
ALLIANCE_ROOT = alliance.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1046_distributed_receiver_atlas"
)

RELATIVE_DEPTH_SLOTS = (2, 3, 4)
COALITION_MASKS = {
    "query_boundary": ("query_nonce", "pre_output"),
    "concept_pair": ("selected_concept", "unselected_concept"),
    "semantic_core": (
        "selected_concept",
        "unselected_concept",
        "query_nonce",
        "pre_output",
    ),
    "full_sequence_reference": ("full_sequence",),
}
MAX_COALITION_TOKENS = 8


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def semantic_role(site: str, target: dict[str, Any]) -> str:
    if site in ("selected_concept", "unselected_concept", "query_nonce"):
        return source.semantic_role(site, target)
    if site == "pre_output":
        return "pre_output"
    raise ValueError(site)


def main() -> None:
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    decision = source_aggregate["automatic_next_decision"]
    if not decision["distributed_receiver_coalition_atlas_needed"]:
        raise RuntimeError("Phase1045 did not authorize Phase1046")
    all_surface2 = read_jsonl(
        SOURCE_ROOT / "protocol" / "targets.jsonl"
    )
    discovery = [
        row for row in all_surface2
        if int(row["query"]) == alliance.selected_query(row)
    ]
    for index, row in enumerate(discovery):
        row["coalition_index"] = index
    if len(discovery) != 120:
        raise RuntimeError("Phase1046 discovery count drift")

    untouched = read_jsonl(
        ALLIANCE_ROOT / "protocol" / "reserved_holdout_targets.jsonl"
    )
    reserved_confirmation = [
        row for row in untouched
        if int(row["surface_index"]) == 0
        and int(row["query"]) != alliance.selected_query(row)
    ]
    if len(reserved_confirmation) != 120:
        raise RuntimeError("Phase1046 reserved confirmation drift")
    if set(int(row["target_index"]) for row in discovery).intersection(
        int(row["target_index"]) for row in reserved_confirmation
    ):
        raise RuntimeError("Phase1046 discovery/confirmation overlap")

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "discovery_targets.jsonl", discovery)
    write_jsonl(
        protocol_dir / "reserved_confirmation_targets.jsonl",
        reserved_confirmation,
    )
    model_audits = {}
    for model_name in MODELS:
        rows = read_jsonl(
            SOURCE_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", rows
        )
        cases = {int(row["case_index"]): row for row in rows}
        span_failures = []
        max_tokens = {mask: 0 for mask in COALITION_MASKS}
        for target in discovery:
            case = cases[int(target["target_case_index"])]
            for mask_name, sites in COALITION_MASKS.items():
                if sites == ("full_sequence",):
                    length = len(case["input_ids"])
                else:
                    length = 0
                    active_positions = []
                    for site in sites:
                        role = semantic_role(site, target)
                        start, end = (
                            int(value)
                            for value in case["anchor_spans"][role]
                        )
                        span = list(range(start, end + 1))
                        active_positions.extend(span)
                        length += len(span)
                    if len(active_positions) != len(set(active_positions)):
                        span_failures.append({
                            "target_index": int(target["target_index"]),
                            "mask": mask_name,
                            "reason": "overlapping positions",
                        })
                    if length > MAX_COALITION_TOKENS:
                        span_failures.append({
                            "target_index": int(target["target_index"]),
                            "mask": mask_name,
                            "reason": f"{length} tokens exceeds budget",
                        })
                max_tokens[mask_name] = max(
                    max_tokens[mask_name], length
                )
        checks = {
            "target_count_120": len(discovery) == 120,
            "source_cases_present": bool(rows),
            "coalition_spans_valid": not span_failures,
            "surface_2_discovery": all(
                int(row["surface_index"]) == 2 for row in discovery
            ),
        }
        model_audits[model_name] = {
            "model": model_name,
            "case_count": len(rows),
            "max_tokens": max_tokens,
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
            "receiver_depths": {
                str(slot): int(
                    read_json(
                        trajectory.OUT_ROOT
                        / "protocol"
                        / "preregistration.json"
                    )["model_depths"][model_name]["receiver_depths"][
                        slot - 1
                    ]
                )
                for slot in RELATIVE_DEPTH_SLOTS
            },
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
        "relative_depth_slots": RELATIVE_DEPTH_SLOTS,
        "coalition_masks": COALITION_MASKS,
        "model_depths": model_depths,
        "discovery_target_indices": [
            int(row["target_index"]) for row in discovery
        ],
        "reserved_confirmation_target_indices": [
            int(row["target_index"]) for row in reserved_confirmation
        ],
    }
    prereg = {
        "schema_version": "phase1046_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "sample_plan": {
            "discovery_targets": len(discovery),
            "reserved_confirmation_targets": len(
                reserved_confirmation
            ),
            "depth_slots": len(RELATIVE_DEPTH_SLOTS),
            "coalition_masks": len(COALITION_MASKS),
            "paired_forward_rows_per_model": (
                len(discovery)
                * len(RELATIVE_DEPTH_SLOTS)
                * len(COALITION_MASKS)
                * 2
            ),
        },
        "intervention_semantics": (
            "Every row pair receives the same frozen early selected-concept "
            "full-state source edit in row 0 and zero payload in row 1. At "
            "one receiver depth, the specified state coalition is swapped "
            "between rows. Row 0 measures reset; row 1 measures replay. No "
            "later intervention is applied."
        ),
        "discovery_gate": {
            "source_shift_positive_rate_min": 0.8,
            "blocked_positive_rate_min": 0.65,
            "mediation_fraction_median_min": 0.2,
            "replay_positive_rate_min": 0.65,
            "replay_recovery_median_min": 0.2,
            "minimum_models": 2,
            "full_sequence_reset_fraction_min": 0.9,
            "full_sequence_replay_fraction_min": 0.9,
        },
        "selection_rule": (
            "Among non-reference masks passing in at least two models, "
            "choose the fewest semantic sites, then the earliest relative "
            "depth. Freeze at most two candidates."
        ),
        "automatic_followup": {
            "if_candidate": (
                "Run only the frozen candidate masks on the untouched "
                "surface-0 complementary-query set."
            ),
            "if_no_candidate": (
                "Stop semantic-anchor coalition tuning. Conclude that the "
                "source effect is distributed beyond these four anchors "
                "or is encoded in a different state variable."
            ),
        },
        "claim_limits": [
            "A coalition pass is local mediation for an artificial family "
            "routing task, not a universal language mechanism.",
            "Full-sequence swapping is a deterministic intervention upper "
            "bound, not a discovered mechanism.",
            "The test covers four semantic anchors, not all prompt tokens, "
            "heads, neurons, or KV states.",
            "No result establishes biological optimality.",
        ],
        "model_audits": model_audits,
        "all_model_audits_passed": all(
            row["all_checks_passed"]
            for row in model_audits.values()
        ),
    }
    if not prereg["all_model_audits_passed"]:
        raise RuntimeError("Phase1046 protocol audit failed")
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.json", {
        "schema_version": "phase1046_protocol_audit.v1",
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
