#!/usr/bin/env python3
"""Freeze independent concept-pair mediation and constituent confirmation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1044_natural_recompute_trajectory_protocol as trajectory
import phase1046_distributed_receiver_protocol as source


PHASE = 1047
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = source.OUT_ROOT
MATERIAL_ROOT = material.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1047_concept_pair_confirmation"
)

RELATIVE_DEPTH_SLOT = 2
CONFIRMATION_MASKS = {
    "selected_concept": ("selected_concept",),
    "unselected_concept": ("unselected_concept",),
    "concept_pair": ("selected_concept", "unselected_concept"),
    "query_boundary": ("query_nonce", "pre_output"),
    "full_sequence_reference": ("full_sequence",),
}
MAX_COALITION_TOKENS = 4
MAX_SOURCE_SPAN = 2


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
    if site == "pre_output":
        return "pre_output"
    raise ValueError(site)


def main() -> None:
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    decision = source_aggregate["automatic_next_decision"]
    if not decision["confirmation_needed"]:
        raise RuntimeError("Phase1046 did not authorize confirmation")
    candidate = decision["frozen_candidates"][0]
    if (
        int(candidate["relative_depth_slot"]) != RELATIVE_DEPTH_SLOT
        or candidate["coalition_mask"] != "concept_pair"
    ):
        raise RuntimeError("Phase1046 frozen candidate drift")

    targets = read_jsonl(
        SOURCE_ROOT
        / "protocol"
        / "reserved_confirmation_targets.jsonl"
    )
    for index, row in enumerate(targets):
        row["confirmation_index"] = index
    if len(targets) != 120:
        raise RuntimeError("Phase1047 target count drift")
    used = {
        int(case_index)
        for row in targets
        for case_index in (
            row["target_case_index"],
            row["cross_family_case_index"],
        )
    }
    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)
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
            raise RuntimeError(f"{model_name} case subset drift")
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", rows
        )
        cases = {int(row["case_index"]): row for row in rows}
        failures = []
        max_tokens = {mask: 0 for mask in CONFIRMATION_MASKS}
        for target in targets:
            target_case = cases[int(target["target_case_index"])]
            donor_case = cases[int(target["cross_family_case_index"])]
            role = semantic_role("selected_concept", target)
            target_span = target_case["anchor_spans"][role]
            donor_span = donor_case["anchor_spans"][role]
            lengths = (
                int(target_span[1]) - int(target_span[0]) + 1,
                int(donor_span[1]) - int(donor_span[0]) + 1,
            )
            if lengths[0] != lengths[1] or lengths[0] > MAX_SOURCE_SPAN:
                failures.append({
                    "target_index": int(target["target_index"]),
                    "reason": f"source span mismatch {lengths}",
                })
            for mask_name, sites in CONFIRMATION_MASKS.items():
                if sites == ("full_sequence",):
                    length = len(target_case["input_ids"])
                else:
                    positions = []
                    for site in sites:
                        current_role = semantic_role(site, target)
                        start, end = (
                            int(value)
                            for value in target_case["anchor_spans"][
                                current_role
                            ]
                        )
                        positions.extend(range(start, end + 1))
                    length = len(positions)
                    if (
                        len(positions) != len(set(positions))
                        or length > MAX_COALITION_TOKENS
                    ):
                        failures.append({
                            "target_index": int(
                                target["target_index"]
                            ),
                            "reason": f"invalid mask {mask_name}",
                        })
                max_tokens[mask_name] = max(
                    max_tokens[mask_name], length
                )
        checks = {
            "target_count_120": len(targets) == 120,
            "case_count_expected": len(rows) == len(used),
            "all_spans_valid": not failures,
            "surface_0_only": all(
                int(row["surface_index"]) == 0 for row in targets
            ),
        }
        model_audits[model_name] = {
            "model": model_name,
            "case_count": len(rows),
            "max_tokens": max_tokens,
            "failures": failures,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }

    trajectory_prereg = read_json(
        trajectory.OUT_ROOT / "protocol" / "preregistration.json"
    )
    model_depths = {
        model_name: {
            "source_depth": int(
                trajectory_prereg["model_depths"][model_name][
                    "source_depth"
                ]
            ),
            "receiver_depth": int(
                trajectory_prereg["model_depths"][model_name][
                    "receiver_depths"
                ][RELATIVE_DEPTH_SLOT - 1]
            ),
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
        "relative_depth_slot": RELATIVE_DEPTH_SLOT,
        "confirmation_masks": CONFIRMATION_MASKS,
        "model_depths": model_depths,
        "target_indices": [
            int(row["target_index"]) for row in targets
        ],
    }
    prereg = {
        "schema_version": "phase1047_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "sample_plan": {
            "targets": len(targets),
            "masks": len(CONFIRMATION_MASKS),
            "paired_baseline_rows_per_model": len(targets) * 2,
            "paired_swap_rows_per_model": (
                len(targets) * len(CONFIRMATION_MASKS) * 2
            ),
        },
        "confirmation_gate": {
            "concept_pair_source_positive_rate_min": 0.8,
            "concept_pair_blocked_positive_rate_min": 0.8,
            "concept_pair_mediation_fraction_median_min": 0.5,
            "concept_pair_replay_positive_rate_min": 0.8,
            "concept_pair_replay_recovery_median_min": 0.5,
            "pair_minus_best_constituent_mediation_min": 0.1,
            "pair_minus_best_constituent_replay_min": 0.1,
            "pair_minus_query_boundary_mediation_min": 0.1,
            "minimum_models": 2,
        },
        "interpretation_rules": {
            "pair_passes_and_synergy_passes": (
                "Evidence for a two-position competitive joint state."
            ),
            "pair_passes_but_selected_matches_pair": (
                "Evidence for persistent selected-source state, not a "
                "two-position alliance."
            ),
            "pair_fails": (
                "Phase1046 discovery does not replicate; preserve it as a "
                "discovery-only response."
            ),
        },
        "claim_limits": [
            "Even a confirmed concept-pair mediator is specific to this "
            "controlled family-routing task.",
            "A selected-position result may reflect persistent residual "
            "memory rather than abstract semantic transport.",
            "The test does not identify heads, neurons, KV channels, or a "
            "universal formula.",
            "No biological optimality is inferred.",
        ],
        "model_audits": model_audits,
        "all_model_audits_passed": all(
            row["all_checks_passed"]
            for row in model_audits.values()
        ),
    }
    if not prereg["all_model_audits_passed"]:
        raise RuntimeError("Phase1047 protocol audit failed")
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.json", {
        "schema_version": "phase1047_protocol_audit.v1",
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
