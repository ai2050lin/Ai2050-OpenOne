#!/usr/bin/env python3
"""Freeze the Phase1044 one-source natural-recomputation trajectory atlas."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1041_position_write_alliance_protocol as alliance
import phase1042_role_depth_write_atlas_protocol as source


PHASE = 1044
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = source.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1044_natural_recompute_trajectory"
)

SOURCE_MODES = ("mlp_write", "full_state")
CONDITIONS = (
    "cross_selected_l0",
    "cross_selected_l1",
    "cross_unselected_l0",
    "cross_unselected_l1",
    "cross_shuffled_l0",
    "cross_shuffled_l1",
    "same_family_lexical_l0",
)
SEMANTIC_SITES = source.SEMANTIC_SITES
CHANNELS = source.CHANNELS
RECEIVER_SITES = ("query_nonce", "pre_output")
MAX_ROLE_SPAN = alliance.MAX_ROLE_SPAN
MAX_SOURCE_SPAN = 2


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def condition_spec(
    target: dict[str, Any],
    condition: str,
    targets_by_atlas: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if condition.endswith("_l1"):
        target_world = "b0l1"
        donor_world = "b1l1"
    else:
        target_world = "b0l0"
        donor_world = "b1l0"

    donor_target = target
    source_site = "selected_concept"
    if condition.startswith("cross_unselected"):
        source_site = "unselected_concept"
    elif condition.startswith("cross_shuffled"):
        donor_target = targets_by_atlas[
            int(target["shuffled_atlas_index"])
        ]
    elif condition == "same_family_lexical_l0":
        donor_world = "b0l1"
    elif not condition.startswith("cross_selected"):
        raise ValueError(condition)

    return {
        "target_world": target_world,
        "donor_world": donor_world,
        "source_site": source_site,
        "donor_atlas_index": int(donor_target["atlas_index"]),
        "target_case_index": int(
            target["world_case_indices"][target_world]
        ),
        "donor_case_index": int(
            donor_target["world_case_indices"][donor_world]
        ),
    }


def model_audit(
    model_name: str,
    targets: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cases = {int(row["case_index"]): row for row in rows}
    targets_by_atlas = {
        int(row["atlas_index"]): row for row in targets
    }
    span_counts: dict[str, int] = {}
    parity_failures = []
    for target in targets:
        for condition in CONDITIONS:
            spec = condition_spec(target, condition, targets_by_atlas)
            donor_target = targets_by_atlas[
                int(spec["donor_atlas_index"])
            ]
            target_role = source.semantic_role(
                str(spec["source_site"]), target
            )
            donor_role = source.semantic_role(
                str(spec["source_site"]), donor_target
            )
            target_span = cases[int(spec["target_case_index"])][
                "anchor_spans"
            ][target_role]
            donor_span = cases[int(spec["donor_case_index"])][
                "anchor_spans"
            ][donor_role]
            target_length = int(target_span[1]) - int(target_span[0]) + 1
            donor_length = int(donor_span[1]) - int(donor_span[0]) + 1
            key = f"{condition}/{target_length}"
            span_counts[key] = span_counts.get(key, 0) + 1
            if (
                target_length != donor_length
                or target_length > MAX_SOURCE_SPAN
            ):
                parity_failures.append({
                    "atlas_index": int(target["atlas_index"]),
                    "condition": condition,
                    "target_length": target_length,
                    "donor_length": donor_length,
                })
    checks = {
        "target_count_120": len(targets) == 120,
        "case_count_480": len(rows) == 480,
        "all_case_indices_unique": len(cases) == len(rows),
        "all_source_spans_aligned": not parity_failures,
        "all_worlds_present": all(
            set(row["world_case_indices"])
            == {"b0l0", "b1l0", "b0l1", "b1l1"}
            for row in targets
        ),
    }
    return {
        "model": model_name,
        "target_count": len(targets),
        "case_count": len(rows),
        "span_counts": span_counts,
        "parity_failures": parity_failures,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    targets = read_jsonl(SOURCE_ROOT / "protocol" / "targets.jsonl")
    if len(targets) != 120:
        raise RuntimeError("Phase1042 discovery target count drift")

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)
    audits = {}
    case_counts = {}
    for model_name in MODELS:
        rows = read_jsonl(
            SOURCE_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl", rows
        )
        audits[model_name] = model_audit(model_name, targets, rows)
        case_counts[model_name] = len(rows)

    model_depths = {
        model_name: {
            "source_depth": int(
                source_prereg["model_physical_depths"][model_name][0]
            ),
            "receiver_depths": [
                int(value)
                for value in source_prereg[
                    "model_physical_depths"
                ][model_name][1:]
            ],
        }
        for model_name in MODELS
    }
    payload = {
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase": source.PHASE,
        "source_protocol_digest": source_prereg["protocol_digest"],
        "source_aggregate_digest": source_aggregate[
            "protocol_digest"
        ],
        "models": MODELS,
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": MODELS,
        "source_modes": SOURCE_MODES,
        "conditions": CONDITIONS,
        "semantic_sites": SEMANTIC_SITES,
        "receiver_sites": RECEIVER_SITES,
        "channels": CHANNELS,
        "model_depths": model_depths,
        "target_indices": [
            int(row["target_index"]) for row in targets
        ],
    }
    prereg = {
        "schema_version": "phase1044_preregistration.v1",
        **payload,
        "protocol_digest": digest(payload),
        "research_question": (
            "After one controlled early source intervention, do later "
            "naturally recomputed states form a role- and condition-specific "
            "trajectory that repeats across lexical surfaces and models?"
        ),
        "sample_plan": {
            "targets": len(targets),
            "cases_per_model": case_counts,
            "source_modes": len(SOURCE_MODES),
            "conditions": len(CONDITIONS),
            "paired_forward_rows_per_model": (
                len(targets)
                * len(SOURCE_MODES)
                * len(CONDITIONS)
                * 2
            ),
            "receiver_depths_per_model": 6,
            "semantic_sites": len(SEMANTIC_SITES),
            "channels": len(CHANNELS),
        },
        "intervention_semantics": {
            "mlp_write": (
                "At the frozen early layer, add the clean donor-minus-target "
                "complete MLP output over the aligned selected or unselected "
                "concept span exactly once."
            ),
            "full_state": (
                "At the frozen early layer output, replace the aligned "
                "concept-span state with the clean donor state exactly once."
            ),
            "natural_recomputation": (
                "No later patch is applied. All later Attention, MLP, "
                "normalization, residual, and readout computations run on "
                "the intervened state."
            ),
            "paired_zero": (
                "Every patched row is evaluated beside an identical input "
                "with an exactly zero payload; downstream response is their "
                "within-forward-pair difference."
            ),
        },
        "descriptive_repetition_gate": {
            "cross_lexical_cosine_median_min": 0.2,
            "matched_minus_shuffled_cosine_median_min": 0.1,
            "advantage_positive_rate_min": 0.65,
            "family_to_same_lexical_norm_ratio_min": 1.0,
            "minimum_models": 2,
            "claim_limit": (
                "Passing identifies a repeated causal response cell after "
                "the source intervention. It is not yet a transport edge, "
                "receiver, sufficient mechanism, or language equation."
            ),
        },
        "behavior_relevance_gate": {
            "selected_cross_margin_shift_median_min": 0.0,
            "selected_cross_positive_rate_min": 0.6,
            "selected_minus_unselected_median_min": 0.0,
            "full_state_required_for_confirmation": True,
        },
        "automatic_followup": {
            "if_cross_model_repeated_and_behavior_relevant": (
                "Freeze no more than three trajectory cells and run an "
                "independent held-out receiver/mediation confirmation."
            ),
            "otherwise": (
                "Preserve the causal response atlas, stop tuning this "
                "controlled-family source, and move to a different natural "
                "language pattern block."
            ),
        },
        "claim_limits": [
            "The task is an artificial two-binding family-routing task.",
            "A cached donor state is used only for the single source edit; "
            "natural recomputation refers to every downstream computation.",
            "A downstream causal response does not prove that the response "
            "is a necessary or sufficient relay.",
            "Cross-model comparison uses relative depth and functional "
            "roles, not equal physical layer numbers.",
            "No result establishes biological optimality, brain-model "
            "isomorphism, or a universal language mechanism.",
        ],
        "model_audits": audits,
        "all_model_audits_passed": all(
            row["all_checks_passed"] for row in audits.values()
        ),
    }
    if not prereg["all_model_audits_passed"]:
        raise RuntimeError("Phase1044 model protocol audit failed")
    write_json(protocol_dir / "preregistration.json", prereg)
    write_json(protocol_dir / "audit.json", {
        "schema_version": "phase1044_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": audits,
        "all_checks_passed": prereg["all_model_audits_passed"],
    })
    print(
        f"Phase{PHASE} protocol frozen: {prereg['protocol_digest']}"
    )


if __name__ == "__main__":
    main()
