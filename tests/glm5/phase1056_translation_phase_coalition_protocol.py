#!/usr/bin/env python3
"""Freeze translation depth-phase and K/V coalition localization."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1054_joint_kv_rollout_protocol as source
import phase1055_pattern_family_transfer_protocol as transfer


PHASE = 1056
PROTOCOL_REVISION = 1
MODELS = transfer.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
TRANSFER_ROOT = transfer.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1056_translation_phase_coalition"
)
BEAM_WIDTH = 3
SEARCH_RETENTION_FRACTION = 0.80
SEARCH_ABSOLUTE_RATE_MIN = 0.30
SEARCH_MAX_EVALUATIONS = {
    "qwen3": 180,
    "glm4": 80,
    "deepseek7b": 80,
}
ROLLOUT_STEPS = 8
ROLLOUT_PAIR_LIMIT = 48
CONFIRMATION_CONDITIONS = (
    "source_postsource_full",
    "source_joint_rectangle",
    "source_all_layers",
    "source_early_only",
    "operator_joint_rectangle",
    "target_language_joint_rectangle",
)
GATES = {
    "discovery_clean_pair_count_min": 40,
    "confirmation_clean_pair_count_min": 40,
    "joint_both_counterfactual_rate_min": 0.30,
    "joint_both_counterfactual_count_min": 20,
    "joint_retained_fraction_min": 0.80,
    "source_minus_control_rate_min": 0.20,
    "maximum_block_fraction": 0.75,
    "rollout_pair_count_min": 20,
    "eos_censored_both_match_rate_min": 0.50,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def main() -> None:
    transfer_aggregate = read_json(TRANSFER_ROOT / "aggregate.json")
    transfer_prereg = read_json(
        TRANSFER_ROOT / "protocol" / "preregistration.json"
    )
    source_prereg = read_json(
        source.OUT_ROOT / "protocol" / "preregistration.json"
    )
    model_plans: dict[str, Any] = {}
    model_audits: dict[str, Any] = {}
    for model_name in MODELS:
        transfer_summary = read_json(
            TRANSFER_ROOT / "atlas" / model_name / "summary.json"
        )
        transfer_plan = transfer_prereg["model_plans"][model_name]
        source_plan = source_prereg["model_plans"][model_name]
        targets = read_jsonl(
            TRANSFER_ROOT
            / "protocol"
            / f"targets.{model_name}.jsonl"
        )
        cases = read_jsonl(
            TRANSFER_ROOT
            / "protocol"
            / f"cases.{model_name}.jsonl"
        )
        frozen_variant = str(transfer_summary["frozen_variant"])
        selected_targets = [
            row for row in targets
            if row["variant"] == frozen_variant
        ]
        selected_case_indices = {
            int(row[key])
            for row in selected_targets
            for key in ("target_case_index", "cross_case_index")
        }
        selected_cases = [
            row for row in cases
            if int(row["semantic_case_index"]) in selected_case_indices
        ]
        write_jsonl(
            OUT_ROOT
            / "protocol"
            / f"targets.{model_name}.jsonl",
            selected_targets,
        )
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            selected_cases,
        )

        all_depths = [
            int(value)
            for value in source_plan["all_postsource_depths"]
        ]
        first_postsource = min(all_depths)
        early_depths = list(range(1, first_postsource))
        all_layers = list(
            range(1, int(transfer_plan["n_layers"]) + 1)
        )
        model_plans[model_name] = {
            "behavior_eligible": bool(
                transfer_summary["behavior_gate_passed"]
            ),
            "frozen_variant": frozen_variant,
            "n_layers": int(transfer_plan["n_layers"]),
            "n_kv_heads": int(transfer_plan["n_kv_heads"]),
            "all_groups": list(
                range(int(transfer_plan["n_kv_heads"]))
            ),
            "depth_slots": [
                [int(value) for value in slot]
                for slot in source_plan["depth_slots"]
            ],
            "all_postsource_depths": all_depths,
            "early_depths": early_depths,
            "all_layers": all_layers,
            "search_max_evaluations": SEARCH_MAX_EVALUATIONS[
                model_name
            ],
        }
        split_counts = {
            split: sum(
                row["split"] == split for row in selected_targets
            )
            for split in ("discovery", "confirmation")
        }
        model_audits[model_name] = {
            "frozen_variant": frozen_variant,
            "target_counts": split_counts,
            "case_count": len(selected_cases),
            "discovery_confirmation_concept_overlap": bool(
                {
                    value
                    for row in selected_targets
                    if row["split"] == "discovery"
                    for value in (
                        row["target_concept_id"],
                        row["cross_concept_id"],
                    )
                }.intersection({
                    value
                    for row in selected_targets
                    if row["split"] == "confirmation"
                    for value in (
                        row["target_concept_id"],
                        row["cross_concept_id"],
                    )
                })
            ),
            "slots_flatten_to_postsource": [
                depth
                for slot in source_plan["depth_slots"]
                for depth in slot
            ] == all_depths,
            "early_postsource_overlap": bool(
                set(early_depths).intersection(all_depths)
            ),
            "all_layers_complete": (
                early_depths + all_depths == all_layers
            ),
        }

    audit = {
        "schema_version": "phase1056_protocol_audit.v1",
        "phase": PHASE,
        "source_phase1055_route": transfer_aggregate[
            "automatic_next_decision"
        ],
        "models": model_audits,
    }
    audit["all_checks_passed"] = all(
        row["target_counts"]["discovery"] >= 50
        and row["target_counts"]["confirmation"] >= 50
        and not row["discovery_confirmation_concept_overlap"]
        and row["slots_flatten_to_postsource"]
        and not row["early_postsource_overlap"]
        and row["all_layers_complete"]
        for row in model_audits.values()
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1056 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1056_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1055_digest": transfer_prereg["protocol_digest"],
        "authorization": (
            "Method audit prompted by the preregistered Phase1055 "
            "postsource-positive/all-layers-negative Qwen3 result."
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "model_plans": model_plans,
        "beam_width": BEAM_WIDTH,
        "search_retention_fraction": SEARCH_RETENTION_FRACTION,
        "search_absolute_rate_min": SEARCH_ABSOLUTE_RATE_MIN,
        "search_rule": (
            "On discovery concepts, jointly delete one KV group or one "
            "normalized postsource depth slot. Keep the three feasible "
            "rectangles with the fewest Cartesian blocks, then higher "
            "full-vocabulary flip rate. Confirm the frozen rectangle on "
            "lexically disjoint confirmation concepts."
        ),
        "confirmation_conditions": list(CONFIRMATION_CONDITIONS),
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "gates": GATES,
        "automatic_next": {
            "translation_rectangle_repeats": (
                "stop_at_translation_coalition_milestone"
            ),
            "otherwise": "stop_with_model_specific_translation_phase",
        },
        "interpretation_limits": [
            "Phase1055 already measured broad conditions on confirmation.",
            "The frozen rectangle itself is new to confirmation.",
            "A rectangle is not an arbitrary sparse group-depth cell set.",
            "Early-plus-postsource nonadditivity is a contrast, not a law.",
            "Only one lexical translation direction is covered.",
            "No result proves a universal language or brain mechanism.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", payload)
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"models={len(MODELS)}"
    )


if __name__ == "__main__":
    main()
