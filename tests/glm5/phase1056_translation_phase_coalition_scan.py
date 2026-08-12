#!/usr/bin/env python3
"""Localize translation K/V phase rectangles and early antagonism."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1052_full_vocab_kv_bridge_scan as bridge_scan
import phase1054_joint_kv_rollout_scan as joint_tools
import phase1056_translation_phase_coalition_protocol as protocol


PAIR_BATCH_SIZE = bridge_scan.PAIR_BATCH_SIZE


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in (
            "both_counterfactual_mask",
            "valid_target_indices",
        )
    }


def rate(result: dict[str, Any]) -> float:
    return float(result["both_counterfactual_top1_rate"])


def valid_targets(
    rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    clean: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    mask, _ = bridge_scan.clean_mask_and_coverage(rows, cases, clean)
    return [row for row, keep in zip(rows, mask) if keep]


def flatten_slots(
    slot_ids: tuple[int, ...] | list[int],
    slots: list[list[int]],
) -> list[int]:
    return [
        depth
        for slot_id in slot_ids
        for depth in slots[int(slot_id)]
    ]


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1056 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    discovery_rows = [
        row for row in targets if row["split"] == "discovery"
    ]
    confirmation_rows = [
        row for row in targets if row["split"] == "confirmation"
    ]
    plan = prereg["model_plans"][model_name]
    all_groups = tuple(int(value) for value in plan["all_groups"])
    slots = [
        [int(value) for value in slot]
        for slot in plan["depth_slots"]
    ]
    all_slot_ids = tuple(range(len(slots)))
    all_postsource = [
        int(value) for value in plan["all_postsource_depths"]
    ]
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        width = bridge_scan.projection_width(
            layers[0].self_attn.k_proj
        )
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("KV projection geometry drift")
        head_dim = width // n_kv_heads
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        clean_discovery = bridge_scan.run_condition(
            model,
            device,
            layers,
            discovery_rows,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        discovery_valid = valid_targets(
            discovery_rows, cases, clean_discovery
        )
        clean_discovery_valid = bridge_scan.run_condition(
            model,
            device,
            layers,
            discovery_valid,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        all_valid = np.ones(len(discovery_valid), dtype=bool)
        cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            dict[str, Any],
        ] = {}
        budget = int(plan["search_max_evaluations"])

        def evaluate(
            groups: tuple[int, ...] | list[int],
            slot_ids: tuple[int, ...] | list[int],
        ) -> dict[str, Any]:
            key = (
                tuple(sorted(int(value) for value in groups)),
                tuple(sorted(int(value) for value in slot_ids)),
            )
            if key not in cache:
                if len(cache) >= budget:
                    raise RuntimeError("search evaluation budget exhausted")
                patched = bridge_scan.run_condition(
                    model,
                    device,
                    layers,
                    discovery_valid,
                    cases,
                    {
                        "site": "source_term",
                        "groups": list(key[0]),
                        "depths": flatten_slots(key[1], slots),
                    },
                    head_dim=head_dim,
                    pad_token_id=int(pad_token_id),
                    pair_batch_size=PAIR_BATCH_SIZE[model_name],
                )
                cache[key] = bridge_scan.condition_metrics(
                    discovery_valid,
                    cases,
                    clean_discovery_valid,
                    patched,
                    all_valid,
                )
            return cache[key]

        full_discovery = evaluate(all_groups, all_slot_ids)
        full_rate = rate(full_discovery)
        minimum_rate = max(
            float(prereg["search_absolute_rate_min"]),
            full_rate * float(prereg["search_retention_fraction"]),
        )
        feasible = {(all_groups, all_slot_ids): full_discovery}
        frontier = [(all_groups, all_slot_ids)]
        beam_trace = []
        budget_exhausted = False
        while frontier:
            children = {}
            step = []
            for groups, slot_ids in frontier:
                proposals = []
                proposals.extend(
                    (
                        tuple(v for v in groups if v != removed),
                        slot_ids,
                        "group",
                        int(removed),
                    )
                    for removed in groups
                    if len(groups) > 1
                )
                proposals.extend(
                    (
                        groups,
                        tuple(v for v in slot_ids if v != removed),
                        "slot",
                        int(removed),
                    )
                    for removed in slot_ids
                    if len(slot_ids) > 1
                )
                for child_groups, child_slots, kind, removed in proposals:
                    try:
                        result = evaluate(child_groups, child_slots)
                    except RuntimeError as exc:
                        if "budget exhausted" not in str(exc):
                            raise
                        budget_exhausted = True
                        break
                    is_feasible = rate(result) >= minimum_rate
                    step.append({
                        "parent_groups": list(groups),
                        "parent_slots": list(slot_ids),
                        "groups": list(child_groups),
                        "slots": list(child_slots),
                        "removed_type": kind,
                        "removed": removed,
                        "rate": rate(result),
                        "block_count": (
                            len(child_groups) * len(child_slots)
                        ),
                        "feasible": is_feasible,
                    })
                    if is_feasible:
                        key = (child_groups, child_slots)
                        children[key] = result
                        feasible[key] = result
                if budget_exhausted:
                    break
            if not children:
                beam_trace.append({
                    "trials": step,
                    "selected": [],
                    "budget_exhausted": budget_exhausted,
                })
                break
            ranked = sorted(
                children,
                key=lambda key: (
                    len(key[0]) * len(key[1]),
                    -rate(children[key]),
                    len(key[0]) + len(key[1]),
                    key,
                ),
            )
            frontier = ranked[:int(prereg["beam_width"])]
            beam_trace.append({
                "trials": step,
                "selected": [
                    {
                        "groups": list(key[0]),
                        "slots": list(key[1]),
                        "rate": rate(children[key]),
                        "block_count": len(key[0]) * len(key[1]),
                    }
                    for key in frontier
                ],
                "budget_exhausted": budget_exhausted,
            })
            if budget_exhausted:
                break

        best_key = min(
            feasible,
            key=lambda key: (
                len(key[0]) * len(key[1]),
                -rate(feasible[key]),
                len(key[0]) + len(key[1]),
                key,
            ),
        )
        frozen_groups, frozen_slot_ids = best_key
        frozen_depths = flatten_slots(frozen_slot_ids, slots)
        best_discovery = feasible[best_key]

        clean_confirmation = bridge_scan.run_condition(
            model,
            device,
            layers,
            confirmation_rows,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        confirmation_valid = valid_targets(
            confirmation_rows, cases, clean_confirmation
        )
        clean_confirmation_valid = bridge_scan.run_condition(
            model,
            device,
            layers,
            confirmation_valid,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        confirmation_mask = np.ones(
            len(confirmation_valid), dtype=bool
        )
        specs = {
            "source_postsource_full": {
                "site": "source_term",
                "groups": list(all_groups),
                "depths": all_postsource,
            },
            "source_joint_rectangle": {
                "site": "source_term",
                "groups": list(frozen_groups),
                "depths": frozen_depths,
            },
            "source_all_layers": {
                "site": "source_term",
                "groups": list(all_groups),
                "depths": [
                    int(value) for value in plan["all_layers"]
                ],
            },
            "source_early_only": {
                "site": "source_term",
                "groups": list(all_groups),
                "depths": [
                    int(value) for value in plan["early_depths"]
                ],
            },
            "operator_joint_rectangle": {
                "site": "operator",
                "groups": list(frozen_groups),
                "depths": frozen_depths,
            },
            "target_language_joint_rectangle": {
                "site": "target_language",
                "groups": list(frozen_groups),
                "depths": frozen_depths,
            },
        }
        confirmation_results = {}
        for name in prereg["confirmation_conditions"]:
            patched = bridge_scan.run_condition(
                model,
                device,
                layers,
                confirmation_valid,
                cases,
                specs[name],
                head_dim=head_dim,
                pad_token_id=int(pad_token_id),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
            confirmation_results[name] = bridge_scan.condition_metrics(
                confirmation_valid,
                cases,
                clean_confirmation_valid,
                patched,
                confirmation_mask,
            )

        full_confirm = confirmation_results[
            "source_postsource_full"
        ]
        joint_confirm = confirmation_results[
            "source_joint_rectangle"
        ]
        all_layers_result = confirmation_results["source_all_layers"]
        early_result = confirmation_results["source_early_only"]
        control_rate = max(
            rate(confirmation_results["operator_joint_rectangle"]),
            rate(
                confirmation_results[
                    "target_language_joint_rectangle"
                ]
            ),
        )
        full_confirm_rate = rate(full_confirm)
        joint_confirm_rate = rate(joint_confirm)
        all_layers_rate = rate(all_layers_result)
        early_rate = rate(early_result)
        retained = (
            joint_confirm_rate / full_confirm_rate
            if full_confirm_rate > 0 else 0.0
        )
        block_fraction = (
            len(frozen_groups) * len(frozen_slot_ids)
        ) / (len(all_groups) * len(all_slot_ids))
        gates = prereg["gates"]
        baseline_gate = (
            bool(plan["behavior_eligible"])
            and len(discovery_valid)
            >= gates["discovery_clean_pair_count_min"]
            and len(confirmation_valid)
            >= gates["confirmation_clean_pair_count_min"]
        )
        coalition_gate = (
            baseline_gate
            and joint_confirm["both_counterfactual_top1_count"]
            >= gates["joint_both_counterfactual_count_min"]
            and joint_confirm_rate
            >= gates["joint_both_counterfactual_rate_min"]
            and retained >= gates["joint_retained_fraction_min"]
            and joint_confirm_rate - control_rate
            >= gates["source_minus_control_rate_min"]
            and block_fraction <= gates["maximum_block_fraction"]
        )

        successful_targets = [
            row for row, success in zip(
                confirmation_valid,
                joint_confirm["both_counterfactual_mask"],
            )
            if success
        ]
        raw_rollouts = []
        legacy_rollout = {
            "pair_count": 0,
            "both_match_other_clean_rate": 0.0,
        }
        if successful_targets:
            raw_rollouts, legacy_rollout = bridge_scan.rollout_pairs(
                model,
                tokenizer,
                device,
                layers,
                successful_targets,
                cases,
                specs["source_joint_rectangle"],
                head_dim=head_dim,
                steps=int(prereg["rollout_steps"]),
                pair_limit=int(prereg["rollout_pair_limit"]),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
        eos_ids = joint_tools.eos_token_ids(model, tokenizer)
        audited_rollouts, eos_rollout = joint_tools.audit_rollouts(
            raw_rollouts, eos_ids
        )
        rollout_gate = (
            coalition_gate
            and eos_rollout["pair_count"]
            >= gates["rollout_pair_count_min"]
            and eos_rollout["both_match_other_clean_rate"]
            >= gates["eos_censored_both_match_rate_min"]
        )

        summary = {
            "schema_version": "phase1056_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "behavior_eligible": bool(plan["behavior_eligible"]),
            "discovery_clean_pair_count": len(discovery_valid),
            "confirmation_clean_pair_count": len(
                confirmation_valid
            ),
            "discovery_postsource_full": compact(full_discovery),
            "discovery_minimum_rate": minimum_rate,
            "search_evaluation_count": len(cache),
            "search_evaluation_budget": budget,
            "search_budget_exhausted": budget_exhausted,
            "feasible_state_count": len(feasible),
            "beam_trace": beam_trace,
            "frozen_groups": list(frozen_groups),
            "frozen_depth_slot_ids": list(frozen_slot_ids),
            "frozen_depths": frozen_depths,
            "block_fraction": block_fraction,
            "discovery_joint_rectangle": compact(best_discovery),
            "confirmation_results": {
                key: compact(value)
                for key, value in confirmation_results.items()
            },
            "confirmation_joint_retained_fraction": retained,
            "confirmation_selected_minus_control_rate": (
                joint_confirm_rate - control_rate
            ),
            "early_plus_postsource_suppression_contrast": (
                full_confirm_rate - all_layers_rate
            ),
            "early_only_rate": early_rate,
            "coalition_gate_passed": coalition_gate,
            "eos_token_ids": eos_ids,
            "legacy_rollout_summary": legacy_rollout,
            "eos_rollout_summary": eos_rollout,
            "rollout_gate_passed": rollout_gate,
            "rollouts": audited_rollouts,
            "elapsed_seconds": float(time.time() - started),
        }
        out = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_json(out / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "eligible": plan["behavior_eligible"],
            "discovery_clean": len(discovery_valid),
            "confirmation_clean": len(confirmation_valid),
            "groups": list(frozen_groups),
            "slots": list(frozen_slot_ids),
            "block_fraction": block_fraction,
            "discovery_full": full_rate,
            "discovery_joint": rate(best_discovery),
            "confirmation_postsource": full_confirm_rate,
            "confirmation_joint": joint_confirm_rate,
            "confirmation_all_layers": all_layers_rate,
            "confirmation_early": early_rate,
            "suppression_contrast": (
                full_confirm_rate - all_layers_rate
            ),
            "control": control_rate,
            "retained": retained,
            "coalition_gate": coalition_gate,
            "eos_rollout": eos_rollout,
            "rollout_gate": rollout_gate,
            "evaluations": len(cache),
            "budget_exhausted": budget_exhausted,
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
