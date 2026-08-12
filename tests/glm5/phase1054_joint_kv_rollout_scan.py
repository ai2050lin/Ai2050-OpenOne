#!/usr/bin/env python3
"""Search joint K/V rectangles and audit EOS-aware output trajectories."""

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
import phase1054_joint_kv_rollout_protocol as protocol


PAIR_BATCH_SIZE = bridge_scan.PAIR_BATCH_SIZE


def filtered_clean_targets(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    clean: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[str]]:
    mask, coverage = bridge_scan.clean_mask_and_coverage(
        target_rows, cases, clean
    )
    return [
        row for row, keep in zip(target_rows, mask) if keep
    ], coverage


def rate(result: dict[str, Any]) -> float:
    return float(result["both_counterfactual_top1_rate"])


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in (
            "both_counterfactual_mask",
            "valid_target_indices",
        )
    }


def flatten_slots(
    slot_ids: tuple[int, ...] | list[int],
    slots: list[list[int]],
) -> list[int]:
    return [
        depth
        for slot_id in slot_ids
        for depth in slots[int(slot_id)]
    ]


def eos_token_ids(model, tokenizer) -> list[int]:
    values = []
    for source in (
        getattr(model.generation_config, "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            values.extend(int(value) for value in source)
        else:
            values.append(int(source))
    result = sorted(set(values))
    if not result:
        raise RuntimeError("model has no EOS token id")
    return result


def censor_at_eos(
    values: list[int],
    eos_ids: set[int],
) -> tuple[list[int], bool, int | None]:
    for index, value in enumerate(values):
        if int(value) in eos_ids:
            return [int(v) for v in values[:index + 1]], True, index
    return [int(v) for v in values], False, None


def audit_rollouts(
    records: list[dict[str, Any]],
    eos_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eos_set = set(eos_ids)
    audited = []
    left_exact = []
    right_exact = []
    both_exact = []
    left_suffix = []
    right_suffix = []
    both_suffix = []
    clean_terminated = []
    patched_terminated = []
    termination_step_matches = []
    for record in records:
        clean_left, clean_left_end, clean_left_step = censor_at_eos(
            record["clean"]["target"], eos_set
        )
        clean_right, clean_right_end, clean_right_step = censor_at_eos(
            record["clean"]["cross"], eos_set
        )
        patched_left, patched_left_end, patched_left_step = censor_at_eos(
            record["patched"]["target"], eos_set
        )
        patched_right, patched_right_end, patched_right_step = censor_at_eos(
            record["patched"]["cross"], eos_set
        )
        left_hit = patched_left == clean_right
        right_hit = patched_right == clean_left
        left_suffix_hit = patched_left[1:] == clean_right[1:]
        right_suffix_hit = patched_right[1:] == clean_left[1:]
        step_hit = (
            patched_left_step == clean_right_step
            and patched_right_step == clean_left_step
        )
        left_exact.append(left_hit)
        right_exact.append(right_hit)
        both_exact.append(left_hit and right_hit)
        left_suffix.append(left_suffix_hit)
        right_suffix.append(right_suffix_hit)
        both_suffix.append(left_suffix_hit and right_suffix_hit)
        clean_terminated.extend((clean_left_end, clean_right_end))
        patched_terminated.extend((patched_left_end, patched_right_end))
        termination_step_matches.append(step_hit)
        audited.append({
            "target_index": int(record["target_index"]),
            "clean_target_censored": clean_left,
            "clean_cross_censored": clean_right,
            "patched_target_censored": patched_left,
            "patched_cross_censored": patched_right,
            "target_matches_other_clean": left_hit,
            "cross_matches_other_clean": right_hit,
            "both_match_other_clean": left_hit and right_hit,
            "both_suffix_after_label_matches": (
                left_suffix_hit and right_suffix_hit
            ),
            "termination_steps_match": step_hit,
            "legacy_both_match_other_clean": bool(
                record["both_match_other_clean"]
            ),
        })
    count = len(records)

    def mean(values: list[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    return audited, {
        "pair_count": count,
        "eos_token_ids": eos_ids,
        "target_matches_other_clean_rate": mean(left_exact),
        "cross_matches_other_clean_rate": mean(right_exact),
        "both_match_other_clean_rate": mean(both_exact),
        "target_suffix_after_label_match_rate": mean(left_suffix),
        "cross_suffix_after_label_match_rate": mean(right_suffix),
        "both_suffix_after_label_match_rate": mean(both_suffix),
        "clean_termination_rate": mean(clean_terminated),
        "patched_termination_rate": mean(patched_terminated),
        "both_termination_step_match_rate": mean(
            termination_step_matches
        ),
        "legacy_both_match_other_clean_rate": mean([
            bool(record["both_match_other_clean"])
            for record in records
        ]),
    }


def state_payload(
    groups: tuple[int, ...],
    slot_ids: tuple[int, ...],
    slots: list[list[int]],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "groups": list(groups),
        "slot_ids": list(slot_ids),
        "depths": flatten_slots(slot_ids, slots),
        "block_count": len(groups) * len(slot_ids),
        "rate": rate(result),
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1054 protocol audit failed")
    discovery_targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "discovery_targets.jsonl"
    )
    confirmation_targets = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / "confirmation_targets.jsonl"
    )
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    plan = prereg["model_plans"][model_name]
    all_groups = tuple(int(value) for value in plan["all_groups"])
    slots = [
        [int(value) for value in slot]
        for slot in plan["depth_slots"]
    ]
    all_slot_ids = tuple(range(len(slots)))
    all_depths = flatten_slots(all_slot_ids, slots)
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
            discovery_targets,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        discovery_valid, discovery_coverage = filtered_clean_targets(
            discovery_targets, cases, clean_discovery
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
        discovery_mask = np.ones(len(discovery_valid), dtype=bool)
        cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            dict[str, Any],
        ] = {}
        evaluation_budget = int(plan["search_max_evaluations"])

        def evaluate(
            groups: tuple[int, ...] | list[int],
            slot_ids: tuple[int, ...] | list[int],
        ) -> dict[str, Any]:
            key = (
                tuple(sorted(int(value) for value in groups)),
                tuple(sorted(int(value) for value in slot_ids)),
            )
            if not key[0] or not key[1]:
                raise ValueError("empty group-depth rectangle")
            if key not in cache:
                if len(cache) >= evaluation_budget:
                    raise RuntimeError("search evaluation budget exhausted")
                patched = bridge_scan.run_condition(
                    model,
                    device,
                    layers,
                    discovery_valid,
                    cases,
                    {
                        "site": "selected_concept",
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
                    discovery_mask,
                )
            return cache[key]

        full_discovery = evaluate(all_groups, all_slot_ids)
        full_rate = rate(full_discovery)
        minimum_rate = max(
            float(prereg["search_absolute_rate_min"]),
            full_rate * float(prereg["search_retention_fraction"]),
        )
        frontier = [(all_groups, all_slot_ids)]
        feasible = {
            (all_groups, all_slot_ids): full_discovery,
        }
        beam_trace = []
        budget_exhausted = False
        while frontier:
            children = {}
            parent_rows = []
            for groups, slot_ids in frontier:
                proposals = []
                for removed in groups:
                    candidate_groups = tuple(
                        value for value in groups if value != removed
                    )
                    if candidate_groups:
                        proposals.append((
                            candidate_groups,
                            slot_ids,
                            "group",
                            int(removed),
                        ))
                for removed in slot_ids:
                    candidate_slots = tuple(
                        value for value in slot_ids if value != removed
                    )
                    if candidate_slots:
                        proposals.append((
                            groups,
                            candidate_slots,
                            "slot",
                            int(removed),
                        ))
                trial_rows = []
                for (
                    candidate_groups,
                    candidate_slots,
                    removal_type,
                    removed,
                ) in proposals:
                    key = (candidate_groups, candidate_slots)
                    try:
                        result = evaluate(
                            candidate_groups, candidate_slots
                        )
                    except RuntimeError as exc:
                        if "budget exhausted" not in str(exc):
                            raise
                        budget_exhausted = True
                        break
                    payload = state_payload(
                        candidate_groups,
                        candidate_slots,
                        slots,
                        result,
                    )
                    payload.update({
                        "removal_type": removal_type,
                        "removed": removed,
                        "feasible": rate(result) >= minimum_rate,
                    })
                    trial_rows.append(payload)
                    if payload["feasible"]:
                        children[key] = result
                        feasible[key] = result
                parent_rows.append({
                    "parent_groups": list(groups),
                    "parent_slot_ids": list(slot_ids),
                    "trials": trial_rows,
                })
                if budget_exhausted:
                    break
            if not children:
                beam_trace.append({
                    "parents": parent_rows,
                    "selected_children": [],
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
                "parents": parent_rows,
                "selected_children": [
                    state_payload(
                        groups, slot_ids, slots, children[
                            (groups, slot_ids)
                        ]
                    )
                    for groups, slot_ids in frontier
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
        individual_cells = {}
        for group in all_groups:
            for slot_id in all_slot_ids:
                key = (str(group), str(slot_id))
                cache_key = ((group,), (slot_id,))
                if cache_key in cache:
                    individual_cells[f"{key[0]}:{key[1]}"] = compact(
                        cache[cache_key]
                    )

        clean_confirmation = bridge_scan.run_condition(
            model,
            device,
            layers,
            confirmation_targets,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        confirmation_valid, confirmation_coverage = (
            filtered_clean_targets(
                confirmation_targets, cases, clean_confirmation
            )
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
        confirmation_specs = {
            "selected_full": {
                "site": "selected_concept",
                "groups": list(all_groups),
                "depths": all_depths,
            },
            "selected_joint_rectangle": {
                "site": "selected_concept",
                "groups": list(frozen_groups),
                "depths": frozen_depths,
            },
            "unselected_joint_rectangle": {
                "site": "unselected_concept",
                "groups": list(frozen_groups),
                "depths": frozen_depths,
            },
            "query_joint_rectangle": {
                "site": "query_nonce",
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
                confirmation_specs[name],
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

        full_confirm = confirmation_results["selected_full"]
        joint_confirm = confirmation_results[
            "selected_joint_rectangle"
        ]
        control_rate = max(
            rate(confirmation_results["unselected_joint_rectangle"]),
            rate(confirmation_results["query_joint_rectangle"]),
        )
        full_confirm_rate = rate(full_confirm)
        joint_confirm_rate = rate(joint_confirm)
        retained = (
            joint_confirm_rate / full_confirm_rate
            if full_confirm_rate > 0 else 0.0
        )
        total_blocks = len(all_groups) * len(all_slot_ids)
        frozen_blocks = len(frozen_groups) * len(frozen_slot_ids)
        block_fraction = frozen_blocks / total_blocks
        gates = prereg["gates"]
        baseline_gate = (
            bool(plan["behavior_eligible"])
            and len(confirmation_valid)
            >= gates["confirmation_clean_pair_count_min"]
            and len(confirmation_coverage)
            >= gates["confirmation_family_coverage_min"]
        )
        coalition_gate = (
            baseline_gate
            and joint_confirm_rate
            >= gates["joint_both_counterfactual_rate_min"]
            and retained >= gates["joint_retained_fraction_min"]
            and joint_confirm_rate - control_rate
            >= gates["selected_minus_control_rate_min"]
            and block_fraction <= gates["maximum_block_fraction"]
        )

        successful_targets = [
            target
            for target, success in zip(
                confirmation_valid,
                joint_confirm["both_counterfactual_mask"],
            )
            if success
        ]
        raw_rollouts = []
        legacy_rollout_summary = {
            "pair_count": 0,
            "both_match_other_clean_rate": 0.0,
        }
        if successful_targets:
            raw_rollouts, legacy_rollout_summary = (
                bridge_scan.rollout_pairs(
                    model,
                    tokenizer,
                    device,
                    layers,
                    successful_targets,
                    cases,
                    confirmation_specs["selected_joint_rectangle"],
                    head_dim=head_dim,
                    steps=int(prereg["rollout_steps"]),
                    pair_limit=int(prereg["rollout_pair_limit"]),
                    pair_batch_size=PAIR_BATCH_SIZE[model_name],
                )
            )
        eos_ids = eos_token_ids(model, tokenizer)
        audited_rollouts, eos_rollout_summary = audit_rollouts(
            raw_rollouts, eos_ids
        )
        prior_summary = protocol.read_json(
            protocol.LOCALIZATION_ROOT
            / "atlas"
            / model_name
            / "summary.json"
        )
        _, prior_eos_reanalysis = audit_rollouts(
            prior_summary.get("rollouts", []), eos_ids
        )
        rollout_gate = (
            coalition_gate
            and eos_rollout_summary["pair_count"]
            >= gates["rollout_pair_count_min"]
            and eos_rollout_summary["both_match_other_clean_rate"]
            >= gates["eos_censored_both_match_rate_min"]
        )

        summary = {
            "schema_version": "phase1054_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "behavior_eligible": bool(plan["behavior_eligible"]),
            "discovery_clean_pair_count": len(discovery_valid),
            "discovery_family_coverage": discovery_coverage,
            "discovery_full": compact(full_discovery),
            "discovery_minimum_rate": minimum_rate,
            "search_evaluation_count": len(cache),
            "search_evaluation_budget": evaluation_budget,
            "search_budget_exhausted": budget_exhausted,
            "beam_trace": beam_trace,
            "feasible_state_count": len(feasible),
            "frozen_groups": list(frozen_groups),
            "frozen_depth_slot_ids": list(frozen_slot_ids),
            "frozen_depths": frozen_depths,
            "total_block_count": total_blocks,
            "frozen_block_count": frozen_blocks,
            "block_fraction": block_fraction,
            "discovery_joint_rectangle": compact(best_discovery),
            "observed_individual_cells": individual_cells,
            "confirmation_clean_pair_count": len(
                confirmation_valid
            ),
            "confirmation_family_coverage": confirmation_coverage,
            "confirmation_results": {
                key: compact(value)
                for key, value in confirmation_results.items()
            },
            "confirmation_joint_retained_fraction": retained,
            "confirmation_selected_minus_control_rate": (
                joint_confirm_rate - control_rate
            ),
            "coalition_gate_passed": coalition_gate,
            "eos_token_ids": eos_ids,
            "legacy_phase1053_eos_reanalysis": prior_eos_reanalysis,
            "legacy_rollout_summary": legacy_rollout_summary,
            "eos_rollout_summary": eos_rollout_summary,
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
            "full_discovery_rate": full_rate,
            "groups": list(frozen_groups),
            "slots": list(frozen_slot_ids),
            "block_fraction": block_fraction,
            "joint_discovery_rate": rate(best_discovery),
            "confirmation_clean": len(confirmation_valid),
            "full_confirmation_rate": full_confirm_rate,
            "joint_confirmation_rate": joint_confirm_rate,
            "control_rate": control_rate,
            "retained": retained,
            "coalition_gate": coalition_gate,
            "legacy_rollout": legacy_rollout_summary,
            "eos_rollout": eos_rollout_summary,
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
