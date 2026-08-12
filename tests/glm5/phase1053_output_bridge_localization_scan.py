#!/usr/bin/env python3
"""Localize distributed K/V group-depth coalitions."""

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
import phase1053_output_bridge_localization_protocol as protocol


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


def all_true_mask(count: int) -> np.ndarray:
    return np.ones(count, dtype=bool)


def rate(
    result: dict[str, Any],
) -> float:
    return float(result["both_counterfactual_top1_rate"])


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1053 protocol audit failed")
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
    all_groups = [int(value) for value in plan["all_groups"]]
    slots = [
        [int(value) for value in slot]
        for slot in plan["depth_slots"]
    ]
    all_depths = [
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
        clean_valid = bridge_scan.run_condition(
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
        discovery_mask = all_true_mask(len(discovery_valid))
        cache: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}

        def evaluate(
            groups: list[int],
            depths: list[int],
        ) -> dict[str, Any]:
            key = (tuple(sorted(groups)), tuple(sorted(depths)))
            if key not in cache:
                condition = {
                    "site": "selected_concept",
                    "groups": list(key[0]),
                    "depths": list(key[1]),
                }
                patched = bridge_scan.run_condition(
                    model,
                    device,
                    layers,
                    discovery_valid,
                    cases,
                    condition,
                    head_dim=head_dim,
                    pad_token_id=int(pad_token_id),
                    pair_batch_size=PAIR_BATCH_SIZE[model_name],
                )
                cache[key] = bridge_scan.condition_metrics(
                    discovery_valid,
                    cases,
                    clean_valid,
                    patched,
                    discovery_mask,
                )
            return cache[key]

        full_discovery = evaluate(all_groups, all_depths)
        full_rate = rate(full_discovery)
        minimum_rate = max(
            float(prereg["discovery_absolute_rate_min"]),
            full_rate * float(prereg["retention_fraction"]),
        )

        individual_groups = {
            str(group): evaluate([group], all_depths)
            for group in all_groups
        }
        group_trace = []
        coalition_groups = list(all_groups)
        while len(coalition_groups) > 1:
            trials = []
            for removed in coalition_groups:
                candidate = [
                    value for value in coalition_groups
                    if value != removed
                ]
                result = evaluate(candidate, all_depths)
                trials.append({
                    "removed": removed,
                    "remaining": candidate,
                    "rate": rate(result),
                })
            best = max(
                trials,
                key=lambda row: (
                    float(row["rate"]),
                    -int(row["removed"]),
                ),
            )
            accepted = float(best["rate"]) >= minimum_rate
            group_trace.append({
                "current": list(coalition_groups),
                "trials": trials,
                "accepted": accepted,
                "selected_removal": (
                    int(best["removed"]) if accepted else None
                ),
            })
            if not accepted:
                break
            coalition_groups = [
                int(value) for value in best["remaining"]
            ]

        individual_slots = {
            str(slot): evaluate(all_groups, slots[slot])
            for slot in range(len(slots))
        }
        depth_trace = []
        coalition_slot_ids = list(range(len(slots)))
        while len(coalition_slot_ids) > 1:
            trials = []
            for removed in coalition_slot_ids:
                candidate_ids = [
                    value for value in coalition_slot_ids
                    if value != removed
                ]
                candidate_depths = [
                    depth
                    for slot_id in candidate_ids
                    for depth in slots[slot_id]
                ]
                result = evaluate(all_groups, candidate_depths)
                trials.append({
                    "removed": removed,
                    "remaining": candidate_ids,
                    "rate": rate(result),
                })
            best = max(
                trials,
                key=lambda row: (
                    float(row["rate"]),
                    -int(row["removed"]),
                ),
            )
            accepted = float(best["rate"]) >= minimum_rate
            depth_trace.append({
                "current": list(coalition_slot_ids),
                "trials": trials,
                "accepted": accepted,
                "selected_removal": (
                    int(best["removed"]) if accepted else None
                ),
            })
            if not accepted:
                break
            coalition_slot_ids = [
                int(value) for value in best["remaining"]
            ]
        coalition_depths = [
            depth
            for slot_id in coalition_slot_ids
            for depth in slots[slot_id]
        ]
        joint_discovery = evaluate(
            coalition_groups, coalition_depths
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
        confirmation_mask = all_true_mask(len(confirmation_valid))
        confirmation_specs = {
            "selected_full": {
                "site": "selected_concept",
                "groups": all_groups,
                "depths": all_depths,
            },
            "selected_joint_coalition": {
                "site": "selected_concept",
                "groups": coalition_groups,
                "depths": coalition_depths,
            },
            "unselected_joint_coalition": {
                "site": "unselected_concept",
                "groups": coalition_groups,
                "depths": coalition_depths,
            },
            "query_joint_coalition": {
                "site": "query_nonce",
                "groups": coalition_groups,
                "depths": coalition_depths,
            },
        }
        confirmation_results = {}
        confirmation_raw = {}
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
            confirmation_raw[name] = patched
            confirmation_results[name] = (
                bridge_scan.condition_metrics(
                    confirmation_valid,
                    cases,
                    clean_confirmation_valid,
                    patched,
                    confirmation_mask,
                )
            )

        full_confirm = confirmation_results["selected_full"]
        joint_confirm = confirmation_results[
            "selected_joint_coalition"
        ]
        controls = (
            confirmation_results["unselected_joint_coalition"],
            confirmation_results["query_joint_coalition"],
        )
        full_confirm_rate = rate(full_confirm)
        joint_confirm_rate = rate(joint_confirm)
        control_rate = max(rate(row) for row in controls)
        retained = (
            joint_confirm_rate / full_confirm_rate
            if full_confirm_rate > 0 else 0.0
        )
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
        )

        success_mask = joint_confirm["both_counterfactual_mask"]
        successful_targets = [
            target
            for target, success in zip(
                confirmation_valid, success_mask
            )
            if success
        ]
        rollouts = []
        rollout_summary = {
            "pair_count": 0,
            "target_matches_other_clean_rate": 0.0,
            "cross_matches_other_clean_rate": 0.0,
            "both_match_other_clean_rate": 0.0,
        }
        if successful_targets:
            rollouts, rollout_summary = bridge_scan.rollout_pairs(
                model,
                tokenizer,
                device,
                layers,
                successful_targets,
                cases,
                confirmation_specs["selected_joint_coalition"],
                head_dim=head_dim,
                steps=int(prereg["rollout_steps"]),
                pair_limit=int(prereg["rollout_pair_limit"]),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
        rollout_gate = (
            coalition_gate
            and rollout_summary["pair_count"]
            >= gates["rollout_pair_count_min"]
            and rollout_summary["both_match_other_clean_rate"]
            >= gates["rollout_both_match_other_clean_rate_min"]
        )

        def compact(result: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value for key, value in result.items()
                if key not in (
                    "both_counterfactual_mask",
                    "valid_target_indices",
                )
            }

        summary = {
            "schema_version": "phase1053_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "behavior_eligible": bool(plan["behavior_eligible"]),
            "discovery_clean_pair_count": len(discovery_valid),
            "discovery_family_coverage": discovery_coverage,
            "discovery_full": compact(full_discovery),
            "discovery_minimum_retained_rate": minimum_rate,
            "individual_group_results": {
                key: compact(value)
                for key, value in individual_groups.items()
            },
            "group_deletion_trace": group_trace,
            "frozen_group_coalition": coalition_groups,
            "individual_depth_slot_results": {
                key: compact(value)
                for key, value in individual_slots.items()
            },
            "depth_deletion_trace": depth_trace,
            "frozen_depth_slot_coalition": coalition_slot_ids,
            "frozen_depths": coalition_depths,
            "discovery_joint": compact(joint_discovery),
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
            "rollout_summary": rollout_summary,
            "rollout_gate_passed": rollout_gate,
            "rollouts": rollouts,
            "elapsed_seconds": float(time.time() - started),
        }
        out = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_json(out / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "eligible": plan["behavior_eligible"],
            "discovery_clean": len(discovery_valid),
            "discovery_full_rate": full_rate,
            "groups": coalition_groups,
            "depth_slots": coalition_slot_ids,
            "discovery_joint_rate": rate(joint_discovery),
            "confirmation_clean": len(confirmation_valid),
            "confirmation_full_rate": full_confirm_rate,
            "confirmation_joint_rate": joint_confirm_rate,
            "control_rate": control_rate,
            "retained": retained,
            "coalition_gate": coalition_gate,
            "rollout": rollout_summary,
            "rollout_gate": rollout_gate,
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
