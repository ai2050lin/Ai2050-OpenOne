#!/usr/bin/env python3
"""Test natural translation as a new K/V transport pattern family."""

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
import phase1054_joint_kv_rollout_scan as rollout_tools
import phase1055_pattern_family_transfer_protocol as protocol


PAIR_BATCH_SIZE = bridge_scan.PAIR_BATCH_SIZE


def clean_metrics(
    targets: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    clean: dict[str, np.ndarray],
) -> dict[str, Any]:
    mask, labels = bridge_scan.clean_mask_and_coverage(
        targets, cases, clean
    )
    concepts = {
        concept
        for row, keep in zip(targets, mask)
        if keep
        for concept in (
            str(row["target_concept_id"]),
            str(row["cross_concept_id"]),
        )
    }
    count = len(targets)
    return {
        "pair_count": count,
        "correct_pair_count": int(mask.sum()),
        "pair_accuracy": float(mask.mean()) if count else 0.0,
        "correct_concept_coverage": len(concepts),
        "correct_target_label_coverage": len(labels),
        "correct_pair_mask": [bool(value) for value in mask],
    }


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in (
            "both_counterfactual_mask",
            "valid_target_indices",
        )
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1055 protocol audit failed")
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    plan = prereg["model_plans"][model_name]
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

        discovery_metrics = {}
        for variant in prereg["variant_order"]:
            panel_targets = [
                row for row in targets
                if row["split"] == "discovery"
                and row["variant"] == variant
            ]
            clean = bridge_scan.run_condition(
                model,
                device,
                layers,
                panel_targets,
                cases,
                None,
                head_dim=head_dim,
                pad_token_id=int(pad_token_id),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
            discovery_metrics[variant] = clean_metrics(
                panel_targets, cases, clean
            )
        order_index = {
            value: index
            for index, value in enumerate(prereg["variant_order"])
        }
        frozen_variant = max(
            prereg["variant_order"],
            key=lambda variant: (
                discovery_metrics[variant]["pair_accuracy"],
                discovery_metrics[variant]["correct_pair_count"],
                discovery_metrics[variant][
                    "correct_concept_coverage"
                ],
                -order_index[variant],
            ),
        )

        confirmation_targets = [
            row for row in targets
            if row["split"] == "confirmation"
            and row["variant"] == frozen_variant
        ]
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
        confirmation_behavior = clean_metrics(
            confirmation_targets, cases, clean_confirmation
        )
        valid_mask = np.asarray(
            confirmation_behavior["correct_pair_mask"],
            dtype=bool,
        )
        valid_targets = [
            row for row, keep in zip(
                confirmation_targets, valid_mask
            )
            if keep
        ]
        clean_valid = bridge_scan.run_condition(
            model,
            device,
            layers,
            valid_targets,
            cases,
            None,
            head_dim=head_dim,
            pad_token_id=int(pad_token_id),
            pair_batch_size=PAIR_BATCH_SIZE[model_name],
        )
        all_valid = np.ones(len(valid_targets), dtype=bool)
        gates = prereg["gates"]
        behavior_gate = (
            confirmation_behavior["correct_pair_count"]
            >= gates["behavior_correct_pair_count_min"]
            and confirmation_behavior["pair_accuracy"]
            >= gates["behavior_pair_accuracy_min"]
            and confirmation_behavior["correct_concept_coverage"]
            >= gates["behavior_concept_coverage_min"]
        )

        condition_results = {}
        condition_raw = {}
        for name in prereg["condition_order"]:
            patched = bridge_scan.run_condition(
                model,
                device,
                layers,
                valid_targets,
                cases,
                plan["conditions"][name],
                head_dim=head_dim,
                pad_token_id=int(pad_token_id),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
            condition_raw[name] = patched
            condition_results[name] = bridge_scan.condition_metrics(
                valid_targets,
                cases,
                clean_valid,
                patched,
                all_valid,
            )

        broad = condition_results[
            "source_all_groups_all_layers"
        ]
        fact = condition_results["source_fact_rectangle"]
        controls = (
            condition_results["operator_all_groups_all_layers"],
            condition_results[
                "target_language_all_groups_all_layers"
            ],
        )
        broad_rate = float(
            broad["both_counterfactual_top1_rate"]
        )
        fact_rate = float(fact["both_counterfactual_top1_rate"])
        control_rate = max(
            float(row["both_counterfactual_top1_rate"])
            for row in controls
        )
        broad_gate = (
            behavior_gate
            and broad["both_counterfactual_top1_count"]
            >= gates["broad_both_counterfactual_count_min"]
            and broad_rate
            >= gates["broad_both_counterfactual_rate_min"]
            and broad_rate - control_rate
            >= gates["source_minus_control_rate_min"]
        )
        fact_retained = (
            fact_rate / broad_rate if broad_rate > 0 else 0.0
        )
        fact_reuse_gate = (
            broad_gate
            and fact_rate >= gates["fact_rectangle_rate_min"]
            and fact_retained
            >= gates["fact_rectangle_retained_fraction_min"]
        )

        successful_targets = [
            row for row, success in zip(
                valid_targets,
                broad["both_counterfactual_mask"],
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
                plan["conditions"][
                    "source_all_groups_all_layers"
                ],
                head_dim=head_dim,
                steps=int(prereg["rollout_steps"]),
                pair_limit=int(prereg["rollout_pair_limit"]),
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
            )
        eos_ids = rollout_tools.eos_token_ids(model, tokenizer)
        audited_rollouts, eos_rollout = (
            rollout_tools.audit_rollouts(raw_rollouts, eos_ids)
        )
        rollout_gate = (
            broad_gate
            and eos_rollout["pair_count"]
            >= gates["rollout_pair_count_min"]
            and eos_rollout["both_match_other_clean_rate"]
            >= gates["eos_censored_both_match_rate_min"]
        )

        summary = {
            "schema_version": "phase1055_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "frozen_variant": frozen_variant,
            "discovery_behavior": discovery_metrics,
            "confirmation_behavior": {
                key: value
                for key, value in confirmation_behavior.items()
                if key != "correct_pair_mask"
            },
            "behavior_gate_passed": behavior_gate,
            "valid_confirmation_pair_count": len(valid_targets),
            "condition_results": {
                key: compact(value)
                for key, value in condition_results.items()
            },
            "maximum_control_rate": control_rate,
            "broad_bridge_gate_passed": broad_gate,
            "fact_rectangle_retained_fraction": fact_retained,
            "fact_rectangle_reuse_gate_passed": fact_reuse_gate,
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
            "variant": frozen_variant,
            "discovery": discovery_metrics[frozen_variant],
            "confirmation": summary["confirmation_behavior"],
            "behavior_gate": behavior_gate,
            "valid_pairs": len(valid_targets),
            "fact_rectangle_rate": fact_rate,
            "postsource_rate": condition_results[
                "source_all_groups_postsource"
            ]["both_counterfactual_top1_rate"],
            "all_layers_rate": broad_rate,
            "control_rate": control_rate,
            "broad_gate": broad_gate,
            "fact_reuse_gate": fact_reuse_gate,
            "legacy_rollout": legacy_rollout,
            "eos_rollout": eos_rollout,
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
