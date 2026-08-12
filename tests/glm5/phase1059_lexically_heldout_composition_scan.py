#!/usr/bin/env python3
"""Run held-out phrase transport and fixed support-width scans."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as engine
import phase1059_lexically_heldout_composition_protocol as protocol


PAIR_BATCH_SIZE = bridge.PAIR_BATCH_SIZE


def condition_specifications(
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    all_groups = [int(value) for value in plan["all_groups"]]
    post = [int(value) for value in plan["postsource_depths"]]
    common = {
        "channels": ["k", "v"],
        "groups": all_groups,
        "depths": post,
        "pair_limit": protocol.PAIR_LIMIT,
    }
    specs = {}
    for split in ("discovery", "confirmation"):
        for family, site in (
            ("phrase", "source_phrase"),
            ("color", "source_color"),
            ("noun", "source_noun"),
        ):
            specs[f"{split}_{family}_post_kv"] = {
                **common,
                "split": split,
                "pair_family": family,
                "site": site,
            }
    specs.update({
        "confirmation_phrase_early_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "depths": [int(value) for value in plan["early_depths"]],
        },
        "confirmation_phrase_all_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "depths": [int(value) for value in plan["all_layers"]],
        },
        "confirmation_phrase_post_k_only": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["k"],
        },
        "confirmation_phrase_post_v_only": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["v"],
        },
        "confirmation_phrase_late_half_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "depths": [
                int(value) for value in plan["late_half_depths"]
            ],
        },
        "confirmation_phrase_late_quarter_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "depths": [
                int(value) for value in plan["late_quarter_depths"]
            ],
        },
        "confirmation_phrase_even_groups_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "groups": [int(value) for value in plan["even_groups"]],
        },
        "confirmation_phrase_odd_groups_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "groups": [int(value) for value in plan["odd_groups"]],
        },
        "confirmation_phrase_frozen_rectangle": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "source_phrase",
            "groups": [
                int(value) for value in plan["frozen_groups"]
            ],
            "depths": [
                int(value) for value in plan["frozen_depths"]
            ],
        },
        "confirmation_operator_post_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "operator",
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
        "confirmation_target_language_post_kv": {
            **common,
            "split": "confirmation",
            "pair_family": "phrase",
            "site": "target_language",
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
    })
    return specs


def support_record(
    name: str,
    spec: dict[str, Any],
    rate: float,
    baseline: float,
    plan: dict[str, Any],
) -> dict[str, Any]:
    depth_fraction = (
        len(spec["depths"]) / len(plan["postsource_depths"])
    )
    group_fraction = len(spec["groups"]) / len(plan["all_groups"])
    return {
        "condition": name,
        "depth_count": len(spec["depths"]),
        "group_count": len(spec["groups"]),
        "depth_fraction_of_post": depth_fraction,
        "group_fraction": group_fraction,
        "tested_area_fraction": depth_fraction * group_fraction,
        "eos_exact_rate": rate,
        "retention_vs_full_post": (
            rate / baseline if baseline > 0 else 0.0
        ),
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1059 protocol audit failed")
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    target_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
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
        width = bridge.projection_width(layers[0].self_attn.k_proj)
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("KV projection geometry drift")
        head_dim = width // n_kv_heads
        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        if not eos_ids:
            raise RuntimeError("no EOS token ids discovered")
        batch_size = PAIR_BATCH_SIZE[model_name]
        clean_outputs = engine.generate_case_outputs(
            model,
            device,
            case_rows,
            eos_ids=eos_ids,
            batch_size=batch_size,
            steps=int(prereg["generation_steps"]),
        )
        behavior_summary, exact_indices = engine.clean_behavior(
            cases, clean_outputs, eos_ids
        )
        valid = {}
        for split in ("discovery", "confirmation"):
            for family in protocol.PAIR_FAMILIES:
                valid[(split, family)] = engine.valid_targets(
                    [
                        row for row in target_rows
                        if row["split"] == split
                        and row["pair_family"] == family
                    ],
                    exact_indices,
                    clean_outputs,
                    eos_ids,
                )
        gates = prereg["gates"]
        behavior_gate = (
            all(
                behavior_summary["exact_case_counts"].get(split, 0)
                >= gates["exact_case_count_per_split_min"]
                for split in ("discovery", "confirmation")
            )
            and all(
                len(valid[(split, family)])
                >= gates["valid_pair_count_per_split_family_min"]
                for split in ("discovery", "confirmation")
                for family in protocol.PAIR_FAMILIES
            )
        )
        specs = condition_specifications(plan)
        condition_results = {}
        condition_records = {}
        selected_target_indices = {}
        for name, spec in specs.items():
            selected = engine.evenly_spaced(
                valid[(str(spec["split"]), str(spec["pair_family"]))],
                int(spec["pair_limit"]),
            )
            selected_target_indices[name] = [
                int(row["target_index"]) for row in selected
            ]
            if not selected or not spec["groups"] or not spec["depths"]:
                condition_results[name] = {
                    "pair_count": 0,
                    "both_match_other_clean_eos_rate": 0.0,
                }
                condition_records[name] = []
                continue
            patched = engine.generate_paired_cached(
                model,
                device,
                layers,
                selected,
                cases,
                spec,
                head_dim=head_dim,
                eos_ids=eos_ids,
                pair_batch_size=batch_size,
                steps=int(prereg["generation_steps"]),
            )
            metrics, records = engine.evaluate_condition(
                selected, clean_outputs, patched, eos_ids
            )
            condition_results[name] = metrics
            condition_records[name] = records
            print(json.dumps({
                "model": model_name,
                "condition": name,
                "pairs": metrics["pair_count"],
                "eos_exact": metrics[
                    "both_match_other_clean_eos_rate"
                ],
                "content_exact": metrics[
                    "both_match_other_clean_content_rate"
                ],
            }), flush=True)

        def rate(name: str) -> float:
            return float(
                condition_results[name][
                    "both_match_other_clean_eos_rate"
                ]
            )

        component_rates = {
            split: {
                family: rate(f"{split}_{family}_post_kv")
                for family in protocol.PAIR_FAMILIES
            }
            for split in ("discovery", "confirmation")
        }
        control_rate = max(
            rate("confirmation_operator_post_kv"),
            rate("confirmation_target_language_post_kv"),
        )
        holdout_gate = (
            behavior_gate
            and all(
                component_rates[split]["phrase"]
                >= gates["phrase_post_eos_exact_rate_min"]
                and component_rates[split]["color"]
                >= gates["component_post_eos_exact_rate_min"]
                and component_rates[split]["noun"]
                >= gates["component_post_eos_exact_rate_min"]
                for split in ("discovery", "confirmation")
            )
            and component_rates["confirmation"]["phrase"] - control_rate
            >= gates["source_minus_control_rate_min"]
        )
        phase_rates = {
            "early": rate("confirmation_phrase_early_kv"),
            "post": rate("confirmation_phrase_post_kv"),
            "all": rate("confirmation_phrase_all_kv"),
        }
        if (
            phase_rates["early"] >= 0.30
            and phase_rates["post"] >= 0.50
            and phase_rates["all"] <= 0.10
        ):
            phase_class = "early_post_conflict"
        elif (
            phase_rates["early"] <= 0.10
            and phase_rates["post"] >= 0.50
            and phase_rates["all"] >= 0.50
        ):
            phase_class = "late_dominant"
        else:
            phase_class = "mixed_or_unresolved"
        baseline = phase_rates["post"]
        support_names = (
            "confirmation_phrase_post_kv",
            "confirmation_phrase_late_half_kv",
            "confirmation_phrase_late_quarter_kv",
            "confirmation_phrase_even_groups_kv",
            "confirmation_phrase_odd_groups_kv",
            "confirmation_phrase_frozen_rectangle",
        )
        support_map = [
            support_record(
                name,
                specs[name],
                rate(name),
                baseline,
                plan,
            )
            for name in support_names
        ]
        summary = {
            "schema_version": "phase1059_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "eos_token_ids": sorted(eos_ids),
            "clean_behavior": behavior_summary,
            "valid_pair_counts": {
                f"{split}.{family}": len(rows)
                for (split, family), rows in sorted(valid.items())
            },
            "behavior_gate_passed": behavior_gate,
            "condition_results": condition_results,
            "condition_records": condition_records,
            "selected_target_indices": selected_target_indices,
            "component_rates": component_rates,
            "phase_rates": phase_rates,
            "phase_class": phase_class,
            "channel_rates": {
                "k_only": rate(
                    "confirmation_phrase_post_k_only"
                ),
                "v_only": rate(
                    "confirmation_phrase_post_v_only"
                ),
                "kv": baseline,
            },
            "support_map": support_map,
            "maximum_role_control_rate": control_rate,
            "fully_heldout_composition_gate_passed": holdout_gate,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(
            protocol.OUT_ROOT / "atlas" / model_name / "summary.json",
            summary,
        )
        print(json.dumps({
            "model": model_name,
            "behavior_gate": behavior_gate,
            "exact_cases": behavior_summary["exact_case_counts"],
            "component_rates": component_rates,
            "phase_rates": phase_rates,
            "phase_class": phase_class,
            "channel_rates": summary["channel_rates"],
            "support_map": support_map,
            "control": control_rate,
            "holdout_gate": holdout_gate,
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
