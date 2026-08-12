#!/usr/bin/env python3
"""Audit translation equivalence and rerun donor-clean transport."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


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
import phase1061_translation_equivalence_protocol as protocol


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
    for family, site in (
        ("phrase", "source_phrase"),
        ("color", "source_color"),
        ("noun", "source_noun"),
    ):
        specs[f"{family}_post_kv"] = {
            **common,
            "pair_family": family,
            "site": site,
        }
    primary = {"pair_family": "phrase", "site": "source_phrase"}
    specs.update({
        "phrase_early_kv": {
            **common,
            **primary,
            "depths": [int(value) for value in plan["early_depths"]],
        },
        "phrase_all_kv": {
            **common,
            **primary,
            "depths": [int(value) for value in plan["all_layers"]],
        },
        "phrase_post_k_only": {
            **common,
            **primary,
            "channels": ["k"],
        },
        "phrase_post_v_only": {
            **common,
            **primary,
            "channels": ["v"],
        },
        "phrase_late_half_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["late_half_depths"]
            ],
        },
        "phrase_late_quarter_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["late_quarter_depths"]
            ],
        },
        "phrase_even_groups_kv": {
            **common,
            **primary,
            "groups": [int(value) for value in plan["even_groups"]],
        },
        "phrase_odd_groups_kv": {
            **common,
            **primary,
            "groups": [int(value) for value in plan["odd_groups"]],
        },
        "phrase_frozen_rectangle": {
            **common,
            **primary,
            "groups": [
                int(value) for value in plan["frozen_groups"]
            ],
            "depths": [
                int(value) for value in plan["frozen_depths"]
            ],
        },
        "operator_post_kv": {
            **common,
            **primary,
            "site": "operator",
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
        "target_language_post_kv": {
            **common,
            **primary,
            "site": "target_language",
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
    })
    return specs


def qualify_behavior(
    cases: dict[int, dict[str, Any]],
    clean_outputs: dict[int, list[int]],
    eos_ids: set[int],
) -> tuple[dict[str, Any], set[int], set[int]]:
    canonical = set()
    accepted = set()
    color_counts = Counter()
    color_totals = Counter()
    alternatives = Counter()
    examples = []
    for index, row in cases.items():
        generated = engine.content_tokens(clean_outputs[index], eos_ids)
        terminated = engine.terminated(clean_outputs[index], eos_ids)
        canonical_ids = [int(value) for value in row["expected_token_ids"]]
        acceptable_ids = [
            [int(value) for value in values]
            for values in row["acceptable_token_ids"]
        ]
        color = str(row["color_id"])
        color_totals[color] += 1
        if terminated and generated == canonical_ids:
            canonical.add(index)
        if terminated and generated in acceptable_ids:
            accepted.add(index)
            color_counts[color] += 1
            alternatives[str(row["acceptable_labels"][
                acceptable_ids.index(generated)
            ])] += 1
        elif len(examples) < 20:
            examples.append({
                "case_key": str(row["case_key"]),
                "acceptable_labels": list(row["acceptable_labels"]),
                "generated_token_ids": [
                    int(value) for value in clean_outputs[index]
                ],
            })
    return {
        "case_count": len(cases),
        "canonical_exact_count": len(canonical),
        "canonical_exact_rate": len(canonical) / len(cases),
        "multi_reference_exact_count": len(accepted),
        "multi_reference_exact_rate": len(accepted) / len(cases),
        "rescued_case_count": len(accepted - canonical),
        "accepted_counts_by_color": dict(color_counts),
        "total_counts_by_color": dict(color_totals),
        "accepted_rates_by_color": {
            color: color_counts[color] / color_totals[color]
            for color in color_totals
        },
        "accepted_label_counts": dict(alternatives),
        "mismatch_examples": examples,
    }, canonical, accepted


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1061 protocol audit failed")
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
        batch_size = PAIR_BATCH_SIZE[model_name]
        clean_outputs = engine.generate_case_outputs(
            model,
            device,
            case_rows,
            eos_ids=eos_ids,
            batch_size=batch_size,
            steps=int(prereg["generation_steps"]),
        )
        behavior_summary, canonical_indices, accepted_indices = (
            qualify_behavior(cases, clean_outputs, eos_ids)
        )
        valid = {
            family: engine.valid_targets(
                [
                    row for row in target_rows
                    if row["pair_family"] == family
                ],
                accepted_indices,
                clean_outputs,
                eos_ids,
            )
            for family in protocol.PAIR_FAMILIES
        }
        gates = prereg["gates"]
        behavior_gate = (
            len(accepted_indices) >= gates["accepted_case_count_min"]
            and all(
                len(valid[family])
                >= gates["valid_pair_count_per_family_min"]
                for family in protocol.PAIR_FAMILIES
            )
        )
        specs = condition_specifications(plan)
        condition_results = {}
        condition_records = {}
        selected_target_indices = {}
        for name, spec in specs.items():
            selected = engine.evenly_spaced(
                valid[str(spec["pair_family"])],
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
            }), flush=True)

        def rate(name: str) -> float:
            return float(
                condition_results[name][
                    "both_match_other_clean_eos_rate"
                ]
            )

        component_rates = {
            family: rate(f"{family}_post_kv")
            for family in protocol.PAIR_FAMILIES
        }
        control_rate = max(
            rate("operator_post_kv"),
            rate("target_language_post_kv"),
        )
        transport_gate = (
            behavior_gate
            and component_rates["phrase"]
            >= gates["phrase_post_eos_exact_rate_min"]
            and component_rates["color"]
            >= gates["component_post_eos_exact_rate_min"]
            and component_rates["noun"]
            >= gates["component_post_eos_exact_rate_min"]
            and component_rates["phrase"] - control_rate
            >= gates["source_minus_control_rate_min"]
        )
        phase_rates = {
            "early": rate("phrase_early_kv"),
            "post": rate("phrase_post_kv"),
            "all": rate("phrase_all_kv"),
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
            "phrase_post_kv",
            "phrase_late_half_kv",
            "phrase_late_quarter_kv",
            "phrase_even_groups_kv",
            "phrase_odd_groups_kv",
            "phrase_frozen_rectangle",
        )
        support_map = []
        for name in support_names:
            spec = specs[name]
            value = rate(name)
            depth_fraction = (
                len(spec["depths"]) / len(plan["postsource_depths"])
            )
            group_fraction = (
                len(spec["groups"]) / len(plan["all_groups"])
            )
            support_map.append({
                "condition": name,
                "depth_fraction_of_post": depth_fraction,
                "group_fraction": group_fraction,
                "tested_area_fraction": depth_fraction * group_fraction,
                "eos_exact_rate": value,
                "retention_vs_full_post": (
                    value / baseline if baseline > 0 else 0.0
                ),
            })
        summary = {
            "schema_version": "phase1061_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "behavior_summary": behavior_summary,
            "canonical_accepted_overlap_count": len(
                canonical_indices & accepted_indices
            ),
            "valid_pair_counts": {
                family: len(rows) for family, rows in valid.items()
            },
            "behavior_gate_passed": behavior_gate,
            "condition_results": condition_results,
            "condition_records": condition_records,
            "selected_target_indices": selected_target_indices,
            "component_rates": component_rates,
            "phase_rates": phase_rates,
            "phase_class": phase_class,
            "channel_rates": {
                "k_only": rate("phrase_post_k_only"),
                "v_only": rate("phrase_post_v_only"),
                "kv": baseline,
            },
            "support_map": support_map,
            "maximum_role_control_rate": control_rate,
            "equivalence_qualified_transport_gate_passed": transport_gate,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(
            protocol.OUT_ROOT / "atlas" / model_name / "summary.json",
            summary,
        )
        print(json.dumps({
            "model": model_name,
            "behavior": behavior_summary,
            "valid_pairs": summary["valid_pair_counts"],
            "behavior_gate": behavior_gate,
            "component_rates": component_rates,
            "phase_rates": phase_rates,
            "channel_rates": summary["channel_rates"],
            "support_map": support_map,
            "control": control_rate,
            "transport_gate": transport_gate,
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
