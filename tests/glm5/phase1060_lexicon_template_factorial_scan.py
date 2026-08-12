#!/usr/bin/env python3
"""Run the Phase1060 lexicon-by-template factorial audit."""

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
import phase1060_lexicon_template_factorial_protocol as protocol


PAIR_BATCH_SIZE = bridge.PAIR_BATCH_SIZE


def clean_behavior_factorial(
    cases: dict[int, dict[str, Any]],
    clean_outputs: dict[int, list[int]],
    eos_ids: set[int],
) -> tuple[dict[str, Any], set[int]]:
    exact_indices = set()
    exact_counts = Counter()
    termination_counts = Counter()
    total_counts = Counter()
    examples = []
    for index, row in cases.items():
        cell = str(row["cell"])
        total_counts[cell] += 1
        generated = clean_outputs[index]
        if engine.terminated(generated, eos_ids):
            termination_counts[cell] += 1
        exact = (
            engine.content_tokens(generated, eos_ids)
            == [int(value) for value in row["expected_token_ids"]]
            and engine.terminated(generated, eos_ids)
        )
        if exact:
            exact_indices.add(index)
            exact_counts[cell] += 1
        elif len(examples) < 16:
            examples.append({
                "case_key": str(row["case_key"]),
                "expected": [
                    int(value) for value in row["expected_token_ids"]
                ],
                "generated": [int(value) for value in generated],
            })
    return {
        "exact_case_counts": dict(exact_counts),
        "total_case_counts": dict(total_counts),
        "termination_counts": dict(termination_counts),
        "exact_case_rates": {
            cell: exact_counts[cell] / total_counts[cell]
            for cell in total_counts
        },
        "mismatch_examples": examples,
    }, exact_indices


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
    for cell in protocol.CELLS:
        specs[f"{cell}_phrase_post_kv"] = {
            **common,
            "cell": cell,
            "pair_family": "phrase",
            "site": "source_phrase",
        }
    for cell in ("old_old", "new_old"):
        for family, site in (
            ("color", "source_color"),
            ("noun", "source_noun"),
        ):
            specs[f"{cell}_{family}_post_kv"] = {
                **common,
                "cell": cell,
                "pair_family": family,
                "site": site,
            }
    primary = {
        "cell": "new_old",
        "pair_family": "phrase",
        "site": "source_phrase",
    }
    specs.update({
        "new_old_phrase_early_kv": {
            **common,
            **primary,
            "depths": [int(value) for value in plan["early_depths"]],
        },
        "new_old_phrase_all_kv": {
            **common,
            **primary,
            "depths": [int(value) for value in plan["all_layers"]],
        },
        "new_old_phrase_post_k_only": {
            **common,
            **primary,
            "channels": ["k"],
        },
        "new_old_phrase_post_v_only": {
            **common,
            **primary,
            "channels": ["v"],
        },
        "new_old_phrase_late_half_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["late_half_depths"]
            ],
        },
        "new_old_phrase_late_quarter_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["late_quarter_depths"]
            ],
        },
        "new_old_phrase_even_groups_kv": {
            **common,
            **primary,
            "groups": [int(value) for value in plan["even_groups"]],
        },
        "new_old_phrase_odd_groups_kv": {
            **common,
            **primary,
            "groups": [int(value) for value in plan["odd_groups"]],
        },
        "new_old_phrase_frozen_rectangle": {
            **common,
            **primary,
            "groups": [
                int(value) for value in plan["frozen_groups"]
            ],
            "depths": [
                int(value) for value in plan["frozen_depths"]
            ],
        },
        "new_old_operator_post_kv": {
            **common,
            **primary,
            "site": "operator",
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
        "new_old_target_language_post_kv": {
            **common,
            **primary,
            "site": "target_language",
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
    })
    return specs


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1060 protocol audit failed")
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
        behavior_summary, exact_indices = clean_behavior_factorial(
            cases, clean_outputs, eos_ids
        )
        exact_by_cell = Counter(
            str(row["cell"])
            for index, row in cases.items()
            if index in exact_indices
        )
        valid = {}
        for cell in protocol.CELLS:
            for family in protocol.PAIR_FAMILIES:
                valid[(cell, family)] = engine.valid_targets(
                    [
                        row for row in target_rows
                        if row["cell"] == cell
                        and row["pair_family"] == family
                    ],
                    exact_indices,
                    clean_outputs,
                    eos_ids,
                )
        gates = prereg["gates"]
        cell_behavior_pass = {
            cell: (
                exact_by_cell[cell]
                >= gates["exact_case_count_per_primary_cell_min"]
                and all(
                    len(valid[(cell, family)])
                    >= gates[
                        "valid_pair_count_per_primary_family_min"
                    ]
                    for family in protocol.PAIR_FAMILIES
                )
            )
            for cell in protocol.CELLS
        }
        specs = condition_specifications(plan)
        condition_results = {}
        condition_records = {}
        selected_target_indices = {}
        for name, spec in specs.items():
            selected = engine.evenly_spaced(
                valid[(str(spec["cell"]), str(spec["pair_family"]))],
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

        phrase_rates = {
            cell: rate(f"{cell}_phrase_post_kv")
            for cell in protocol.CELLS
        }
        component_rates = {
            cell: {
                "phrase": phrase_rates[cell],
                "color": (
                    rate(f"{cell}_color_post_kv")
                    if f"{cell}_color_post_kv" in condition_results
                    else None
                ),
                "noun": (
                    rate(f"{cell}_noun_post_kv")
                    if f"{cell}_noun_post_kv" in condition_results
                    else None
                ),
            }
            for cell in protocol.CELLS
        }
        control_rate = max(
            rate("new_old_operator_post_kv"),
            rate("new_old_target_language_post_kv"),
        )
        primary_gate = (
            cell_behavior_pass["old_old"]
            and cell_behavior_pass["new_old"]
            and component_rates["new_old"]["phrase"]
            >= gates["phrase_post_eos_exact_rate_min"]
            and component_rates["new_old"]["color"]
            >= gates["component_post_eos_exact_rate_min"]
            and component_rates["new_old"]["noun"]
            >= gates["component_post_eos_exact_rate_min"]
            and component_rates["new_old"]["phrase"] - control_rate
            >= gates["source_minus_control_rate_min"]
        )
        phase_rates = {
            "early": rate("new_old_phrase_early_kv"),
            "post": rate("new_old_phrase_post_kv"),
            "all": rate("new_old_phrase_all_kv"),
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
            "new_old_phrase_post_kv",
            "new_old_phrase_late_half_kv",
            "new_old_phrase_late_quarter_kv",
            "new_old_phrase_even_groups_kv",
            "new_old_phrase_odd_groups_kv",
            "new_old_phrase_frozen_rectangle",
        )
        support_map = []
        for name in support_names:
            spec = specs[name]
            depth_fraction = (
                len(spec["depths"]) / len(plan["postsource_depths"])
            )
            group_fraction = (
                len(spec["groups"]) / len(plan["all_groups"])
            )
            value = rate(name)
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
        behavior_factor_contrasts = {
            "template_effect_on_old_lexicon": (
                exact_by_cell["old_new"] / 112
                - exact_by_cell["old_old"] / 112
            ),
            "template_effect_on_new_lexicon": (
                exact_by_cell["new_new"] / 112
                - exact_by_cell["new_old"] / 112
            ),
            "lexicon_effect_under_old_templates": (
                exact_by_cell["new_old"] / 112
                - exact_by_cell["old_old"] / 112
            ),
            "lexicon_effect_under_new_templates": (
                exact_by_cell["new_new"] / 112
                - exact_by_cell["old_new"] / 112
            ),
        }
        summary = {
            "schema_version": "phase1060_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "clean_behavior": behavior_summary,
            "exact_case_counts_by_cell": dict(exact_by_cell),
            "exact_case_rates_by_cell": {
                cell: exact_by_cell[cell] / 112
                for cell in protocol.CELLS
            },
            "valid_pair_counts": {
                f"{cell}.{family}": len(rows)
                for (cell, family), rows in sorted(valid.items())
            },
            "cell_behavior_passed": cell_behavior_pass,
            "behavior_factor_contrasts": behavior_factor_contrasts,
            "condition_results": condition_results,
            "condition_records": condition_records,
            "selected_target_indices": selected_target_indices,
            "component_rates": component_rates,
            "phase_rates": phase_rates,
            "phase_class": phase_class,
            "channel_rates": {
                "k_only": rate("new_old_phrase_post_k_only"),
                "v_only": rate("new_old_phrase_post_v_only"),
                "kv": baseline,
            },
            "support_map": support_map,
            "maximum_role_control_rate": control_rate,
            "new_lexicon_old_template_gate_passed": primary_gate,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(
            protocol.OUT_ROOT / "atlas" / model_name / "summary.json",
            summary,
        )
        print(json.dumps({
            "model": model_name,
            "exact_by_cell": summary["exact_case_counts_by_cell"],
            "cell_behavior": cell_behavior_pass,
            "factor_contrasts": behavior_factor_contrasts,
            "component_rates": component_rates,
            "phase_rates": phase_rates,
            "channel_rates": summary["channel_rates"],
            "support_map": support_map,
            "control": control_rate,
            "primary_gate": primary_gate,
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
