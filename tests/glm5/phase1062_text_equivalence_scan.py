#!/usr/bin/env python3
"""Run decoded-text behavior qualification and K/V transport."""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
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
import phase1061_translation_equivalence_scan as source_scan
import phase1062_text_equivalence_protocol as protocol


PAIR_BATCH_SIZE = bridge.PAIR_BATCH_SIZE


def normalize_text(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFC", value).strip().split()
    ).casefold()


def decode_content(
    tokenizer,
    values: list[int],
    eos_ids: set[int],
) -> str:
    content = engine.content_tokens(values, eos_ids)
    return normalize_text(
        tokenizer.decode(content, skip_special_tokens=True)
    )


def qualify_behavior(
    tokenizer,
    cases: dict[int, dict[str, Any]],
    clean_outputs: dict[int, list[int]],
    eos_ids: set[int],
) -> tuple[dict[str, Any], set[int], set[int]]:
    token_accepted = set()
    text_accepted = set()
    color_counts = Counter()
    color_totals = Counter()
    accepted_labels = Counter()
    examples = []
    for index, row in cases.items():
        generated_tokens = engine.content_tokens(
            clean_outputs[index], eos_ids
        )
        generated_text = decode_content(
            tokenizer, clean_outputs[index], eos_ids
        )
        acceptable_tokens = [
            [int(value) for value in values]
            for values in row["acceptable_token_ids"]
        ]
        acceptable_texts = [
            normalize_text(str(value))
            for value in row["acceptable_labels"]
        ]
        terminated = engine.terminated(clean_outputs[index], eos_ids)
        color = str(row["color_id"])
        color_totals[color] += 1
        if terminated and generated_tokens in acceptable_tokens:
            token_accepted.add(index)
        if terminated and generated_text in acceptable_texts:
            text_accepted.add(index)
            color_counts[color] += 1
            accepted_labels[str(row["acceptable_labels"][
                acceptable_texts.index(generated_text)
            ])] += 1
        elif len(examples) < 24:
            examples.append({
                "case_key": str(row["case_key"]),
                "generated_text": generated_text,
                "acceptable_texts": acceptable_texts,
            })
    return {
        "case_count": len(cases),
        "token_multi_reference_count": len(token_accepted),
        "token_multi_reference_rate": len(token_accepted) / len(cases),
        "text_multi_reference_count": len(text_accepted),
        "text_multi_reference_rate": len(text_accepted) / len(cases),
        "same_text_different_token_rescue_count": len(
            text_accepted - token_accepted
        ),
        "accepted_counts_by_color": dict(color_counts),
        "total_counts_by_color": dict(color_totals),
        "accepted_rates_by_color": {
            color: color_counts[color] / color_totals[color]
            for color in color_totals
        },
        "accepted_label_counts": dict(accepted_labels),
        "mismatch_examples": examples,
    }, token_accepted, text_accepted


def valid_targets_text(
    tokenizer,
    rows: list[dict[str, Any]],
    accepted_indices: set[int],
    clean_outputs: dict[int, list[int]],
    eos_ids: set[int],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        left = int(row["target_case_index"])
        right = int(row["cross_case_index"])
        if left not in accepted_indices or right not in accepted_indices:
            continue
        if decode_content(
            tokenizer, clean_outputs[left], eos_ids
        ) == decode_content(tokenizer, clean_outputs[right], eos_ids):
            continue
        output.append(row)
    return output


def add_text_metrics(
    tokenizer,
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    eos_ids: set[int],
) -> dict[str, Any]:
    text_rows = []
    for row in records:
        clean_target = decode_content(
            tokenizer, row["clean_target"], eos_ids
        )
        clean_cross = decode_content(
            tokenizer, row["clean_cross"], eos_ids
        )
        patched_target = decode_content(
            tokenizer, row["patched_target"], eos_ids
        )
        patched_cross = decode_content(
            tokenizer, row["patched_cross"], eos_ids
        )
        text_rows.append({
            "target_matches_other_clean_text": (
                patched_target == clean_cross
            ),
            "cross_matches_other_clean_text": (
                patched_cross == clean_target
            ),
            "both_match_other_clean_text": (
                patched_target == clean_cross
                and patched_cross == clean_target
            ),
            "clean_target_text": clean_target,
            "clean_cross_text": clean_cross,
            "patched_target_text": patched_target,
            "patched_cross_text": patched_cross,
        })
    count = len(text_rows)
    enriched = dict(metrics)
    enriched["both_match_other_clean_text_count"] = sum(
        bool(row["both_match_other_clean_text"]) for row in text_rows
    )
    enriched["both_match_other_clean_text_rate"] = (
        enriched["both_match_other_clean_text_count"] / count
        if count else 0.0
    )
    enriched["text_records"] = text_rows
    return enriched


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1062 protocol audit failed")
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
        behavior_summary, token_indices, text_indices = qualify_behavior(
            tokenizer, cases, clean_outputs, eos_ids
        )
        valid = {
            family: valid_targets_text(
                tokenizer,
                [
                    row for row in target_rows
                    if row["pair_family"] == family
                ],
                text_indices,
                clean_outputs,
                eos_ids,
            )
            for family in protocol.PAIR_FAMILIES
        }
        gates = prereg["gates"]
        behavior_gate = (
            len(text_indices) >= gates["accepted_case_count_min"]
            and all(
                len(valid[family])
                >= gates["valid_pair_count_per_family_min"]
                for family in protocol.PAIR_FAMILIES
            )
        )
        specs = source_scan.condition_specifications(plan)
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
                    "both_match_other_clean_text_rate": 0.0,
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
            metrics = add_text_metrics(
                tokenizer, metrics, records, eos_ids
            )
            condition_results[name] = metrics
            condition_records[name] = records
            print(json.dumps({
                "model": model_name,
                "condition": name,
                "pairs": metrics["pair_count"],
                "token_eos_exact": metrics[
                    "both_match_other_clean_eos_rate"
                ],
                "text_exact": metrics[
                    "both_match_other_clean_text_rate"
                ],
            }), flush=True)

        def text_rate(name: str) -> float:
            return float(
                condition_results[name][
                    "both_match_other_clean_text_rate"
                ]
            )

        def token_rate(name: str) -> float:
            return float(
                condition_results[name][
                    "both_match_other_clean_eos_rate"
                ]
            )

        component_text_rates = {
            family: text_rate(f"{family}_post_kv")
            for family in protocol.PAIR_FAMILIES
        }
        component_token_rates = {
            family: token_rate(f"{family}_post_kv")
            for family in protocol.PAIR_FAMILIES
        }
        control_text_rate = max(
            text_rate("operator_post_kv"),
            text_rate("target_language_post_kv"),
        )
        transport_gate = (
            behavior_gate
            and component_text_rates["phrase"]
            >= gates["phrase_post_text_exact_rate_min"]
            and component_text_rates["color"]
            >= gates["component_post_text_exact_rate_min"]
            and component_text_rates["noun"]
            >= gates["component_post_text_exact_rate_min"]
            and component_text_rates["phrase"] - control_text_rate
            >= gates["source_minus_control_text_rate_min"]
        )
        phase_text_rates = {
            "early": text_rate("phrase_early_kv"),
            "post": text_rate("phrase_post_kv"),
            "all": text_rate("phrase_all_kv"),
        }
        phase_token_rates = {
            "early": token_rate("phrase_early_kv"),
            "post": token_rate("phrase_post_kv"),
            "all": token_rate("phrase_all_kv"),
        }
        if (
            phase_text_rates["early"] >= 0.30
            and phase_text_rates["post"] >= 0.50
            and phase_text_rates["all"] <= 0.10
        ):
            phase_class = "early_post_conflict"
        elif (
            phase_text_rates["early"] <= 0.10
            and phase_text_rates["post"] >= 0.50
            and phase_text_rates["all"] >= 0.50
        ):
            phase_class = "late_dominant"
        else:
            phase_class = "mixed_or_unresolved"
        baseline = phase_text_rates["post"]
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
            value = text_rate(name)
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
                "text_exact_rate": value,
                "token_eos_exact_rate": token_rate(name),
                "retention_vs_full_post_text": (
                    value / baseline if baseline > 0 else 0.0
                ),
            })
        summary = {
            "schema_version": "phase1062_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "behavior_summary": behavior_summary,
            "token_text_accepted_overlap_count": len(
                token_indices & text_indices
            ),
            "valid_pair_counts": {
                family: len(rows) for family, rows in valid.items()
            },
            "behavior_gate_passed": behavior_gate,
            "condition_results": condition_results,
            "condition_records": condition_records,
            "selected_target_indices": selected_target_indices,
            "component_text_rates": component_text_rates,
            "component_token_rates": component_token_rates,
            "phase_text_rates": phase_text_rates,
            "phase_token_rates": phase_token_rates,
            "phase_class": phase_class,
            "channel_text_rates": {
                "k_only": text_rate("phrase_post_k_only"),
                "v_only": text_rate("phrase_post_v_only"),
                "kv": baseline,
            },
            "support_map": support_map,
            "maximum_role_control_text_rate": control_text_rate,
            "text_equivalence_transport_gate_passed": transport_gate,
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
            "component_text_rates": component_text_rates,
            "component_token_rates": component_token_rates,
            "phase_text_rates": phase_text_rates,
            "phase_token_rates": phase_token_rates,
            "channel_text_rates": summary["channel_text_rates"],
            "support_map": support_map,
            "control_text": control_text_rate,
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
