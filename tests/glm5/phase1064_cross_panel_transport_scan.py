#!/usr/bin/env python3
"""Run cross-panel K/V transport with clean-replay parity."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
import phase1062_text_equivalence_scan as text_engine
import phase1063_lexical_behavior_atlas_protocol as source
import phase1063_lexical_behavior_atlas_scan as behavior_scan
import phase1064_cross_panel_transport_protocol as protocol


PAIR_BATCH_SIZE = bridge.PAIR_BATCH_SIZE


def condition_specifications(
    panel: str,
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    all_groups = [int(value) for value in plan["all_groups"]]
    post = [int(value) for value in plan["postsource_depths"]]
    common = {
        "channels": ["k", "v"],
        "groups": all_groups,
        "depths": post,
        "pair_limit": int(protocol.PAIR_LIMITS[panel]),
    }
    primary = {"pair_family": "phrase", "site": "source_phrase"}
    return {
        f"{panel}.phrase_post_kv": {
            **common,
            **primary,
        },
        f"{panel}.color_post_kv": {
            **common,
            "pair_family": "color",
            "site": "source_color",
        },
        f"{panel}.noun_post_kv": {
            **common,
            "pair_family": "noun",
            "site": "source_noun",
        },
        f"{panel}.phrase_early_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["early_depths"]
            ],
        },
        f"{panel}.phrase_all_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["all_layers"]
            ],
        },
        f"{panel}.phrase_post_k_only": {
            **common,
            **primary,
            "channels": ["k"],
        },
        f"{panel}.phrase_post_v_only": {
            **common,
            **primary,
            "channels": ["v"],
        },
        f"{panel}.phrase_late_half_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["late_half_depths"]
            ],
        },
        f"{panel}.phrase_late_quarter_kv": {
            **common,
            **primary,
            "depths": [
                int(value) for value in plan["late_quarter_depths"]
            ],
        },
        f"{panel}.phrase_frozen_rectangle": {
            **common,
            **primary,
            "groups": [
                int(value) for value in plan["frozen_groups"]
            ],
            "depths": [
                int(value) for value in plan["frozen_depths"]
            ],
        },
        f"{panel}.operator_post_kv": {
            **common,
            **primary,
            "site": "operator",
            "pair_limit": int(
                protocol.CONTROL_PAIR_LIMITS[panel]
            ),
        },
        f"{panel}.target_language_post_kv": {
            **common,
            **primary,
            "site": "target_language",
            "pair_limit": int(
                protocol.CONTROL_PAIR_LIMITS[panel]
            ),
        },
    }


def selected_rows(
    rows: list[dict[str, Any]],
    panel: str,
    family: str,
    valid_target_indices: set[int],
    limit: int,
) -> list[dict[str, Any]]:
    eligible = [
        row for row in rows
        if row["panel"] == panel
        and row["pair_family"] == family
        and int(row["target_index"]) in valid_target_indices
    ]
    return engine.evenly_spaced(eligible, limit)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"text_records"}
    }


def rate(
    results: dict[str, dict[str, Any]],
    name: str,
    field: str,
) -> float:
    return float(results[name][field])


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    source_prereg = source.read_json(
        source.OUT_ROOT / "protocol" / "preregistration.json"
    )
    source_summary = source.read_json(
        source.OUT_ROOT / "atlas" / model_name / "summary.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1064 protocol audit failed")
    if source_prereg["protocol_digest"] != prereg[
        "source_phase1063_digest"
    ]:
        raise RuntimeError("Phase1063 source digest drift")
    case_rows = source.read_jsonl(
        source.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    target_rows = source.read_jsonl(
        source.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    source_clean_rows = source.read_jsonl(
        source.OUT_ROOT
        / "atlas"
        / model_name
        / "clean_outputs.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    source_clean = {
        int(row["semantic_case_index"]): [
            int(value) for value in row["generated_token_ids"]
        ]
        for row in source_clean_rows
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
        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        clean_outputs = engine.generate_case_outputs(
            model,
            device,
            case_rows,
            eos_ids=eos_ids,
            batch_size=PAIR_BATCH_SIZE[model_name],
            steps=int(prereg["generation_steps"]),
        )
        parity_count = sum(
            clean_outputs[index] == source_clean[index]
            for index in cases
        )
        parity_rate = parity_count / len(cases)
        if parity_rate < prereg["gates"]["clean_replay_parity_min"]:
            raise RuntimeError(
                f"clean replay drift for {model_name}: {parity_rate}"
            )
        (
            behavior_summary,
            _,
            text_accepted,
            _,
        ) = behavior_scan.qualify(
            tokenizer, cases, clean_outputs, eos_ids
        )
        valid = {}
        valid_index_sets = {}
        for panel in protocol.PANELS:
            for family in protocol.PAIR_FAMILIES:
                key = f"{panel}.{family}"
                rows = text_engine.valid_targets_text(
                    tokenizer,
                    [
                        row for row in target_rows
                        if row["panel"] == panel
                        and row["pair_family"] == family
                    ],
                    text_accepted,
                    clean_outputs,
                    eos_ids,
                )
                valid[key] = rows
                valid_index_sets[key] = {
                    int(row["target_index"]) for row in rows
                }
                expected = set(
                    int(value)
                    for value in source_summary[
                        "valid_target_indices"
                    ][key]
                )
                if valid_index_sets[key] != expected:
                    raise RuntimeError(
                        f"valid target drift for {model_name} {key}"
                    )

        eligible = bool(
            source_summary["primary_behavior_gate_passed"]
        )
        layers = list(get_layers(model))
        width = bridge.projection_width(
            layers[0].self_attn.k_proj
        )
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("KV projection geometry drift")
        head_dim = width // n_kv_heads
        specs = {}
        for panel in protocol.PANELS:
            specs.update(condition_specifications(panel, plan))
        condition_results = {}
        selected_target_indices = {}
        output_records = []
        for name, spec in specs.items():
            panel = name.split(".", 1)[0]
            family = str(spec["pair_family"])
            selected = []
            if eligible:
                selected = selected_rows(
                    target_rows,
                    panel,
                    family,
                    valid_index_sets[f"{panel}.{family}"],
                    int(spec["pair_limit"]),
                )
            selected_target_indices[name] = [
                int(row["target_index"]) for row in selected
            ]
            if not selected or not spec["groups"] or not spec["depths"]:
                condition_results[name] = {
                    "pair_count": 0,
                    "both_match_other_clean_eos_count": 0,
                    "both_match_other_clean_eos_rate": 0.0,
                    "both_match_other_clean_text_count": 0,
                    "both_match_other_clean_text_rate": 0.0,
                }
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
                pair_batch_size=PAIR_BATCH_SIZE[model_name],
                steps=int(prereg["generation_steps"]),
            )
            metrics, records = engine.evaluate_condition(
                selected, clean_outputs, patched, eos_ids
            )
            metrics = text_engine.add_text_metrics(
                tokenizer, metrics, records, eos_ids
            )
            text_records = list(metrics.get("text_records", []))
            condition_results[name] = compact_metrics(metrics)
            for index, record in enumerate(records):
                output_records.append({
                    "schema_version": "phase1064_condition_record.v1",
                    "phase": protocol.PHASE,
                    "model": model_name,
                    "condition": name,
                    "target_index": int(
                        selected[index]["target_index"]
                    ),
                    **record,
                    **text_records[index],
                })
            print(json.dumps({
                "model": model_name,
                "condition": name,
                "pairs": metrics["pair_count"],
                "token_exact": metrics[
                    "both_match_other_clean_eos_rate"
                ],
                "text_exact": metrics[
                    "both_match_other_clean_text_rate"
                ],
            }), flush=True)

        panel_results = {}
        for panel in protocol.PANELS:
            prefix = f"{panel}."

            def text_rate(condition: str) -> float:
                return rate(
                    condition_results,
                    prefix + condition,
                    "both_match_other_clean_text_rate",
                )

            def token_rate(condition: str) -> float:
                return rate(
                    condition_results,
                    prefix + condition,
                    "both_match_other_clean_eos_rate",
                )

            component_text = {
                family: text_rate(f"{family}_post_kv")
                for family in protocol.PAIR_FAMILIES
            }
            component_token = {
                family: token_rate(f"{family}_post_kv")
                for family in protocol.PAIR_FAMILIES
            }
            phase_text = {
                "early": text_rate("phrase_early_kv"),
                "post": text_rate("phrase_post_kv"),
                "all": text_rate("phrase_all_kv"),
            }
            channel_text = {
                "k_only": text_rate("phrase_post_k_only"),
                "v_only": text_rate("phrase_post_v_only"),
                "kv": text_rate("phrase_post_kv"),
            }
            control = max(
                text_rate("operator_post_kv"),
                text_rate("target_language_post_kv"),
            )
            baseline = phase_text["post"]
            support = []
            for condition in (
                "phrase_post_kv",
                "phrase_late_half_kv",
                "phrase_late_quarter_kv",
                "phrase_frozen_rectangle",
            ):
                full_name = prefix + condition
                spec = specs[full_name]
                value = text_rate(condition)
                depth_fraction = (
                    len(spec["depths"])
                    / len(plan["postsource_depths"])
                )
                group_fraction = (
                    len(spec["groups"])
                    / len(plan["all_groups"])
                )
                support.append({
                    "condition": condition,
                    "depth_fraction_of_post": depth_fraction,
                    "group_fraction": group_fraction,
                    "tested_area_fraction": (
                        depth_fraction * group_fraction
                    ),
                    "text_exact_rate": value,
                    "token_exact_rate": token_rate(condition),
                    "retention_vs_full_post": (
                        value / baseline if baseline else 0.0
                    ),
                })
            gates = prereg["gates"]
            transport_gate = (
                eligible
                and component_text["phrase"]
                >= gates["phrase_post_text_rate_min"]
                and component_text["color"]
                >= gates["component_post_text_rate_min"]
                and component_text["noun"]
                >= gates["component_post_text_rate_min"]
                and component_text["phrase"] - control
                >= gates["source_minus_control_text_rate_min"]
            )
            panel_results[panel] = {
                "component_text_rates": component_text,
                "component_token_rates": component_token,
                "phase_text_rates": phase_text,
                "channel_text_rates": channel_text,
                "support_map": support,
                "maximum_role_control_text_rate": control,
                "transport_gate_passed": transport_gate,
            }
        cross_panel_gate = (
            eligible
            and all(
                panel_results[panel]["transport_gate_passed"]
                for panel in protocol.PANELS
            )
        )
        summary = {
            "schema_version": "phase1064_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "source_phase1063_digest": source_prereg[
                "protocol_digest"
            ],
            "precision": precision,
            "placement": placement,
            "clean_replay_parity_count": parity_count,
            "clean_replay_case_count": len(cases),
            "clean_replay_parity_rate": parity_rate,
            "behavior_summary": behavior_summary,
            "source_behavior_gate_passed": eligible,
            "valid_pair_counts": {
                key: len(rows) for key, rows in valid.items()
            },
            "condition_results": condition_results,
            "selected_target_indices": selected_target_indices,
            "panel_results": panel_results,
            "cross_panel_transport_gate_passed": cross_panel_gate,
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg["interpretation_limits"],
        }
        atlas_dir = protocol.OUT_ROOT / "atlas" / model_name
        protocol.write_json(atlas_dir / "summary.json", summary)
        material_write_jsonl(
            atlas_dir / "condition_records.jsonl", output_records
        )
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "eligible": eligible,
            "replay_parity": parity_rate,
            "panel_results": panel_results,
            "cross_panel_gate": cross_panel_gate,
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)


def material_write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    source.write_jsonl(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=list(protocol.MODELS),
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
