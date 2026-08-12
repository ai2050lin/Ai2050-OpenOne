#!/usr/bin/env python3
"""Run the Phase1063 FP16 lexical behavior atlas."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1040_expanded_mlp_replication_protocol as material
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1058_multitoken_translation_scan as engine
import phase1062_text_equivalence_scan as text_engine
import phase1063_lexical_behavior_atlas_protocol as protocol


BATCH_SIZE = bridge.PAIR_BATCH_SIZE


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def width_bin(value: int) -> str:
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    return "5+"


def qualify(
    tokenizer,
    cases: dict[int, dict[str, Any]],
    clean_outputs: dict[int, list[int]],
    eos_ids: set[int],
) -> tuple[
    dict[str, Any],
    set[int],
    set[int],
    list[dict[str, Any]],
]:
    token_accepted = set()
    text_accepted = set()
    terminated_indices = set()
    panel_totals = Counter()
    panel_token = Counter()
    panel_text = Counter()
    panel_terminated = Counter()
    panel_color_total = Counter()
    panel_color_text = Counter()
    panel_noun_total = Counter()
    panel_noun_text = Counter()
    source_width_total = Counter()
    source_width_text = Counter()
    target_width_total = Counter()
    target_width_text = Counter()
    accepted_labels = Counter()
    records = []
    mismatch_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in cases.items():
        panel = str(row["panel"])
        color = str(row["color_id"])
        noun = str(row["noun_id"])
        generated_tokens = engine.content_tokens(
            clean_outputs[index], eos_ids
        )
        generated_text = text_engine.decode_content(
            tokenizer, clean_outputs[index], eos_ids
        )
        acceptable_tokens = [
            [int(value) for value in values]
            for values in row["acceptable_token_ids"]
        ]
        acceptable_texts = [
            text_engine.normalize_text(str(value))
            for value in row["acceptable_labels"]
        ]
        terminated = engine.terminated(clean_outputs[index], eos_ids)
        token_ok = terminated and generated_tokens in acceptable_tokens
        text_ok = terminated and generated_text in acceptable_texts
        panel_totals[panel] += 1
        panel_color_total[(panel, color)] += 1
        panel_noun_total[(panel, noun)] += 1
        source_width = width_bin(int(row["source_token_width"]))
        target_width = width_bin(
            int(row["canonical_target_token_width"])
        )
        source_width_total[(panel, source_width)] += 1
        target_width_total[(panel, target_width)] += 1
        if terminated:
            terminated_indices.add(index)
            panel_terminated[panel] += 1
        if token_ok:
            token_accepted.add(index)
            panel_token[panel] += 1
        if text_ok:
            text_accepted.add(index)
            panel_text[panel] += 1
            panel_color_text[(panel, color)] += 1
            panel_noun_text[(panel, noun)] += 1
            source_width_text[(panel, source_width)] += 1
            target_width_text[(panel, target_width)] += 1
            accepted_labels[generated_text] += 1
        elif len(mismatch_examples[panel]) < 20:
            mismatch_examples[panel].append({
                "case_key": str(row["case_key"]),
                "generated_text": generated_text,
                "acceptable_texts": acceptable_texts,
                "terminated": terminated,
            })
        records.append({
            "schema_version": "phase1063_clean_output.v1",
            "phase": protocol.PHASE,
            "model": str(row["model"]),
            "semantic_case_index": index,
            "case_key": str(row["case_key"]),
            "panel": panel,
            "color_id": color,
            "noun_id": noun,
            "generated_token_ids": [
                int(value) for value in clean_outputs[index]
            ],
            "generated_text": generated_text,
            "terminated": terminated,
            "token_accepted": token_ok,
            "text_accepted": text_ok,
        })

    def keyed_rates(
        totals: Counter,
        successes: Counter,
    ) -> dict[str, float]:
        return {
            ".".join(str(value) for value in key): rate(
                successes[key], total
            )
            for key, total in sorted(totals.items())
        }

    summary = {
        "case_count": len(cases),
        "terminated_count": len(terminated_indices),
        "token_accepted_count": len(token_accepted),
        "text_accepted_count": len(text_accepted),
        "same_text_different_token_rescue_count": len(
            text_accepted - token_accepted
        ),
        "panel_counts": {
            panel: {
                "total": panel_totals[panel],
                "terminated": panel_terminated[panel],
                "token_accepted": panel_token[panel],
                "text_accepted": panel_text[panel],
                "termination_rate": rate(
                    panel_terminated[panel], panel_totals[panel]
                ),
                "token_accepted_rate": rate(
                    panel_token[panel], panel_totals[panel]
                ),
                "text_accepted_rate": rate(
                    panel_text[panel], panel_totals[panel]
                ),
            }
            for panel in protocol.PANELS
        },
        "panel_color_text_rates": keyed_rates(
            panel_color_total, panel_color_text
        ),
        "panel_noun_text_rates": keyed_rates(
            panel_noun_total, panel_noun_text
        ),
        "source_width_text_rates": keyed_rates(
            source_width_total, source_width_text
        ),
        "target_width_text_rates": keyed_rates(
            target_width_total, target_width_text
        ),
        "accepted_label_counts": dict(accepted_labels),
        "mismatch_examples": dict(mismatch_examples),
    }
    return summary, token_accepted, text_accepted, records


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1063 protocol audit failed")
    case_rows = material_read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    target_rows = material_read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
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
        if not eos_ids:
            raise RuntimeError("no EOS token ids discovered")
        clean_outputs = engine.generate_case_outputs(
            model,
            device,
            case_rows,
            eos_ids=eos_ids,
            batch_size=BATCH_SIZE[model_name],
            steps=int(prereg["generation_steps"]),
        )
        (
            behavior_summary,
            _,
            text_accepted,
            records,
        ) = qualify(tokenizer, cases, clean_outputs, eos_ids)
        valid_pair_counts = {}
        valid_target_indices = {}
        panel_gates = {}
        for panel in protocol.PANELS:
            panel_gates[panel] = True
            for family in protocol.PAIR_FAMILIES:
                valid = text_engine.valid_targets_text(
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
                key = f"{panel}.{family}"
                valid_pair_counts[key] = len(valid)
                valid_target_indices[key] = [
                    int(row["target_index"]) for row in valid
                ]
                panel_gates[panel] = (
                    panel_gates[panel]
                    and len(valid)
                    >= prereg["gates"]["panel_valid_pair_min"][panel]
                )
            panel_gates[panel] = (
                panel_gates[panel]
                and behavior_summary["panel_counts"][panel][
                    "text_accepted"
                ]
                >= prereg["gates"]["panel_accepted_case_min"][panel]
            )
        primary_gate = all(
            panel_gates[panel]
            for panel in prereg["primary_panels"]
        )
        summary = {
            "schema_version": "phase1063_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "behavior_summary": behavior_summary,
            "valid_pair_counts": valid_pair_counts,
            "valid_target_indices": valid_target_indices,
            "panel_behavior_gates": panel_gates,
            "primary_behavior_gate_passed": primary_gate,
            "elapsed_seconds": time.time() - started,
            "interpretation_limits": prereg["interpretation_limits"],
        }
        atlas_dir = (
            protocol.OUT_ROOT / "atlas" / model_name
        )
        protocol.write_json(
            atlas_dir / "summary.json", summary
        )
        protocol.write_jsonl(
            atlas_dir / "clean_outputs.jsonl", records
        )
        print(json.dumps({
            "phase": protocol.PHASE,
            "model": model_name,
            "panels": behavior_summary["panel_counts"],
            "valid_pairs": valid_pair_counts,
            "gates": panel_gates,
            "primary_gate": primary_gate,
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            release_fp16(model)


def material_read_jsonl(path: Path) -> list[dict[str, Any]]:
    return material.read_jsonl(path)


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
