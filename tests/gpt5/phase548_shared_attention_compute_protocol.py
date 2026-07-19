#!/usr/bin/env python3
"""Freeze the Phase548 matched-control compute-edge qualification protocol.

Phase548 is staged. Natural behavior is tested on all three models first. Only
the two Phase546-qualified model topologies may enter the frozen-window
observer gate, and interventions remain forbidden until that gate separates a
functional flip from entity, answer-token, and template controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


PHASE = "Phase548"
SCHEMA_VERSION = "phase548_shared_attention_compute.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("category", "negated_attribute")
SPLITS = ("discovery", "independent_confirmation")
VARIANTS = (
    "base_plus",
    "functional_minus",
    "identity_control",
    "answer_token_control",
    "template_control",
)
PAIR_UNITS_PER_SPLIT = 73
OUT_DIR = ROOT / "tests/gpt5/result/phase548_shared_attention_compute"
CASES_PATH = OUT_DIR / "phase548_registered_cases.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase548_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase548_static_audit.json"
WINDOWS = {
    "qwen3": {"target_layers": [28, 29, 30], "wrong_layers": [12, 13, 14]},
    "glm4": {"target_layers": [34, 35, 36], "wrong_layers": [15, 16, 17]},
    "deepseek7b": {"target_layers": [], "wrong_layers": []},
}
Z = 1.96

CATEGORY_WORDS = (
    "bird", "tool", "plant", "vehicle", "mammal", "instrument", "fruit", "mineral",
    "fabric", "vessel", "device", "flower", "beverage", "furniture", "building", "garment",
)
ATTRIBUTE_WORDS = (
    "amber", "violet", "silver", "crimson", "teal", "ivory", "copper", "indigo",
    "scarlet", "golden", "navy", "maroon", "white", "black", "green", "yellow",
)
SYLLABLE_A = ("ba", "ce", "di", "fo", "ga", "hu", "ji", "ke", "lu", "mi", "no", "pa", "ri")
SYLLABLE_B = ("lan", "mer", "tin", "vor", "sen", "dak", "pel", "rin", "sol", "wen", "yas", "kor", "zen")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1.0 + Z * Z / n
    center = (p + Z * Z / (2.0 * n)) / denominator
    radius = Z * math.sqrt((p * (1.0 - p) + Z * Z / (4.0 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def entity_word(index: int, split: str) -> str:
    shifted = index + (1200 if split == "independent_confirmation" else 400)
    a = SYLLABLE_A[shifted % len(SYLLABLE_A)]
    b = SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
    c = SYLLABLE_A[(shifted // (len(SYLLABLE_A) * len(SYLLABLE_B))) % len(SYLLABLE_A)]
    return (a + b + c).capitalize()


def three_distinct(words: tuple[str, ...], base: int) -> tuple[str, str, str]:
    a = words[base % len(words)]
    b = words[(base * 5 + 3) % len(words)]
    c = words[(base * 7 + 9) % len(words)]
    cursor = 1
    while len({a, b, c}) < 3:
        c = words[(base + 9 + cursor) % len(words)]
        cursor += 1
    return a, b, c


def case_spec(mechanism: str, split: str, pair_index: int, variant: str) -> dict[str, Any]:
    split_offset = 0 if split == "discovery" else PAIR_UNITS_PER_SPLIT
    base = pair_index + split_offset
    entity_a = entity_word(base * 4, split)
    entity_b = entity_word(base * 4 + 1, split)
    entity_c = entity_word(base * 4 + 2, split)
    entity_d = entity_word(base * 4 + 3, split)

    if mechanism == "category":
        answer_a, answer_b, answer_c = three_distinct(CATEGORY_WORDS, base)
        if variant == "base_plus":
            context = (
                f"In the stated local taxonomy, {entity_a} is a {answer_a}, and "
                f"{entity_b} is a {answer_b}. The selected entry is {entity_a}."
            )
            target = answer_a
        elif variant == "functional_minus":
            context = (
                f"In the stated local taxonomy, {entity_a} is a {answer_a}, and "
                f"{entity_b} is a {answer_b}. The selected entry is {entity_b}."
            )
            target = answer_b
        elif variant == "identity_control":
            context = (
                f"In the stated local taxonomy, {entity_c} is a {answer_a}, and "
                f"{entity_d} is a {answer_b}. The selected entry is {entity_c}."
            )
            target = answer_a
        elif variant == "answer_token_control":
            context = (
                f"In the stated local taxonomy, {entity_a} is a {answer_c}, and "
                f"{entity_b} is a {answer_b}. The selected entry is {entity_a}."
            )
            target = answer_c
        elif variant == "template_control":
            context = (
                f"The local taxonomy lists {entity_a} under {answer_a} and {entity_b} under "
                f"{answer_b}; use the entry selected as {entity_a}."
            )
            target = answer_a
        else:
            raise KeyError(variant)
        question = "What category belongs to the selected entry?"
        candidates = [answer_a, answer_b, answer_c]
        operation = "select_explicit_category"
    elif mechanism == "negated_attribute":
        answer_a, answer_b, answer_c = three_distinct(ATTRIBUTE_WORDS, base)
        if variant == "base_plus":
            context = (
                f"In the attribute record for {entity_a}, {answer_a} is marked applicable, "
                f"while {answer_b} is marked inapplicable."
            )
            target = answer_a
        elif variant == "functional_minus":
            context = (
                f"In the attribute record for {entity_a}, {answer_a} is marked inapplicable, "
                f"while {answer_b} is marked applicable."
            )
            target = answer_b
        elif variant == "identity_control":
            context = (
                f"In the attribute record for {entity_c}, {answer_a} is marked applicable, "
                f"while {answer_b} is marked inapplicable."
            )
            target = answer_a
        elif variant == "answer_token_control":
            context = (
                f"In the attribute record for {entity_a}, {answer_c} is marked applicable, "
                f"while {answer_b} is marked inapplicable."
            )
            target = answer_c
        elif variant == "template_control":
            context = (
                f"{entity_a}'s attribute record accepts {answer_a} and explicitly rejects "
                f"{answer_b}."
            )
            target = answer_a
        else:
            raise KeyError(variant)
        question = "Which attribute is applicable?"
        candidates = [answer_a, answer_b, answer_c]
        operation = "select_non_negated_attribute"
    else:
        raise KeyError(mechanism)

    distractors = [value for value in candidates if value != target]
    instruction = "Return only the requested answer word and no explanation."
    raw_prompt = f"Context: {context}\nQuestion: {question}\nInstruction: {instruction}"
    return {
        "context": context,
        "question": question,
        "instruction": instruction,
        "raw_prompt": raw_prompt,
        "source_fragment": f"Context: {context}",
        "query_fragment": f"Question: {question}",
        "target": target,
        "target_aliases": [target],
        "distractors": distractors,
        "all_candidates": candidates,
        "operation": operation,
        "entity_key": f"{entity_a}:{entity_b}:{entity_c}:{entity_d}",
    }


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_chat(tokenizer: Any, model: str, content: str) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], **kwargs)
    if model == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return rendered


def token_edit_distance(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, 1):
        current = [i]
        for j, right_value in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + int(left_value != right_value),
            ))
        previous = current
    return previous[-1]


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    created_at = now()
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        for mechanism in MECHANISMS:
            for split in SPLITS:
                for pair_index in range(PAIR_UNITS_PER_SPLIT):
                    anchor_id = f"phase548_{mechanism}_{split}_{pair_index:03d}"
                    anchor_rows = []
                    for variant in VARIANTS:
                        spec = case_spec(mechanism, split, pair_index, variant)
                        prompt = render_chat(tokenizer, model, spec["raw_prompt"])
                        prompt_ids = [int(value) for value in tokenizer(
                            prompt, add_special_tokens=True,
                        )["input_ids"]]
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": PHASE,
                            "created_at": created_at,
                            "case_id": f"{anchor_id}_{model}_{variant}",
                            "anchor_id": anchor_id,
                            "model": model,
                            "family_id": "content_knowledge",
                            "mechanism_id": mechanism,
                            "split": split,
                            "pair_index": pair_index,
                            "variant": variant,
                            "contrast_kind": variant.replace("_control", ""),
                            "raw_prompt": spec["raw_prompt"],
                            "prompt": prompt,
                            "source_fragment": spec["source_fragment"],
                            "query_fragment": spec["query_fragment"],
                            "target": spec["target"],
                            "target_aliases": spec["target_aliases"],
                            "distractors": spec["distractors"],
                            "all_candidates": spec["all_candidates"],
                            "strict_expected": spec["target"],
                            "strict_kind": "plain",
                            "operation": spec["operation"],
                            "entity_key": spec["entity_key"],
                            "prompt_token_count": len(prompt_ids),
                            "semantic_event_is_natural_answer": True,
                            "arbitrary_label_output": False,
                            "behavior_gate_required": True,
                            "observer_gate_required_before_intervention": True,
                            "sealed": False,
                            "head_channel_neuron_scan_allowed": False,
                        }
                        anchor_rows.append((row, prompt_ids))
                    base_ids = next(ids for row, ids in anchor_rows if row["variant"] == "base_plus")
                    for row, ids in anchor_rows:
                        row["token_edit_distance_from_base"] = token_edit_distance(base_ids, ids)
                        row["token_length_delta_from_base"] = len(ids) - len(base_ids)
                        rows.append(row)
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = len(MODELS) * len(MECHANISMS) * len(SPLITS) * PAIR_UNITS_PER_SPLIT * len(VARIANTS)
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["mechanism_id"], row["split"], row["pair_index"])].append(row)
    target_errors = 0
    edit_errors = 0
    for group in groups.values():
        by_variant = {row["variant"]: row for row in group}
        base = by_variant["base_plus"]["target"]
        if not (
            base == by_variant["identity_control"]["target"]
            == by_variant["template_control"]["target"]
            and len({base, by_variant["functional_minus"]["target"], by_variant["answer_token_control"]["target"]}) == 3
        ):
            target_errors += 1
        if any(
            by_variant[name]["token_edit_distance_from_base"] <= 0
            for name in VARIANTS if name != "base_plus"
        ):
            edit_errors += 1
    discovery_entities = {
        row["entity_key"] for row in rows if row["split"] == "discovery"
    }
    confirmation_entities = {
        row["entity_key"] for row in rows if row["split"] == "independent_confirmation"
    }
    perfect_lcb, _ = wilson(PAIR_UNITS_PER_SPLIT, PAIR_UNITS_PER_SPLIT)
    _, zero_ucb = wilson(0, PAIR_UNITS_PER_SPLIT)
    payload = {
        "schema_version": "phase548_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "expected_case_count": expected,
        "model_case_counts": dict(Counter(row["model"] for row in rows)),
        "mechanism_count": len({row["mechanism_id"] for row in rows}),
        "anchor_group_count": len(groups),
        "rows_per_anchor": sorted({len(group) for group in groups.values()}),
        "variant_set_count": sum({row["variant"] for row in group} == set(VARIANTS) for group in groups.values()),
        "target_relation_error_count": target_errors,
        "nonbase_zero_edit_error_count": edit_errors,
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_prompt_count": len(rows) - len({(row["model"], row["prompt"]) for row in rows}),
        "discovery_confirmation_entity_overlap_count": len(discovery_entities & confirmation_entities),
        "prompt_token_count_range_by_model": {
            model: [
                min(row["prompt_token_count"] for row in rows if row["model"] == model),
                max(row["prompt_token_count"] for row in rows if row["model"] == model),
            ]
            for model in MODELS
        },
        "token_edit_distance_range_by_model_variant": {
            f"{model}:{variant}": [
                min(row["token_edit_distance_from_base"] for row in rows if row["model"] == model and row["variant"] == variant),
                max(row["token_edit_distance_from_base"] for row in rows if row["model"] == model and row["variant"] == variant),
            ]
            for model in MODELS for variant in VARIANTS
        },
        "perfect_anchor_lcb95": perfect_lcb,
        "zero_unrecoverable_ucb95": zero_ucb,
        "sealed_row_count": sum(bool(row["sealed"]) for row in rows),
        "arbitrary_label_row_count": sum(bool(row["arbitrary_label_output"]) for row in rows),
    }
    payload["valid"] = (
        payload["registered_case_count"] == expected
        and set(payload["model_case_counts"].values()) == {expected // len(MODELS)}
        and payload["mechanism_count"] == 2
        and payload["rows_per_anchor"] == [5]
        and payload["variant_set_count"] == len(groups)
        and max(maximum for _minimum, maximum in payload["prompt_token_count_range_by_model"].values()) <= 512
        and all(payload[key] == 0 for key in (
            "target_relation_error_count", "nonbase_zero_edit_error_count", "duplicate_case_id_count",
            "duplicate_prompt_count", "discovery_confirmation_entity_overlap_count", "sealed_row_count",
            "arbitrary_label_row_count",
        ))
        and perfect_lcb >= 0.90
        and zero_ucb <= 0.05
    )
    payload["status"] = "static_pass_no_model_run" if payload["valid"] else "static_fail"
    return payload


def build_protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Matched-control qualification of the shared late-attention observer platform",
        "models_in_required_execution_order": list(MODELS),
        "mechanisms": list(MECHANISMS),
        "splits": list(SPLITS),
        "independent_worlds_per_mechanism_split": PAIR_UNITS_PER_SPLIT,
        "matched_variants": list(VARIANTS),
        "frozen_windows": WINDOWS,
        "behavior_gate": {
            "independent_anchor_all_variants_correct_lcb95_min": 0.90,
            "unrecoverable_anchor_ucb95_max": 0.05,
            "all_splits_required": True,
            "all_variants_required": True,
        },
        "observer_deconfounding_gate": {
            "frozen_platform_aggregation": "three-layer concatenated Euclidean geometry",
            "functional_delta_median_must_exceed": [
                "identity_delta", "answer_token_delta", "template_delta",
            ],
            "paired_dominance_fraction_min": 0.70,
            "one_sided_sign_flip_permutation_p_max": 0.01,
            "permutation_count": 1024,
            "discovery_and_independent_confirmation_both_required": True,
            "selection_updates_allowed": False,
        },
        "intervention_gate": {
            "authorized_only_after_behavior_and_observer_gates": True,
            "target_component": "attention_output",
            "target_role": "current_prompt_end",
            "alphas": [0.5, 1.0],
            "required_controls": [
                "wrong_layer", "wrong_role", "norm_matched_random", "orthogonal",
                "identity", "answer_token", "template", "reverse_sign",
            ],
            "natural_semantic_event_is_primary": True,
            "candidate_margin_is_secondary": True,
            "single_neuron_scan_allowed": False,
        },
        "stopping_rules": {
            "behavior_failure": "no physical collection for that model-mechanism cell",
            "observer_reconfirmation_failure": "no intervention for that cell",
            "control_explains_observer": "close shared functional-platform interpretation",
            "control_matches_intervention": "record distributed perturbation, not compute edge",
            "margin_only_effect": "not a compute edge",
            "one_model_only": "model-specific candidate only",
            "both_mechanisms_fail": "close the current shared late-attention compute route",
            "post_result_threshold_or_window_change_allowed": False,
        },
        "evidence_boundaries": {
            "phase546_observer_is_compute_edge": False,
            "whole_attention_output_is_attention_head_mechanism": False,
            "content_identity_control_required": True,
            "new_sealed_split_read": False,
            "head_channel_neuron_search": False,
            "deepseek_physical_window_registered": False,
            "small_model_error_percent_empirically_measured": False,
        },
        "source_phase546_frozen_events": "tests/gpt5/result/phase546_upstream_physical_prediction/phase546_frozen_upstream_events.jsonl",
        "registered_cases_path": str(CASES_PATH.relative_to(ROOT)),
        "registered_cases_sha256": sha256_file(CASES_PATH),
        "static_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }


def register() -> dict[str, Any]:
    rows = build_rows()
    write_jsonl(CASES_PATH, rows)
    audit = validate(rows)
    write_json(AUDIT_PATH, audit)
    write_json(PROTOCOL_PATH, build_protocol())
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(register(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
