#!/usr/bin/env python3
"""Freeze a model-specific natural full-vocabulary behavior protocol."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1050_head_group_natural_validation_protocol as previous


PHASE = 1051
PROTOCOL_REVISION = 1
MODELS = material.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = material.OUT_ROOT
PREVIOUS_ROOT = previous.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1051_natural_behavior_protocol"
)
PARTITION_UNIT_COUNTS_PER_FAMILY = {
    "discovery": 3,
    "confirmation": 10,
    "causal_holdout": 11,
}
VARIANTS = (
    {
        "variant": "raw_natural_cloze",
        "render_mode": "raw",
        "content_mode": "natural_cloze",
        "assistant_prefill": "",
        "continuation_prefix": " ",
    },
    {
        "variant": "raw_explicit",
        "render_mode": "raw",
        "content_mode": "explicit",
        "assistant_prefill": "",
        "continuation_prefix": " ",
    },
    {
        "variant": "raw_table",
        "render_mode": "raw",
        "content_mode": "table",
        "assistant_prefill": "",
        "continuation_prefix": " ",
    },
    {
        "variant": "raw_fewshot",
        "render_mode": "raw",
        "content_mode": "fewshot",
        "assistant_prefill": "",
        "continuation_prefix": " ",
    },
    {
        "variant": "chat_explicit",
        "render_mode": "native_chat",
        "content_mode": "explicit",
        "assistant_prefill": "",
        "continuation_prefix": "",
    },
    {
        "variant": "chat_table",
        "render_mode": "native_chat",
        "content_mode": "table",
        "assistant_prefill": "",
        "continuation_prefix": "",
    },
    {
        "variant": "chat_fewshot",
        "render_mode": "native_chat",
        "content_mode": "fewshot",
        "assistant_prefill": "",
        "continuation_prefix": "",
    },
    {
        "variant": "chat_system",
        "render_mode": "native_chat_system",
        "content_mode": "explicit",
        "assistant_prefill": "",
        "continuation_prefix": "",
    },
    {
        "variant": "chat_category_prefill",
        "render_mode": "native_chat",
        "content_mode": "explicit",
        "assistant_prefill": "Category:",
        "continuation_prefix": " ",
    },
    {
        "variant": "chat_answer_prefill",
        "render_mode": "native_chat",
        "content_mode": "fewshot",
        "assistant_prefill": "Answer:",
        "continuation_prefix": " ",
    },
)
VARIANT_BY_NAME = {row["variant"]: row for row in VARIANTS}
SELECTION_ORDER = tuple(row["variant"] for row in VARIANTS)
GATES = {
    "discovery_finite_rate_min": 0.99,
    "confirmation_finite_rate_min": 0.99,
    "confirmation_arm_accuracy_min": 0.80,
    "confirmation_pair_accuracy_min": 0.60,
    "confirmation_correct_pair_count_min": 100,
    "confirmation_min_family_accuracy_min": 0.50,
    "minimum_passing_models": 2,
}
ROLLOUT_STEPS = 3
ROLLOUT_PAIR_LIMIT = 40


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def role_fragment(
    prompt: str,
    value: str,
    *,
    occurrence: str,
) -> tuple[int, int, str]:
    if occurrence == "first":
        start = prompt.find(value)
    elif occurrence == "last":
        start = prompt.rfind(value)
    else:
        raise ValueError(occurrence)
    if start < 0:
        raise RuntimeError(f"missing role fragment {value!r}")
    return start, start + len(value), value


def prompt_content(
    row: dict[str, Any],
    content_mode: str,
) -> tuple[str, dict[str, tuple[int, int, str]]]:
    a = str(row["concept_a"])
    b = str(row["concept_b"])
    nonce_a, nonce_b = material.NONCE_PAIRS[
        int(row["template_index"])
    ][int(row["surface_index"])]
    q = str(row["query_nonce"])
    q0_slot = str(row["q0_slot"])
    selected_is_a = (
        q0_slot == "a"
        if int(row["query"]) == 0
        else q0_slot != "a"
    )
    if q not in (nonce_a, nonce_b):
        raise RuntimeError("query nonce reconstruction drift")
    labels = ", ".join(material.FAMILIES)
    if content_mode == "natural_cloze":
        content = (
            f"In a temporary dictionary, {nonce_a} means {a} and "
            f"{nonce_b} means {b}. In one word, the concept named by "
            f"{q} is a kind of"
        )
    elif content_mode == "explicit":
        content = (
            "Follow this temporary symbol dictionary.\n"
            f"- {nonce_a} maps to {a}\n"
            f"- {nonce_b} maps to {b}\n"
            f"Requested symbol: {q}\n"
            f"Allowed category labels: {labels}.\n"
            "Return exactly one category label and nothing else.\n"
            "Answer:"
        )
    elif content_mode == "table":
        content = (
            "Category lookup. Use the mapping, not the spelling of the "
            "symbol.\n"
            f"Mapping {nonce_a} means {a}\n"
            f"Mapping {nonce_b} means {b}\n"
            f"Lookup symbol: {q}\n"
            f"Labels: {labels}.\n"
            "Category:"
        )
    elif content_mode == "fewshot":
        content = (
            "Infer the category of the concept assigned to the requested "
            "symbol. Output one label only.\n"
            "Example 1: zoric -> cello; paven -> copper. Query zoric. "
            "Answer: music\n"
            "Example 2: mirel -> cedar; nuvak -> triangle. Query nuvak. "
            "Answer: shape\n"
            f"Task: {nonce_a} -> {a}; {nonce_b} -> {b}. Query {q}.\n"
            f"Allowed labels: {labels}.\n"
            "Answer:"
        )
    else:
        raise KeyError(content_mode)
    fragments = {
        "concept_a": role_fragment(content, a, occurrence="last"),
        "concept_b": role_fragment(content, b, occurrence="last"),
        "query_nonce": role_fragment(content, q, occurrence="last"),
    }
    selected_role = "concept_a" if selected_is_a else "concept_b"
    fragments["selected_concept"] = fragments[selected_role]
    fragments["unselected_concept"] = fragments[
        "concept_b" if selected_is_a else "concept_a"
    ]
    return content, fragments


def render_native(
    tokenizer,
    model_name: str,
    content: str,
    *,
    with_system: bool,
) -> str:
    messages = []
    if with_system:
        messages.append({
            "role": "system",
            "content": (
                "You perform exact symbolic lookup. Follow temporary "
                "bindings and return only the requested category label."
            ),
        })
    messages.append({"role": "user", "content": content})
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_name == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if model_name == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return str(rendered)


def continuation_ids(
    tokenizer,
    rendered: str,
    prefix: str,
    label: str,
) -> list[int]:
    base = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    extended = [
        int(value)
        for value in tokenizer.encode(
            rendered + prefix + label,
            add_special_tokens=False,
        )
    ]
    if extended[:len(base)] != base:
        raise RuntimeError(
            f"continuation retokenized prompt for {label!r}"
        )
    suffix = extended[len(base):]
    if not suffix:
        raise RuntimeError(f"empty continuation for {label!r}")
    return suffix


def model_case(
    tokenizer,
    model_name: str,
    row: dict[str, Any],
    variant: dict[str, str],
) -> dict[str, Any]:
    content, fragments = prompt_content(row, variant["content_mode"])
    if variant["render_mode"] == "raw":
        rendered = content
    else:
        rendered = render_native(
            tokenizer,
            model_name,
            content,
            with_system=variant["render_mode"] == "native_chat_system",
        )
    rendered += variant["assistant_prefill"]
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    located = offset_token_spans(
        tokenizer, rendered, content, fragments
    )
    spans = {
        role: [int(start), int(end)]
        for role, (start, end) in located.items()
    }
    candidates = {
        label: continuation_ids(
            tokenizer,
            rendered,
            variant["continuation_prefix"],
            label,
        )
        for label in material.FAMILIES
    }
    first_ids = [values[0] for values in candidates.values()]
    if len(set(first_ids)) != len(first_ids):
        raise RuntimeError(
            f"{model_name}/{variant['variant']} first-token collision"
        )
    expected = str(row["expected_label"])
    result = {
        "schema_version": "phase1051_model_case.v1",
        "phase": PHASE,
        "model": model_name,
        "variant": variant["variant"],
        "semantic_case_index": int(row["case_index"]),
        "unit_index": int(row["unit_index"]),
        "case_key": str(row["case_key"]),
        "partition": str(row["partition"]),
        "template_index": int(row["template_index"]),
        "surface_index": int(row["surface_index"]),
        "surface_stratum": str(row["surface_stratum"]),
        "binding": int(row["binding"]),
        "query": int(row["query"]),
        "lexical": int(row["lexical"]),
        "expected_index": int(row["expected_index"]),
        "expected_label": expected,
        "raw_content": content,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "role_spans": spans,
        "anchor_spans": {
            **spans,
            "pre_output": [len(input_ids) - 1, len(input_ids) - 1],
        },
        "candidate_token_ids": candidates,
        "candidate_first_token_ids": {
            label: values[0] for label, values in candidates.items()
        },
        "expected_token_ids": candidates[expected],
        "expected_first_token_id": int(candidates[expected][0]),
        "prompt_token_count": len(input_ids),
    }
    return result


def untouched_units() -> list[dict[str, Any]]:
    units = read_jsonl(SOURCE_ROOT / "protocol" / "units.jsonl")
    previous_targets = read_jsonl(
        PREVIOUS_ROOT / "protocol" / "targets.jsonl"
    )
    used = {int(row["unit_index"]) for row in previous_targets}
    untouched = [
        dict(row) for row in units
        if int(row["unit_index"]) not in used
    ]
    if len(used) != 120 or len(untouched) != 240:
        raise RuntimeError(
            f"unit reserve drift: used={len(used)} untouched={len(untouched)}"
        )
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in untouched:
        by_family[str(row["target_family"])].append(row)
    partition_order = tuple(PARTITION_UNIT_COUNTS_PER_FAMILY)
    for family, rows in by_family.items():
        rows.sort(key=lambda item: (
            int(item["template_index"]),
            str(item["surface_stratum"]),
            int(item["unit_key"].split(".")[1][1:]),
            int(item["donor_offset"]),
            int(item["unit_index"]),
        ))
        if len(rows) != 24:
            raise RuntimeError(f"{family} untouched count {len(rows)}")
        cursor = 0
        for partition in partition_order:
            count = PARTITION_UNIT_COUNTS_PER_FAMILY[partition]
            for item in rows[cursor:cursor + count]:
                item["partition"] = partition
            cursor += count
        if cursor != len(rows):
            raise RuntimeError("partition allocation drift")
    return sorted(untouched, key=lambda row: int(row["unit_index"]))


def build_targets_and_cases() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    units = untouched_units()
    source_cases = read_jsonl(
        SOURCE_ROOT / "protocol" / "cases.common.jsonl"
    )
    case_lookup = {
        (
            int(row["unit_index"]),
            int(row["binding"]),
            int(row["query"]),
            int(row["lexical"]),
        ): row
        for row in source_cases
    }
    targets: list[dict[str, Any]] = []
    needed: set[int] = set()
    for unit in units:
        for lexical in (0, 1):
            for query in (0, 1):
                left = case_lookup[
                    (int(unit["unit_index"]), 0, query, lexical)
                ]
                right = case_lookup[
                    (int(unit["unit_index"]), 1, query, lexical)
                ]
                needed.update((int(left["case_index"]), int(right["case_index"])))
                q0_slot = str(unit["q0_slot"])
                selected_slot = (
                    q0_slot
                    if query == 0
                    else ("b" if q0_slot == "a" else "a")
                )
                targets.append({
                    "schema_version": "phase1051_target.v1",
                    "phase": PHASE,
                    "target_index": len(targets),
                    "partition": str(unit["partition"]),
                    "unit_index": int(unit["unit_index"]),
                    "template_index": int(unit["template_index"]),
                    "surface_index": int(
                        unit["unit_key"].split(".")[1][1:]
                    ),
                    "surface_stratum": str(unit["surface_stratum"]),
                    "query": query,
                    "lexical": lexical,
                    "selected_role": f"concept_{selected_slot}",
                    "unselected_role": (
                        "concept_b" if selected_slot == "a" else "concept_a"
                    ),
                    "target_case_index": int(left["case_index"]),
                    "cross_case_index": int(right["case_index"]),
                    "target_expected_label": str(left["expected_label"]),
                    "cross_expected_label": str(right["expected_label"]),
                    "target_family": str(unit["target_family"]),
                    "donor_family": str(unit["donor_family"]),
                })
    cases = [
        dict(row) for row in source_cases
        if int(row["case_index"]) in needed
    ]
    partition_by_unit = {
        int(row["unit_index"]): str(row["partition"]) for row in units
    }
    for row in cases:
        row["partition"] = partition_by_unit[int(row["unit_index"])]
    return targets, cases


def main() -> None:
    targets, semantic_cases = build_targets_and_cases()
    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "targets.jsonl", targets)
    for partition in PARTITION_UNIT_COUNTS_PER_FAMILY:
        write_jsonl(
            protocol_dir / f"{partition}_targets.jsonl",
            [
                row for row in targets
                if row["partition"] == partition
            ],
        )
    write_jsonl(protocol_dir / "semantic_cases.jsonl", semantic_cases)

    model_audits: dict[str, Any] = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        model_cases = []
        for variant in VARIANTS:
            for row in semantic_cases:
                model_cases.append(
                    model_case(tokenizer, model_name, row, variant)
                )
        write_jsonl(
            protocol_dir / f"cases.{model_name}.jsonl",
            model_cases,
        )
        by_key = {
            (row["variant"], int(row["semantic_case_index"])): row
            for row in model_cases
        }
        pair_alignment = {}
        for variant in SELECTION_ORDER:
            checks = []
            first_ids = None
            for target in targets:
                left = by_key[
                    (variant, int(target["target_case_index"]))
                ]
                right = by_key[
                    (variant, int(target["cross_case_index"]))
                ]
                aligned = (
                    len(left["input_ids"]) == len(right["input_ids"])
                    and left["role_spans"]["selected_concept"]
                    == right["role_spans"]["selected_concept"]
                    and left["role_spans"]["unselected_concept"]
                    == right["role_spans"]["unselected_concept"]
                )
                checks.append(aligned)
                observed = tuple(
                    left["candidate_first_token_ids"][label]
                    for label in material.FAMILIES
                )
                if first_ids is None:
                    first_ids = observed
                elif first_ids != observed:
                    raise RuntimeError(
                        f"{model_name}/{variant} candidate id drift"
                    )
            pair_alignment[variant] = {
                "pair_count": len(checks),
                "all_aligned": all(checks),
                "candidate_first_token_ids": list(first_ids or ()),
            }
        model_audits[model_name] = {
            "case_count": len(model_cases),
            "pair_alignment": pair_alignment,
            "all_pairs_aligned": all(
                row["all_aligned"] for row in pair_alignment.values()
            ),
        }

    partition_target_counts = Counter(
        row["partition"] for row in targets
    )
    expected_counts = {
        partition: count * len(material.FAMILIES) * 4
        for partition, count in PARTITION_UNIT_COUNTS_PER_FAMILY.items()
    }
    audit = {
        "schema_version": "phase1051_protocol_audit.v1",
        "phase": PHASE,
        "unused_source_unit_count": len({
            int(row["unit_index"]) for row in targets
        }),
        "partition_target_counts": dict(partition_target_counts),
        "expected_partition_target_counts": expected_counts,
        "partition_unit_overlap": False,
        "model_audits": model_audits,
    }
    unit_sets = {
        partition: {
            int(row["unit_index"]) for row in targets
            if row["partition"] == partition
        }
        for partition in PARTITION_UNIT_COUNTS_PER_FAMILY
    }
    audit["partition_unit_overlap"] = any(
        unit_sets[left].intersection(unit_sets[right])
        for index, left in enumerate(unit_sets)
        for right in list(unit_sets)[index + 1:]
    )
    audit["all_checks_passed"] = (
        dict(partition_target_counts) == expected_counts
        and not audit["partition_unit_overlap"]
        and all(
            row["all_pairs_aligned"] for row in model_audits.values()
        )
    )
    write_json(protocol_dir / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1051 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1051_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "source_phase": material.PHASE,
        "excluded_phase1050_unit_count": 120,
        "reserved_unit_count": 240,
        "partitions": expected_counts,
        "variants": [dict(row) for row in VARIANTS],
        "selection_order": list(SELECTION_ORDER),
        "selection_rule": (
            "Among variants with discovery finite rate >= gate, maximize "
            "(pair accuracy, minimum family accuracy, arm accuracy, "
            "mean expected-vs-best-other margin), then use fixed order."
        ),
        "gates": GATES,
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "automatic_next": {
            "if_two_models_pass": "phase1052_full_vocab_kv_bridge",
            "otherwise": "stop_and_redesign_behavior_protocol",
        },
        "interpretation_limits": [
            "Discovery selects an output protocol, not a mechanism.",
            "Confirmation and causal holdout share no semantic unit.",
            "Full-vocabulary top-1 is not candidate-set accuracy.",
            "Native chat assistant prefills are reported separately.",
            "No behavior result establishes brain homology or optimality.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(protocol_dir / "preregistration.json", payload)
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"targets={len(targets)}"
    )


if __name__ == "__main__":
    main()
