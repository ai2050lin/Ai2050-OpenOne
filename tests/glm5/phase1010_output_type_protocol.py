#!/usr/bin/env python3
"""Freeze the Phase1010 output-type decoupling protocol.

The source semantic worlds are inherited from Phase1009. Each world is
rendered with four disjoint answer vocabularies through an explicit response
map. The experiment compares within-output operation responses across output
types; it never treats a raw prompt difference as a semantic mechanism.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1006_autoregressive_temporal_aggregation_protocol import ANSWER_PREFIX
from phase1009_crossfamily_response_protocol import (
    FAMILIES,
    MODELS,
    NATURAL_STATES,
    PAIR_OPERATIONS,
    ROLE_CLASSES,
    SPLITS,
    TEMPLATES_BY_SPLIT,
    TIME_STAGES,
    canonical,
    digest,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


PHASE = 1010
PROTOCOL_REVISION = 1
OUTPUT_TYPES = ("person", "number", "code", "color")
ANALYSIS_OPERATIONS = PAIR_OPERATIONS + ("X",)
LABEL_SETS = {
    "number": ("one", "two", "three", "four", "five", "six"),
    "code": ("alpha", "beta", "gamma", "delta", "omega", "sigma"),
    "color": ("red", "blue", "green", "yellow", "black", "white"),
}
SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1009_crossfamily_response_atlas"
)
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1010_output_type_decoupling"
)
SOURCE_INSTRUCTION = "\nReply exactly as Answer:"


def labels_for(
    output_type: str,
    candidate_names: list[str],
) -> list[str]:
    if output_type == "person":
        return list(candidate_names)
    labels = list(LABEL_SETS[output_type])
    if len(labels) != len(candidate_names):
        raise RuntimeError(
            f"{output_type}: label/name width mismatch "
            f"{len(labels)} != {len(candidate_names)}"
        )
    return labels


def response_mapping(
    source_case: dict[str, Any],
    output_type: str,
) -> dict[str, str]:
    names = list(source_case["candidate_names"])
    labels = labels_for(output_type, names)
    if output_type == "person":
        return dict(zip(names, names))
    offset = (
        int(source_case["template"])
        + int(source_case["name_pool"])
        + int(source_case["world_index"])
    ) % len(labels)
    rotated = labels[offset:] + labels[:offset]
    return dict(zip(names, rotated))


def answer_specs(
    tokenizer,
    model_name: str,
    labels: list[str],
) -> dict[str, Any]:
    answers = {
        label: [
            int(value)
            for value in tokenizer.encode(
                f"{ANSWER_PREFIX[model_name]}{label} done",
                add_special_tokens=False,
            )
        ]
        for label in labels
    }
    widths = {len(values) for values in answers.values()}
    if len(widths) != 1:
        raise RuntimeError(
            f"{model_name}: answer width drift for labels {labels}: {widths}"
        )
    width = next(iter(widths))
    varying = [
        index
        for index in range(width)
        if len({values[index] for values in answers.values()}) > 1
    ]
    if len(varying) != 1:
        raise RuntimeError(
            f"{model_name}: expected one semantic answer step, got {varying}"
        )
    semantic_step = int(varying[0])
    prefixes = {tuple(values[:semantic_step]) for values in answers.values()}
    suffixes = {
        tuple(values[semantic_step + 1 :]) for values in answers.values()
    }
    if len(prefixes) != 1 or len(suffixes) != 1:
        raise RuntimeError(f"{model_name}: answer framing drift")
    suffix = list(next(iter(suffixes)))
    if len(suffix) != 1:
        raise RuntimeError(f"{model_name}: expected one function token")
    label_ids = {
        label: int(values[semantic_step])
        for label, values in answers.items()
    }
    if len(set(label_ids.values())) != len(label_ids):
        raise RuntimeError(f"{model_name}: output label token collision")
    return {
        "answers": answers,
        "semantic_step": semantic_step,
        "function_step": semantic_step + 1,
        "protocol_prefix_ids": list(next(iter(prefixes))),
        "function_token_id": int(suffix[0]),
        "label_ids": label_ids,
    }


def semantic_prompt(source_case: dict[str, Any]) -> str:
    raw = str(source_case["raw_prompt"])
    if SOURCE_INSTRUCTION not in raw:
        raise RuntimeError(
            f"{source_case['record_id']}: source instruction not found"
        )
    return raw.split(SOURCE_INSTRUCTION, 1)[0].rstrip()


def common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def build_case(
    *,
    tokenizer,
    model_name: str,
    source_case: dict[str, Any],
    output_type: str,
    answer: dict[str, Any],
) -> dict[str, Any]:
    mapping = response_mapping(source_case, output_type)
    names = list(source_case["candidate_names"])
    labels = [mapping[name] for name in names]
    mapping_text = "; ".join(
        f"{name}={mapping[name]}" for name in names
    )
    raw_prompt = (
        semantic_prompt(source_case)
        + f"\nResponse map: {mapping_text}."
        + "\nFirst determine the correct person from the facts and question, "
        "then return that person's mapped label."
        + "\nReply exactly as Answer: LABEL done. Replace LABEL with one "
        "label from the response map. Add nothing else."
    )
    rendered = render_chat(tokenizer, model_name, raw_prompt)
    input_ids = [
        int(value)
        for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    source_ids = [int(value) for value in source_case["input_ids"]]
    shared_prefix = common_prefix_length(source_ids, input_ids)
    role_positions = {}
    for role, position_value in source_case["role_positions"].items():
        if role == "answer_boundary":
            continue
        position = int(position_value)
        if position >= shared_prefix:
            raise RuntimeError(
                f"{source_case['record_id']}/{output_type}: "
                f"role {role} leaves shared semantic prefix "
                f"({position} >= {shared_prefix})"
            )
        if source_ids[position] != input_ids[position]:
            raise RuntimeError(
                f"{source_case['record_id']}/{output_type}: role token drift"
            )
        role_positions[role] = position
    role_positions["answer_boundary"] = len(input_ids) - 1
    if set(role_positions) != set(ROLE_CLASSES[source_case["family"]]):
        raise RuntimeError(
            f"{source_case['record_id']}/{output_type}: role coverage drift"
        )

    gold_entity = str(source_case["gold"])
    foil_entity = str(source_case["foil"])
    gold_label = mapping[gold_entity]
    foil_label = mapping[foil_entity]
    answer_ids = list(answer["answers"][gold_label])
    extended = [
        int(value)
        for value in tokenizer.encode(
            rendered + f"{ANSWER_PREFIX[model_name]}{gold_label} done",
            add_special_tokens=False,
        )
    ]
    if extended != input_ids + answer_ids:
        raise RuntimeError(
            f"{source_case['record_id']}/{output_type}: answer boundary drift"
        )

    source_unit_id = str(source_case["unit_id"])
    unit_id = f"{source_unit_id}.{output_type}"
    state = str(source_case["state"])
    candidate_label_ids = {
        label: int(answer["label_ids"][label]) for label in labels
    }
    return {
        "schema_version": "phase1010_case.v1",
        "phase": PHASE,
        "model": model_name,
        "family": source_case["family"],
        "output_type": output_type,
        "split": source_case["split"],
        "template": int(source_case["template"]),
        "name_pool": int(source_case["name_pool"]),
        "world_index": int(source_case["world_index"]),
        "source_unit_id": source_unit_id,
        "source_record_id": source_case["record_id"],
        "unit_id": unit_id,
        "record_id": f"{unit_id}.{state}",
        "state": state,
        "operation": state,
        "raw_prompt": raw_prompt,
        "rendered_prompt": rendered,
        "input_ids": input_ids,
        "shared_semantic_prefix_length": shared_prefix,
        "role_positions": role_positions,
        "role_classes": ROLE_CLASSES[source_case["family"]],
        "response_mapping": mapping,
        "gold_entity": gold_entity,
        "foil_entity": foil_entity,
        "gold": gold_label,
        "foil": foil_label,
        "candidate_entities": names,
        "candidate_labels": labels,
        "candidate_label_ids": candidate_label_ids,
        # Compatibility alias for Phase1009 measurement helpers.
        "candidate_names": labels,
        "candidate_name_ids": candidate_label_ids,
        "answer_text": f"{ANSWER_PREFIX[model_name]}{gold_label} done",
        "answer_token_ids": answer_ids,
        "semantic_step": int(answer["semantic_step"]),
        "function_step": int(answer["function_step"]),
        "protocol_prefix_ids": list(answer["protocol_prefix_ids"]),
        "function_token_id": int(answer["function_token_id"]),
    }


def build_model(model_name: str) -> dict[str, Any]:
    source_protocol = read_json(SOURCE_ROOT / "protocol" / "protocol.json")
    source_cases = read_jsonl(
        SOURCE_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    source_units = read_jsonl(
        SOURCE_ROOT / "protocol" / model_name / "units.jsonl"
    )
    source_case_by_id = {
        case["record_id"]: case for case in source_cases
    }
    tokenizer = tokenizer_for(model_name)
    labels_by_type = {
        "person": sorted({
            name
            for case in source_cases
            for name in case["candidate_names"]
        }),
        **{
            output_type: list(labels)
            for output_type, labels in LABEL_SETS.items()
        },
    }
    specs = {
        output_type: answer_specs(
            tokenizer,
            model_name,
            labels_by_type[output_type],
        )
        for output_type in OUTPUT_TYPES
    }
    all_ids = {
        output_type: set(specs[output_type]["label_ids"].values())
        for output_type in OUTPUT_TYPES
    }
    for left_index, left in enumerate(OUTPUT_TYPES):
        for right in OUTPUT_TYPES[left_index + 1 :]:
            overlap = all_ids[left] & all_ids[right]
            if overlap:
                raise RuntimeError(
                    f"{model_name}: output token collision "
                    f"{left}/{right}: {sorted(overlap)}"
                )

    cases = []
    units = []
    widths: dict[tuple[str, str, int, str, str], set[int]] = defaultdict(set)
    for source_unit in source_units:
        for output_type in OUTPUT_TYPES:
            unit_id = f"{source_unit['unit_id']}.{output_type}"
            case_ids = {}
            gold_entities = {}
            gold_labels = {}
            for state in NATURAL_STATES:
                source_case = source_case_by_id[
                    source_unit["case_ids"][state]
                ]
                case = build_case(
                    tokenizer=tokenizer,
                    model_name=model_name,
                    source_case=source_case,
                    output_type=output_type,
                    answer=specs[output_type],
                )
                cases.append(case)
                case_ids[state] = case["record_id"]
                gold_entities[state] = case["gold_entity"]
                gold_labels[state] = case["gold"]
                widths[(
                    case["family"],
                    case["split"],
                    int(case["template"]),
                    output_type,
                    state,
                )].add(len(case["input_ids"]))
            for operation in ("F", "Q"):
                if gold_entities[operation] == gold_entities["base"]:
                    raise RuntimeError(
                        f"{unit_id}: {operation} entity failed to change"
                    )
                if gold_labels[operation] == gold_labels["base"]:
                    raise RuntimeError(
                        f"{unit_id}: {operation} label failed to change"
                    )
            for operation in ("FQ", "E", "O", "N", "S"):
                if gold_entities[operation] != gold_entities["base"]:
                    raise RuntimeError(
                        f"{unit_id}: {operation} entity invariant failed"
                    )
                if gold_labels[operation] != gold_labels["base"]:
                    raise RuntimeError(
                        f"{unit_id}: {operation} label invariant failed"
                    )
            operation_pairs = {
                operation: {
                    "base": case_ids["base"],
                    "variant": (
                        case_ids["base"]
                        if operation == "I"
                        else case_ids[operation]
                    ),
                }
                for operation in PAIR_OPERATIONS
            }
            units.append({
                "schema_version": "phase1010_unit.v1",
                "phase": PHASE,
                "model": model_name,
                "family": source_unit["family"],
                "output_type": output_type,
                "split": source_unit["split"],
                "template": int(source_unit["template"]),
                "name_pool": int(source_unit["name_pool"]),
                "world_index": int(source_unit["world_index"]),
                "source_unit_id": source_unit["unit_id"],
                "unit_id": unit_id,
                "case_ids": case_ids,
                "operation_pairs": operation_pairs,
                "gold_entities": gold_entities,
                "gold_labels": gold_labels,
            })
    width_ranges = {
        canonical(key): {
            "minimum": min(values),
            "maximum": max(values),
        }
        for key, values in widths.items()
    }
    output_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(output_root / "cases.jsonl", cases)
    write_jsonl(output_root / "units.jsonl", units)
    family_type_counts = Counter(
        (unit["family"], unit["output_type"]) for unit in units
    )
    summary = {
        "schema_version": "phase1010_protocol_model.v1",
        "phase": PHASE,
        "model": model_name,
        "source_phase1009_digest": source_protocol[
            "preregistration_digest"
        ],
        "case_count": len(cases),
        "unit_count": len(units),
        "pair_count": len(units) * len(PAIR_OPERATIONS),
        "family_output_type_unit_counts": {
            f"{family}:{output_type}": family_type_counts[
                (family, output_type)
            ]
            for family in FAMILIES
            for output_type in OUTPUT_TYPES
        },
        "output_label_ids": {
            output_type: specs[output_type]["label_ids"]
            for output_type in OUTPUT_TYPES
        },
        "output_token_sets_disjoint": True,
        "input_width_ranges": width_ranges,
        "variable_lengths_are_grouped_during_forward": True,
        "raw_prompt_difference_is_semantic_evidence": False,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def build_protocol() -> dict[str, Any]:
    source = read_json(SOURCE_ROOT / "protocol" / "protocol.json")
    model_summaries = [build_model(model_name) for model_name in MODELS]
    payload = {
        "schema_version": "phase1010_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Output-type decoupling atlas without a predeclared shared "
            "interface or transport path"
        ),
        "source_phase1009_digest": source["preregistration_digest"],
        "models_in_required_execution_order": list(MODELS),
        "families": list(FAMILIES),
        "output_types": list(OUTPUT_TYPES),
        "output_labels": {
            key: list(value) for key, value in LABEL_SETS.items()
        },
        "natural_states": list(NATURAL_STATES),
        "pair_operations": list(PAIR_OPERATIONS),
        "analysis_operations": list(ANALYSIS_OPERATIONS),
        "time_stages": list(TIME_STAGES),
        "splits": list(SPLITS),
        "templates_by_split": {
            key: list(value)
            for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "measurement_contract": {
            "primary_comparison": (
                "compare the within-output F/Q/FQ/control response shape "
                "across disjoint output vocabularies"
            ),
            "forbidden_shortcut": (
                "a raw difference between two output-type prompts is never "
                "called a semantic feature"
            ),
            "amplitude_and_direction_separate": True,
            "co_response_is_not_transport": True,
            "formula_status": "measurement definitions only",
        },
        "causal_gate": {
            "selection": (
                "Phase1008-frozen head groups only; Phase1010 data may not "
                "select the tested heads"
            ),
            "output_general_evidence": (
                "at least one non-person output type passes the established "
                "paired selected-over-control local-contribution criterion "
                "in at least two held-out language families"
            ),
            "person_specific_evidence": (
                "person cells pass while all adequately powered non-person "
                "cells fail; this is evidence for type specificity, not a "
                "complete mechanism"
            ),
            "no_arbitrary_effect_ratio_threshold": True,
        },
        "bf16_gate": {
            "required_model": "glm4",
            "required_scope": (
                "person cells and every non-person output type that is "
                "positive in the frozen 8-bit causal screen"
            ),
            "fixed_batch_order": True,
        },
        "upstream_gate": {
            "authorized_only_if": (
                "a non-person output type retains frozen-head causal "
                "contribution after precision audit"
            ),
            "reason": (
                "otherwise an upstream scan would mainly map a person-name "
                "selector and would not resolve the language-rule question"
            ),
        },
        "model_summaries": model_summaries,
    }
    payload["preregistration_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "protocol.json", payload)
    return payload


def main() -> None:
    payload = build_protocol()
    print(json.dumps({
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "protocol_digest": payload["preregistration_digest"],
        "models": [
            {
                "model": row["model"],
                "cases": row["case_count"],
                "units": row["unit_count"],
            }
            for row in payload["model_summaries"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
