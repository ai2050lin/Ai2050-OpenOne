#!/usr/bin/env python3
"""Independent registered cases for the Phase329 competition/mediation atlas."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase326_distributed_carrier_case_bank as phase326  # noqa: E402
import phase327_natural_retrieval_case_bank as phase327  # noqa: E402


PHASE = "Phase329"
SCHEMA_VERSION = "7.0.0"
OUT = ROOT / "tests/gpt5/result/phase329_full_vocabulary_mediation"
TEMPLATES = ("template_h", "template_i")
VARIANTS = (
    "correct_object",
    "same_target_object",
    "same_semantic_wrong_target",
    "unrelated_wrong_target",
)


OBJECTS: dict[str, list[dict[str, str]]] = {
    "color_retrieval": [
        {"subject": "stop sign", "target": "red", "semantic_group": "artifact"},
        {"subject": "cardinal", "target": "red", "semantic_group": "natural_entity"},
        {"subject": "school bus", "target": "yellow", "semantic_group": "artifact"},
        {"subject": "daffodil", "target": "yellow", "semantic_group": "natural_entity"},
        {"subject": "billiard table felt", "target": "green", "semantic_group": "artifact"},
        {"subject": "pea pod", "target": "green", "semantic_group": "natural_entity"},
        {"subject": "cobalt glass", "target": "blue", "semantic_group": "artifact"},
        {"subject": "lapis stone", "target": "blue", "semantic_group": "natural_entity"},
        {"subject": "rubber tire", "target": "black", "semantic_group": "artifact"},
        {"subject": "panther", "target": "black", "semantic_group": "natural_entity"},
        {"subject": "chalk stick", "target": "white", "semantic_group": "artifact"},
        {"subject": "egret", "target": "white", "semantic_group": "natural_entity"},
    ],
    "category_retrieval": [
        {"subject": "heron", "target": "bird", "semantic_group": "natural_entity"},
        {"subject": "albatross", "target": "bird", "semantic_group": "natural_entity"},
        {"subject": "crowbar", "target": "tool", "semantic_group": "artifact"},
        {"subject": "mallet", "target": "tool", "semantic_group": "artifact"},
        {"subject": "orchid", "target": "plant", "semantic_group": "natural_entity"},
        {"subject": "sequoia", "target": "plant", "semantic_group": "natural_entity"},
        {"subject": "clarinet", "target": "instrument", "semantic_group": "artifact"},
        {"subject": "cello", "target": "instrument", "semantic_group": "artifact"},
        {"subject": "tram", "target": "vehicle", "semantic_group": "artifact"},
        {"subject": "kayak", "target": "vehicle", "semantic_group": "artifact"},
        {"subject": "giraffe", "target": "mammal", "semantic_group": "natural_entity"},
        {"subject": "badger", "target": "mammal", "semantic_group": "natural_entity"},
    ],
    "habitat_retrieval": [
        {"subject": "swordfish", "target": "ocean", "semantic_group": "animal"},
        {"subject": "sea turtle", "target": "ocean", "semantic_group": "animal"},
        {"subject": "lynx", "target": "forest", "semantic_group": "animal"},
        {"subject": "chipmunk", "target": "forest", "semantic_group": "animal"},
        {"subject": "jerboa", "target": "desert", "semantic_group": "animal"},
        {"subject": "horned viper", "target": "desert", "semantic_group": "animal"},
        {"subject": "ptarmigan", "target": "arctic", "semantic_group": "animal"},
        {"subject": "beluga", "target": "arctic", "semantic_group": "animal"},
        {"subject": "mosquito larva", "target": "pond", "semantic_group": "animal"},
        {"subject": "freshwater snail", "target": "pond", "semantic_group": "animal"},
        {"subject": "blind shrimp", "target": "cave", "semantic_group": "animal"},
        {"subject": "guano beetle", "target": "cave", "semantic_group": "animal"},
    ],
}


LABELS = {
    "color_retrieval": "color",
    "category_retrieval": "category",
    "habitat_retrieval": "habitat",
}


ALIASES = {
    "red": ["red"],
    "yellow": ["yellow"],
    "green": ["green"],
    "blue": ["blue"],
    "black": ["black"],
    "white": ["white"],
    "bird": ["bird", "avian"],
    "tool": ["tool", "implement"],
    "plant": ["plant"],
    "instrument": ["instrument", "musical instrument"],
    "vehicle": ["vehicle"],
    "mammal": ["mammal"],
    "ocean": ["ocean", "sea"],
    "forest": ["forest", "woodland"],
    "desert": ["desert"],
    "arctic": ["arctic", "tundra"],
    "pond": ["pond", "wetland"],
    "cave": ["cave", "cavern"],
}


BLOCKER_TAXONOMY = (
    "target",
    "target_alias",
    "registered_wrong_answer",
    "punctuation_or_whitespace",
    "continuation_function",
    "protocol_or_format",
    "subject_copy",
    "semantic_content_other",
)


def prompt_for(subject: str, mechanism: str, template: str) -> tuple[str, str]:
    label = LABELS[mechanism]
    if template == "template_h":
        return f"For the {subject}, its ordinary {label} is", f"ordinary {label}"
    if template == "template_i":
        return f"With respect to the {subject}, the usual {label} is", f"usual {label}"
    raise KeyError(template)


def choose_controls(items: list[dict[str, str]], index: int) -> dict[str, dict[str, str]]:
    item = items[index]
    candidates = [row for offset, row in enumerate(items) if offset != index]

    def first(predicate: Any) -> dict[str, str]:
        return next(row for row in candidates if predicate(row))

    same_target = first(lambda row: row["target"] == item["target"])
    same_semantic = first(
        lambda row: row["semantic_group"] == item["semantic_group"]
        and row["target"] != item["target"]
    )
    unrelated_pool = [
        row for row in candidates
        if row["semantic_group"] != item["semantic_group"] and row["target"] != item["target"]
    ]
    unrelated = unrelated_pool[-1] if unrelated_pool else candidates[-1]
    if unrelated["target"] == item["target"]:
        unrelated = first(lambda row: row["target"] != item["target"])
    return {
        "same_target_object": same_target,
        "same_semantic_wrong_target": same_semantic,
        "unrelated_wrong_target": unrelated,
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for mechanism, items in OBJECTS.items():
        targets = sorted({row["target"] for row in items})
        for object_index, item in enumerate(items):
            controls = choose_controls(items, object_index)
            variants = {"correct_object": item, **controls}
            base_case_id = f"phase329_{mechanism}_{object_index:03d}"
            for template in TEMPLATES:
                prompt, query = prompt_for(item["subject"], mechanism, template)
                cases.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "case_id": f"{base_case_id}_{template}",
                    "base_case_id": base_case_id,
                    "family_id": "content_knowledge",
                    "mechanism_id": mechanism,
                    "split": "registered_independent",
                    "template_id": template,
                    "object_index": object_index,
                    "subject": item["subject"],
                    "semantic_group": item["semantic_group"],
                    "target": item["target"],
                    "target_aliases": ALIASES[item["target"]],
                    "distractors": [value for value in targets if value != item["target"]],
                    "prompt": prompt,
                    "source_fragments": [item["subject"]],
                    "query_fragment": query,
                    "variants": {
                        name: {
                            "subject": row["subject"],
                            "natural_target": row["target"],
                            "semantic_group": row["semantic_group"],
                            "prompt": prompt_for(row["subject"], mechanism, template)[0],
                        }
                        for name, row in variants.items()
                    },
                    "target_absent_from_prompt": item["target"].lower() not in prompt.lower(),
                    "residual_selection_frozen_from": "Phase327 registered_primary",
                    "carrier_selection_frozen_from": "Phase326",
                    "selection_updates_allowed": False,
                })
    return cases


def prior_subjects() -> dict[str, set[str]]:
    phase326_cases = [*phase326.build_cases(), *phase326.build_confirmation_cases()]
    result = {mechanism: set() for mechanism in OBJECTS}
    for case in phase326_cases:
        if case["mechanism_id"] in result:
            result[case["mechanism_id"]].add(str(case["source_fragments"][0]).lower())
    for mechanism, items in phase327.OBJECTS.items():
        if mechanism in result:
            result[mechanism].update(row["subject"].lower() for row in items)
    return result


def validate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    prior = prior_subjects()
    overlaps = [
        f"{mechanism}:{row['subject']}"
        for mechanism, items in OBJECTS.items()
        for row in items
        if row["subject"].lower() in prior[mechanism]
    ]
    counts = Counter(case["mechanism_id"] for case in cases)
    variant_rows = [
        (case["case_id"], name, row)
        for case in cases
        for name, row in case["variants"].items()
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "mechanism_count": len(OBJECTS),
        "independent_object_count": sum(len(items) for items in OBJECTS.values()),
        "objects_per_mechanism": {key: len(value) for key, value in OBJECTS.items()},
        "templates": list(TEMPLATES),
        "prompt_case_count": len(cases),
        "prompt_cases_per_mechanism": dict(counts),
        "natural_variant_count": len(variant_rows),
        "variant_types": list(VARIANTS),
        "blocker_taxonomy": list(BLOCKER_TAXONOMY),
        "target_leak_count": sum(not case["target_absent_from_prompt"] for case in cases),
        "phase326_phase327_same_mechanism_subject_overlap_count": len(overlaps),
        "phase326_phase327_same_mechanism_subject_overlaps": overlaps,
        "duplicate_case_id_count": len(cases) - len({case["case_id"] for case in cases}),
        "variant_contract_error_count": sum(set(case["variants"]) != set(VARIANTS) for case in cases),
        "wrong_control_target_error_count": sum(
            row["natural_target"] == case["target"]
            for case in cases
            for name, row in case["variants"].items()
            if name not in {"correct_object", "same_target_object"}
        ),
        "same_target_control_error_count": sum(
            case["variants"]["same_target_object"]["natural_target"] != case["target"]
            for case in cases
        ),
        "causal_role_order_error_count": sum(
            case["prompt"].index(case["source_fragments"][0])
            >= case["prompt"].index(case["query_fragment"])
            for case in cases
        ),
        "selection_frozen": True,
    }
    zero_fields = (
        "target_leak_count",
        "phase326_phase327_same_mechanism_subject_overlap_count",
        "duplicate_case_id_count",
        "variant_contract_error_count",
        "wrong_control_target_error_count",
        "same_target_control_error_count",
        "causal_role_order_error_count",
    )
    result["valid"] = (
        all(result[field] == 0 for field in zero_fields)
        and len(cases) == 72
        and len(variant_rows) == 288
        and all(value == 24 for value in counts.values())
    )
    return result


def protocol() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "primary_goal": "full_vocabulary_blocker_and_tokenwise_query_to_carrier_mediation_atlas",
        "registered_prompt_count_per_model": 72,
        "registered_independent_object_count_per_model": 36,
        "top_k_competitor_count": 50,
        "blocker_definition": "all vocabulary tokens with logit strictly above the target first token",
        "blocker_taxonomy": list(BLOCKER_TAXONOMY),
        "query_conditions": [
            "recipient_baseline",
            "recipient_tokenwise_correct",
            "recipient_pooled_correct",
            "recipient_tokenwise_same_target",
            "recipient_tokenwise_wrong_target",
            "recipient_tokenwise_unrelated",
            "recipient_tokenwise_norm_matched_unrelated",
            "recipient_tokenwise_shuffled",
            "recipient_tokenwise_correct_wrong_layer",
            "recipient_natural_carrier_correct",
            "correct_baseline",
            "correct_carrier_joint_zero",
        ],
        "generation_conditions": [
            "correct_baseline",
            "correct_carrier_joint_zero",
            "recipient_baseline",
            "recipient_tokenwise_correct",
            "recipient_pooled_correct",
            "recipient_natural_carrier_correct",
        ],
        "registered_thresholds": {
            "positive_consistency_min": 0.65,
            "max_mean_js_divergence": 0.05,
            "cross_model_support_min": 2,
            "mechanism_top1_unlock_model_min": 2,
        },
        "single_unit_intervention_gate": (
            "Do not run the Phase288 full single-neuron CUDA intervention unless a mechanism "
            "passes cross-model tokenwise, blocker, carrier-member, and generation criteria."
        ),
        "selection_updates_allowed": False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="full_vocabulary_mediation")
    args = parser.parse_args()
    cases = build_cases()
    validation = validate_cases(cases)
    output = OUT / args.round
    write_jsonl(output / "phase329_registered_cases.jsonl", cases)
    write_json(output / "phase329_case_bank_validation.json", validation)
    write_json(output / "phase329_protocol.json", protocol())
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
