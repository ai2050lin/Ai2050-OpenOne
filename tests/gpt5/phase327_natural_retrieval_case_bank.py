#!/usr/bin/env python3
"""Registered independent cases for Phase327 natural retrieval-path tests."""

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


PHASE = "Phase327"
SCHEMA_VERSION = "6.0.0"
OUT = ROOT / "tests/gpt5/result/phase327_natural_retrieval_path"
TEMPLATES = ("template_f", "template_g")
VARIANTS = (
    "correct_object",
    "same_target_object",
    "same_semantic_wrong_target",
    "token_length_wrong_target",
    "unrelated_wrong_target",
)


OBJECTS: dict[str, list[dict[str, str]]] = {
    "color_retrieval": [
        {"subject": "fire engine", "target": "red", "semantic_group": "vehicle"},
        {"subject": "taxi cab", "target": "yellow", "semantic_group": "vehicle"},
        {"subject": "farm tractor", "target": "green", "semantic_group": "vehicle"},
        {"subject": "postal collection box", "target": "blue", "semantic_group": "vehicle"},
        {"subject": "limousine", "target": "black", "semantic_group": "vehicle"},
        {"subject": "ambulance", "target": "white", "semantic_group": "vehicle"},
        {"subject": "ladybug", "target": "red", "semantic_group": "animal"},
        {"subject": "canary", "target": "yellow", "semantic_group": "animal"},
        {"subject": "parrot", "target": "green", "semantic_group": "animal"},
        {"subject": "kingfisher", "target": "blue", "semantic_group": "animal"},
        {"subject": "crow", "target": "black", "semantic_group": "animal"},
        {"subject": "swan", "target": "white", "semantic_group": "animal"},
        {"subject": "cranberry", "target": "red", "semantic_group": "food"},
        {"subject": "sweet corn", "target": "yellow", "semantic_group": "food"},
        {"subject": "broccoli", "target": "green", "semantic_group": "food"},
        {"subject": "mold veined cheese", "target": "blue", "semantic_group": "food"},
        {"subject": "ripe olive", "target": "black", "semantic_group": "food"},
        {"subject": "plain rice", "target": "white", "semantic_group": "food"},
    ],
    "category_retrieval": [
        {"subject": "finch", "target": "bird", "semantic_group": "natural_entity"},
        {"subject": "penguin", "target": "bird", "semantic_group": "natural_entity"},
        {"subject": "woodpecker", "target": "bird", "semantic_group": "natural_entity"},
        {"subject": "pliers", "target": "tool", "semantic_group": "artifact"},
        {"subject": "chisel", "target": "tool", "semantic_group": "artifact"},
        {"subject": "handsaw", "target": "tool", "semantic_group": "artifact"},
        {"subject": "fern", "target": "plant", "semantic_group": "natural_entity"},
        {"subject": "cactus", "target": "plant", "semantic_group": "natural_entity"},
        {"subject": "moss", "target": "plant", "semantic_group": "natural_entity"},
        {"subject": "flute", "target": "instrument", "semantic_group": "artifact"},
        {"subject": "drum", "target": "instrument", "semantic_group": "artifact"},
        {"subject": "harp", "target": "instrument", "semantic_group": "artifact"},
        {"subject": "train", "target": "vehicle", "semantic_group": "artifact"},
        {"subject": "canoe", "target": "vehicle", "semantic_group": "artifact"},
        {"subject": "scooter", "target": "vehicle", "semantic_group": "artifact"},
        {"subject": "horse", "target": "mammal", "semantic_group": "natural_entity"},
        {"subject": "elephant", "target": "mammal", "semantic_group": "natural_entity"},
        {"subject": "otter", "target": "mammal", "semantic_group": "natural_entity"},
    ],
    "habitat_retrieval": [
        {"subject": "tuna", "target": "ocean", "semantic_group": "animal"},
        {"subject": "octopus", "target": "ocean", "semantic_group": "animal"},
        {"subject": "manta ray", "target": "ocean", "semantic_group": "animal"},
        {"subject": "deer", "target": "forest", "semantic_group": "animal"},
        {"subject": "porcupine", "target": "forest", "semantic_group": "animal"},
        {"subject": "squirrel", "target": "forest", "semantic_group": "animal"},
        {"subject": "rattlesnake", "target": "desert", "semantic_group": "animal"},
        {"subject": "fennec", "target": "desert", "semantic_group": "animal"},
        {"subject": "sidewinder", "target": "desert", "semantic_group": "animal"},
        {"subject": "walrus", "target": "arctic", "semantic_group": "animal"},
        {"subject": "narwhal", "target": "arctic", "semantic_group": "animal"},
        {"subject": "musk ox", "target": "arctic", "semantic_group": "animal"},
        {"subject": "tadpole", "target": "pond", "semantic_group": "animal"},
        {"subject": "newt", "target": "pond", "semantic_group": "animal"},
        {"subject": "dragonfly nymph", "target": "pond", "semantic_group": "animal"},
        {"subject": "swiftlet", "target": "cave", "semantic_group": "animal"},
        {"subject": "salamander", "target": "cave", "semantic_group": "animal"},
        {"subject": "horseshoe bat", "target": "cave", "semantic_group": "animal"},
    ],
}


LABELS = {
    "color_retrieval": "color",
    "category_retrieval": "category",
    "habitat_retrieval": "habitat",
}


def prompt_for(subject: str, mechanism: str, template: str) -> tuple[str, str]:
    label = LABELS[mechanism]
    if template == "template_f":
        return (
            f"For the {subject}, the common {label} is",
            f"common {label}",
        )
    if template == "template_g":
        return (
            f"Regarding the {subject}, its typical {label} is",
            f"typical {label}",
        )
    raise KeyError(template)


def choose_controls(items: list[dict[str, str]], index: int) -> dict[str, dict[str, str]]:
    item = items[index]
    candidates = [(offset, row) for offset, row in enumerate(items) if offset != index]

    def first(predicate: Any) -> dict[str, str]:
        return next(row for _offset, row in candidates if predicate(row))

    same_target = first(lambda row: row["target"] == item["target"])
    same_semantic = first(
        lambda row: row["semantic_group"] == item["semantic_group"] and row["target"] != item["target"]
    )
    wrong = [(offset, row) for offset, row in candidates if row["target"] != item["target"]]
    token_length = min(
        wrong,
        key=lambda pair: (abs(len(pair[1]["subject"]) - len(item["subject"])), pair[0]),
    )[1]
    unrelated = max(
        wrong,
        key=lambda pair: (
            pair[1]["semantic_group"] != item["semantic_group"],
            abs(len(pair[1]["subject"]) - len(item["subject"])),
            -pair[0],
        ),
    )[1]
    return {
        "same_target_object": same_target,
        "same_semantic_wrong_target": same_semantic,
        "token_length_wrong_target": token_length,
        "unrelated_wrong_target": unrelated,
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for mechanism, items in OBJECTS.items():
        targets = sorted({row["target"] for row in items})
        for object_index, item in enumerate(items):
            controls = choose_controls(items, object_index)
            base_case_id = f"phase327_{mechanism}_{object_index:03d}"
            split = "registered_primary" if object_index < 12 else "registered_confirmation"
            for template in TEMPLATES:
                prompt, query = prompt_for(item["subject"], mechanism, template)
                variants: dict[str, dict[str, str]] = {"correct_object": item, **controls}
                cases.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "case_id": f"{base_case_id}_{template}",
                    "base_case_id": base_case_id,
                    "family_id": "content_knowledge",
                    "mechanism_id": mechanism,
                    "split": split,
                    "template_id": template,
                    "object_index": object_index,
                    "subject": item["subject"],
                    "semantic_group": item["semantic_group"],
                    "target": item["target"],
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
                    "frozen_selection_phase": "Phase326",
                    "selection_updates_allowed": False,
                })
    return cases


def phase326_subjects() -> dict[str, set[str]]:
    prior = [*phase326.build_cases(), *phase326.build_confirmation_cases()]
    return {
        mechanism: {
            str(case["source_fragments"][0]).lower()
            for case in prior
            if case["mechanism_id"] == mechanism
        }
        for mechanism in OBJECTS
    }


def validate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    prior = phase326_subjects()
    overlaps = []
    for mechanism, items in OBJECTS.items():
        overlaps.extend(
            f"{mechanism}:{row['subject']}"
            for row in items
            if row["subject"].lower() in prior[mechanism]
        )
    variant_rows = [
        (case["case_id"], name, row)
        for case in cases
        for name, row in case["variants"].items()
    ]
    counts = Counter(case["mechanism_id"] for case in cases)
    result = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "mechanism_count": len(OBJECTS),
        "independent_object_count": sum(len(items) for items in OBJECTS.values()),
        "objects_per_mechanism": {key: len(value) for key, value in OBJECTS.items()},
        "template_count": len(TEMPLATES),
        "prompt_case_count": len(cases),
        "prompt_cases_per_mechanism": dict(counts),
        "natural_variant_count": len(variant_rows),
        "variant_types": list(VARIANTS),
        "target_leak_count": sum(not case["target_absent_from_prompt"] for case in cases),
        "phase326_same_mechanism_subject_overlap_count": len(overlaps),
        "phase326_same_mechanism_subject_overlaps": overlaps,
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
    result["valid"] = all(
        result[key] == 0
        for key in (
            "target_leak_count",
            "phase326_same_mechanism_subject_overlap_count",
            "duplicate_case_id_count",
            "variant_contract_error_count",
            "wrong_control_target_error_count",
            "same_target_control_error_count",
            "causal_role_order_error_count",
        )
    ) and len(cases) == 108 and len(variant_rows) == 540
    return result


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="natural_retrieval_path")
    args = parser.parse_args()
    cases = build_cases()
    validation = validate_cases(cases)
    output = OUT / args.round
    write_jsonl(output / "phase327_registered_cases.jsonl", cases)
    write_json(output / "phase327_case_bank_validation.json", validation)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
