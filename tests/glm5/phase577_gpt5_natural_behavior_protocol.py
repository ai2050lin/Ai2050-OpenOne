#!/usr/bin/env python3
"""Phase577 GPT5 natural-knowledge behavior denominator preregistration.

This stage is CPU-only.  It creates a new evidence denominator after Phase576
was blocked at behavior admission.  It deliberately contains no model loading,
activation coordinates, attention-head choices, or mechanism formulas.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import sys
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase577"
SCHEMA = "phase577_gpt5_natural_behavior_protocol.v2"
OUT_DIR = ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol"
PRIVATE_DIR_RELATIVE = "protocol/private"

SPLITS = ("development", "confirmation", "heldout_novel_entities", "sealed")
OPEN_SPLITS = SPLITS[:-1]
MODELS = ("qwen3", "glm4", "deepseek7b")
RELATIONS = ("fruit_membership", "citrus_membership")
DIRECT_SURFACES = tuple(range(6))
SELECTION_SURFACES = tuple(range(4))
ORDERS = (0, 1)
QUERY_POLARITIES = ("positive", "negative")
MAX_NEW_TOKENS = 24
BEHAVIOR_REPEATS = ("repeat1", "repeat2")
GENERATION_BATCH_SIZE = 8
DESIGN_SEED = "phase577-gpt5-natural-behavior-denominator-v2"

SCRIPT_RELATIVES = (
    "tests/glm5/phase577_gpt5_natural_behavior_protocol.py",
    "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py",
    "tests/glm5/phase577_gpt5_natural_behavior_audit.py",
)
LEGACY_PHASE577_RELATIVE = "tests/glm5/phase577_retrieval_circuit.py"
MODEL_REGISTRY_RELATIVE = "tests/gpt5/model_registry.py"
MODEL_DIR_RELATIVES = {
    "qwen3": "models/hf/qwen3-4b",
    "glm4": "models/hf/glm4-9b-chat-hf",
    "deepseek7b": "models/hf/deepseek-r1-distill-qwen-7b",
}
TOKENIZER_INPUT_NAMES = (
    "config.json", "generation_config.json", "merges.txt", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
)
STAGE_PUBLIC_RELATIVES = (
    "phase577_development_cases.jsonl",
    "phase577_confirmation_cases.jsonl",
    "phase577_heldout_novel_entities_cases.jsonl",
    "phase577_preregistered_protocol.json",
    "phase577_dataset_audit.json",
    "phase577_sealed_commitment.json",
)
STAGE_REQUIRED_RELATIVES = frozenset(STAGE_PUBLIC_RELATIVES) | {
    "phase577_stage_commit.json",
    f"{PRIVATE_DIR_RELATIVE}/phase577_sealed_cases.jsonl",
}
QUALIFICATION_RELATIVES = {
    "phase577_tokenizer_precheck.json",
    "phase577_independent_audit.json",
    "phase577_freeze_commit.json",
}

PILOT_IDENTITIES = {
    "phase576_protocol": (
        "tests/glm5/result/phase576r2_gpt5_fruit_structure/"
        "phase576_frozen_protocol.json"
    ),
    "phase576_behavior_decision": (
        "tests/glm5/result/phase576r2_gpt5_fruit_structure/"
        "phase576_discovery_behavior_decision.json"
    ),
    "phase576_trace_receipt": (
        "tests/glm5/result/phase576r2_gpt5_fruit_structure/"
        "natural_trace/discovery/phase576_discovery_trace_execution_receipt.json"
    ),
}

PRIOR_OPEN_SOURCES = (
    (
        "phase556_open_cases",
        "tests/gpt5/result/phase556_fruit_encoding/phase556_open_cases.jsonl",
        "jsonl",
        ("raw_prompt", "prompt"),
    ),
    (
        "phase557_open_cases",
        "tests/gpt5/result/phase557_fruit_composite/phase557_open_cases.jsonl",
        "jsonl",
        ("raw_prompt", "prompt"),
    ),
    (
        "phase576_open_cases",
        "tests/glm5/result/phase576r2_gpt5_fruit_structure/phase576_open_cases.jsonl",
        "jsonl",
        ("raw_prompt",),
    ),
    (
        "phase990_dataset",
        "tests/glm5/result/phase990_delayed_binding_protocol/dataset.json",
        "json_records",
        ("prompt",),
    ),
)

ATTACHMENT_RELATIVE_EXTERNAL = (
    "C:/Users/Admin/.codex/attachments/"
    "deba539a-1816-44e3-a6ca-25756b8b98a4/pasted-text.txt"
)
PREVIOUS_ATTACHMENT_RELATIVE_EXTERNAL = (
    "C:/Users/Admin/.codex/attachments/"
    "f4fadaf5-bc84-4a2e-8f20-4350bb78e8fb/pasted-text.txt"
)


OBJECTS: dict[str, dict[str, tuple[str, ...]]] = {
    "development": {
        "citrus": (
            "eureka lemon", "persian lime", "grapefruit", "clementine", "pomelo",
            "satsuma",
        ),
        "noncitrus_fruit": (
            "apple", "pear", "peach", "pineapple", "watermelon", "papaya",
        ),
        "nonfruit_food": ("potato", "onion"),
        "nonfruit_animal": ("sparrow", "eagle"),
        "nonfruit_object": ("hammer", "copper"),
    },
    "confirmation": {
        "citrus": (
            "navel orange", "key lime", "blood orange", "yuzu", "kumquat",
            "dancy tangerine",
        ),
        "noncitrus_fruit": (
            "raspberry", "blackberry", "plum", "apricot", "nectarine", "lychee",
        ),
        "nonfruit_food": ("celery", "cabbage"),
        "nonfruit_animal": ("robin", "falcon"),
        "nonfruit_object": ("table", "granite"),
    },
    "heldout_novel_entities": {
        "citrus": (
            "mandarin orange", "meyer lemon", "finger lime", "calamansi",
            "kabosu", "sudachi",
        ),
        "noncitrus_fruit": (
            "gooseberry", "granadilla", "black sapote", "dragon fruit",
            "passion fruit", "jackfruit",
        ),
        "nonfruit_food": ("spinach", "lettuce"),
        "nonfruit_animal": ("owl", "dolphin"),
        "nonfruit_object": ("violin", "silver"),
    },
    "sealed": {
        "citrus": (
            "valencia orange", "rangpur lime", "makrut lime", "tangelo",
            "minneola tangelo", "oro blanco",
        ),
        "noncitrus_fruit": (
            "honeydew melon", "custard apple", "breadfruit", "rambutan",
            "carambola", "white currant",
        ),
        "nonfruit_food": ("cauliflower", "asparagus"),
        "nonfruit_animal": ("rhinoceros", "salamander"),
        "nonfruit_object": ("screwdriver", "telescope"),
    },
}

RELATION_SPECS = {
    "fruit_membership": {
        "label": "fruit",
        "contract_id": "culinary_fruit_membership.v2",
        "positive_classes": ("citrus", "noncitrus_fruit"),
        "negative_classes": (
            "nonfruit_food", "nonfruit_animal", "nonfruit_object",
        ),
    },
    "citrus_membership": {
        "label": "citrus fruit",
        "contract_id": "citrus_membership.v1",
        "positive_classes": ("citrus",),
        "negative_classes": ("noncitrus_fruit",),
    },
}

DIRECT_TEMPLATES = (
    {
        "paraphrase_id": 0,
        "contract": "semantic_label_first",
        "text": (
            "Taxonomy task. Item: {focus}. Question: Is this item a {relation_label}? "
            "Begin the first line with YES or NO; a brief explanation may follow."
        ),
    },
    {
        "paraphrase_id": 0,
        "contract": "exact_short",
        "text": (
            "Taxonomy task. Item: {focus}. Question: Is this item a {relation_label}? "
            "Return exactly YES or NO and nothing else."
        ),
    },
    {
        "paraphrase_id": 1,
        "contract": "semantic_label_first",
        "text": (
            "Classify {focus} using everyday food taxonomy. Does it count as a "
            "{relation_label}? Put YES or NO first; an explanation is optional."
        ),
    },
    {
        "paraphrase_id": 1,
        "contract": "exact_short",
        "text": (
            "Classify {focus} using everyday food taxonomy. Does it count as a "
            "{relation_label}? "
            "Return exactly YES or NO and nothing else."
        ),
    },
    {
        "paraphrase_id": 2,
        "contract": "semantic_label_first",
        "text": (
            "For the named item {focus}, use everyday food taxonomy to decide whether "
            "it belongs to the class "
            "{relation_label}. Start with YES or NO; reasoning may follow."
        ),
    },
    {
        "paraphrase_id": 2,
        "contract": "exact_short",
        "text": (
            "For the named item {focus}, use everyday food taxonomy to decide whether "
            "it belongs to the class "
            "{relation_label}. Return exactly YES or NO and nothing else."
        ),
    },
)

SELECTION_TEMPLATES = (
    {
        "paraphrase_id": 0,
        "contract": "semantic_label_first",
        "text": (
            "Taxonomy comparison. Options: {left} | {right}. Which option is "
            "{polarity_phrase}? Begin with the option text; a brief explanation may follow."
        ),
    },
    {
        "paraphrase_id": 0,
        "contract": "exact_short",
        "text": (
            "Taxonomy comparison. Options: {left} | {right}. Which option is "
            "{polarity_phrase}? Return exactly one option and nothing else."
        ),
    },
    {
        "paraphrase_id": 1,
        "contract": "semantic_label_first",
        "text": (
            "Choose between {left} and {right}. Identify the one that is "
            "{polarity_phrase}. Put the chosen option first; explanation is optional."
        ),
    },
    {
        "paraphrase_id": 1,
        "contract": "exact_short",
        "text": (
            "Choose between {left} and {right}. Identify the one that is "
            "{polarity_phrase}. Return exactly one option and nothing else."
        ),
    },
)

BEHAVIOR_GATE = {
    "primary_metric": "both_repeats_semantic_prefix_correct",
    "direct_unit_minimum_stable_surfaces_of_6": 5,
    "selection_unit_minimum_stable_cases_of_16": 14,
    "selection_unit_minimum_each_query_polarity_of_8": 7,
    "minimum_stable_fruit_direct_units_of_12": 10,
    "minimum_stable_citrus_direct_units_of_12": 10,
    "minimum_stable_fruit_selection_units_of_6": 5,
    "minimum_stable_citrus_selection_units_of_6": 5,
    "fruit_direct_nonfruit_food_units_required_of_2": 2,
    "fruit_selection_nonfruit_food_pairs_required_of_2": 2,
    "minimum_passing_analysis_units_of_36": 30,
    "case_level_semantic_stable_micro_rate_secondary_floor": 0.85,
    "answer_prefix_resolution_token_budget": 8,
    "exact_short_compliance_is_separate_diagnostic": True,
    "eos_before_generation_budget_is_separate_diagnostic": True,
    "surface_order_robustness_is_reported_separately": True,
    "semantic_gate_must_not_be_recomputed_with_phase576_aliases": True,
    "semantic_correctness_required_in_both_repeats": True,
    "full_generated_identity_is_separate_diagnostic": True,
    "full_generated_identity_required_for_semantic_gate": False,
    "analysis_units_are_aggregation_units_not_independent_samples": True,
    "case_micro_rate_is_not_an_equal_object_or_unit_average": True,
    "per_object_class_and_nonfruit_subclass_results_required": True,
    "statistical_independence_claimed": False,
    "model_failure_is_behavior_blocked_not_mechanism_absence": True,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_to: Path | None = ROOT) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    label = (
        str(resolved.relative_to(relative_to.resolve(strict=True))).replace("\\", "/")
        if relative_to is not None else str(resolved).replace("\\", "/")
    )
    return {
        "path": label,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "is_symlink": path.is_symlink(),
    }


def confined_path(base: Path, relative: Any, *, strict: bool = True) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact identity path must be a non-empty string")
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != relative
    ):
        raise ValueError(f"unsafe artifact identity path: {relative!r}")
    base_resolved = base.resolve(strict=True)
    candidate = base.joinpath(*pure.parts)
    resolved = candidate.resolve(strict=strict)
    resolved.relative_to(base_resolved)
    cursor = candidate
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError(f"symlink is forbidden in artifact path: {relative!r}")
        cursor = cursor.parent
    return candidate


def result_file_relatives() -> set[str]:
    if not OUT_DIR.is_dir() or OUT_DIR.is_symlink():
        raise RuntimeError("Phase577 result root is missing or is a symlink")
    relatives: set[str] = set()
    allowed_directories = {"protocol", "protocol/private"}
    for path in OUT_DIR.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Phase577 result symlink is forbidden: {path}")
        if path.is_file():
            relatives.add(str(path.relative_to(OUT_DIR)).replace("\\", "/"))
        elif path.is_dir():
            relative = str(path.relative_to(OUT_DIR)).replace("\\", "/")
            if relative not in allowed_directories:
                raise RuntimeError(f"unexpected Phase577 result directory: {path}")
        else:
            raise RuntimeError(f"unsupported Phase577 filesystem entry: {path}")
    return relatives


def normalize_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(row) + b"\n" for row in rows)


def atomic_write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    if path.exists():
        raise RuntimeError(f"refusing to overwrite Phase577 artifact: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def remove_failed_staging(path: Path) -> None:
    resolved = path.resolve(strict=True)
    expected_parent = OUT_DIR.parent.resolve(strict=True)
    if (
        resolved.parent != expected_parent
        or not resolved.name.startswith(f".{OUT_DIR.name}.staging-")
        or resolved == expected_parent
    ):
        raise RuntimeError(f"refusing unsafe Phase577 staging cleanup: {resolved}")
    shutil.rmtree(resolved)


def raw_spans(prompt: str, values: dict[str, str | None]) -> dict[str, Any]:
    spans: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            spans[key] = None
            continue
        starts = [match.start() for match in re.finditer(re.escape(value), prompt)]
        if not starts:
            raise RuntimeError(f"role fragment missing: {key}={value!r}")
        start = starts[-1] if key == "query_anchor" else starts[0]
        spans[key] = {"start": start, "end": start + len(value), "text": value}
    return spans


def class_membership(object_class: str, relation: str) -> bool:
    return object_class in RELATION_SPECS[relation]["positive_classes"]


def seeded_rotation(values: Iterable[str], split: str, relation: str, role: str) -> list[str]:
    bank = list(values)
    digest = hashlib.sha256(
        f"{DESIGN_SEED}|{split}|{relation}|{role}".encode("utf-8")
    ).digest()
    offset = int.from_bytes(digest[:4], "big") % len(bank)
    return bank[offset:] + bank[:offset]


def length_balanced_pairs(
    positives: Iterable[str], negatives: Iterable[str], split: str, relation: str
) -> list[tuple[str, str]]:
    positive_bank = seeded_rotation(positives, split, relation, "positive_objects")
    negative_bank = list(negatives)
    if len(positive_bank) != 6 or len(negative_bank) != 6:
        raise RuntimeError("Phase577 paired interfaces require six-by-six object banks")
    candidates = []
    for permutation in itertools.permutations(negative_bank):
        positive_longer = sum(
            len(positive) > len(negative)
            for positive, negative in zip(positive_bank, permutation)
        )
        negative_longer = sum(
            len(negative) > len(positive)
            for positive, negative in zip(positive_bank, permutation)
        )
        ties = 6 - positive_longer - negative_longer
        tie_break = hashlib.sha256(
            (
                f"{DESIGN_SEED}|{split}|{relation}|length_pairing|"
                + "|".join(permutation)
            ).encode("utf-8")
        ).hexdigest()
        candidates.append((
            (abs(positive_longer - negative_longer), -ties, tie_break),
            permutation,
        ))
    _, best = min(candidates, key=lambda item: item[0])
    return list(zip(positive_bank, best))


def split_records(split: str) -> list[dict[str, Any]]:
    groups = OBJECTS[split]
    sealed = split == "sealed"
    nonfruit_objects = (
        list(groups["nonfruit_food"])
        + list(groups["nonfruit_animal"])
        + list(groups["nonfruit_object"])
    )
    label_to_class = {
        label: object_class
        for object_class, labels in groups.items()
        for label in labels
    }
    rows: list[dict[str, Any]] = []

    direct_objects = {
        "fruit_membership": (
            list(groups["citrus"][:3])
            + list(groups["noncitrus_fruit"][:3])
            + nonfruit_objects
        ),
        "citrus_membership": (
            list(groups["citrus"]) + list(groups["noncitrus_fruit"])
        ),
    }
    for relation in RELATIONS:
        relation_spec = RELATION_SPECS[relation]
        for focus in direct_objects[relation]:
            object_class = label_to_class[focus]
            positive = class_membership(object_class, relation)
            target = "yes" if positive else "no"
            unit_id = f"phase577_{split}_direct_{relation}_{focus.replace(' ', '_')}"
            for surface_id, template in enumerate(DIRECT_TEMPLATES):
                prompt = template["text"].format(
                    focus=focus, relation_label=relation_spec["label"]
                )
                case_id = f"{unit_id}_s{surface_id}"
                rows.append({
                    "schema_version": "phase577_gpt5_natural_behavior_case.v2",
                    "phase_id": PHASE,
                    "case_id": case_id,
                    "split": split,
                    "sealed": sealed,
                    "interface": "direct",
                    "relation": relation,
                    "relation_contract_id": relation_spec["contract_id"],
                    "output_contract": template["contract"],
                    "surface_id": surface_id,
                    "paraphrase_id": template["paraphrase_id"],
                    "order": None,
                    "query_polarity": "affirmative",
                    "target_truth_polarity": positive and "positive" or "negative",
                    "analysis_unit_id": unit_id,
                    "focus_object": focus,
                    "focus_object_class": object_class,
                    "comparison_object": None,
                    "comparison_object_class": None,
                    "positive_object": focus if positive else None,
                    "negative_object": None if positive else focus,
                    "left_option": None,
                    "right_option": None,
                    "target": target,
                    "foil": "no" if target == "yes" else "yes",
                    "candidate_groups": {"yes": ["yes"], "no": ["no"]},
                    "raw_prompt": prompt,
                    "normalized_prompt_sha256": hashlib.sha256(
                        normalize_prompt(prompt).encode("utf-8")
                    ).hexdigest(),
                    "raw_role_char_spans": raw_spans(prompt, {
                        "focus": focus,
                        "comparison": None,
                        "query_anchor": relation_spec["label"],
                    }),
                    "candidate_layer": None,
                    "candidate_head": None,
                    "candidate_neuron": None,
                    "candidate_direction": None,
                    "candidate_mechanism_formula": None,
                })

    fruit_positives = list(groups["citrus"][3:]) + list(
        groups["noncitrus_fruit"][3:]
    )
    fruit_pairs = length_balanced_pairs(
        fruit_positives, nonfruit_objects, split, "fruit_membership"
    )
    citrus_pairs = length_balanced_pairs(
        groups["citrus"], groups["noncitrus_fruit"], split,
        "citrus_membership"
    )
    selection_pairs = {
        "fruit_membership": fruit_pairs,
        "citrus_membership": citrus_pairs,
    }
    for relation in RELATIONS:
        relation_spec = RELATION_SPECS[relation]
        for pair_index, (positive_object, negative_object) in enumerate(
            selection_pairs[relation]
        ):
            unit_id = f"phase577_{split}_selection_{relation}_pair{pair_index:02d}"
            for surface_id, template in enumerate(SELECTION_TEMPLATES):
                for order in ORDERS:
                    left, right = (
                        (positive_object, negative_object)
                        if order == 0 else (negative_object, positive_object)
                    )
                    for polarity in QUERY_POLARITIES:
                        target = positive_object if polarity == "positive" else negative_object
                        foil = negative_object if polarity == "positive" else positive_object
                        phrase = (
                            f"a {relation_spec['label']}"
                            if polarity == "positive"
                            else f"not a {relation_spec['label']}"
                        )
                        prompt = template["text"].format(
                            left=left, right=right, polarity_phrase=phrase
                        )
                        case_id = (
                            f"{unit_id}_s{surface_id}_o{order}_q{polarity[0]}"
                        )
                        rows.append({
                            "schema_version": "phase577_gpt5_natural_behavior_case.v2",
                            "phase_id": PHASE,
                            "case_id": case_id,
                            "split": split,
                            "sealed": sealed,
                            "interface": "selection",
                            "relation": relation,
                            "relation_contract_id": relation_spec["contract_id"],
                            "output_contract": template["contract"],
                            "surface_id": surface_id,
                            "paraphrase_id": template["paraphrase_id"],
                            "order": order,
                            "query_polarity": polarity,
                            "target_truth_polarity": polarity,
                            "analysis_unit_id": unit_id,
                            "focus_object": target,
                            "focus_object_class": label_to_class[target],
                            "comparison_object": foil,
                            "comparison_object_class": label_to_class[foil],
                            "positive_object": positive_object,
                            "negative_object": negative_object,
                            "left_option": left,
                            "right_option": right,
                            "target": target,
                            "foil": foil,
                            "candidate_groups": {
                                positive_object: [positive_object],
                                negative_object: [negative_object],
                            },
                            "raw_prompt": prompt,
                            "normalized_prompt_sha256": hashlib.sha256(
                                normalize_prompt(prompt).encode("utf-8")
                            ).hexdigest(),
                            "raw_role_char_spans": raw_spans(prompt, {
                                "focus": target,
                                "comparison": foil,
                                "query_anchor": phrase,
                            }),
                            "candidate_layer": None,
                            "candidate_head": None,
                            "candidate_neuron": None,
                            "candidate_direction": None,
                            "candidate_mechanism_formula": None,
                        })
    return sorted(rows, key=lambda row: row["case_id"])


def validate_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    if len(rows) != 336:
        raise RuntimeError(f"{split}: expected 336 cases, observed {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"{split}: duplicate case IDs")
    if len({row["normalized_prompt_sha256"] for row in rows}) != len(rows):
        raise RuntimeError(f"{split}: duplicate normalized prompts")
    expected_keys = set(rows[0])
    if any(set(row) != expected_keys for row in rows):
        raise RuntimeError(f"{split}: non-exact case schema")
    for row in rows:
        if (
            row["phase_id"] != PHASE
            or row["split"] != split
            or row["sealed"] is (split != "sealed")
            or row["normalized_prompt_sha256"]
            != hashlib.sha256(
                normalize_prompt(row["raw_prompt"]).encode("utf-8")
            ).hexdigest()
            or any(row[key] is not None for key in (
                "candidate_layer", "candidate_head", "candidate_neuron",
                "candidate_direction", "candidate_mechanism_formula",
            ))
        ):
            raise RuntimeError(f"{split}/{row['case_id']}: case identity drift")

    by_interface_relation = Counter(
        (row["interface"], row["relation"]) for row in rows
    )
    expected_counts = {
        ("direct", "fruit_membership"): 72,
        ("direct", "citrus_membership"): 72,
        ("selection", "fruit_membership"): 96,
        ("selection", "citrus_membership"): 96,
    }
    if dict(by_interface_relation) != expected_counts:
        raise RuntimeError(f"{split}: stratum denominator drift")

    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[row["analysis_unit_id"]].append(row)
    if len(by_unit) != 36:
        raise RuntimeError(f"{split}: expected 36 analysis units")
    family_units = Counter()
    for unit_rows in by_unit.values():
        interface = unit_rows[0]["interface"]
        relation = unit_rows[0]["relation"]
        family_units[(interface, relation)] += 1
        if interface == "direct":
            if len(unit_rows) != 6 or {row["surface_id"] for row in unit_rows} != set(
                DIRECT_SURFACES
            ):
                raise RuntimeError(f"{split}: direct unit grid drift")
            if Counter(
                (row["paraphrase_id"], row["output_contract"])
                for row in unit_rows
            ) != {
                (paraphrase, contract): 1
                for paraphrase in range(3)
                for contract in ("semantic_label_first", "exact_short")
            }:
                raise RuntimeError(f"{split}: direct contract/paraphrase grid drift")
        else:
            expected_grid = {
                (surface, order, polarity)
                for surface in SELECTION_SURFACES
                for order in ORDERS
                for polarity in QUERY_POLARITIES
            }
            actual_grid = {
                (row["surface_id"], row["order"], row["query_polarity"])
                for row in unit_rows
            }
            if len(unit_rows) != 16 or actual_grid != expected_grid:
                raise RuntimeError(f"{split}: selection unit grid drift")
            if Counter(
                (row["paraphrase_id"], row["output_contract"], row["order"],
                 row["query_polarity"])
                for row in unit_rows
            ) != {
                (paraphrase, contract, order, polarity): 1
                for paraphrase in range(2)
                for contract in ("semantic_label_first", "exact_short")
                for order in ORDERS
                for polarity in QUERY_POLARITIES
            }:
                raise RuntimeError(f"{split}: selection factorial grid drift")
    if dict(family_units) != {
        ("direct", "fruit_membership"): 12,
        ("direct", "citrus_membership"): 12,
        ("selection", "fruit_membership"): 6,
        ("selection", "citrus_membership"): 6,
    }:
        raise RuntimeError(f"{split}: unit family drift")
    groups = OBJECTS[split]
    expected_pair_inputs = {
        "fruit_membership": (
            list(groups["citrus"][3:]) + list(groups["noncitrus_fruit"][3:]),
            list(groups["nonfruit_food"])
            + list(groups["nonfruit_animal"])
            + list(groups["nonfruit_object"]),
        ),
        "citrus_membership": (
            list(groups["citrus"]), list(groups["noncitrus_fruit"]),
        ),
    }
    for relation, (positives, negatives) in expected_pair_inputs.items():
        expected_pairs = set(length_balanced_pairs(
            positives, negatives, split, relation
        ))
        representatives = {
            unit_id: unit_rows[0]
            for unit_id, unit_rows in by_unit.items()
            if unit_rows[0]["interface"] == "selection"
            and unit_rows[0]["relation"] == relation
        }
        observed_pairs = {
            (row["positive_object"], row["negative_object"])
            for row in representatives.values()
        }
        if (
            len(representatives) != 6
            or len({row["positive_object"] for row in representatives.values()}) != 6
            or len({row["negative_object"] for row in representatives.values()}) != 6
            or observed_pairs != expected_pairs
        ):
            raise RuntimeError(f"{split}/{relation}: selection object pairing drift")

    balance: dict[str, Any] = {}
    for relation in RELATIONS:
        direct = [
            row for row in rows
            if row["interface"] == "direct" and row["relation"] == relation
        ]
        selection = [
            row for row in rows
            if row["interface"] == "selection" and row["relation"] == relation
        ]
        direct_target = Counter(row["target"] for row in direct)
        selection_polarity = Counter(row["query_polarity"] for row in selection)
        target_position = Counter(
            "left" if row["target"] == row["left_option"] else "right"
            for row in selection
        )
        output_contract = Counter(row["output_contract"] for row in direct + selection)
        if direct_target != {"yes": 36, "no": 36}:
            raise RuntimeError(f"{split}/{relation}: direct target imbalance")
        if selection_polarity != {"positive": 48, "negative": 48}:
            raise RuntimeError(f"{split}/{relation}: query-polarity imbalance")
        if target_position != {"left": 48, "right": 48}:
            raise RuntimeError(f"{split}/{relation}: target-position imbalance")
        if output_contract != {
            "semantic_label_first": 84, "exact_short": 84,
        }:
            raise RuntimeError(f"{split}/{relation}: output-contract imbalance")
        direct_class_counts = Counter(row["focus_object_class"] for row in direct)
        selection_negative_class_counts = Counter(
            row["negative_object"] and row["comparison_object_class"]
            if row["comparison_object"] == row["negative_object"]
            else row["focus_object_class"]
            for row in selection
        )
        expected_direct_classes = (
            {
                "citrus": 18,
                "noncitrus_fruit": 18,
                "nonfruit_food": 12,
                "nonfruit_animal": 12,
                "nonfruit_object": 12,
            }
            if relation == "fruit_membership"
            else {"citrus": 36, "noncitrus_fruit": 36}
        )
        expected_selection_negative_classes = (
            {
                "nonfruit_food": 32,
                "nonfruit_animal": 32,
                "nonfruit_object": 32,
            }
            if relation == "fruit_membership"
            else {"noncitrus_fruit": 96}
        )
        if dict(direct_class_counts) != expected_direct_classes:
            raise RuntimeError(f"{split}/{relation}: direct class balance drift")
        if dict(selection_negative_class_counts) != expected_selection_negative_classes:
            raise RuntimeError(f"{split}/{relation}: selection negative-class drift")
        balance[relation] = {
            "direct_target_counts": dict(sorted(direct_target.items())),
            "selection_query_polarity_counts": dict(sorted(selection_polarity.items())),
            "selection_target_position_counts": dict(sorted(target_position.items())),
            "output_contract_counts": dict(sorted(output_contract.items())),
            "direct_object_class_counts": dict(sorted(direct_class_counts.items())),
            "selection_negative_object_class_counts": dict(
                sorted(selection_negative_class_counts.items())
            ),
        }
    return {
        "case_count": len(rows),
        "analysis_unit_count": len(by_unit),
        "stratum_case_counts": {
            "|".join(key): value
            for key, value in sorted(by_interface_relation.items())
        },
        "family_unit_counts": {
            "|".join(key): value for key, value in sorted(family_units.items())
        },
        "balance": balance,
    }


def iter_prior_prompts(
    path: Path, kind: str, fields: tuple[str, ...]
) -> Iterable[str]:
    if kind == "jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in fields:
                    value = row.get(field)
                    if isinstance(value, str) and value:
                        yield value
                        break
    elif kind == "json_records":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            for field in fields:
                value = row.get(field)
                if isinstance(value, str) and value:
                    yield value
                    break
    else:
        raise RuntimeError(f"unsupported prior prompt source kind: {kind}")


def prior_prompt_audit(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    new_prompts = {normalize_prompt(row["raw_prompt"]) for row in all_rows}
    sources = []
    total_overlap: set[str] = set()
    for name, relative, kind, fields in PRIOR_OPEN_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing prior open source: {relative}")
        prior = {normalize_prompt(value) for value in iter_prior_prompts(path, kind, fields)}
        overlap = new_prompts & prior
        total_overlap.update(overlap)
        sources.append({
            "name": name,
            "identity": file_identity(path),
            "normalized_prompt_count": len(prior),
            "normalized_overlap_count": len(overlap),
        })
    if total_overlap:
        raise RuntimeError("Phase577 normalized prompts overlap prior open prompts")
    return {
        "sources": sources,
        "source_count": len(sources),
        "new_normalized_prompt_count": len(new_prompts),
        "total_normalized_overlap_count": 0,
    }


def phase576_open_object_audit(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = ROOT / (
        "tests/glm5/result/phase576r2_gpt5_fruit_structure/phase576_open_cases.jsonl"
    )
    old_objects: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("focus_object_label", "comparison_object_label"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    old_objects.add(value.casefold())
    new_objects = {
        value.casefold()
        for row in all_rows
        for value in (row.get("focus_object"), row.get("comparison_object"))
        if isinstance(value, str) and value
    }
    overlap = sorted(new_objects & old_objects)
    if overlap:
        raise RuntimeError(f"Phase577 objects overlap Phase576 open objects: {overlap}")
    old_atoms = {
        atom for label in old_objects for atom in re.findall(r"[a-z0-9]+", label)
    }
    new_atoms = {
        atom for label in new_objects for atom in re.findall(r"[a-z0-9]+", label)
    }
    lexical_atom_overlap = sorted(old_atoms & new_atoms)
    return {
        "phase576_open_identity": file_identity(path),
        "phase576_open_object_count": len(old_objects),
        "phase577_object_count": len(new_objects),
        "exact_casefold_object_overlap": overlap,
        "overlap_count": 0,
        "whole_token_lexical_atom_overlap": lexical_atom_overlap,
        "lexical_atom_overlap_count": len(lexical_atom_overlap),
        "exact_entity_disjoint_does_not_mean_lexically_disjoint": True,
    }


def deterministic_shortcut_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    markers = ("orange", "lemon", "lime", "grapefruit")

    def marked(label: str) -> bool:
        words = set(re.findall(r"[a-z0-9]+", label.casefold()))
        return any(marker in words for marker in markers)

    citrus_direct_units = {
        row["analysis_unit_id"]: row for row in rows
        if row["interface"] == "direct"
        and row["relation"] == "citrus_membership"
    }
    citrus_marker_direct_correct_units = sum(
        ("yes" if marked(row["focus_object"]) else "no") == row["target"]
        for row in citrus_direct_units.values()
    )
    fruit_direct_units = {
        row["analysis_unit_id"]: row for row in rows
        if row["interface"] == "direct"
        and row["relation"] == "fruit_membership"
    }
    fruit_foodness_correct_units = sum(
        (
            "no" if row["focus_object_class"] in {
                "nonfruit_animal", "nonfruit_object",
            } else "yes"
        ) == row["target"]
        for row in fruit_direct_units.values()
    )
    fruit_foodness_nonfruit_food_correct_units = sum(
        "yes" == row["target"]
        for row in fruit_direct_units.values()
        if row["focus_object_class"] == "nonfruit_food"
    )
    length_reports = {}
    marker_selection_correct_units = 0
    marker_selection_resolved_units = 0
    for relation in RELATIONS:
        units = {
            row["analysis_unit_id"]: row for row in rows
            if row["interface"] == "selection" and row["relation"] == relation
        }
        positive_longer = sum(
            len(row["positive_object"]) > len(row["negative_object"])
            for row in units.values()
        )
        negative_longer = sum(
            len(row["negative_object"]) > len(row["positive_object"])
            for row in units.values()
        )
        ties = len(units) - positive_longer - negative_longer
        length_reports[relation] = {
            "analysis_unit_count": len(units),
            "positive_longer_correct_units_for_length_heuristic": positive_longer,
            "negative_longer_incorrect_units_for_length_heuristic": negative_longer,
            "unresolved_equal_length_units": ties,
            "can_meet_selection_family_gate_of_5": positive_longer >= 5,
        }
        if relation == "citrus_membership":
            for row in units.values():
                positive_marked = marked(row["positive_object"])
                negative_marked = marked(row["negative_object"])
                if positive_marked != negative_marked:
                    marker_selection_resolved_units += 1
                    marker_selection_correct_units += positive_marked
    shortcut_can_pass = (
        citrus_marker_direct_correct_units >= 10
        or marker_selection_correct_units >= 5
        or any(
            report["can_meet_selection_family_gate_of_5"]
            for report in length_reports.values()
        )
        or (
            fruit_foodness_correct_units >= 10
            and fruit_foodness_nonfruit_food_correct_units >= 2
        )
    )
    return {
        "purpose": "diagnose deterministic surface shortcuts; never a behavior gate",
        "citrus_name_markers": list(markers),
        "citrus_direct_marker_correct_analysis_units": (
            citrus_marker_direct_correct_units
        ),
        "citrus_direct_marker_total_analysis_units": len(citrus_direct_units),
        "citrus_selection_marker_resolved_analysis_units": (
            marker_selection_resolved_units
        ),
        "citrus_selection_marker_correct_analysis_units": (
            marker_selection_correct_units
        ),
        "fruit_direct_foodness_correct_analysis_units": fruit_foodness_correct_units,
        "fruit_direct_foodness_total_analysis_units": len(fruit_direct_units),
        "fruit_direct_foodness_nonfruit_food_correct_units_of_2": (
            fruit_foodness_nonfruit_food_correct_units
        ),
        "selection_character_length_heuristic": length_reports,
        "shortcut_can_pass_all_relevant_family_and_subgroup_gates": shortcut_can_pass,
        "passing_models_cannot_be_interpreted_as_abstract_taxonomy_without_"
        "additional_shortcut_controls": True,
    }


def source_identities() -> dict[str, Any]:
    identities = {}
    for relative in SCRIPT_RELATIVES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing Phase577 source before freeze: {relative}")
        identities[relative] = file_identity(path)
    return identities


def tokenizer_input_identities() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        directory = ROOT / MODEL_DIR_RELATIVES[model]
        if not directory.is_dir():
            raise RuntimeError(f"invalid tokenizer directory for {model}: {directory}")
        identities = {}
        for name in TOKENIZER_INPUT_NAMES:
            path = directory / name
            if path.is_file():
                identities[name] = {
                    "path": f"{MODEL_DIR_RELATIVES[model]}/{name}",
                    "resolved_path": str(path.resolve(strict=True)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "leaf_is_symlink": path.is_symlink(),
                }
        if not {"tokenizer_config.json", "tokenizer.json"}.issubset(identities):
            raise RuntimeError(f"required tokenizer inputs are missing for {model}")
        registry[model] = {
            "entry_path": MODEL_DIR_RELATIVES[model],
            "entry_is_symlink": directory.is_symlink(),
            "resolved_directory": str(directory.resolve(strict=True)).replace("\\", "/"),
            "files": identities,
        }
    return registry


def build_bundle(created_at: str) -> dict[str, Any]:
    cases = {split: split_records(split) for split in SPLITS}
    split_audits = {
        split: validate_split(rows, split) for split, rows in cases.items()
    }
    all_rows = [row for split in SPLITS for row in cases[split]]
    if len({row["case_id"] for row in all_rows}) != 4 * 336:
        raise RuntimeError("Phase577 global case IDs are not unique")
    if len({row["normalized_prompt_sha256"] for row in all_rows}) != 4 * 336:
        raise RuntimeError("Phase577 normalized prompts are not globally unique")

    split_object_sets = {
        split: {
            label
            for labels in OBJECTS[split].values()
            for label in labels
        }
        for split in SPLITS
    }
    object_intersections = {}
    split_lexical_atoms = {
        split: {
            atom
            for label in split_object_sets[split]
            for atom in re.findall(r"[a-z0-9]+", label.casefold())
        }
        for split in SPLITS
    }
    lexical_atom_intersections = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = split_object_sets[left] & split_object_sets[right]
            object_intersections[f"{left}|{right}"] = sorted(overlap)
            lexical_atom_intersections[f"{left}|{right}"] = sorted(
                split_lexical_atoms[left] & split_lexical_atoms[right]
            )
            if overlap:
                raise RuntimeError(f"Phase577 object leakage: {left}/{right}")

    prior_audit = prior_prompt_audit(all_rows)
    phase576_object_audit = phase576_open_object_audit(all_rows)
    shortcut_audits = {
        split: deterministic_shortcut_audit(cases[split]) for split in SPLITS
    }
    if any(
        report["shortcut_can_pass_all_relevant_family_and_subgroup_gates"]
        for report in shortcut_audits.values()
    ):
        raise RuntimeError(
            "a preregistered deterministic string shortcut can pass Phase577 gates"
        )
    sources = source_identities()
    tokenizer_inputs = tokenizer_input_identities()
    legacy_path = ROOT / LEGACY_PHASE577_RELATIVE
    legacy_identity = file_identity(legacy_path)
    model_registry_identity = file_identity(ROOT / MODEL_REGISTRY_RELATIVE)
    pilot = {
        name: file_identity(ROOT / relative)
        for name, relative in PILOT_IDENTITIES.items()
    }
    attachment_path = Path(ATTACHMENT_RELATIVE_EXTERNAL)
    attachment_identity = file_identity(attachment_path, relative_to=None)
    previous_attachment_identity = file_identity(
        Path(PREVIOUS_ATTACHMENT_RELATIVE_EXTERNAL), relative_to=None
    )
    if attachment_identity["sha256"] != previous_attachment_identity["sha256"]:
        raise RuntimeError("new Phase990 attachment differs from its previous copy")

    sealed_payload = jsonl_bytes(cases["sealed"])
    per_split_payloads = {
        split: jsonl_bytes(cases[split]) for split in OPEN_SPLITS
    }
    sealed_truth_registry = {
        label: object_class
        for object_class, labels in OBJECTS["sealed"].items()
        for label in labels
    }
    sealed_commitment = {
        "schema_version": "phase577_sealed_commitment.v2",
        "phase_id": PHASE,
        "created_at_utc": created_at,
        "sealed_split": "sealed",
        "sealed_case_count": len(cases["sealed"]),
        "sealed_payload_sha256": hashlib.sha256(sealed_payload).hexdigest(),
        "sealed_payload_size_bytes": len(sealed_payload),
        "sealed_truth_registry_sha256": stable_hash(sealed_truth_registry),
        "sealed_truth_registry_entry_count": len(sealed_truth_registry),
        "sealed_payload_path": (
            f"{PRIVATE_DIR_RELATIVE}/phase577_sealed_cases.jsonl"
        ),
        "payload_generated_but_not_model_accessed": True,
        "not_external_blind": True,
        "candidate_coordinates_frozen": [],
        "candidate_mechanism_formulas_frozen": [],
        "prior_phase576_sealed_payload_read": False,
    }

    template_registry = {
        "direct": list(DIRECT_TEMPLATES),
        "selection": list(SELECTION_TEMPLATES),
    }
    protocol = {
        "schema_version": SCHEMA,
        "phase_id": PHASE,
        "created_at_utc": created_at,
        "research_question": (
            "Can natural fruit and citrus membership be semantically reached in "
            "both repeats across paired instruction paraphrases, order, and query "
            "polarity before any internal trajectory is inspected? Exact-format "
            "compliance and full-output identity are separate diagnostics."
        ),
        "design_seed": DESIGN_SEED,
        "models_in_required_order": list(MODELS),
        "splits": list(SPLITS),
        "open_splits": list(OPEN_SPLITS),
        "cases_per_split": 336,
        "analysis_units_per_split": 36,
        "behavior_repeats": list(BEHAVIOR_REPEATS),
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "relations": list(RELATIONS),
        "relation_specs": RELATION_SPECS,
        "taxonomy_contract": {
            "sense": "ordinary English culinary-food taxonomy for each exact named referent",
            "fruit_membership": (
                "citrus and non-citrus culinary fruits are positive; food vegetables, "
                "animals, and objects/materials are negative"
            ),
            "citrus_membership": (
                "citrus fruits are positive and non-citrus fruits are hard negatives; "
                "this relation does not estimate rejection of arbitrary nonfruit items"
            ),
            "labels_are_preregistered_task_ground_truth_not_discovered_mechanism": True,
            "independent_audit_duplicates_open_truth_registry": True,
            "independent_audit_duplicates_sealed_truth_registry_without_opening_"
            "private_cases": True,
            "external_ontology_certificate": False,
        },
        "template_registry": template_registry,
        "behavior_gate": BEHAVIOR_GATE,
        "scoring_contract": {
            "semantic_primary": (
                "first normalized answer prefix resolves uniquely to target before "
                "the frozen eight-token prefix budget"
            ),
            "strict_exact_short_separate": True,
            "eos_and_budget_separate": True,
            "contradictory_later_candidate_mention_reported_separately": True,
            "repeat_identity_fields": [
                "normalized_generated", "generated_token_ids_before_eos",
                "first_eos_token_id", "full_generated_suffix_token_ids",
            ],
            "both_repeats_need_only_resolve_semantically_to_target_for_primary_gate": True,
            "polarity_fields": {
                "query_polarity": (
                    "affirmative for every direct question; positive or negative for "
                    "selection instructions"
                ),
                "target_truth_polarity": (
                    "positive or negative semantic class of the registered target"
                ),
            },
            "semantic_prefix_parser": {
                "normalization": "Unicode NFKC, casefold, collapse whitespace",
                "leading_ignored_characters": (
                    "ASCII whitespace plus quote, bracket, and bullet punctuation only"
                ),
                "resolution": (
                    "after leading ignored characters, exactly one registered candidate "
                    "must be the first complete word-sequence prefix; the following "
                    "character must be end-of-text or non-word punctuation/whitespace"
                ),
                "correct": "the uniquely resolved prefix candidate equals target",
                "incorrect": "the uniquely resolved prefix candidate equals foil",
                "unresolved": "no candidate or multiple candidates resolve at the prefix",
                "prefix_must_finish_within_generated_tokens": 8,
                "later_foil_mentions_are_reported_but_do_not_change_primary_label": True,
            },
            "exact_short_parser": {
                "normalization": (
                    "semantic normalization plus outer terminal punctuation trim"
                ),
                "correct": "normalized complete output equals one target alias",
            },
        },
        "evidence_order": [
            "preregister_and_freeze",
            "tokenizer_precheck",
            "independent_public_audit",
            "separate_runner_and_executable_scorer_source_freeze",
            "future_engineering_qualification",
            "development_behavior",
            "behavior_gate",
            "eligible_model_full_natural_trace",
            "post_trace_candidate_discovery",
            "candidate_and_confirmation_rule_freeze",
            "confirmation",
            "heldout_novel_entities",
            "separate_sealed_decision",
        ],
        "phase576_role": "pilot_and_hypothesis_generation_only_not_phase577_evidence",
        "relation_novelty": {
            "fruit_membership": (
                "revised v2 contract informed by Phase576 pilot but evaluated only "
                "on new Phase577 objects and prompts"
            ),
            "citrus_membership": "new subset relation not evaluated in Phase576",
        },
        "phase576_pilot_artifact_identities": pilot,
        "attachment_identity": attachment_identity,
        "attachment_comparison": {
            "previous_identity": previous_attachment_identity,
            "byte_identical": True,
            "same_sha256": attachment_identity["sha256"],
        },
        "attachment_has_new_evidence": False,
        "source_identities": sources,
        "tokenizer_input_identities": tokenizer_inputs,
        "tokenizer_input_registry_sha256": stable_hash(tokenizer_inputs),
        "stage_public_relative_paths": list(STAGE_PUBLIC_RELATIVES),
        "model_registry_identity": model_registry_identity,
        "legacy_phase577_collision": {
            "identity": legacy_identity,
            "status": "excluded_not_executed_not_imported",
            "reason": (
                "legacy script presupposes an attention-retrieval circuit and "
                "preselects heads/ablations before current natural evidence"
            ),
        },
        "candidate_coordinates_before_trace": [],
        "candidate_mechanism_formulas_before_trace": [],
        "internal_activation_access_authorized": False,
        "causal_intervention_authorized": False,
        "gpu_behavior_authorized_before_final_freeze": False,
        "gpu_behavior_authorized_by_phase577_final_freeze": False,
        "future_gpu_behavior_requires_separate_runner_and_scorer_freeze": True,
        "cross_model_internal_comparison_requires_all_models_behavior_pass": True,
        "single_model_trace_requires_that_model_behavior_pass": True,
        "sealed_policy": {
            "payload_is_hash_committed_not_external_blind": True,
            "no_model_access_before_confirmation_and_heldout_pass": True,
            "generator_validates_sealed_rows_in_memory_before_private_write": True,
            "tokenizer_precheck_and_independent_audit_must_not_open_private_file": True,
            "private_file_is_not_a_third_party_blind": True,
            "old_phase576_sealed_payload_must_remain_unread": True,
        },
        "split_access_policy": {
            "no_combined_open_case_file": True,
            "tokenizer_and_independent_audit_non_model_access_to_all_open_splits": True,
            "future_development_runner_may_read_development_only": True,
            "confirmation_requires_post_development_gate_lease": True,
            "heldout_novel_entities_requires_post_confirmation_gate_lease": True,
            "future_runner_source_and_runtime_path_allowlist_required": True,
        },
        "scientific_limits": [
            "protocol gates are design choices, not discovered internal laws",
            "behavior success would not itself prove an internal mechanism",
            "deterministic repeats are paired checks, not independent samples",
            "culinary taxonomy is a bounded English-language task",
            "sealed payload is locally committed, not third-party blinded",
            "36 analysis units are repeated-measures aggregation units, not independent samples",
            "336 cases per split must not be reported as 336 independent observations",
            "the heldout split contains exact-label-novel entities, not a compositional recombination test",
            "exact entity disjointness does not imply lexical-atom disjointness",
            "shared prompt families prevent claims of unseen-surface generalization",
            "entity frequency, token length, and visible name fragments remain shortcut confounds",
            "Phase577 is behavior admission only; any post-development candidate is exploratory, and a stable repeated internal-structure claim requires a later larger object-independent confirmation round",
        ],
    }

    dataset_audit = {
        "schema_version": "phase577_dataset_audit.v2",
        "phase_id": PHASE,
        "created_at_utc": created_at,
        "valid": True,
        "split_audits": split_audits,
        "global_case_count": len(all_rows),
        "global_unique_case_id_count": len({row["case_id"] for row in all_rows}),
        "global_unique_normalized_prompt_count": len({
            row["normalized_prompt_sha256"] for row in all_rows
        }),
        "split_object_intersections": object_intersections,
        "split_whole_token_lexical_atom_intersections": lexical_atom_intersections,
        "exact_entity_disjoint_does_not_claim_lexical_isolation": True,
        "prior_open_prompt_audit": prior_audit,
        "phase576_open_object_audit": phase576_object_audit,
        "deterministic_shortcut_audits": shortcut_audits,
        "old_phase576_sealed_payload_read": False,
        "sealed_rows_generated_and_validated_in_memory": True,
        "private_sealed_file_reopened_during_bundle_build": False,
        "candidate_coordinates_observed": [],
        "candidate_mechanism_formulas_observed": [],
    }
    return {
        "cases": cases,
        "per_split_payloads": per_split_payloads,
        "sealed_payload": sealed_payload,
        "sealed_commitment": sealed_commitment,
        "protocol": protocol,
        "dataset_audit": dataset_audit,
    }


def artifact_identity_from_bytes(relative: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": relative,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_stage() -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError(
            "Phase577 directory publication is frozen to Windows no-replace rename semantics"
        )
    if os.path.lexists(OUT_DIR):
        raise RuntimeError(f"Phase577 result root already exists: {OUT_DIR}")
    created_at = now()
    bundle = build_bundle(created_at)
    staging = OUT_DIR.with_name(f".{OUT_DIR.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        payloads: dict[str, bytes] = {
            "phase577_preregistered_protocol.json": json_bytes(bundle["protocol"]),
            "phase577_dataset_audit.json": json_bytes(bundle["dataset_audit"]),
            "phase577_sealed_commitment.json": json_bytes(bundle["sealed_commitment"]),
        }
        for split, payload in bundle["per_split_payloads"].items():
            payloads[f"phase577_{split}_cases.jsonl"] = payload
        private_relative = f"{PRIVATE_DIR_RELATIVE}/phase577_sealed_cases.jsonl"
        payloads[private_relative] = bundle["sealed_payload"]
        for relative, payload in payloads.items():
            atomic_write_new(staging / relative, payload)

        public_identities = {
            relative: artifact_identity_from_bytes(relative, payload)
            for relative, payload in payloads.items()
            if relative != private_relative
        }
        if set(public_identities) != set(STAGE_PUBLIC_RELATIVES):
            raise RuntimeError("Phase577 public artifact registry drift before publish")
        stage_commit = {
            "schema_version": "phase577_stage_commit.v2",
            "phase_id": PHASE,
            "created_at_utc": created_at,
            "stage_complete": True,
            "public_artifact_identities": public_identities,
            "sealed_commitment_identity": public_identities[
                "phase577_sealed_commitment.json"
            ],
            "sealed_payload_identity_not_republished": True,
            "sealed_payload_path": private_relative,
            "source_identities": bundle["protocol"]["source_identities"],
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
            "gpu_used": False,
            "model_weights_loaded": False,
            "old_phase576_sealed_payload_read": False,
        }
        atomic_write_new(staging / "phase577_stage_commit.json", json_bytes(stage_commit))
        OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(OUT_DIR):
            raise RuntimeError("Phase577 result root appeared during publish")
        # On Windows, os.rename refuses an existing destination.  This is the
        # directory-level no-overwrite primitive for the frozen environment.
        os.rename(staging, OUT_DIR)
    except BaseException:
        if staging.exists():
            remove_failed_staging(staging)
        raise
    return verify_stage(require_private_untouched=False)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifact_identity(
    base: Path, identity: Any, expected_relative: str
) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "path", "size_bytes", "sha256",
    }:
        return False
    if identity.get("path") != expected_relative:
        return False
    try:
        path = confined_path(base, expected_relative)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == identity.get("size_bytes")
        and isinstance(identity.get("sha256"), str)
        and len(identity["sha256"]) == 64
        and sha256_file(path) == identity["sha256"]
    )


def verify_stage(*, require_private_untouched: bool = True) -> dict[str, Any]:
    commit_path = OUT_DIR / "phase577_stage_commit.json"
    if not commit_path.is_file():
        raise RuntimeError("Phase577 stage commit is missing")
    commit = read_json(commit_path)
    public = commit.get("public_artifact_identities", {})
    observed_files = result_file_relatives()
    expected_public = set(STAGE_PUBLIC_RELATIVES)
    checks = {
        "stage_exact_schema_keys": set(commit) == {
            "schema_version", "phase_id", "created_at_utc", "stage_complete",
            "public_artifact_identities", "sealed_commitment_identity",
            "sealed_payload_identity_not_republished", "sealed_payload_path",
            "source_identities", "candidate_coordinates",
            "candidate_mechanism_formulas", "gpu_used", "model_weights_loaded",
            "old_phase576_sealed_payload_read",
        },
        "stage_schema": commit.get("schema_version") == "phase577_stage_commit.v2",
        "stage_phase": commit.get("phase_id") == PHASE,
        "stage_complete": commit.get("stage_complete") is True,
        "public_identity_registry": isinstance(public, dict)
        and set(public) == expected_public,
        "public_artifacts_match": all(
            verify_artifact_identity(OUT_DIR, public.get(relative), relative)
            for relative in STAGE_PUBLIC_RELATIVES
        ) if isinstance(public, dict) else False,
        "sealed_commitment_identity_exact": isinstance(public, dict)
        and commit.get("sealed_commitment_identity")
        == public.get("phase577_sealed_commitment.json"),
        "sealed_identity_not_republished": commit.get(
            "sealed_payload_identity_not_republished"
        ) is True,
        "recursive_file_allowlist": STAGE_REQUIRED_RELATIVES.issubset(observed_files)
        and observed_files.issubset(
            STAGE_REQUIRED_RELATIVES | QUALIFICATION_RELATIVES
        ),
        "qualification_order_consistent": (
            "phase577_freeze_commit.json" not in observed_files
            or {
                "phase577_tokenizer_precheck.json",
                "phase577_independent_audit.json",
            }.issubset(observed_files)
        ),
        "source_identities_match": commit.get("source_identities")
        == source_identities(),
        "no_candidates": commit.get("candidate_coordinates") == []
        and commit.get("candidate_mechanism_formulas") == [],
        "no_gpu": commit.get("gpu_used") is False,
        "no_weights": commit.get("model_weights_loaded") is False,
        "old_sealed_unread": commit.get("old_phase576_sealed_payload_read") is False,
    }
    protocol_path = OUT_DIR / "phase577_preregistered_protocol.json"
    audit_path = OUT_DIR / "phase577_dataset_audit.json"
    commitment_path = OUT_DIR / "phase577_sealed_commitment.json"
    protocol = read_json(protocol_path) if protocol_path.is_file() else {}
    audit = read_json(audit_path) if audit_path.is_file() else {}
    sealed_commitment = read_json(commitment_path) if commitment_path.is_file() else {}
    checks.update({
        "protocol_schema": protocol.get("schema_version") == SCHEMA,
        "protocol_stage_public_paths": protocol.get(
            "stage_public_relative_paths"
        ) == list(STAGE_PUBLIC_RELATIVES),
        "protocol_source_chain": protocol.get("source_identities")
        == source_identities(),
        "tokenizer_input_chain": protocol.get("tokenizer_input_identities")
        == tokenizer_input_identities()
        and protocol.get("tokenizer_input_registry_sha256")
        == stable_hash(tokenizer_input_identities()),
        "gpu_still_unauthorized": protocol.get(
            "gpu_behavior_authorized_by_phase577_final_freeze"
        ) is False
        and protocol.get(
            "future_gpu_behavior_requires_separate_runner_and_scorer_freeze"
        ) is True,
        "protocol_no_internal": (
            protocol.get("candidate_coordinates_before_trace") == []
            and protocol.get("candidate_mechanism_formulas_before_trace") == []
            and protocol.get("internal_activation_access_authorized") is False
            and protocol.get("causal_intervention_authorized") is False
        ),
        "audit_valid": audit.get("valid") is True
        and audit.get("global_case_count") == 1344
        and audit.get("prior_open_prompt_audit", {}).get(
            "total_normalized_overlap_count"
        ) == 0
        and audit.get("phase576_open_object_audit", {}).get("overlap_count") == 0
        and audit.get("sealed_rows_generated_and_validated_in_memory") is True
        and audit.get("private_sealed_file_reopened_during_bundle_build") is False,
        "sealed_commitment_schema": sealed_commitment.get("schema_version")
        == "phase577_sealed_commitment.v2",
        "sealed_commitment_exact_keys": set(sealed_commitment) == {
            "schema_version", "phase_id", "created_at_utc", "sealed_split",
            "sealed_case_count", "sealed_payload_sha256",
            "sealed_payload_size_bytes", "sealed_truth_registry_sha256",
            "sealed_truth_registry_entry_count", "sealed_payload_path",
            "payload_generated_but_not_model_accessed", "not_external_blind",
            "candidate_coordinates_frozen", "candidate_mechanism_formulas_frozen",
            "prior_phase576_sealed_payload_read",
        },
        "sealed_commitment_semantics": sealed_commitment.get("phase_id") == PHASE
        and sealed_commitment.get("sealed_split") == "sealed"
        and sealed_commitment.get("sealed_payload_path")
        == f"{PRIVATE_DIR_RELATIVE}/phase577_sealed_cases.jsonl"
        and isinstance(sealed_commitment.get("sealed_payload_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", sealed_commitment.get(
            "sealed_payload_sha256", ""
        )))
        and isinstance(sealed_commitment.get("sealed_payload_size_bytes"), int)
        and sealed_commitment.get("sealed_payload_size_bytes", 0) > 0
        and isinstance(sealed_commitment.get("sealed_truth_registry_sha256"), str)
        and bool(re.fullmatch(r"[0-9a-f]{64}", sealed_commitment.get(
            "sealed_truth_registry_sha256", ""
        )))
        and sealed_commitment.get("sealed_truth_registry_entry_count") == 18
        and sealed_commitment.get("payload_generated_but_not_model_accessed") is True
        and sealed_commitment.get("candidate_coordinates_frozen") == []
        and sealed_commitment.get("candidate_mechanism_formulas_frozen") == []
        and sealed_commitment.get("prior_phase576_sealed_payload_read") is False,
        "sealed_commitment_count": sealed_commitment.get("sealed_case_count") == 336,
        "sealed_not_external_blind": sealed_commitment.get("not_external_blind") is True,
    })
    private_relative = commit.get("sealed_payload_path")
    try:
        private_path = confined_path(OUT_DIR, private_relative)
        private_safe = private_relative == (
            f"{PRIVATE_DIR_RELATIVE}/phase577_sealed_cases.jsonl"
        )
    except (OSError, RuntimeError, ValueError):
        private_path = OUT_DIR / "__invalid_private_path__"
        private_safe = False
    checks["private_path_exact_and_confined"] = private_safe
    checks["private_payload_exists_without_content_read"] = (
        private_safe
        and private_path.is_file()
        and not private_path.is_symlink()
        and private_path.stat().st_size
        == sealed_commitment.get("sealed_payload_size_bytes")
    )
    if require_private_untouched:
        checks["private_payload_not_opened_by_verifier"] = True
    if not all(checks.values()):
        raise RuntimeError(f"Phase577 stage verification failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "files_written": False,
        "gpu_used": False,
        "model_weights_loaded": False,
        "sealed_payload_read": False,
        "stage_commit_sha256": sha256_file(commit_path),
    }


def load_frozen_module(module_name: str, relative: str) -> Any:
    identity = source_identities().get(relative)
    if not isinstance(identity, dict):
        raise RuntimeError(f"missing frozen source identity: {relative}")
    path = confined_path(ROOT, relative)
    if (
        identity.get("sha256") != sha256_file(path)
        or identity.get("size_bytes") != path.stat().st_size
        or identity.get("is_symlink") is not False
    ):
        raise RuntimeError(f"frozen qualification source drift: {relative}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen qualification source: {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_qualifications() -> dict[str, Any]:
    tokenizer_path = OUT_DIR / "phase577_tokenizer_precheck.json"
    independent_path = OUT_DIR / "phase577_independent_audit.json"
    if not tokenizer_path.is_file() or not independent_path.is_file():
        raise RuntimeError("tokenizer precheck and independent audit are required")
    tokenizer = read_json(tokenizer_path)
    independent = read_json(independent_path)
    tokenizer_module = load_frozen_module(
        "phase577_frozen_tokenizer_verifier",
        "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py",
    )
    audit_module = load_frozen_module(
        "phase577_frozen_independent_verifier",
        "tests/glm5/phase577_gpt5_natural_behavior_audit.py",
    )
    tokenizer_checks = tokenizer_module.validate_payload(tokenizer)
    independent_verification = audit_module.verify()
    checks = {
        "tokenizer_exact_qualification": isinstance(tokenizer_checks, dict)
        and bool(tokenizer_checks) and all(tokenizer_checks.values()),
        "tokenizer_schema_phase": tokenizer.get("schema_version")
        == "phase577_tokenizer_precheck.v2"
        and tokenizer.get("phase_id") == PHASE
        and tokenizer.get("passed") is True,
        "tokenizer_source_chain": tokenizer.get("tokenizer_source_sha256")
        == source_identities()[
            "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py"
        ]["sha256"],
        "tokenizer_protocol_stage_chain": tokenizer.get("protocol_sha256")
        == sha256_file(OUT_DIR / "phase577_preregistered_protocol.json")
        and tokenizer.get("stage_commit_sha256")
        == sha256_file(OUT_DIR / "phase577_stage_commit.json"),
        "tokenizer_model_order": tokenizer.get("models_in_observed_order")
        == list(MODELS),
        "tokenizer_no_execution_or_private_access": tokenizer.get("cuda_used") is False
        and tokenizer.get("model_weights_loaded") is False
        and tokenizer.get("weight_file_open_attempts") == []
        and tokenizer.get("unregistered_model_input_attempts") == []
        and tokenizer.get("private_content_open_attempts") == []
        and tokenizer.get("sealed_payload_read") is False
        and tokenizer.get("old_phase576_sealed_payload_read") is False,
        "independent_exact_qualification": independent_verification.get("passed") is True
        and all(independent_verification.get("checks", {}).values()),
        "independent_schema_phase": independent.get("schema_version")
        == "phase577_independent_audit.v2"
        and independent.get("phase_id") == PHASE
        and independent.get("passed") is True
        and all(independent.get("checks", {}).values()),
        "independent_source_chain": independent.get("audit_source_sha256")
        == source_identities()[
            "tests/glm5/phase577_gpt5_natural_behavior_audit.py"
        ]["sha256"],
        "independent_artifact_chains": independent.get("protocol_sha256")
        == sha256_file(OUT_DIR / "phase577_preregistered_protocol.json")
        and independent.get("dataset_audit_sha256")
        == sha256_file(OUT_DIR / "phase577_dataset_audit.json")
        and independent.get("stage_commit_sha256")
        == sha256_file(OUT_DIR / "phase577_stage_commit.json")
        and independent.get("sealed_commitment_sha256")
        == sha256_file(OUT_DIR / "phase577_sealed_commitment.json")
        and independent.get("tokenizer_precheck_sha256")
        == sha256_file(tokenizer_path),
        "independent_no_execution_or_private_access": independent.get("cuda_used") is False
        and independent.get("model_weights_loaded") is False
        and independent.get("private_content_open_attempts") == []
        and independent.get("sealed_payload_read") is False
        and independent.get("old_phase576_sealed_payload_read") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase577 qualification verification failed: {checks}")
    return {
        "checks": checks,
        "tokenizer_checks": tokenizer_checks,
        "independent_verification_checks": independent_verification["checks"],
        "tokenizer_precheck_sha256": sha256_file(tokenizer_path),
        "independent_audit_sha256": sha256_file(independent_path),
    }


def finalize() -> dict[str, Any]:
    stage = verify_stage()
    tokenizer_path = OUT_DIR / "phase577_tokenizer_precheck.json"
    independent_path = OUT_DIR / "phase577_independent_audit.json"
    final_path = OUT_DIR / "phase577_freeze_commit.json"
    if final_path.exists():
        raise RuntimeError("Phase577 final freeze commit already exists")
    qualifications = verify_qualifications()
    payload = {
        "schema_version": "phase577_freeze_commit.v2",
        "phase_id": PHASE,
        "created_at_utc": now(),
        "freeze_complete": True,
        "stage_commit_sha256": stage["stage_commit_sha256"],
        "protocol_sha256": sha256_file(
            OUT_DIR / "phase577_preregistered_protocol.json"
        ),
        "dataset_audit_sha256": sha256_file(
            OUT_DIR / "phase577_dataset_audit.json"
        ),
        "sealed_commitment_sha256": sha256_file(
            OUT_DIR / "phase577_sealed_commitment.json"
        ),
        "tokenizer_precheck_sha256": qualifications["tokenizer_precheck_sha256"],
        "independent_audit_sha256": qualifications["independent_audit_sha256"],
        "qualification_checks_sha256": stable_hash(qualifications["checks"]),
        "tokenizer_verification_checks_sha256": stable_hash(
            qualifications["tokenizer_checks"]
        ),
        "independent_verification_checks_sha256": stable_hash(
            qualifications["independent_verification_checks"]
        ),
        "source_identities": source_identities(),
        "models_in_required_future_order": list(MODELS),
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "gpu_behavior_run_count": 0,
        "internal_trace_run_count": 0,
        "gpu_behavior_authorized": False,
        "next_required_stage": "freeze_separate_gpu_runner_and_executable_scorer",
        "sealed_model_access_count": 0,
        "sealed_payload_read_for_finalize": False,
        "old_phase576_sealed_payload_read": False,
    }
    atomic_write_new(final_path, json_bytes(payload))
    return verify_final()


def verify_final() -> dict[str, Any]:
    stage = verify_stage()
    qualifications = verify_qualifications()
    final_path = OUT_DIR / "phase577_freeze_commit.json"
    if not final_path.is_file():
        raise RuntimeError("Phase577 final freeze commit is missing")
    final = read_json(final_path)
    checks = {
        "exact_schema_keys": set(final) == {
            "schema_version", "phase_id", "created_at_utc", "freeze_complete",
            "stage_commit_sha256", "protocol_sha256", "dataset_audit_sha256",
            "sealed_commitment_sha256", "tokenizer_precheck_sha256",
            "independent_audit_sha256", "qualification_checks_sha256",
            "tokenizer_verification_checks_sha256",
            "independent_verification_checks_sha256", "source_identities",
            "models_in_required_future_order", "candidate_coordinates",
            "candidate_mechanism_formulas", "gpu_behavior_run_count",
            "internal_trace_run_count", "gpu_behavior_authorized",
            "next_required_stage", "sealed_model_access_count",
            "sealed_payload_read_for_finalize", "old_phase576_sealed_payload_read",
        },
        "schema": final.get("schema_version") == "phase577_freeze_commit.v2",
        "phase": final.get("phase_id") == PHASE,
        "complete": final.get("freeze_complete") is True,
        "stage_chain": final.get("stage_commit_sha256")
        == stage["stage_commit_sha256"],
        "protocol_chain": final.get("protocol_sha256") == sha256_file(
            OUT_DIR / "phase577_preregistered_protocol.json"
        ),
        "dataset_audit_chain": final.get("dataset_audit_sha256") == sha256_file(
            OUT_DIR / "phase577_dataset_audit.json"
        ),
        "sealed_commitment_chain": final.get("sealed_commitment_sha256")
        == sha256_file(OUT_DIR / "phase577_sealed_commitment.json"),
        "tokenizer_chain": final.get("tokenizer_precheck_sha256") == sha256_file(
            OUT_DIR / "phase577_tokenizer_precheck.json"
        ),
        "independent_chain": final.get("independent_audit_sha256") == sha256_file(
            OUT_DIR / "phase577_independent_audit.json"
        ),
        "qualification_chain": final.get("qualification_checks_sha256")
        == stable_hash(qualifications["checks"])
        and final.get("tokenizer_verification_checks_sha256")
        == stable_hash(qualifications["tokenizer_checks"])
        and final.get("independent_verification_checks_sha256")
        == stable_hash(qualifications["independent_verification_checks"]),
        "source_chain": final.get("source_identities") == source_identities(),
        "model_order": final.get("models_in_required_future_order") == list(MODELS),
        "no_candidates": final.get("candidate_coordinates") == []
        and final.get("candidate_mechanism_formulas") == [],
        "no_execution": final.get("gpu_behavior_run_count") == 0
        and final.get("internal_trace_run_count") == 0
        and final.get("gpu_behavior_authorized") is False
        and final.get("next_required_stage")
        == "freeze_separate_gpu_runner_and_executable_scorer",
        "sealed_unopened": final.get("sealed_model_access_count") == 0
        and final.get("sealed_payload_read_for_finalize") is False,
        "old_sealed_unread": final.get("old_phase576_sealed_payload_read") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase577 final verification failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "files_written": False,
        "gpu_used": False,
        "model_weights_loaded": False,
        "sealed_payload_read": False,
        "freeze_commit_sha256": sha256_file(final_path),
    }


def self_test() -> dict[str, Any]:
    bundle = build_bundle("SELF_TEST_TIMESTAMP")
    return {
        "passed": True,
        "case_counts": {
            split: len(rows) for split, rows in bundle["cases"].items()
        },
        "analysis_units_per_split": 36,
        "global_normalized_prompt_count": bundle["dataset_audit"][
            "global_unique_normalized_prompt_count"
        ],
        "prior_open_overlap_count": bundle["dataset_audit"][
            "prior_open_prompt_audit"
        ]["total_normalized_overlap_count"],
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "gpu_used": False,
        "model_weights_loaded": False,
        "files_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify-stage", action="store_true")
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.write:
        result = write_stage()
    elif args.verify_stage:
        result = verify_stage()
    elif args.finalize:
        result = finalize()
    else:
        result = verify_final()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
