#!/usr/bin/env python3
"""Freeze the Phase590 natural semantic-event observer protocol."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase585_object_swap_protocol as phase585  # noqa: E402
import phase589_food_attribute_protocol as phase589  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase590"
MODELS = phase585.MODELS
OPEN_SPLITS = ("observer_discovery", "observer_confirmation", "observer_heldout")
SEALED_SPLIT = "sealed"
SPLITS = OPEN_SPLITS + (SEALED_SPLIT,)
NOOP_REPEATS = ("natural1", "natural2")
FIXED_BATCH_SIZE = 16
MAX_NEW_TOKENS = 24

MIN_COVERAGE = 0.90
MIN_SEMANTIC_ACCURACY = 0.90
MIN_GROUP_ACCURACY = 0.85
MIN_REPEAT_POLARITY_RATE = 0.99
MIN_STABLE_SURFACES_PER_OBJECT = 10
MIN_QUALIFIED_BY_SPLIT_GROUP = {
    "observer_discovery": 10,
    "observer_confirmation": 5,
    "observer_heldout": 2,
}

OUT_DIR = ROOT / "tests/gpt5/result/phase590_natural_semantic_event"
OPEN_CASES_PATH = OUT_DIR / "phase590_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase590_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase590_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase590_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase590_static_audit.json"
PUBLIC_OBJECTS_PATH = OUT_DIR / "phase590_public_object_bank.json"


OBJECT_LABELS = {
    "fruit": (
        "bilberry", "boysenberry", "gooseberry", "kumquat", "pomelo", "satsuma",
        "plantain", "quince", "soursop", "breadfruit", "longan", "mangosteen",
        "salak", "cherimoya", "feijoa", "loquat", "jujube", "acai",
        "blackcurrant", "redcurrant", "tamarind", "langsat", "jabuticaba", "cloudberry",
    ),
    "edible_nonfruit": (
        "black bean", "kidney bean", "soybean", "oats", "wheat", "barley",
        "quinoa", "couscous", "tofu", "bread", "cheese", "yogurt",
        "pasta", "noodles", "peanut", "almond", "walnut", "cashew",
        "hazelnut", "egg", "milk", "chicken", "salmon", "beef",
    ),
    "hazardous_nonfood_plant": (
        "foxglove", "yew", "lily of the valley", "castor bean", "daffodil", "rhododendron",
        "azalea", "monkshood", "jimsonweed", "lantana", "mistletoe", "wisteria",
        "delphinium", "laburnum", "dogbane", "deadly nightshade", "horse chestnut", "autumn crocus",
        "white snakeroot", "water hemlock", "angel's trumpet", "rosary pea", "poison hemlock", "oleander flower",
    ),
    "artifact": (
        "ratchet", "hex key", "torque wrench", "pry bar", "bolt cutter", "wire stripper",
        "crimping tool", "caulking gun", "nail gun", "chainsaw", "pickaxe", "plunger",
        "flashlight", "sedan", "hatchback", "limousine", "submarine", "moped",
        "golf cart", "excavator", "leaf blower", "vacuum cleaner", "fire extinguisher", "sewing machine",
    ),
}

EXPECTED_POLARITY = {
    "fruit": "positive",
    "edible_nonfruit": "positive",
    "hazardous_nonfood_plant": "negative",
    "artifact": "negative",
}

SPLIT_SLICES = {
    "observer_discovery": (0, 12),
    "observer_confirmation": (12, 18),
    "observer_heldout": (18, 21),
    "sealed": (21, 24),
}

SPLIT_SURFACES = {
    "observer_discovery": tuple(range(0, 12)),
    "observer_confirmation": tuple(range(12, 24)),
    "observer_heldout": tuple(range(24, 36)),
    "sealed": tuple(range(36, 48)),
}

SURFACE_TEMPLATES = (
    "Write one short ordinary factual sentence about whether people commonly eat {object} as food.",
    "Using everyday knowledge, describe in one short sentence how {object} relates to human food.",
    "Give one concise real-world sentence about people consuming {object} as food.",
    "State one ordinary fact, in a single short sentence, about eating {object} as food.",
    "Describe briefly whether {object} is normally part of what people eat.",
    "In one short factual sentence, explain the usual human food use of {object}.",
    "Provide a concise everyday statement about {object} and human consumption.",
    "Write a brief factual sentence about the normal food status of {object} for people.",
    "Give a plain one-sentence fact about whether humans ordinarily consume {object}.",
    "Summarize in one short sentence the everyday relationship between {object} and food.",
    "State briefly how {object} is normally treated as something people might eat.",
    "Answer with one ordinary factual sentence about {object} as human food.",
    "Explain in one concise sentence whether people typically eat {object}.",
    "Write a short common-knowledge sentence about human consumption of {object}.",
    "Give one plain factual sentence about the usual edibility of {object} for people.",
    "Describe in a single brief sentence whether {object} is ordinarily eaten by humans.",
    "Provide one short real-world fact about {object} in relation to people's diet.",
    "State, in one concise sentence, the normal role of {object} in human food.",
    "Write one brief everyday sentence about people using {object} as food.",
    "Give a short factual description of whether {object} is a normal food for humans.",
    "In one sentence, describe the ordinary food use or non-use of {object} by people.",
    "Provide one concise common-sense fact about eating {object}.",
    "State one brief fact about whether {object} is usually consumed as food.",
    "Write a single short sentence about the normal human consumption status of {object}.",
    "Give one short factual statement about whether {object} belongs in an ordinary human diet.",
    "Explain briefly, in one sentence, the usual relation of {object} to human meals.",
    "Write one concise everyday fact about whether people would normally eat {object}.",
    "State in a short sentence how {object} is ordinarily regarded as human food.",
    "Provide one plain factual sentence about the food use of {object} among people.",
    "Give a brief common-knowledge statement about {object} and eating.",
    "In one short sentence, say whether {object} is normally consumed by humans as food.",
    "Write one concise fact about the everyday place of {object} in people's food.",
    "Describe briefly the usual human consumption of {object}, using one factual sentence.",
    "State a short ordinary fact about whether {object} is something people eat.",
    "Give one concise sentence about the typical food status of {object}.",
    "Write one brief factual statement about {object} in an ordinary diet.",
    "In one short sentence, explain whether {object} is commonly eaten by people.",
    "Provide a concise everyday fact about humans eating {object} as food.",
    "State one plain sentence about the normal food relationship of {object} to people.",
    "Write a short factual sentence about whether {object} is generally used as human food.",
    "Give one brief common-sense sentence about consuming {object}.",
    "Describe in one concise sentence the ordinary human food status of {object}.",
    "Provide one short factual sentence about whether people generally eat {object}.",
    "State briefly, in one sentence, the usual role of {object} in human consumption.",
    "Write one plain real-world sentence about {object} as something humans may eat.",
    "Give one concise ordinary fact about whether {object} is part of human food.",
    "Explain in one short factual sentence how people normally treat {object} as food or non-food.",
    "Provide one brief sentence stating whether {object} is ordinarily eaten by humans.",
)

DIAGNOSTIC_CONTINUATIONS = {
    "positive": " Yes. People commonly eat it as food.",
    "negative": " No. People do not commonly eat it as food.",
}

NEGATIVE_PATTERNS = (
    r"\bnot (?:commonly|normally|typically|generally|usually|ordinarily) (?:eaten|consumed)\b",
    r"\b(?:is|are) not (?:edible|food)\b",
    r"\bnot edible\b",
    r"\binedible\b",
    r"\b(?:cannot|can't|can not) be eaten\b",
    r"\bshould not be (?:eaten|consumed)\b",
    r"\bunsafe to (?:eat|consume)\b",
    r"\b(?:toxic|poisonous)\b",
    r"\bnot (?:used|consumed|prepared|regarded|considered) as (?:a )?food\b",
    r"\bpeople (?:do not|don't) (?:commonly |normally |typically |generally |usually |ordinarily )?(?:eat|consume)\b",
    r"\bnot (?:intended|meant|suitable|fit) for (?:human )?consumption\b",
    r"\bnot something (?:people|humans) (?:eat|consume)\b",
    r"\bnever (?:eaten|consumed)\b",
    r"\bnon[- ]food\b",
)

POSITIVE_PATTERNS = (
    r"\b(?:commonly|normally|typically|generally|usually|often|widely|ordinarily) (?:eaten|consumed)\b",
    r"\b(?:is|are) edible\b",
    r"\bsafe to eat\b",
    r"\bcan be eaten\b",
    r"\bpeople (?:commonly |normally |typically |generally |usually |often |ordinarily )?(?:eat|consume)\b",
    r"\b(?:used|prepared|consumed|regarded|considered) as (?:a )?food\b",
    r"\b(?<!not )(?<!not generally )(?<!not normally )(?:fit|suitable) for human consumption\b",
    r"\bpart of (?:a|the) (?:human )?diet\b",
    r"\bfood (?:item|source|product)\b",
    r"\bhuman food\b",
)

CONDITIONAL_PATTERNS = (
    r"\bonly (?:after|if|when)\b",
    r"\b(?:some|certain) (?:parts|varieties)\b",
    r"\brequires? (?:cooking|processing|preparation)\b",
    r"\bin small (?:amounts|quantities)\b",
    r"\bdepending on\b",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def classify_semantic_text(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    folded = normalized.casefold()
    negative_hits = [pattern for pattern in NEGATIVE_PATTERNS if re.search(pattern, folded)]
    positive_hits = [pattern for pattern in POSITIVE_PATTERNS if re.search(pattern, folded)]
    conditional_hits = [pattern for pattern in CONDITIONAL_PATTERNS if re.search(pattern, folded)]
    if re.match(r"^\s*no\b", folded):
        negative_hits.append("leading_no")
    if re.match(r"^\s*yes\b", folded):
        positive_hits.append("leading_yes")
    if negative_hits and positive_hits:
        polarity = "ambiguous"
        event = "positive_negative_conflict"
    elif negative_hits:
        polarity = "negative"
        event = "negative"
    elif positive_hits and conditional_hits:
        polarity = "ambiguous"
        event = "conditional_positive"
    elif positive_hits:
        polarity = "positive"
        event = "positive"
    else:
        polarity = "unresolved"
        event = "no_frozen_semantic_event"
    return {
        "normalized_generated": normalized,
        "semantic_polarity": polarity,
        "semantic_event": event,
        "positive_pattern_hits": positive_hits,
        "negative_pattern_hits": negative_hits,
        "conditional_pattern_hits": conditional_hits,
        "semantic_event_observed": polarity in {"positive", "negative"},
    }


def objects_for(split: str) -> list[dict[str, str]]:
    start, stop = SPLIT_SLICES[split]
    return [
        {
            "object_id": slug(label),
            "object_label": label,
            "semantic_group": group,
            "expected_polarity": EXPECTED_POLARITY[group],
        }
        for group, labels in OBJECT_LABELS.items()
        for label in labels[start:stop]
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        destination = sealed_rows if split == SEALED_SPLIT else open_rows
        for item in objects_for(split):
            for surface_id in SPLIT_SURFACES[split]:
                raw_prompt = SURFACE_TEMPLATES[surface_id].format(object=item["object_label"])
                prompt_counts: dict[str, int] = {}
                candidate_ids: dict[str, dict[str, list[int]]] = {}
                for model, tokenizer in tokenizers.items():
                    rendered = render_chat(tokenizer, model, raw_prompt)
                    prompt_counts[model] = len(
                        tokenizer(rendered, add_special_tokens=True)["input_ids"]
                    )
                    candidate_ids[model] = {
                        name: [
                            int(value)
                            for value in tokenizer(text, add_special_tokens=False)["input_ids"]
                        ]
                        for name, text in DIAGNOSTIC_CONTINUATIONS.items()
                    }
                destination.append(
                    {
                        "schema_version": "phase590_natural_semantic_event_case.v1",
                        "phase_id": PHASE,
                        "created_at": now(),
                        "case_id": f"phase590_{split}_{item['object_id']}_surface{surface_id:02d}",
                        "split": split,
                        **item,
                        "surface_id": surface_id,
                        "raw_prompt": raw_prompt,
                        "diagnostic_continuations": DIAGNOSTIC_CONTINUATIONS,
                        "diagnostic_token_ids_by_model": candidate_ids,
                        "prompt_token_count_by_model": prompt_counts,
                        "explicit_answer_label_in_prompt": bool(
                            re.search(r"(?<!\w)(yes|no|positive|negative|edible|inedible)(?!\w)", raw_prompt, re.I)
                        ),
                        "observer_only": True,
                        "causal": False,
                        "sealed": split == SEALED_SPLIT,
                    }
                )
    return open_rows, sealed_rows


def _prior_object_ids() -> set[str]:
    phase585_ids = {
        item["object_id"] for group in phase585.OBJECT_GROUPS.values() for item in group
    }
    phase589_ids = {slug(label) for labels in phase589.OBJECT_LABELS.values() for label in labels}
    return phase585_ids | phase589_ids


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = open_rows + sealed_rows
    expected = {
        "observer_discovery": 576,
        "observer_confirmation": 288,
        "observer_heldout": 144,
        "sealed": 144,
    }
    current_ids = {row["object_id"] for row in rows}
    audit = {
        "schema_version": "phase590_natural_semantic_event_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "expected_case_count_by_split": expected,
        "object_count_by_split_group": {
            f"{split}:{group}": len(
                {
                    row["object_id"]
                    for row in rows
                    if row["split"] == split and row["semantic_group"] == group
                }
            )
            for split in SPLITS
            for group in OBJECT_LABELS
        },
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_split_prompt_count": len(rows)
        - len({(row["split"], row["raw_prompt"]) for row in rows}),
        "prior_object_overlap_count": len(current_ids & _prior_object_ids()),
        "explicit_answer_label_in_prompt_count": sum(
            row["explicit_answer_label_in_prompt"] for row in rows
        ),
        "empty_diagnostic_tokenization_count": sum(
            not token_ids
            for row in rows
            for model_candidates in row["diagnostic_token_ids_by_model"].values()
            for token_ids in model_candidates.values()
        ),
        "max_prompt_token_count": max(
            count for row in rows for count in row["prompt_token_count_by_model"].values()
        ),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
    }
    audit["valid"] = bool(
        len(rows) == 1152
        and len(open_rows) == 1008
        and len(sealed_rows) == 144
        and audit["case_count_by_split"] == expected
        and audit["max_prompt_token_count"] <= 160
        and all(
            audit[key] == 0
            for key in (
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
                "prior_object_overlap_count",
                "explicit_answer_label_in_prompt_count",
                "empty_diagnostic_tokenization_count",
                "open_contains_sealed_count",
                "sealed_flag_missing_count",
            )
        )
    )
    audit["status"] = "static_pass_no_model_run" if audit["valid"] else "static_fail"
    return audit


def register() -> dict[str, Any]:
    open_rows, sealed_rows = build_rows()
    audit = validate(open_rows, sealed_rows)
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    write_json(
        PUBLIC_OBJECTS_PATH,
        {
            "schema_version": "phase590_public_object_bank.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "objects_by_group": OBJECT_LABELS,
            "expected_polarity_by_group": EXPECTED_POLARITY,
            "open_split_slices": {split: SPLIT_SLICES[split] for split in OPEN_SPLITS},
        },
    )
    write_json(
        SEALED_COMMITMENT_PATH,
        {
            "schema_version": "phase590_sealed_commitment.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "sealed_case_count": len(sealed_rows),
            "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
            "sealed_split_read_for_analysis": False,
        },
    )
    write_json(AUDIT_PATH, audit)
    frozen = {
        "schema_version": "phase590_natural_semantic_event_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Prospective natural food-status semantic-event observer",
        "narrow_proposition": "whether people ordinarily eat the object as food",
        "models_in_required_execution_order": list(MODELS),
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "noop_repeats": list(NOOP_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "diagnostic_continuations": DIAGNOSTIC_CONTINUATIONS,
        "frozen_semantic_parser": {
            "negative_patterns": NEGATIVE_PATTERNS,
            "positive_patterns": POSITIVE_PATTERNS,
            "conditional_patterns": CONDITIONAL_PATTERNS,
        },
        "gate": {
            "minimum_coverage": MIN_COVERAGE,
            "minimum_semantic_accuracy": MIN_SEMANTIC_ACCURACY,
            "minimum_each_group_accuracy": MIN_GROUP_ACCURACY,
            "minimum_repeat_polarity_rate": MIN_REPEAT_POLARITY_RATE,
            "minimum_stable_surfaces_per_object": MIN_STABLE_SURFACES_PER_OBJECT,
            "minimum_qualified_objects_each_group_by_split": MIN_QUALIFIED_BY_SPLIT_GROUP,
            "all_three_open_splits_must_pass": True,
        },
        "evidence_policy": {
            "observer_frozen_before_model_execution": True,
            "independent_human_gold_standard_available": False,
            "automatic_parser_is_not_human_gold": True,
            "natural_generation_observer_only": True,
            "teacher_forced_token_ledger_is_auxiliary_only": True,
            "may_authorize_exploratory_open_hidden_capture": True,
            "may_not_authorize_mechanism_or_causal_claim": True,
            "sealed_split_read": False,
        },
        "open_cases_path": str(OPEN_CASES_PATH.relative_to(ROOT)),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
