#!/usr/bin/env python3
"""Freeze a prospective, label-free food-attribute response protocol."""

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

import phase585_object_swap_protocol as prior_objects  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase589"
MODELS = prior_objects.MODELS
OPEN_SPLITS = ("prospective_confirmation", "prospective_heldout")
SEALED_SPLIT = "sealed"
SPLITS = OPEN_SPLITS + (SEALED_SPLIT,)
NOOP_REPEATS = ("score1", "score2")
FIXED_BATCH_SIZE = 12
MAX_REPEAT_SCORE_DELTA = 1e-6

MIN_EDIBLE_VS_NONEDIBLE_AUC = 0.85
MIN_EDIBLE_VS_NONFOOD_PLANT_AUC = 0.75
MIN_EDIBLE_GROUP_VS_ARTIFACT_AUC = 0.90
MIN_SURFACE_AUC = 0.80
MIN_QUALIFIED_SURFACES = 10

OUT_DIR = ROOT / "tests/gpt5/result/phase589_food_attribute"
OPEN_CASES_PATH = OUT_DIR / "phase589_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase589_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase589_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase589_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase589_static_audit.json"


OBJECT_LABELS = {
    "fruit": (
        "apricot", "avocado", "blackberry", "raspberry", "cranberry", "grapefruit",
        "tangerine", "clementine", "nectarine", "guava", "passionfruit", "dragonfruit",
        "cantaloupe", "honeydew", "fig", "date", "persimmon", "coconut",
        "starfruit", "lychee", "rambutan", "jackfruit", "durian", "mulberry",
    ),
    "edible_nonfruit": (
        "asparagus", "beet", "cauliflower", "kale", "garlic", "ginger",
        "turnip", "parsnip", "pea", "corn", "mushroom", "zucchini",
        "pumpkin", "sweet potato", "artichoke", "leek", "okra", "yam",
        "bok choy", "Brussels sprout", "green bean", "lentil", "chickpea", "rice",
    ),
    "nonfood_plant": (
        "oak", "pine", "maple", "rose", "tulip", "daisy",
        "fern", "moss", "oleander", "ivy", "hemlock", "grass",
        "cedar", "spruce", "orchid", "eucalyptus", "sequoia", "poison ivy",
        "juniper", "willow", "redwood", "marigold", "petunia", "hydrangea",
    ),
    "tool": (
        "chisel", "rake", "hoe", "crowbar", "clamp", "metal file",
        "spirit level", "tape measure", "paintbrush", "soldering iron", "trowel", "mallet",
        "socket set", "utility knife", "stapler", "vise", "awl", "hand plane",
        "lathe", "anvil", "pruning shears", "caliper", "workbench", "sandpaper",
    ),
    "vehicle": (
        "motorcycle", "scooter", "subway", "tram", "taxi", "ferry",
        "canoe", "kayak", "sailboat", "jet", "glider", "tractor",
        "ambulance", "fire engine", "rocket", "skateboard", "snowmobile", "bulldozer",
        "yacht", "hovercraft", "forklift", "monorail", "pickup", "rickshaw",
    ),
}

SPLIT_SLICES = {
    "prospective_confirmation": (0, 12),
    "prospective_heldout": (12, 18),
    "sealed": (18, 24),
}

SPLIT_SURFACES = {
    "prospective_confirmation": tuple(range(0, 12)),
    "prospective_heldout": tuple(range(12, 24)),
    "sealed": tuple(range(24, 36)),
}

SURFACE_TEMPLATES = (
    "Complete an ordinary-use fact about {object}: it is commonly",
    "Using everyday knowledge, {object} is commonly",
    "A typical statement about {object} is that it is commonly",
    "The ordinary-use completion for {object} is: it is commonly",
    "Finish this factual statement about {object}: it is commonly",
    "In ordinary situations, {object} is commonly",
    "A common practical fact about {object} is that it is commonly",
    "Complete the everyday statement: {object} is commonly",
    "For {object}, the most ordinary continuation is: it is commonly",
    "Supply a factual continuation about {object}: it is commonly",
    "An accurate everyday completion says {object} is commonly",
    "Ordinary knowledge about {object}: it is commonly",
    "Complete one practical fact for {object}: it is commonly",
    "From common experience, {object} is commonly",
    "The requested ordinary fact about {object} ends: it is commonly",
    "Continue accurately: in everyday life, {object} is commonly",
    "A short factual phrase about {object} is: it is commonly",
    "Complete without adding a new topic: {object} is commonly",
    "Everyday-use record for {object}: it is commonly",
    "The natural factual continuation about {object} is: it is commonly",
    "Provide one ordinary completion: {object} is commonly",
    "A broadly correct everyday statement says {object} is commonly",
    "Finish the common-use entry for {object}: it is commonly",
    "One unremarkable fact about {object} is that it is commonly",
    "Complete this everyday fact concerning {object}: it is commonly",
    "According to ordinary knowledge, {object} is commonly",
    "The factual continuation requested for {object} is: it is commonly",
    "In a normal context, {object} is commonly",
    "Supply the ordinary continuation: {object} is commonly",
    "A concise everyday statement about {object}: it is commonly",
    "Complete the practical knowledge slot for {object}: it is commonly",
    "The ordinary statement to finish is: {object} is commonly",
    "Use a normal factual continuation for {object}: it is commonly",
    "One common observation about {object} is that it is commonly",
    "Continue the everyday description of {object}: it is commonly",
    "For the item {object}, complete the fact: it is commonly",
)

FOOD_CONTINUATIONS = {
    "eaten": " eaten as food.",
    "meal": " prepared for meals.",
    "consumed": " consumed by people.",
    "diet": " included in a diet.",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")


def objects_for(split: str) -> list[dict[str, str]]:
    start, stop = SPLIT_SLICES[split]
    return [
        {"object_id": slug(label), "object_label": label, "semantic_group": group}
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
                raw_prompt = SURFACE_TEMPLATES[surface_id].format(
                    object=item["object_label"]
                )
                prompt_counts = {}
                candidate_ids = {}
                for model, tokenizer in tokenizers.items():
                    rendered = render_chat(tokenizer, model, raw_prompt)
                    prompt_counts[model] = len(
                        tokenizer(rendered, add_special_tokens=True)["input_ids"]
                    )
                    candidate_ids[model] = {
                        name: [
                            int(value)
                            for value in tokenizer(text, add_special_tokens=False)[
                                "input_ids"
                            ]
                        ]
                        for name, text in FOOD_CONTINUATIONS.items()
                    }
                destination.append(
                    {
                        "schema_version": "phase589_food_attribute_case.v1",
                        "phase_id": PHASE,
                        "created_at": now(),
                        "case_id": f"phase589_{split}_{item['object_id']}_surface{surface_id:02d}",
                        "split": split,
                        **item,
                        "surface_id": surface_id,
                        "raw_prompt": raw_prompt,
                        "continuations": FOOD_CONTINUATIONS,
                        "candidate_token_ids_by_model": candidate_ids,
                        "prompt_token_count_by_model": prompt_counts,
                        "continuation_fragment_in_prompt": any(
                            text.strip(" .").casefold() in raw_prompt.casefold()
                            for text in FOOD_CONTINUATIONS.values()
                        ),
                        "category_label_in_prompt": bool(
                            re.search(
                                r"(?<!\w)(fruit|vegetable|plant|tool|vehicle|edible|food)(?!\w)",
                                raw_prompt,
                                re.I,
                            )
                        ),
                        "candidate_continuations_inserted_into_model_input": False,
                        "prospective_after_phase588": True,
                        "observer_only": True,
                        "causal": False,
                        "sealed": split == SEALED_SPLIT,
                    }
                )
    return open_rows, sealed_rows


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = open_rows + sealed_rows
    expected = {"prospective_confirmation": 720, "prospective_heldout": 360, "sealed": 360}
    prior_ids = {
        item["object_id"]
        for group in prior_objects.OBJECT_GROUPS.values()
        for item in group
    }
    current_ids = {row["object_id"] for row in rows}
    audit = {
        "schema_version": "phase589_food_attribute_static_audit.v1",
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
        "prior_object_overlap_count": len(current_ids & prior_ids),
        "continuation_fragment_in_prompt_count": sum(
            row["continuation_fragment_in_prompt"] for row in rows
        ),
        "category_label_in_prompt_count": sum(row["category_label_in_prompt"] for row in rows),
        "empty_candidate_tokenization_count": sum(
            not token_ids
            for row in rows
            for candidates in row["candidate_token_ids_by_model"].values()
            for token_ids in candidates.values()
        ),
        "max_prompt_token_count": max(
            count for row in rows for count in row["prompt_token_count_by_model"].values()
        ),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
    }
    audit["valid"] = bool(
        len(rows) == 1440
        and len(open_rows) == 1080
        and len(sealed_rows) == 360
        and audit["case_count_by_split"] == expected
        and audit["max_prompt_token_count"] <= 128
        and all(
            audit[key] == 0
            for key in (
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
                "prior_object_overlap_count",
                "continuation_fragment_in_prompt_count",
                "category_label_in_prompt_count",
                "empty_candidate_tokenization_count",
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
        SEALED_COMMITMENT_PATH,
        {
            "schema_version": "phase589_sealed_commitment.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "sealed_case_count": len(sealed_rows),
            "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
            "sealed_split_read_for_analysis": False,
        },
    )
    write_json(AUDIT_PATH, audit)
    frozen = {
        "schema_version": "phase589_food_attribute_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Prospective food-attribute relative object response",
        "source_hypothesis": "Phase588 post-hoc food-use ordering only",
        "models_in_required_execution_order": list(MODELS),
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "noop_repeats": list(NOOP_REPEATS),
        "food_continuation_variants": FOOD_CONTINUATIONS,
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "gate": {
            "minimum_edible_vs_nonedible_auc_each_variant": MIN_EDIBLE_VS_NONEDIBLE_AUC,
            "minimum_edible_vs_nonfood_plant_auc_each_variant": MIN_EDIBLE_VS_NONFOOD_PLANT_AUC,
            "minimum_fruit_vs_artifact_auc_each_variant": MIN_EDIBLE_GROUP_VS_ARTIFACT_AUC,
            "minimum_edible_nonfruit_vs_artifact_auc_each_variant": MIN_EDIBLE_GROUP_VS_ARTIFACT_AUC,
            "minimum_surface_auc": MIN_SURFACE_AUC,
            "minimum_qualified_surfaces_each_variant": MIN_QUALIFIED_SURFACES,
            "both_open_splits_must_pass": True,
        },
        "evidence_policy": {
            "prospective_independent_objects_and_surfaces": True,
            "external_observer_not_natural_generation": True,
            "external_observer_not_internal_structure": True,
            "external_observer_not_causal_evidence": True,
            "may_authorize_open_hidden_response_capture": True,
            "may_not_authorize_causal_intervention": True,
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
