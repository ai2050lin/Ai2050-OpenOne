#!/usr/bin/env python3
"""Freeze counterbalanced natural continuations for Phase587 behavior scoring."""

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

import phase585_object_swap_protocol as objects


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase587"
MODELS = objects.MODELS
OPEN_SPLITS = objects.OPEN_SPLITS
SEALED_SPLIT = objects.SEALED_SPLIT
SPLITS = OPEN_SPLITS + (SEALED_SPLIT,)
RELATIONS = objects.RELATIONS
NOOP_REPEATS = ("score1", "score2")
FIXED_BATCH_SIZE = 16
MIN_TARGET_WIN_RATE = 0.85
MIN_GROUP_TARGET_WIN_RATE = 0.80
MIN_STABLE_SURFACES_PER_OBJECT = 6
MIN_QUALIFIED_BY_SPLIT_GROUP = objects.MIN_QUALIFIED_BY_SPLIT_GROUP
MAX_REPEAT_SCORE_DELTA = 1e-6

OUT_DIR = ROOT / "tests/gpt5/result/phase587_counterbalanced_continuation"
OPEN_CASES_PATH = OUT_DIR / "phase587_open_cases.jsonl.gz"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase587_sealed_cases.jsonl.gz"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase587_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase587_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase587_static_audit.json"


CONTINUATIONS = {
    "ordinary_origin": {
        "natural_growth": " natural growth as part of a plant.",
        "human_manufacture": " manufacturing and assembly by people.",
    },
    "primary_function": {
        "food_use": " being eaten or prepared as food.",
        "manual_task": " helping people perform a practical manual task.",
        "transport_use": " transporting people or goods between places.",
    },
}


TARGET_CLASS_BY_GROUP = {
    "ordinary_origin": {
        "fruit": "natural_growth",
        "near_food_plant": "natural_growth",
        "tool": "human_manufacture",
        "vehicle": "human_manufacture",
    },
    "primary_function": {
        "fruit": "food_use",
        "near_food_plant": "food_use",
        "tool": "manual_task",
        "vehicle": "transport_use",
    },
}


SURFACE_TEMPLATES: tuple[str, ...] = (
    "Complete one ordinary fact about {object}. Its {field} is best described by",
    "Using common real-world knowledge, the {field} of {object} involves",
    "Factual completion — object: {object}; requested aspect: {field}; continuation:",
    "The most plausible short continuation about the {field} of {object} is",
    "Ordinary knowledge entry for {object}. Complete the {field} with",
    "For the everyday item {object}, its {field} is characterized by",
    "Complete this factual slot: {object} / {field} /",
    "A correct ordinary statement about the {field} of {object} ends with",
    "Knowledge completion: the {field} normally associated with {object} is",
    "For {object}, complete a common fact concerning its {field}:",
    "Subject {object}; aspect {field}. The natural factual continuation is",
    "Complete without changing the requested aspect: {field} of {object} means",
    "In ordinary life, the {field} of {object} can be summarized as",
    "Factual record completion for {object}: {field} =",
    "Use everyday knowledge to continue the {field} statement for {object}:",
    "The requested fact is the {field} of {object}. Complete it with",
    "A short accurate continuation for {object} under {field} is",
    "Complete the ordinary {field} fact about {object} using",
    "Object {object}; requested field {field}; most plausible continuation:",
    "Common-knowledge completion of {field}({object}):",
    "Finish the factual phrase about the {field} of {object} with",
    "For the subject {object}, an accurate {field} continuation is",
    "Ordinary fact slot — {object}, {field}: completion",
    "Supply the most plausible continuation for {object}'s {field}:",
)


FIELD_PHRASES = {
    "ordinary_origin": (
        "broad way it comes into existence",
        "ordinary source process",
        "usual origin process",
        "broad production path",
    ),
    "primary_function": (
        "main practical purpose",
        "ordinary use",
        "typical role",
        "common function",
    ),
}


SPLIT_SURFACES = objects.SPLIT_SURFACES


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_for(
    item: dict[str, Any], relation: str, split: str, surface_id: int
) -> tuple[str, str]:
    if surface_id not in SPLIT_SURFACES[split]:
        raise ValueError(f"Surface {surface_id} is not frozen for {split}")
    field = FIELD_PHRASES[relation][surface_id % len(FIELD_PHRASES[relation])]
    return SURFACE_TEMPLATES[surface_id].format(object=item["label"], field=field), field


def materialize_row(
    tokenizers: dict[str, Any],
    item: dict[str, Any],
    relation: str,
    split: str,
    surface_id: int,
) -> dict[str, Any]:
    raw_prompt, field = prompt_for(item, relation, split, surface_id)
    target_class = TARGET_CLASS_BY_GROUP[relation][item["semantic_group"]]
    prompt_counts = {}
    candidate_token_ids = {}
    for model, tokenizer in tokenizers.items():
        rendered = render_chat(tokenizer, model, raw_prompt)
        prompt_counts[model] = len(tokenizer(rendered, add_special_tokens=True)["input_ids"])
        candidate_token_ids[model] = {
            key: [
                int(token)
                for token in tokenizer(value, add_special_tokens=False)["input_ids"]
            ]
            for key, value in CONTINUATIONS[relation].items()
        }
    prompt_folded = raw_prompt.casefold()
    return {
        "schema_version": "phase587_counterbalanced_continuation_case.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "case_id": f"phase587_{split}_{item['object_id']}_{relation}_surface{surface_id:02d}",
        "split": split,
        "object_group": objects.SPLIT_GROUP[split],
        "object_id": item["object_id"],
        "object_label": item["label"],
        "semantic_group": item["semantic_group"],
        "relation": relation,
        "surface_id": surface_id,
        "field_phrase": field,
        "raw_prompt": raw_prompt,
        "target_continuation_class": target_class,
        "continuations": CONTINUATIONS[relation],
        "candidate_token_ids_by_model": candidate_token_ids,
        "prompt_token_count_by_model": prompt_counts,
        "continuation_fragment_in_prompt": any(
            fragment.strip(" .").casefold() in prompt_folded
            for fragment in CONTINUATIONS[relation].values()
        ),
        "category_label_in_prompt": bool(
            re.search(r"(?<!\w)(fruit|vegetable|tool|vehicle)(?!\w)", raw_prompt, re.I)
        ),
        "candidate_continuations_inserted_into_model_input": False,
        "counterbalanced_across_object_groups": True,
        "natural_behavior_observer_only": True,
        "causal": False,
        "sealed": split == SEALED_SPLIT,
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        destination = sealed_rows if split == SEALED_SPLIT else open_rows
        for item in objects.OBJECT_GROUPS[objects.SPLIT_GROUP[split]]:
            for relation in RELATIONS:
                for surface_id in SPLIT_SURFACES[split]:
                    destination.append(
                        materialize_row(tokenizers, item, relation, split, surface_id)
                    )
    return open_rows, sealed_rows


def validate(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = open_rows + sealed_rows
    expected = {
        "behavior_discovery": 400,
        "behavior_confirmation": 400,
        "heldout_objects": 160,
        "sealed": 160,
    }
    audit = {
        "schema_version": "phase587_counterbalanced_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "case_count_by_split": dict(Counter(row["split"] for row in rows)),
        "expected_case_count_by_split": expected,
        "relation_count": dict(Counter(row["relation"] for row in rows)),
        "target_class_count_by_split_relation": {
            f"{split}:{relation}": dict(
                Counter(
                    row["target_continuation_class"]
                    for row in rows
                    if row["split"] == split and row["relation"] == relation
                )
            )
            for split in SPLITS
            for relation in RELATIONS
        },
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_split_prompt_count": len(rows)
        - len({(row["split"], row["raw_prompt"]) for row in rows}),
        "continuation_fragment_in_prompt_count": sum(
            row["continuation_fragment_in_prompt"] for row in rows
        ),
        "category_label_in_prompt_count": sum(row["category_label_in_prompt"] for row in rows),
        "empty_candidate_tokenization_count": sum(
            not token_ids
            for row in rows
            for model_candidates in row["candidate_token_ids_by_model"].values()
            for token_ids in model_candidates.values()
        ),
        "max_prompt_token_count": max(
            count
            for row in rows
            for count in row["prompt_token_count_by_model"].values()
        ),
        "open_contains_sealed_count": sum(row["sealed"] for row in open_rows),
        "sealed_flag_missing_count": sum(not row["sealed"] for row in sealed_rows),
    }
    audit["valid"] = bool(
        len(rows) == 1120
        and len(open_rows) == 960
        and len(sealed_rows) == 160
        and audit["case_count_by_split"] == expected
        and audit["max_prompt_token_count"] <= 128
        and all(
            audit[key] == 0
            for key in (
                "duplicate_case_id_count",
                "duplicate_split_prompt_count",
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
            "schema_version": "phase587_sealed_commitment.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "sealed_case_count": len(sealed_rows),
            "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
            "sealed_split_read_for_analysis": False,
        },
    )
    write_json(AUDIT_PATH, audit)
    frozen = {
        "schema_version": "phase587_counterbalanced_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Counterbalanced natural continuation behavior observer",
        "models_in_required_execution_order": list(MODELS),
        "open_splits": list(OPEN_SPLITS),
        "sealed_split": SEALED_SPLIT,
        "relations": list(RELATIONS),
        "noop_repeats": list(NOOP_REPEATS),
        "fixed_batch_size": FIXED_BATCH_SIZE,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "score_definition": {
            "score": "mean conditional token log-probability of each external full continuation",
            "margin": "target continuation score minus maximum foil continuation score",
            "candidate_continuations_inserted_into_model_input": False,
            "same_continuation_bank_reused_across_all_objects_in_a_relation": True,
            "continuations_serve_as_targets_and_foils_in_different_object_groups": True,
        },
        "behavior_gate": {
            "minimum_target_win_rate_each_split_relation": MIN_TARGET_WIN_RATE,
            "minimum_target_win_rate_each_semantic_group": MIN_GROUP_TARGET_WIN_RATE,
            "minimum_stable_surfaces_per_object": MIN_STABLE_SURFACES_PER_OBJECT,
            "minimum_qualified_objects_by_split_group": MIN_QUALIFIED_BY_SPLIT_GROUP,
            "maximum_repeat_score_delta": MAX_REPEAT_SCORE_DELTA,
            "all_three_open_splits_must_pass": True,
        },
        "evidence_policy": {
            "external_observer_not_natural_generation": True,
            "external_observer_not_internal_structure": True,
            "external_observer_not_causal_evidence": True,
            "may_authorize_open_full_hidden_response_capture": True,
            "sealed_split_read": False,
            "strict_mechanism_closure_claimed": False,
        },
        "open_cases_path": str(OPEN_CASES_PATH.relative_to(ROOT)),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_path": str(SEALED_COMMITMENT_PATH.relative_to(ROOT)),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "static_audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "static_audit_sha256": sha256_file(AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, frozen)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register(), ensure_ascii=False, indent=2))
