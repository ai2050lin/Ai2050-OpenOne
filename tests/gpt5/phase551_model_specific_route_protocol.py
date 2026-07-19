#!/usr/bin/env python3
"""Freeze model-specific same-answer route scaffold calibration for Phase551."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase548_shared_attention_compute_protocol import (  # noqa: E402
    ATTRIBUTE_WORDS,
    CATEGORY_WORDS,
    MODELS,
    SYLLABLE_A,
    SYLLABLE_B,
    render_chat,
    three_distinct,
    token_edit_distance,
)


PHASE = "Phase551"
SCHEMA_VERSION = "phase551_model_specific_route_contract.v1"
OUT_DIR = ROOT / "tests/gpt5/result/phase551_model_specific_route"
CALIBRATION_CASES_PATH = OUT_DIR / "phase551_calibration_cases.jsonl"
CALIBRATION_AUDIT_PATH = OUT_DIR / "phase551_calibration_static_audit.json"
PROTOCOL_PATH = OUT_DIR / "phase551_frozen_protocol.json"
VALIDATION_PROTOCOL_PATH = OUT_DIR / "phase551_validation_protocol.json"
FROZEN_SCAFFOLDS_PATH = OUT_DIR / "phase551_frozen_scaffolds.json"
VALIDATION_CASES_PATH = OUT_DIR / "phase551_validation_cases.jsonl"
VALIDATION_AUDIT_PATH = OUT_DIR / "phase551_validation_static_audit.json"
MECHANISMS = ("category", "negated_attribute", "transitive_order", "subject_verb_agreement")
FAMILIES = {
    "category": "content_knowledge",
    "negated_attribute": "logical_reasoning",
    "transitive_order": "relational_reasoning",
    "subject_verb_agreement": "syntax_structure",
}
SCAFFOLDS = ("prose", "explicit_candidates", "table", "compact", "guarded")
CELLS = ("route0_answer_a", "route1_answer_a", "route0_answer_b", "route1_answer_b")
SPLITS = ("discovery", "independent_confirmation")
CALIBRATION_WORLDS = 24
VALIDATION_WORLDS = 73
NOUNS = (
    "lantern", "harbor", "violin", "tablet", "garden", "window", "engine", "planet",
    "helmet", "camera", "bridge", "cabin", "pencil", "island", "bucket", "pillow",
    "museum", "ticket", "castle", "bottle", "rocket", "basket", "meadow", "tunnel",
)


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def entity_word(index: int, namespace: str) -> str:
    namespace_code = {"calibration": "c", "discovery": "d", "independent_confirmation": "v"}[namespace]
    a = SYLLABLE_A[index % len(SYLLABLE_A)]
    b = SYLLABLE_B[(index // len(SYLLABLE_A)) % len(SYLLABLE_B)]
    c = SYLLABLE_A[(index // (len(SYLLABLE_A) * len(SYLLABLE_B))) % len(SYLLABLE_A)]
    d = SYLLABLE_B[(index * 7 + 11) % len(SYLLABLE_B)]
    return f"Z{namespace_code}{a}{b}{c}{d}"


def cell_factors(cell: str) -> tuple[int, str]:
    return (0 if cell.startswith("route0") else 1, "a" if cell.endswith("answer_a") else "b")


def category_prompt(index: int, namespace: str, cell: str, scaffold: str) -> dict[str, Any]:
    route, answer_role = cell_factors(cell)
    entity_a = entity_word(index * 3, namespace)
    entity_b = entity_word(index * 3 + 1, namespace)
    answer_a, answer_b, _ = three_distinct(CATEGORY_WORDS, index + 71)
    category_a, category_b = (answer_a, answer_b) if route == 0 else (answer_b, answer_a)
    selected = entity_a if (route == 0) == (answer_role == "a") else entity_b
    target = answer_a if answer_role == "a" else answer_b
    if scaffold == "prose":
        context = f"{entity_a} belongs to category {category_a}. {entity_b} belongs to category {category_b}. Selected entry: {selected}."
    elif scaffold == "explicit_candidates":
        context = f"Entry one is {entity_a}, category {category_a}. Entry two is {entity_b}, category {category_b}. The selected entry is {selected}. Allowed category words: {answer_a}, {answer_b}."
    elif scaffold == "table":
        context = f"Category table: [{entity_a} | {category_a}] [{entity_b} | {category_b}]. Selected={selected}."
    elif scaffold == "compact":
        context = f"{entity_a}=>{category_a}; {entity_b}=>{category_b}; selected=>{selected}."
    else:
        context = f"Local facts only: category({entity_a})={category_a}; category({entity_b})={category_b}. Lookup category({selected}). Candidate category words are {answer_a} and {answer_b}."
    return {
        "context": context,
        "question": "Which category word is assigned to the selected entry?",
        "instruction": f"Reply with exactly one category word from [{answer_a}, {answer_b}]. Never reply with an entity name or an explanation.",
        "target": target,
        "distractors": [answer_b if answer_role == "a" else answer_a],
        "all_candidates": [answer_a, answer_b],
        "entity_key": f"{entity_a}:{entity_b}",
    }


def negation_prompt(index: int, namespace: str, cell: str, scaffold: str) -> dict[str, Any]:
    route, answer_role = cell_factors(cell)
    entity = entity_word(index * 3, namespace)
    answer_a, answer_b, _ = three_distinct(ATTRIBUTE_WORDS, index + 113)
    status_a, status_b = (("applicable", "inapplicable") if route == 0 else ("inapplicable", "applicable"))
    requested = status_a if answer_role == "a" else status_b
    target = answer_a if answer_role == "a" else answer_b
    if scaffold == "prose":
        context = f"For {entity}, attribute {answer_a} is {status_a}, while attribute {answer_b} is {status_b}."
    elif scaffold == "explicit_candidates":
        context = f"Object: {entity}. Attribute one: {answer_a}; status: {status_a}. Attribute two: {answer_b}; status: {status_b}. Allowed attribute words: {answer_a}, {answer_b}."
    elif scaffold == "table":
        context = f"Attribute status table for {entity}: [{answer_a} | {status_a}] [{answer_b} | {status_b}]."
    elif scaffold == "compact":
        context = f"{entity}: {answer_a}={status_a}; {answer_b}={status_b}."
    else:
        context = f"Use this record only: status({entity},{answer_a})={status_a}; status({entity},{answer_b})={status_b}. Candidate attributes are {answer_a} and {answer_b}."
    return {
        "context": context,
        "question": f"Which attribute has status {requested}?",
        "instruction": f"Reply with exactly one attribute word from [{answer_a}, {answer_b}] and no explanation.",
        "target": target,
        "distractors": [answer_b if answer_role == "a" else answer_a],
        "all_candidates": [answer_a, answer_b],
        "entity_key": entity,
    }


def order_prompt(index: int, namespace: str, cell: str, scaffold: str) -> dict[str, Any]:
    route, answer_role = cell_factors(cell)
    first = entity_word(index * 3, namespace)
    middle = entity_word(index * 3 + 1, namespace)
    last = entity_word(index * 3 + 2, namespace)
    if route == 0:
        facts = f"{first} comes before {middle}. {middle} comes before {last}."
        compact = f"{first} < {middle} < {last}"
    else:
        facts = f"{last} comes after {middle}. {middle} comes after {first}."
        compact = f"{last} > {middle} > {first}"
    target = first if answer_role == "a" else last
    requested = "first" if answer_role == "a" else "last"
    if scaffold == "prose":
        context = facts
    elif scaffold == "explicit_candidates":
        context = f"Ordering facts: {facts} Endpoint candidates: {first}, {last}."
    elif scaffold == "table":
        context = f"Order record: [{facts}]"
    elif scaffold == "compact":
        context = f"Order: {compact}."
    else:
        context = f"Use only this chain: {facts} The middle item is {middle}; endpoint candidates are {first} and {last}."
    return {
        "context": context,
        "question": f"Which endpoint item comes {requested}?",
        "instruction": f"Reply with exactly one endpoint name from [{first}, {last}] and no explanation.",
        "target": target,
        "distractors": [last if answer_role == "a" else first],
        "all_candidates": [first, last],
        "entity_key": f"{first}:{middle}:{last}",
    }


def syntax_prompt(index: int, namespace: str, cell: str, scaffold: str) -> dict[str, Any]:
    route, answer_role = cell_factors(cell)
    noun = NOUNS[index % len(NOUNS)]
    subject = f"the {noun}" if answer_role == "a" else f"the {noun}s"
    attractor = f"the {NOUNS[(index + 7) % len(NOUNS)]}{'s' if answer_role == 'a' else ''}"
    sentence = (
        f"{subject} __ ready"
        if route == 0
        else f"Near {attractor}, {subject} __ ready"
    )
    target = "is" if answer_role == "a" else "are"
    distractor = "are" if answer_role == "a" else "is"
    record = "" if namespace == "calibration" else f"Record {entity_word(index * 3, namespace)}. "
    if scaffold == "prose":
        context = f"{record}Complete the sentence: {sentence}."
    elif scaffold == "explicit_candidates":
        context = f"{record}Sentence with one blank: {sentence}. Allowed verbs: is, are. The verb must agree with the grammatical subject {subject}."
    elif scaffold == "table":
        context = f"{record}Grammar record: [subject | {subject}] [sentence | {sentence}] [verb options | is, are]."
    elif scaffold == "compact":
        context = f"{record}SUBJECT={subject}; SENTENCE={sentence}; OPTIONS=is|are."
    else:
        context = f"{record}Use grammatical subject number, not the nearby noun: subject={subject}; sentence={sentence}; candidates=is,are."
    return {
        "context": context,
        "question": "Which verb correctly fills the blank?",
        "instruction": "Reply with exactly one word: is or are. Do not explain.",
        "target": target,
        "distractors": [distractor],
        "all_candidates": ["is", "are"],
        "entity_key": f"{namespace}:{noun}:{index}",
    }


def case_spec(mechanism: str, namespace: str, index: int, cell: str, scaffold: str) -> dict[str, Any]:
    if mechanism == "category":
        spec = category_prompt(index, namespace, cell, scaffold)
    elif mechanism == "negated_attribute":
        spec = negation_prompt(index, namespace, cell, scaffold)
    elif mechanism == "transitive_order":
        spec = order_prompt(index, namespace, cell, scaffold)
    elif mechanism == "subject_verb_agreement":
        spec = syntax_prompt(index, namespace, cell, scaffold)
    else:
        raise KeyError(mechanism)
    raw_prompt = (
        f"Context: {spec['context']}\nQuestion: {spec['question']}\nInstruction: {spec['instruction']}"
    )
    return {
        **spec,
        "raw_prompt": raw_prompt,
        "source_fragment": f"Context: {spec['context']}",
        "query_fragment": f"Question: {spec['question']}",
    }


def row_from_spec(
    tokenizer: Any, model: str, mechanism: str, split: str, world_index: int,
    cell: str, scaffold: str, stage: str,
) -> tuple[dict[str, Any], list[int]]:
    namespace = "calibration" if stage == "calibration" else split
    offset = 0 if stage == "calibration" else (2000 if split == "discovery" else 4000)
    spec = case_spec(mechanism, namespace, world_index + offset, cell, scaffold)
    prompt = render_chat(tokenizer, model, spec["raw_prompt"])
    ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
    route, answer_role = cell_factors(cell)
    anchor_id = f"phase551_{stage}_{model}_{mechanism}_{split}_{scaffold}_{world_index:03d}"
    return ({
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "stage": stage,
        "case_id": f"{anchor_id}_{scaffold}_{cell}",
        "anchor_id": anchor_id,
        "model": model,
        "family_id": FAMILIES[mechanism],
        "mechanism_id": mechanism,
        "split": split,
        "world_index": world_index,
        "scaffold_id": scaffold,
        "factorial_cell": cell,
        "route_factor": route,
        "answer_factor": answer_role,
        "raw_prompt": spec["raw_prompt"],
        "prompt": prompt,
        "source_fragment": spec["source_fragment"],
        "query_fragment": spec["query_fragment"],
        "target": spec["target"],
        "target_aliases": [spec["target"]],
        "distractors": spec["distractors"],
        "all_candidates": spec["all_candidates"],
        "strict_expected": spec["target"],
        "strict_kind": "plain",
        "entity_key": spec["entity_key"],
        "prompt_token_count": len(ids),
        "semantic_event_is_natural_answer": True,
        "arbitrary_label_output": False,
        "observer_only": True,
        "compute_edge": False,
        "causal": False,
        "single_neuron": False,
        "sealed": False,
    }, ids)


def build_calibration_rows() -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        for mechanism in MECHANISMS:
            for scaffold in SCAFFOLDS:
                for world_index in range(CALIBRATION_WORLDS):
                    anchor = []
                    for cell in CELLS:
                        anchor.append(row_from_spec(
                            tokenizer, model, mechanism, "calibration", world_index,
                            cell, scaffold, "calibration",
                        ))
                    reference_ids = anchor[0][1]
                    for row, ids in anchor:
                        row["token_edit_distance_from_route0_answer_a"] = token_edit_distance(reference_ids, ids)
                        row["token_length_delta_from_route0_answer_a"] = len(ids) - len(reference_ids)
                        rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]], expected: int, rows_per_anchor: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["anchor_id"]].append(row)
    relation_errors = 0
    for group in groups.values():
        by_cell = {row["factorial_cell"]: row for row in group}
        if set(by_cell) != set(CELLS):
            relation_errors += 1
            continue
        if not (
            by_cell["route0_answer_a"]["target"] == by_cell["route1_answer_a"]["target"]
            and by_cell["route0_answer_b"]["target"] == by_cell["route1_answer_b"]["target"]
            and by_cell["route0_answer_a"]["target"] != by_cell["route0_answer_b"]["target"]
        ):
            relation_errors += 1
    audit = {
        "registered_case_count": len(rows),
        "expected_case_count": expected,
        "model_case_counts": dict(Counter(row["model"] for row in rows)),
        "anchor_count": len(groups),
        "rows_per_anchor": sorted({len(group) for group in groups.values()}),
        "factorial_relation_error_count": relation_errors,
        "duplicate_case_id_count": len(rows) - len({row["case_id"] for row in rows}),
        "duplicate_prompt_count": len(rows) - len({(row["model"], row["prompt"]) for row in rows}),
        "prompt_token_count_range_by_model": {
            model: [
                min(row["prompt_token_count"] for row in rows if row["model"] == model),
                max(row["prompt_token_count"] for row in rows if row["model"] == model),
            ] for model in MODELS if any(row["model"] == model for row in rows)
        },
        "sealed_row_count": sum(bool(row["sealed"]) for row in rows),
    }
    audit["valid"] = (
        len(rows) == expected
        and audit["rows_per_anchor"] == ([rows_per_anchor] if expected else [])
        and max((maximum for _minimum, maximum in audit["prompt_token_count_range_by_model"].values()), default=0) <= 512
        and all(audit[key] == 0 for key in (
            "factorial_relation_error_count", "duplicate_case_id_count",
            "duplicate_prompt_count", "sealed_row_count",
        ))
    )
    return audit


def register_calibration() -> dict[str, Any]:
    rows = build_calibration_rows()
    expected = len(MODELS) * len(MECHANISMS) * len(SCAFFOLDS) * CALIBRATION_WORLDS * len(CELLS)
    audit = validate_rows(rows, expected, len(CELLS))
    audit.update({
        "schema_version": "phase551_calibration_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "status": "static_pass_no_model_run" if audit["valid"] else "static_fail",
    })
    write_jsonl(CALIBRATION_CASES_PATH, rows)
    write_json(CALIBRATION_AUDIT_PATH, audit)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "title": "Model-specific same-answer route contracts before full-layer observation",
        "models_in_required_execution_order": list(MODELS),
        "mechanisms": list(MECHANISMS),
        "scaffolds": list(SCAFFOLDS),
        "factorial_cells": list(CELLS),
        "calibration_worlds_per_scaffold_mechanism_model": CALIBRATION_WORLDS,
        "validation_worlds_per_split": VALIDATION_WORLDS,
        "calibration_selection_gate": {
            "all_four_correct_anchor_min": 22,
            "each_cell_correct_min": 23,
            "unrecoverable_anchor_max": 1,
            "selection_tie_break": "all_anchor_count,min_cell_count,mean_prompt_length,scaffold_order",
        },
        "validation_behavior_gate": {
            "all_four_cells_correct_lcb95_min": 0.90,
            "unrecoverable_anchor_ucb95_max": 0.05,
            "both_independent_splits_required": True,
        },
        "observer_gate": {
            "full_layer": True,
            "components": ["layer_input", "attention_output", "mlp_output", "layer_output"],
            "roles": ["source", "query", "current"],
            "prompt_end_only": True,
            "head_channel_neuron_search": False,
            "intervention_authorized": False,
        },
        "evidence_boundaries": {
            "calibration_is_mechanism_evidence": False,
            "validation_behavior_is_physical_evidence": False,
            "full_layer_observer_is_compute_edge": False,
            "new_sealed_split_read": False,
        },
        "calibration_cases_path": str(CALIBRATION_CASES_PATH.relative_to(ROOT)),
        "calibration_cases_sha256": sha256_file(CALIBRATION_CASES_PATH),
        "calibration_audit_path": str(CALIBRATION_AUDIT_PATH.relative_to(ROOT)),
        "calibration_audit_sha256": sha256_file(CALIBRATION_AUDIT_PATH),
    }
    write_json(PROTOCOL_PATH, protocol)
    if not audit["valid"]:
        raise SystemExit(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    print(json.dumps(register_calibration(), ensure_ascii=False, indent=2))
