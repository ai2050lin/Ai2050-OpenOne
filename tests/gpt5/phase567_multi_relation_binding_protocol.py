#!/usr/bin/env python3
"""Freeze the Phase567 three-object, three-value, two-relation contract.

Within each counterfactual triplet, the object lexicon, value lexicon, query,
surface, fact order, and token multiset are fixed. Only the assignment for the
queried relation is rotated. The other relation remains unchanged.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402


PHASE = "Phase567"
SCHEMA_VERSION = "phase567_multi_relation_binding.v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLIT_WORLD_COUNTS = {
    "behavior_discovery": 48,
    "behavior_confirmation": 48,
    "role_discovery": 32,
    "role_confirmation": 32,
    "unseen_recombination": 48,
    "sealed": 48,
}
SPLITS = tuple(SPLIT_WORLD_COUNTS)
OPEN_SPLITS = tuple(split for split in SPLITS if split != "sealed")
BINDINGS = (0, 1, 2)
QUERY_OBJECTS = (0, 1, 2)
QUERY_RELATIONS = ("surface", "marker")
SURFACES = (0, 1, 2)
FACT_ORDERS = (0, 1)
ROWS_PER_WORLD = (
    len(BINDINGS)
    * len(QUERY_OBJECTS)
    * len(QUERY_RELATIONS)
    * len(SURFACES)
    * len(FACT_ORDERS)
)
CELLS = tuple(
    f"binding{binding}_query{query}_relation{relation}_surface{surface}_order{order}"
    for binding, query, relation, surface, order in itertools.product(
        BINDINGS, QUERY_OBJECTS, QUERY_RELATIONS, SURFACES, FACT_ORDERS
    )
)

CORE_VALUES = (
    "red", "green", "blue", "yellow", "orange", "purple", "black", "white",
    "brown", "pink", "gray", "gold", "cyan", "lime", "beige",
)
HELDOUT_VALUES = (
    "teal", "silver", "violet", "amber", "crimson", "ivory", "navy", "maroon",
    "coral", "indigo", "magenta", "olive",
)
SYLLABLE_A = ("ba", "ce", "di", "fo", "ga", "hu", "ji", "ke", "lu", "mi", "no", "pa", "ri")
SYLLABLE_B = ("lan", "mer", "tin", "vor", "sen", "dak", "pel", "rin", "sol", "wen", "yas", "kor", "zen")

OUT_DIR = ROOT / "tests/gpt5/result/phase567_multi_relation_binding"
OPEN_CASES_PATH = OUT_DIR / "phase567_open_cases.jsonl"
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase567_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase567_sealed_commitment.json"
PUBLIC_WORLD_BANK_PATH = OUT_DIR / "phase567_public_world_bank.json"
PROTOCOL_PATH = OUT_DIR / "phase567_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase567_static_audit.json"


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


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def alpha_code(value: int) -> str:
    if value < 0:
        raise ValueError(value)
    letters = []
    current = value
    while True:
        current, remainder = divmod(current, 26)
        letters.append(chr(ord("a") + remainder))
        if current == 0:
            return "".join(reversed(letters))


def pseudo_word(index: int, split: str) -> str:
    split_offset = 60000 + list(SPLITS).index(split) * 6000
    shifted = index + split_offset
    stem = (
        SYLLABLE_A[shifted % len(SYLLABLE_A)]
        + SYLLABLE_B[(shifted // len(SYLLABLE_A)) % len(SYLLABLE_B)]
        + SYLLABLE_A[(shifted * 11 + 7) % len(SYLLABLE_A)]
    )
    return (stem + alpha_code(shifted)).capitalize()


def normalized_word_multiset(text: str) -> list[str]:
    return sorted(re.findall(r"[A-Za-z]+", text.casefold()))


def distinct_triple(pool: tuple[str, ...], index: int, stride: int) -> tuple[str, str, str]:
    selected: list[str] = []
    cursor = 0
    while len(selected) < 3:
        value = pool[(index * stride + cursor * (stride + 2) + 3) % len(pool)]
        if value not in selected:
            selected.append(value)
        cursor += 1
        if cursor > len(pool) * 3:
            raise RuntimeError("Could not construct a distinct value triple")
    return selected[0], selected[1], selected[2]


def world_spec(split: str, world_index: int) -> dict[str, Any]:
    objects = tuple(pseudo_word(world_index * 3 + offset, split) for offset in range(3))
    use_heldout = (
        split == "unseen_recombination" and world_index >= SPLIT_WORLD_COUNTS[split] // 2
    ) or (
        split == "sealed" and world_index >= SPLIT_WORLD_COUNTS[split] // 2
    )
    pool = HELDOUT_VALUES if use_heldout else CORE_VALUES
    values = distinct_triple(pool, world_index + list(SPLITS).index(split) * 23, 7)
    base_offset = (world_index + list(SPLITS).index(split)) % 3
    surface_map = tuple((object_index + base_offset) % 3 for object_index in range(3))
    marker_map = tuple((value_index + 1) % 3 for value_index in surface_map)
    return {
        "objects": objects,
        "values": values,
        "value_regime": "heldout_value_labels" if use_heldout else "core_value_labels",
        "base_maps": {"surface": surface_map, "marker": marker_map},
    }


def relation_map(world: dict[str, Any], relation: str, binding: int, query_relation: str) -> tuple[int, ...]:
    base = world["base_maps"][relation]
    if relation != query_relation:
        return tuple(base)
    return tuple((value_index + binding) % 3 for value_index in base)


def relation_label(surface: int, relation: str) -> str:
    if surface == 0:
        return f"{relation} color"
    if surface == 1:
        return f"{relation}-color"
    if surface == 2:
        return f"{relation}_color"
    raise ValueError(surface)


def render_fact(surface: int, obj: str, relation: str, value: str) -> str:
    label = relation_label(surface, relation)
    if surface == 0:
        return f"{obj} has {label} {value}"
    if surface == 1:
        return f"{obj} | {label} | {value}"
    if surface == 2:
        return f"{label}({obj}) = {value}"
    raise ValueError(surface)


def render_context(surface: int, fact_records: list[dict[str, Any]]) -> str:
    facts = [record["text"] for record in fact_records]
    if surface == 0:
        return "Temporary dual-color ledger. " + ". ".join(facts) + "."
    if surface == 1:
        return "Temporary dual-color ledger:\n" + "\n".join(facts)
    if surface == 2:
        return "Temporary dual-color ledger: " + "; ".join(facts) + "."
    raise ValueError(surface)


def render_question(surface: int, query_object: str, query_relation: str) -> tuple[str, str]:
    label = relation_label(surface, query_relation)
    if surface == 0:
        return f"According to the ledger, what {label} does {query_object} have?", label
    if surface == 1:
        return f"Look up the {label} recorded for {query_object}.", label
    if surface == 2:
        return f"Return {label}({query_object}).", label
    raise ValueError(surface)


def ordered_fact_records(
    world: dict[str, Any], query_relation: str, binding: int, surface: int, fact_order: int
) -> list[dict[str, Any]]:
    records = []
    for relation in QUERY_RELATIONS:
        mapping = relation_map(world, relation, binding, query_relation)
        for object_index, value_index in enumerate(mapping):
            obj = world["objects"][object_index]
            value = world["values"][value_index]
            records.append({
                "object_index": object_index,
                "relation": relation,
                "value_index": value_index,
                "object": obj,
                "value": value,
                "relation_label": relation_label(surface, relation),
                "text": render_fact(surface, obj, relation, value),
            })
    if fact_order == 0:
        return records
    by_key = {(record["relation"], record["object_index"]): record for record in records}
    reordered = []
    for object_index in (2, 1, 0):
        reordered.append(by_key[("marker", object_index)])
        reordered.append(by_key[("surface", object_index)])
    return reordered


def controlled_case(
    split: str,
    world_index: int,
    binding: int,
    query_index: int,
    query_relation: str,
    surface: int,
    fact_order: int,
) -> dict[str, Any]:
    world = world_spec(split, world_index)
    records = ordered_fact_records(world, query_relation, binding, surface, fact_order)
    context = render_context(surface, records)
    query_object = world["objects"][query_index]
    question, query_relation_label = render_question(surface, query_object, query_relation)
    instruction = (
        "Use only this temporary ledger. Reply with exactly one lowercase color word "
        "and no explanation."
    )
    raw_prompt = f"{context}\nQuestion: {question}\nInstruction: {instruction}"
    target_record = next(
        record for record in records
        if record["relation"] == query_relation and record["object_index"] == query_index
    )
    same_relation_other = [
        record for record in records
        if record["relation"] == query_relation and record["object_index"] != query_index
    ]
    other_relation = [record for record in records if record["relation"] != query_relation]
    target = target_record["value"]
    distractors = [value for value in world["values"] if value != target]
    anchor_id = f"phase567_{split}_{world_index:03d}"
    triplet_id = (
        f"{anchor_id}_query{query_index}_relation{query_relation}_surface{surface}_order{fact_order}"
    )
    relation_maps = {
        relation: list(relation_map(world, relation, binding, query_relation))
        for relation in QUERY_RELATIONS
    }
    return {
        "raw_prompt": raw_prompt,
        "context": context,
        "question": question,
        "instruction": instruction,
        "objects": list(world["objects"]),
        "values": list(world["values"]),
        "value_regime": world["value_regime"],
        "relation_maps": relation_maps,
        "binding": binding,
        "query_object_index": query_index,
        "query_object": query_object,
        "query_relation": query_relation,
        "other_relation": next(value for value in QUERY_RELATIONS if value != query_relation),
        "query_relation_label": query_relation_label,
        "target": target,
        "target_aliases": [target],
        "distractors": distractors,
        "all_candidates": list(world["values"]),
        "surface_id": surface,
        "fact_order": fact_order,
        "anchor_id": anchor_id,
        "triplet_id": triplet_id,
        "factorial_cell": (
            f"binding{binding}_query{query_index}_relation{query_relation}_"
            f"surface{surface}_order{fact_order}"
        ),
        "factorial_cell_without_binding": (
            f"query{query_index}_relation{query_relation}_surface{surface}_order{fact_order}"
        ),
        "fact_token_multiset_key": stable_hash(normalized_word_multiset(" ".join(
            record["text"] for record in records
        ))),
        "prompt_token_multiset_key": stable_hash(normalized_word_multiset(raw_prompt)),
        "fact_records": records,
        "semantic_fragments": {
            "target_fact": target_record["text"],
            "target_fact_object": target_record["object"],
            "target_fact_value": target_record["value"],
            "target_fact_relation": target_record["relation_label"],
            "same_relation_other_facts": [record["text"] for record in same_relation_other],
            "other_relation_facts": [record["text"] for record in other_relation],
            "query_relation": query_relation_label,
            "query_object": query_object,
        },
    }


def materialize_row(model: str, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    prompt = render_chat(tokenizer, model, row["raw_prompt"])
    prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]]
    target_ids = [int(value) for value in tokenizer(row["target"], add_special_tokens=False)["input_ids"]]
    distractor_ids = {
        value: [int(token) for token in tokenizer(value, add_special_tokens=False)["input_ids"]]
        for value in row["distractors"]
    }
    case_id = (
        f"{row['anchor_id']}_{model}_binding{row['binding']}_query{row['query_object_index']}_"
        f"relation{row['query_relation']}_surface{row['surface_id']}_order{row['fact_order']}"
    )
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "target_token_ids": target_ids,
        "distractor_token_ids": distractor_ids,
        "case_id": case_id,
        "split": row["anchor_id"].split("_")[1] + "_" + row["anchor_id"].split("_")[2]
        if "unseen_recombination" not in row["anchor_id"]
        else "unseen_recombination",
        "sealed": "_sealed_" in row["anchor_id"],
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    open_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    public_worlds: list[dict[str, Any]] = []
    for split, world_count in SPLIT_WORLD_COUNTS.items():
        for world_index in range(world_count):
            world = world_spec(split, world_index)
            public_worlds.append({
                "anchor_id": f"phase567_{split}_{world_index:03d}",
                "split": split,
                "object_count": 3,
                "value_count": 3,
                "relations": list(QUERY_RELATIONS),
                "value_regime": world["value_regime"],
                "sealed": split == "sealed",
            })
            semantic_rows = [
                controlled_case(split, world_index, binding, query, relation, surface, order)
                for binding, query, relation, surface, order in itertools.product(
                    BINDINGS, QUERY_OBJECTS, QUERY_RELATIONS, SURFACES, FACT_ORDERS
                )
            ]
            for model in MODELS:
                target = sealed_rows if split == "sealed" else open_rows
                target.extend(materialize_row(model, tokenizers[model], row) for row in semantic_rows)
    return open_rows, sealed_rows, public_worlds


def derive_split(anchor_id: str) -> str:
    for split in SPLITS:
        if anchor_id.startswith(f"phase567_{split}_"):
            return split
    raise ValueError(anchor_id)


def static_audit(open_rows: list[dict[str, Any]], sealed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_rows = open_rows + sealed_rows
    for row in all_rows:
        row["split"] = derive_split(row["anchor_id"])
        row["sealed"] = row["split"] == "sealed"
    failures: list[str] = []
    expected_open = sum(SPLIT_WORLD_COUNTS[split] for split in OPEN_SPLITS) * ROWS_PER_WORLD * len(MODELS)
    expected_sealed = SPLIT_WORLD_COUNTS["sealed"] * ROWS_PER_WORLD * len(MODELS)
    if len(open_rows) != expected_open:
        failures.append(f"open_count:{len(open_rows)}!={expected_open}")
    if len(sealed_rows) != expected_sealed:
        failures.append(f"sealed_count:{len(sealed_rows)}!={expected_sealed}")
    if len({row["case_id"] for row in all_rows}) != len(all_rows):
        failures.append("duplicate_case_id")
    if any(row["sealed"] for row in open_rows) or any(not row["sealed"] for row in sealed_rows):
        failures.append("sealed_partition")

    by_world: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_triplet: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_world[(row["model"], row["anchor_id"])].append(row)
        by_triplet[(row["model"], row["triplet_id"])].append(row)
    world_sizes = sorted({len(rows) for rows in by_world.values()})
    triplet_sizes = sorted({len(rows) for rows in by_triplet.values()})
    if world_sizes != [ROWS_PER_WORLD]:
        failures.append(f"world_sizes:{world_sizes}")
    if triplet_sizes != [3]:
        failures.append(f"triplet_sizes:{triplet_sizes}")

    triplet_invariant_failures = 0
    for rows in by_triplet.values():
        ordered = sorted(rows, key=lambda row: int(row["binding"]))
        if {row["binding"] for row in ordered} != set(BINDINGS):
            triplet_invariant_failures += 1
            continue
        fixed_keys = (
            "objects", "values", "query_object", "query_relation", "surface_id", "fact_order",
            "fact_token_multiset_key", "prompt_token_multiset_key",
        )
        if any(ordered[index][key] != ordered[0][key] for key in fixed_keys for index in (1, 2)):
            triplet_invariant_failures += 1
            continue
        if {row["target"] for row in ordered} != set(ordered[0]["values"]):
            triplet_invariant_failures += 1
            continue
        other_relation = ordered[0]["other_relation"]
        if len({tuple(row["relation_maps"][other_relation]) for row in ordered}) != 1:
            triplet_invariant_failures += 1
            continue
        queried_maps = {tuple(row["relation_maps"][row["query_relation"]]) for row in ordered}
        if len(queried_maps) != 3:
            triplet_invariant_failures += 1
    if triplet_invariant_failures:
        failures.append(f"triplet_invariant_failures:{triplet_invariant_failures}")

    split_objects: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        if row["model"] == "qwen3":
            split_objects[row["split"]].update(row["objects"])
    split_overlap_count = 0
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            split_overlap_count += len(split_objects[left] & split_objects[right])
    if split_overlap_count:
        failures.append(f"split_object_overlap:{split_overlap_count}")

    per_model_open = Counter(row["model"] for row in open_rows)
    prompt_duplicate_count = 0
    for model in MODELS:
        prompts = [row["prompt"] for row in open_rows if row["model"] == model]
        prompt_duplicate_count += len(prompts) - len(set(prompts))
    if prompt_duplicate_count:
        failures.append(f"prompt_duplicates:{prompt_duplicate_count}")

    return {
        "schema_version": "phase567_static_audit.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "valid": not failures,
        "status": "static_pass_no_model_run" if not failures else "static_fail",
        "failures": failures,
        "registered_case_count": len(all_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "open_case_count_by_model": dict(sorted(per_model_open.items())),
        "rows_per_world": world_sizes,
        "rows_per_counterfactual_triplet": triplet_sizes,
        "counterfactual_triplet_count": len(by_triplet),
        "triplet_invariant_failure_count": triplet_invariant_failures,
        "split_object_overlap_count": split_overlap_count,
        "prompt_duplicate_count": prompt_duplicate_count,
        "factorial_cell_count": len(CELLS),
        "sealed_split_read": False,
        "model_execution_performed": False,
    }


def freeze() -> dict[str, Any]:
    open_rows, sealed_rows, public_worlds = build_rows()
    for row in open_rows + sealed_rows:
        row["split"] = derive_split(row["anchor_id"])
        row["sealed"] = row["split"] == "sealed"
    audit = static_audit(open_rows, sealed_rows)
    if not audit["valid"]:
        raise RuntimeError(f"Phase567 static audit failed: {audit['failures']}")
    write_jsonl(OPEN_CASES_PATH, open_rows)
    write_jsonl(SEALED_CASES_PATH, sealed_rows)
    write_json(PUBLIC_WORLD_BANK_PATH, {
        "schema_version": "phase567_public_world_bank.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "worlds": public_worlds,
        "sealed_world_content_exposed": False,
    })
    sealed_commitment = {
        "schema_version": "phase567_sealed_commitment.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_file(SEALED_CASES_PATH),
        "sealed_split_read_for_analysis": False,
    }
    write_json(SEALED_COMMITMENT_PATH, sealed_commitment)
    protocol = {
        "schema_version": "phase567_frozen_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "split_world_counts": SPLIT_WORLD_COUNTS,
        "open_splits": list(OPEN_SPLITS),
        "rows_per_world": ROWS_PER_WORLD,
        "factorial_cells": list(CELLS),
        "registered_case_count": len(open_rows) + len(sealed_rows),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
        "behavior_gate": {
            "world_all_108_rate_min_per_behavior_split": 0.80,
            "minimum_cell_wilson_95_lcb": 0.85,
            "minimum_axis_wilson_95_lcb": 0.90,
            "unrecoverable_wilson_95_ucb_max": 0.05,
            "minimum_all_correct_role_worlds_per_split": 24,
        },
        "evidence_policy": {
            "three_object_three_value_two_relation_contract": True,
            "queried_relation_assignment_only_changes_within_triplet": True,
            "other_relation_assignment_fixed_within_triplet": True,
            "query_and_lexical_multisets_fixed_within_triplet": True,
            "role_labels_must_map_to_disjoint_physical_coordinates": True,
            "matched_wrong_state_is_counterfactual_sensitivity_not_natural_necessity": True,
            "same_layer_identity_write_is_not_delete_restore": True,
            "true_restore_requires_upstream_damage_and_later_layer_restore": True,
            "fine_scan_before_replicated_coarse_role_edge": False,
            "single_neuron_scan_before_compute_edge": False,
            "sealed_split_read": False,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    write_json(AUDIT_PATH, {
        **audit,
        "open_cases_sha256": sha256_file(OPEN_CASES_PATH),
        "sealed_commitment_sha256": sha256_file(SEALED_COMMITMENT_PATH),
    })
    print(json.dumps({
        "registered": protocol["registered_case_count"],
        "open": protocol["open_case_count"],
        "sealed": protocol["sealed_case_count"],
        "per_model_open": audit["open_case_count_by_model"],
        "rows_per_world": ROWS_PER_WORLD,
        "triplets": audit["counterfactual_triplet_count"],
        "valid": audit["valid"],
    }, ensure_ascii=False, indent=2))
    return protocol


if __name__ == "__main__":
    freeze()
