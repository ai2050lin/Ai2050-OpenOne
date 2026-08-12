#!/usr/bin/env python3
"""Phase1233: one-shot program-identifiable medal-binding behavior gate.

The phase freezes a non-bijective two-entity/three-value material, native Qwen3
tokenization, competing-program ceilings, collision registries, exact batches,
and behavior gates before loading any weights.  It then runs Qwen3 once in
CUDA FP16.  No hidden states, attentions, or interventions are permitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1233
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1233_qwen3_program_identifiable_medal_binding_audit.py"
UPSTREAM_ROOT = TEST_ROOT / "result/phase1232_qwen3_native_boundary_behavior_correction"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_FINAL_AUDIT = UPSTREAM_ROOT / "audit/independent_final_audit.json"
EXPECTED_UPSTREAM_FINAL = "885d40152f901335204af5f87b491acbe56ddfa719fbc4200da5f752bee3d190"
EXPECTED_UPSTREAM_AUDIT = "73c50bbfc05147209c4e8fa674c4b09627d88a5e33d22fc04fdd34cee83eec92"

OUT_ROOT = TEST_ROOT / "result/phase1233_qwen3_program_identifiable_medal_binding"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/medal_binding.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
PROGRAM_AUDIT_PATH = OUT_ROOT / "protocol/competing_program_audit.json"
BATCH_PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

MODEL_PATH = ROOT / "models/hf/qwen3-4b"
SYSTEM_PROMPT = (
    "Use only the supplied medal records. Return exactly one lowercase ordinal "
    "word and no explanation."
)
SPLITS = ("discovery", "confirmation", "natural_use")
VALUES = ("gold", "silver", "bronze")
CANDIDATES = ("first", "second", "third")
VALUE_TO_ANSWER = dict(zip(VALUES, CANDIDATES))
WORLDS_PER_SPLIT = 32
TEMPLATES_PER_SPLIT = 2
EXPECTED_ROWS_PER_SPLIT = 2304
EXPECTED_ROWS = 6912
BATCH_SIZE = 16
TIE_TOLERANCE = 1e-7
MAX_INPUT_LENGTH = 160

THRESHOLDS = {
    "Q0_finite_rate": 1.0,
    "Q1_split_accuracy": 0.90,
    "Q1_worst_marginal": 0.80,
    "Q1_program_ceiling_margin": 0.15,
    "Q2_target_change_triplet": 0.70,
    "Q3_non_target_null_triplet": 0.75,
    "Q4_query_switch_pair": 0.80,
    "Q4_binding_swap_pair": 0.80,
    "Q5_order_pair": 0.85,
    "Q5_template_pair": 0.85,
    "Q6_natural_first_token": 0.80,
}
MARGINAL_AXES = ("gold_candidate", "template_id", "world_id", "query_index", "order_variant")

FIRST_NAMES = {
    "discovery": (
        "Lina", "Pavel", "Rita", "Soren", "Clara", "Felix", "Maya", "Bruno",
        "Elena", "Gavin", "Iris", "Kellan", "Julia", "Derek", "Nina", "Roland",
    ),
    "confirmation": (
        "Laura", "Peter", "Rosa", "Stefan", "Chloe", "Grant", "Hazel", "Wesley",
        "Eva", "Hugo", "Ingrid", "Jonas", "Jasmine", "Edgar", "Naomi", "Lucian",
    ),
    "natural_use": (
        "Leona", "Pierce", "Rhea", "Tobias", "Celia", "Fraser", "Mabel", "Warren",
        "Elsa", "Gideon", "Ida", "Jasper", "Jade", "Desmond", "Nora", "Lionel",
    ),
}
SURNAMES = {
    "discovery": ("Vale", "Stone", "Marsh", "Quill"),
    "confirmation": ("Harbor", "Field", "Crown", "Meadow"),
    "natural_use": ("River", "Grove", "Summit", "Brook"),
}
ANCHOR_LEFT = {
    "discovery": ("Cedar", "Alder", "Birch", "Maple", "Willow", "Juniper", "Elm", "Pine"),
    "confirmation": ("Harbor", "Granite", "Silver", "Orchid", "Cypress", "Laurel", "Aspen", "Hazel"),
    "natural_use": ("River", "Meadow", "Summit", "Garden", "Forest", "Valley", "Prairie", "Lake"),
}
ANCHOR_RIGHT = {
    "discovery": ("Meet", "Registry", "League", "Series"),
    "confirmation": ("Contest", "Circuit", "Trials", "Games"),
    "natural_use": ("Event", "Festival", "Tournament", "Finals"),
}
TEMPLATES = {
    "discovery": (
        ("discovery_results", "In the {anchor} results, {entity}'s medal is {value}."),
        ("discovery_record", "The {anchor} record lists {entity} with a {value} medal."),
    ),
    "confirmation": (
        ("confirmation_award", "At the {anchor}, {entity} received the {value} medal tier."),
        ("confirmation_table", "The medal table for {anchor} marks {entity} as {value}."),
    ),
    "natural_use": (
        ("natural_report", "According to the {anchor} report, {entity} holds a {value} medal."),
        ("natural_listing", "{entity} appears in the {anchor} listing at the {value} medal level."),
    ),
}
QUERIES = {
    "discovery": "What one-word ordinal rank is indicated by {entity}'s medal? Answer:",
    "confirmation": "Which lowercase ordinal word matches {entity}'s medal tier? Answer:",
    "natural_use": "In one lowercase ordinal word, what rank does {entity}'s medal represent? Answer:",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def lexical_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())


def source_hashes() -> dict[str, str]:
    return {"execution": file_sha256(SCRIPT), "independent_audit": file_sha256(AUDIT_SCRIPT)}


def verify_upstream() -> None:
    final = read_json(UPSTREAM_FINAL)
    audit = read_json(UPSTREAM_FINAL_AUDIT)
    if final.get("final_digest") != EXPECTED_UPSTREAM_FINAL:
        raise RuntimeError("Phase1232 final digest mismatch")
    if audit.get("audit_digest") != EXPECTED_UPSTREAM_AUDIT or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1232 final audit mismatch")
    if final.get("authorization", {}).get("hidden_scan") is not False:
        raise RuntimeError("Phase1232 evidence boundary drift")


def world_entities(split: str, world_index: int) -> tuple[str, str]:
    first = FIRST_NAMES[split]
    surnames = SURNAMES[split]
    left_index = 2 * world_index
    right_index = left_index + 1
    return (
        f"{first[left_index % len(first)]} {surnames[(left_index // len(first)) % len(surnames)]}",
        f"{first[right_index % len(first)]} {surnames[(right_index // len(first)) % len(surnames)]}",
    )


def world_anchor(split: str, world_index: int) -> str:
    left = ANCHOR_LEFT[split]
    right = ANCHOR_RIGHT[split]
    return f"{left[world_index % len(left)]} {right[(world_index // len(left)) % len(right)]}"


def add_span(registry: dict[str, list[list[int]]], role: str, start: int, end: int) -> None:
    registry.setdefault(role, []).append([start, end])


def render_row(
    split: str,
    world_index: int,
    template_index: int,
    order_variant: int,
    query_index: int,
    values: tuple[str, str],
) -> dict[str, Any]:
    entities = world_entities(split, world_index)
    anchor = world_anchor(split, world_index)
    template_id, template = TEMPLATES[split][template_index]
    order = (0, 1) if order_variant == 0 else (1, 0)
    spans: dict[str, list[list[int]]] = {}
    pieces: list[str] = []
    cursor = 0
    rendered_records: list[str] = []
    for position, entity_index in enumerate(order):
        if position:
            pieces.append(" ")
            cursor += 1
        entity = entities[entity_index]
        value = values[entity_index]
        record = template.format(anchor=anchor, entity=entity, value=value)
        start = cursor
        pieces.append(record)
        cursor += len(record)
        rendered_records.append(record)
        add_span(spans, "record_full", start, cursor)
        entity_start = start + record.index(entity)
        add_span(spans, "record_subject", entity_start, entity_start + len(entity))
        value_start = start + record.index(value)
        add_span(spans, "record_value", value_start, value_start + len(value))
        relation_start = start + record.lower().index("medal")
        add_span(spans, "record_relation", relation_start, relation_start + len("medal"))
    records_text = "".join(pieces)
    pieces.append(" ")
    cursor += 1
    query_entity = entities[query_index]
    query = QUERIES[split].format(entity=query_entity)
    query_start = cursor
    pieces.append(query)
    cursor += len(query)
    add_span(spans, "query_full", query_start, cursor)
    subject_start = query_start + query.index(query_entity)
    add_span(spans, "query_subject", subject_start, subject_start + len(query_entity))
    relation_start = query_start + query.lower().index("ordinal")
    add_span(spans, "query_relation", relation_start, relation_start + len("ordinal"))
    answer_start = query_start + query.index("Answer:")
    add_span(spans, "answer_boundary", answer_start, answer_start + len("Answer:"))
    prompt = "".join(pieces)
    target_value = values[query_index]
    other_value = values[1 - query_index]
    base = {
        "phase": PHASE,
        "schema_version": "phase1233.medal_binding.row.v1",
        "split": split,
        "world_index": world_index,
        "world_id": f"{split}-world-{world_index:02d}",
        "anchor": anchor,
        "entities": list(entities),
        "template_index": template_index,
        "template_id": template_id,
        "order_variant": order_variant,
        "record_order_indices": list(order),
        "query_index": query_index,
        "query_entity": query_entity,
        "values": list(values),
        "target_value": target_value,
        "other_value": other_value,
        "gold_candidate": VALUE_TO_ANSWER[target_value],
        "candidates": list(CANDIDATES),
        "rendered_records": rendered_records,
        "records_text": records_text,
        "query": query,
        "prompt": prompt,
        "spans": spans,
        "prompt_char_length": len(prompt),
        "prompt_lexical_multiset_digest": digest(sorted(lexical_tokens(prompt))),
        "record_value_bag": sorted(values),
        "target_record_position": order.index(query_index),
    }
    identity = {
        key: base[key]
        for key in (
            "split", "world_index", "template_index", "order_variant", "query_index", "values"
        )
    }
    base["item_id"] = f"p1233-{digest(identity)[:24]}"
    base["target_triplet_id"] = f"target-{digest({**identity, 'values': None, 'target': None, 'other_value': other_value})[:20]}"
    base["null_triplet_id"] = f"null-{digest({**identity, 'values': None, 'target_value': target_value, 'other': None})[:20]}"
    base["query_pair_id"] = f"query-{digest({**identity, 'query_index': None})[:20]}"
    base["order_pair_id"] = f"order-{digest({**identity, 'order_variant': None})[:20]}"
    base["template_pair_id"] = f"template-{digest({**identity, 'template_index': None})[:20]}"
    if values[0] != values[1]:
        base["binding_swap_pair_id"] = f"swap-{digest({**identity, 'values': sorted(values)})[:20]}"
    else:
        base["binding_swap_pair_id"] = None
    base["row_digest"] = digest(base)
    return base


def generate_material() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for world_index in range(WORLDS_PER_SPLIT):
            for template_index in range(TEMPLATES_PER_SPLIT):
                for order_variant in (0, 1):
                    for query_index in (0, 1):
                        for value0 in VALUES:
                            for value1 in VALUES:
                                rows.append(
                                    render_row(
                                        split,
                                        world_index,
                                        template_index,
                                        order_variant,
                                        query_index,
                                        (value0, value1),
                                    )
                                )
    if len(rows) != EXPECTED_ROWS or len({row["item_id"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("Phase1233 material cardinality failure")
    return rows


def render_native(tokenizer: Any, prompt: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def token_span_for_chars(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    indices = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and right > start and left < end
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise RuntimeError("invalid character-to-token span")
    if offsets[indices[0]][0] > start or offsets[indices[-1]][1] < end:
        raise RuntimeError("token span does not cover source characters")
    return [indices[0], indices[-1] + 1]


def build_manifest(material: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer, __version__ as transformers_version

    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    if not getattr(fast, "is_fast", False):
        raise RuntimeError("fast tokenizer unavailable")
    candidate_ids = {candidate: [int(value) for value in slow.encode(candidate, add_special_tokens=False)] for candidate in CANDIDATES}
    if any(len(ids) != 1 for ids in candidate_ids.values()) or len({ids[0] for ids in candidate_ids.values()}) != len(CANDIDATES):
        raise RuntimeError("native candidate IDs are not distinct singleton tokens")
    manifest: list[dict[str, Any]] = []
    slow_fast_mismatch = 0
    candidate_overlap = 0
    span_failures = 0
    suffix_failures = 0
    for execution_index, row in enumerate(material):
        rendered = render_native(slow, row["prompt"])
        input_ids = [int(value) for value in slow.encode(rendered, add_special_tokens=False)]
        encoded = fast(rendered, add_special_tokens=False, return_offsets_mapping=True)
        fast_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
        if input_ids != fast_ids:
            slow_fast_mismatch += 1
        prompt_start = rendered.find(row["prompt"])
        if prompt_start < 0 or rendered.find(row["prompt"], prompt_start + 1) >= 0:
            raise RuntimeError("prompt embedding is not unique")
        roles: dict[str, list[list[int]]] = {}
        try:
            for role, spans in row["spans"].items():
                if role == "answer_boundary":
                    roles[role] = [[len(input_ids) - 1, len(input_ids)]]
                else:
                    roles[role] = [
                        token_span_for_chars(offsets, prompt_start + int(start), prompt_start + int(end))
                        for start, end in spans
                    ]
        except RuntimeError:
            span_failures += 1
            roles = {}
        if any(ids[0] in input_ids for ids in candidate_ids.values()):
            candidate_overlap += 1
        for candidate, ids in candidate_ids.items():
            appended = [int(value) for value in slow.encode(rendered + candidate, add_special_tokens=False)]
            if appended[: len(input_ids)] != input_ids or appended[len(input_ids) :] != ids:
                suffix_failures += 1
        case: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1233.qwen3.behavior_case.v1",
            "execution_index": execution_index,
            "item_id": row["item_id"],
            "material_row_digest": row["row_digest"],
            "input_ids": input_ids,
            "input_ids_digest": digest(input_ids),
            "input_id_multiset_digest": digest(sorted(input_ids)),
            "input_length": len(input_ids),
            "prediction_token_index": len(input_ids) - 1,
            "candidate_token_ids": candidate_ids,
            "gold_candidate": row["gold_candidate"],
            "gold_candidate_token_id": candidate_ids[row["gold_candidate"]][0],
            "role_token_spans": roles,
            "native_prompt_digest": digest(rendered),
            "split": row["split"],
            "world_id": row["world_id"],
            "template_id": row["template_id"],
            "order_variant": row["order_variant"],
            "query_index": row["query_index"],
            "query_entity": row["query_entity"],
            "target_value": row["target_value"],
            "other_value": row["other_value"],
            "target_record_position": row["target_record_position"],
            "target_triplet_id": row["target_triplet_id"],
            "null_triplet_id": row["null_triplet_id"],
            "query_pair_id": row["query_pair_id"],
            "order_pair_id": row["order_pair_id"],
            "template_pair_id": row["template_pair_id"],
            "binding_swap_pair_id": row["binding_swap_pair_id"],
        }
        case["manifest_row_digest"] = digest(case)
        manifest.append(case)
    lengths = [row["input_length"] for row in manifest]
    summary = {
        "transformers_version": transformers_version,
        "slow_tokenizer_class": type(slow).__name__,
        "fast_tokenizer_class": type(fast).__name__,
        "candidate_token_ids": candidate_ids,
        "row_count": len(manifest),
        "input_length_min": min(lengths),
        "input_length_max": max(lengths),
        "input_length_bucket_count": len(set(lengths)),
        "slow_fast_mismatch_count": slow_fast_mismatch,
        "candidate_input_overlap_count": candidate_overlap,
        "span_failure_count": span_failures,
        "native_suffix_failure_count": suffix_failures,
        "manifest_digest": digest(manifest),
        "model_weights_loaded": False,
    }
    summary["tokenizer_gate"] = bool(
        len(manifest) == EXPECTED_ROWS
        and max(lengths) <= MAX_INPUT_LENGTH
        and slow_fast_mismatch == 0
        and candidate_overlap == 0
        and span_failures == 0
        and suffix_failures == 0
    )
    return manifest, summary


def empirical_lookup_accuracy(rows: list[dict[str, Any]], feature: Callable[[dict[str, Any]], Any]) -> float:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[canonical_json(feature(row))][row["gold_candidate"]] += 1
    return sum(max(counts.values()) for counts in table.values()) / len(rows)


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            result[str(row[key])].append(row)
    return result


def build_program_audit(material: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_by_id = {row["item_id"]: row for row in manifest}
    features = {
        "constant": lambda row: "constant",
        "query_entity": lambda row: row["query_entity"],
        "query_index": lambda row: row["query_index"],
        "other_record_value": lambda row: row["other_value"],
        "query_plus_other_value": lambda row: (row["query_index"], row["other_value"]),
        "first_record_value": lambda row: row["values"][row["record_order_indices"][0]],
        "last_record_value": lambda row: row["values"][row["record_order_indices"][-1]],
        "value_bag": lambda row: tuple(sorted(row["values"])),
        "target_record_position": lambda row: row["target_record_position"],
        "template": lambda row: row["template_id"],
        "world": lambda row: row["world_id"],
        "order": lambda row: row["order_variant"],
        "intended_target_value": lambda row: row["target_value"],
    }
    accuracies = {name: empirical_lookup_accuracy(material, feature) for name, feature in features.items()}
    target_groups = grouped(material, "target_triplet_id")
    null_groups = grouped(material, "null_triplet_id")
    query_groups = grouped(material, "query_pair_id")
    order_groups = grouped(material, "order_pair_id")
    template_groups = grouped(material, "template_pair_id")
    swap_groups = grouped(material, "binding_swap_pair_id")
    target_complete = all(
        len(group) == 3
        and len({row["other_value"] for row in group}) == 1
        and {row["gold_candidate"] for row in group} == set(CANDIDATES)
        for group in target_groups.values()
    )
    null_complete = all(
        len(group) == 3
        and len({row["target_value"] for row in group}) == 1
        and len({row["gold_candidate"] for row in group}) == 1
        for group in null_groups.values()
    )
    query_discriminating = all(
        len(group) == 2
        and len({tuple(row["values"]) for row in group}) == 1
        and (
            len({row["gold_candidate"] for row in group}) == 2
            if group[0]["values"][0] != group[0]["values"][1]
            else len({row["gold_candidate"] for row in group}) == 1
        )
        for group in query_groups.values()
    )
    order_invariant = all(
        len(group) == 2
        and len({row["gold_candidate"] for row in group}) == 1
        for group in order_groups.values()
    )
    template_invariant = all(
        len(group) == 2
        and len({row["gold_candidate"] for row in group}) == 1
        for group in template_groups.values()
    )
    swap_discriminating = all(
        len(group) == 2
        and len({row["gold_candidate"] for row in group}) == 2
        and len({row["prompt_lexical_multiset_digest"] for row in group}) == 1
        and len({manifest_by_id[row["item_id"]]["input_id_multiset_digest"] for row in group}) == 1
        for group in swap_groups.values()
    )
    alternative_names = [name for name in accuracies if name != "intended_target_value"]
    max_alternative = max(accuracies[name] for name in alternative_names)
    audit: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1233.competing_program_audit.v1",
        "created_at_utc": utc_now(),
        "scope": "all material and native-token manifests before any Phase1233 model output",
        "cardinality": {"entity_count": 2, "value_count": 3, "value_count_gt_entity_count": True},
        "non_bijective_support": {
            "repeat_value_row_count": sum(row["values"][0] == row["values"][1] for row in material),
            "unused_value_min": min(len(set(VALUES) - set(row["values"])) for row in material),
        },
        "empirical_bayes_accuracy": accuracies,
        "maximum_registered_alternative_accuracy": max_alternative,
        "intended_target_reader_accuracy": accuracies["intended_target_value"],
        "collision_groups": {
            "target_change_triplets": len(target_groups),
            "non_target_null_triplets": len(null_groups),
            "query_pairs": len(query_groups),
            "order_pairs": len(order_groups),
            "template_pairs": len(template_groups),
            "binding_swap_pairs": len(swap_groups),
        },
        "checks": {
            "target_change_complete": target_complete,
            "non_target_null_complete": null_complete,
            "query_switch_discriminating": query_discriminating,
            "order_invariant": order_invariant,
            "template_invariant": template_invariant,
            "binding_swap_discriminating": swap_discriminating,
            "other_record_at_chance": abs(accuracies["query_plus_other_value"] - 1 / 3) < 1e-12,
            "value_bag_below_behavior_gate": accuracies["value_bag"] < THRESHOLDS["Q1_split_accuracy"],
            "first_last_below_behavior_gate": max(accuracies["first_record_value"], accuracies["last_record_value"]) < THRESHOLDS["Q1_split_accuracy"],
            "intended_reader_sufficient": accuracies["intended_target_value"] == 1.0,
            "registered_alternatives_separated": max_alternative + THRESHOLDS["Q1_program_ceiling_margin"] < THRESHOLDS["Q1_split_accuracy"],
        },
        "claim_boundary": (
            "The material distinguishes the registered extensional programs. A behavior pass still cannot prove a unique neural algorithm; "
            "that requires separately frozen causal interventions."
        ),
    }
    audit["program_identifiability_gate"] = all(audit["checks"].values())
    audit["program_audit_digest"] = digest(audit)
    return audit


def build_batch_plan(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        buckets[int(row["input_length"])].append(row)
    batches: list[dict[str, Any]] = []
    for length in sorted(buckets):
        ordered = sorted(buckets[length], key=lambda row: int(row["execution_index"]))
        for start in range(0, len(ordered), BATCH_SIZE):
            members = ordered[start : start + BATCH_SIZE]
            batches.append(
                {
                    "batch_index": len(batches),
                    "input_length": length,
                    "runtime_batch_size": len(members),
                    "item_ids": [row["item_id"] for row in members],
                }
            )
    flat = [item for batch in batches for item in batch["item_ids"]]
    if len(flat) != EXPECTED_ROWS or len(set(flat)) != EXPECTED_ROWS:
        raise RuntimeError("batch plan does not partition the manifest")
    plan = {
        "phase": PHASE,
        "schema_version": "phase1233.qwen3.batch_plan.v1",
        "batch_size": BATCH_SIZE,
        "adaptive_fallback": False,
        "bucket_count": len(buckets),
        "batch_count": len(batches),
        "batches": batches,
    }
    plan["plan_digest"] = digest(plan)
    return plan


def preregister() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("Phase1233 output directory already exists")
    verify_upstream()
    material = generate_material()
    manifest, tokenizer_summary = build_manifest(material)
    program_audit = build_program_audit(material, manifest)
    if not tokenizer_summary["tokenizer_gate"] or not program_audit["program_identifiability_gate"]:
        raise RuntimeError("Phase1233 zero-model qualification failed before preregistration")
    plan = build_batch_plan(manifest)
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1233.qwen3.program_identifiable_medal_binding.v1",
        "created_at_utc": utc_now(),
        "objective": (
            "One-shot Qwen3 behavior qualification on a non-bijective medal-to-ordinal object binding whose registered competing programs are distinguishable."
        ),
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1232_final_digest": EXPECTED_UPSTREAM_FINAL,
            "phase1232_final_audit_digest": EXPECTED_UPSTREAM_AUDIT,
            "phase1232_final_sha256": file_sha256(UPSTREAM_FINAL),
            "phase1232_audit_sha256": file_sha256(UPSTREAM_FINAL_AUDIT),
            "new_user_authorization": (
                "Phase1232 closed the old family; Phase1233 is a separately preregistered object, not an automatic prompt patch."
            ),
        },
        "material": {
            "mapping": VALUE_TO_ANSWER,
            "entity_count": 2,
            "value_count": 3,
            "repeats_allowed": True,
            "unused_values_present": True,
            "full_assignment_factorial": True,
            "full_query_factorial": True,
            "splits": list(SPLITS),
            "row_count": len(material),
            "material_digest": digest(material),
        },
        "interface": {
            "system_prompt": SYSTEM_PROMPT,
            "native_chat_template": True,
            "enable_thinking": False,
            "candidate_continuation": "unprefixed response begins at the native assistant generation boundary",
            "candidate_token_ids": tokenizer_summary["candidate_token_ids"],
            "candidate_input_overlap_count": tokenizer_summary["candidate_input_overlap_count"],
            "manifest_digest": tokenizer_summary["manifest_digest"],
            "maximum_input_length": MAX_INPUT_LENGTH,
        },
        "program_identifiability": {
            "program_audit_digest": program_audit["program_audit_digest"],
            "gate": program_audit["program_identifiability_gate"],
            "registered_programs": [
                "target record", "other record", "first record", "last record", "value bag",
                "position", "query prior", "template prior",
            ],
            "maximum_alternative_accuracy": program_audit["maximum_registered_alternative_accuracy"],
            "behavior_margin_min": THRESHOLDS["Q1_program_ceiling_margin"],
        },
        "execution": {
            "model": "qwen3",
            "device": "cuda",
            "precision": "float16",
            "quantization": "none",
            "input": "exact frozen input_ids",
            "batch_plan_digest": plan["plan_digest"],
            "batch_size": BATCH_SIZE,
            "adaptive_batch_fallback": False,
            "score": "FP32 log_softmax at the final prompt position over unprefixed native candidate token IDs",
            "tie_tolerance": TIE_TOLERANCE,
            "hidden_states": False,
            "attentions": False,
            "intervention": False,
        },
        "thresholds": THRESHOLDS,
        "typed_ledgers": {
            "Q0": "overall and split finite rates",
            "Q1": "split accuracy, active worst marginals, and margin over the strongest registered alternative program",
            "Q2": "target-value change triplets",
            "Q3": "non-target-value null triplets",
            "Q4": "query-switch and same-bag binding-swap collision pairs",
            "Q5": "record-order and template invariance pairs",
            "Q6": "natural_use full-vocabulary first token",
        },
        "authorization_rule": {
            "behavior_gate": "Q0 and Q1 and Q2 and Q3 and Q4 and Q5",
            "construct_gate": "program_identifiability_gate",
            "hidden_eligibility": "behavior_gate and construct_gate",
            "natural_first_token": "hidden_eligibility and Q6",
        },
        "forbidden": [
            "change material, prompt, mapping, candidates, token IDs, batches, thresholds, or denominators after preregistration",
            "run GLM4 or DS7B",
            "save hidden states or attentions",
            "perform interventions",
            "claim a unique neural algorithm from behavior",
            "patch this family after a failed confirmation split",
        ],
        "claim_boundary": [
            "This is an artificial English medal-rank task using a familiar parametric mapping.",
            "Program identifiability covers only the registered extensional alternatives.",
            "Behavioral collision success does not prove target-record-specific neural use.",
            "Candidate scoring and first-token generation do not establish complete open generation.",
            "A pass is Qwen3- and interface-specific until independently confirmed.",
        ],
        "source_artifacts": {
            "tokenizer_summary": tokenizer_summary,
            "program_audit_digest": program_audit["program_audit_digest"],
            "batch_plan_digest": plan["plan_digest"],
        },
    }
    contract["contract_digest"] = digest(contract)
    write_jsonl(MATERIAL_PATH, material)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(PROGRAM_AUDIT_PATH, program_audit)
    write_json(BATCH_PLAN_PATH, plan)
    write_json(CONTRACT_PATH, contract)
    print(
        canonical_json(
            {
                "status": "phase1233_preregistered",
                "contract_digest": contract["contract_digest"],
                "rows": len(material),
                "program_gate": program_audit["program_identifiability_gate"],
                "candidate_ids": tokenizer_summary["candidate_token_ids"],
            }
        )
    )


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    verify_upstream()
    contract = read_json(CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program_audit = read_json(PROGRAM_AUDIT_PATH)
    plan = read_json(BATCH_PLAN_PATH)
    if contract["contract_digest"] != digest(strip_digest(contract, "contract_digest")):
        raise RuntimeError("contract drift")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    if contract["material"]["material_digest"] != digest(material):
        raise RuntimeError("material drift")
    if contract["interface"]["manifest_digest"] != digest(manifest):
        raise RuntimeError("manifest drift")
    if program_audit["program_audit_digest"] != digest(strip_digest(program_audit, "program_audit_digest")):
        raise RuntimeError("program audit drift")
    if plan["plan_digest"] != digest(strip_digest(plan, "plan_digest")):
        raise RuntimeError("batch plan drift")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit failed")
    return contract, material, manifest, program_audit, plan


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1233 behavior outputs already exist")
    contract, _material, manifest, _program, plan = verify_frozen()
    manifest_by_id = {row["item_id"]: row for row in manifest}
    candidate_ids = {candidate: int(ids[0]) for candidate, ids in contract["interface"]["candidate_token_ids"].items()}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    if device.type != "cuda" or precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
        release_fp16(model)
        raise RuntimeError("Phase1233 numerical contract failed")
    raw: list[dict[str, Any]] = []
    try:
        for batch_number, batch in enumerate(plan["batches"], start=1):
            members = [manifest_by_id[item_id] for item_id in batch["item_ids"]]
            expected_length = int(batch["input_length"])
            if any(int(row["input_length"]) != expected_length for row in members):
                raise RuntimeError("batch input length drift")
            input_ids = torch.tensor([row["input_ids"] for row in members], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    logits_to_keep=1,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
            logits = output.logits[:, -1, :].float()
            finite_rows = torch.isfinite(logits).all(dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            top1_ids = logits.argmax(dim=-1)
            for row_index, source in enumerate(members):
                scores = {candidate: float(log_probs[row_index, token_id].item()) for candidate, token_id in candidate_ids.items()}
                order = sorted(CANDIDATES, key=lambda candidate: scores[candidate], reverse=True)
                top_margin = scores[order[0]] - scores[order[1]]
                finite = bool(finite_rows[row_index].item()) and all(math.isfinite(value) for value in scores.values())
                prediction = None if (not finite or top_margin <= TIE_TOLERANCE) else order[0]
                gold = source["gold_candidate"]
                wrong_best = max(value for candidate, value in scores.items() if candidate != gold)
                top1_id = int(top1_ids[row_index].item())
                row: dict[str, Any] = {
                    "phase": PHASE,
                    "schema_version": "phase1233.qwen3.behavior.row.v1",
                    "contract_digest": contract["contract_digest"],
                    "item_id": source["item_id"],
                    "manifest_row_digest": source["manifest_row_digest"],
                    "execution_index": int(source["execution_index"]),
                    "split": source["split"],
                    "world_id": source["world_id"],
                    "template_id": source["template_id"],
                    "order_variant": source["order_variant"],
                    "query_index": source["query_index"],
                    "query_entity": source["query_entity"],
                    "target_value": source["target_value"],
                    "other_value": source["other_value"],
                    "target_record_position": source["target_record_position"],
                    "target_triplet_id": source["target_triplet_id"],
                    "null_triplet_id": source["null_triplet_id"],
                    "query_pair_id": source["query_pair_id"],
                    "order_pair_id": source["order_pair_id"],
                    "template_pair_id": source["template_pair_id"],
                    "binding_swap_pair_id": source["binding_swap_pair_id"],
                    "gold_candidate": gold,
                    "all_vocab_logits_finite": finite,
                    "candidate_scores": scores,
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "unresolved_tie": finite and top_margin <= TIE_TOLERANCE,
                    "top_candidate_margin": top_margin,
                    "gold_margin": scores[gold] - wrong_best,
                    "full_vocab_top1_id": top1_id,
                    "full_vocab_top1_text": tokenizer.decode([top1_id], skip_special_tokens=False),
                    "full_vocab_top1_is_gold_candidate": top1_id == candidate_ids[gold],
                    "input_length": expected_length,
                    "runtime_batch_size": len(members),
                    "batch_index": int(batch["batch_index"]),
                }
                row["behavior_row_digest"] = digest(row)
                raw.append(row)
            del output, logits, log_probs, top1_ids, finite_rows, input_ids
            if batch_number % 50 == 0 or batch_number == len(plan["batches"]):
                print(f"[phase1233] {batch_number}/{len(plan['batches'])} batches", flush=True)
        raw.sort(key=lambda row: row["execution_index"])
        write_jsonl(RAW_PATH, raw)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1233.qwen3.run_summary.v1",
            "created_at_utc": utc_now(),
            "model": "qwen3",
            "contract_digest": contract["contract_digest"],
            "case_count": len(raw),
            "raw_digest": digest(raw),
            "precision_audit": precision,
            "placement": placement,
            "batch_plan_digest": plan["plan_digest"],
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "hidden_states_saved": False,
            "attentions_saved": False,
            "interventions_performed": False,
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
        print(canonical_json({"status": "behavior_complete", "rows": len(raw), "summary_digest": summary["summary_digest"]}))
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def collision_rate(rows: list[dict[str, Any]], key: str, expected_size: int, mode: str) -> float:
    groups = grouped(rows, key)
    outcomes: list[bool] = []
    for values in groups.values():
        success = len(values) == expected_size and all(row["all_vocab_logits_finite"] and row["correct"] for row in values)
        predictions = [row["prediction"] for row in values]
        if mode == "cover":
            success = success and set(predictions) == set(CANDIDATES)
        elif mode == "invariant":
            success = success and len(set(predictions)) == 1
        elif mode == "different":
            success = success and len(set(predictions)) == expected_size
        else:
            raise ValueError(mode)
        outcomes.append(success)
    return sum(outcomes) / len(outcomes) if outcomes else float("nan")


def adjudicate(raw: list[dict[str, Any]], program_audit: dict[str, Any]) -> dict[str, Any]:
    q0_split = {split: rate([row for row in raw if row["split"] == split], "all_vocab_logits_finite") for split in SPLITS}
    q0_overall = rate(raw, "all_vocab_logits_finite")
    q0_pass = q0_overall >= THRESHOLDS["Q0_finite_rate"] and min(q0_split.values()) >= THRESHOLDS["Q0_finite_rate"]
    split_accuracy = {split: rate([row for row in raw if row["split"] == split], "correct") for split in SPLITS}
    marginal_cells: dict[str, dict[str, float]] = {}
    marginal_worst: dict[str, float] = {}
    for axis in MARGINAL_AXES:
        cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw:
            cells[canonical_json(row[axis])].append(row)
        values = {key: rate(group, "correct") for key, group in cells.items()}
        marginal_cells[axis] = values
        marginal_worst[axis] = min(values.values())
    overall_accuracy = rate(raw, "correct")
    alternative_ceiling = float(program_audit["maximum_registered_alternative_accuracy"])
    program_margin = overall_accuracy - alternative_ceiling
    q1_pass = (
        min(split_accuracy.values()) >= THRESHOLDS["Q1_split_accuracy"]
        and min(marginal_worst.values()) >= THRESHOLDS["Q1_worst_marginal"]
        and program_margin >= THRESHOLDS["Q1_program_ceiling_margin"]
    )
    q2 = {split: collision_rate([row for row in raw if row["split"] == split], "target_triplet_id", 3, "cover") for split in SPLITS}
    q2_pass = min(q2.values()) >= THRESHOLDS["Q2_target_change_triplet"]
    q3 = {split: collision_rate([row for row in raw if row["split"] == split], "null_triplet_id", 3, "invariant") for split in SPLITS}
    q3_pass = min(q3.values()) >= THRESHOLDS["Q3_non_target_null_triplet"]
    q4_query: dict[str, float] = {}
    q4_swap: dict[str, float] = {}
    for split in SPLITS:
        selected = [row for row in raw if row["split"] == split and row["target_value"] != row["other_value"]]
        q4_query[split] = collision_rate(selected, "query_pair_id", 2, "different")
        q4_swap[split] = collision_rate(selected, "binding_swap_pair_id", 2, "different")
    q4_pass = min(q4_query.values()) >= THRESHOLDS["Q4_query_switch_pair"] and min(q4_swap.values()) >= THRESHOLDS["Q4_binding_swap_pair"]
    q5_order = {split: collision_rate([row for row in raw if row["split"] == split], "order_pair_id", 2, "invariant") for split in SPLITS}
    q5_template = {split: collision_rate([row for row in raw if row["split"] == split], "template_pair_id", 2, "invariant") for split in SPLITS}
    q5_pass = min(q5_order.values()) >= THRESHOLDS["Q5_order_pair"] and min(q5_template.values()) >= THRESHOLDS["Q5_template_pair"]
    natural = [row for row in raw if row["split"] == "natural_use"]
    q6_accuracy = rate(natural, "full_vocab_top1_is_gold_candidate")
    q6_pass = q6_accuracy >= THRESHOLDS["Q6_natural_first_token"]
    construct_gate = bool(program_audit["program_identifiability_gate"])
    behavior_gate = q0_pass and q1_pass and q2_pass and q3_pass and q4_pass and q5_pass
    return {
        "Q0": {"overall_finite_rate": q0_overall, "split_finite_rates": q0_split, "passed": q0_pass},
        "Q1": {
            "split_accuracy": split_accuracy,
            "worst_marginal_by_axis": marginal_worst,
            "marginal_cells": marginal_cells,
            "overall_accuracy": overall_accuracy,
            "registered_alternative_ceiling": alternative_ceiling,
            "program_ceiling_margin": program_margin,
            "passed": q1_pass,
        },
        "Q2": {"target_change_triplet_success": q2, "passed": q2_pass},
        "Q3": {"non_target_null_triplet_success": q3, "passed": q3_pass},
        "Q4": {"query_switch_pair_success": q4_query, "binding_swap_pair_success": q4_swap, "passed": q4_pass},
        "Q5": {"order_pair_success": q5_order, "template_pair_success": q5_template, "passed": q5_pass},
        "Q6": {"natural_first_token_accuracy": q6_accuracy, "passed": q6_pass},
        "construct_gate": construct_gate,
        "behavior_gate": behavior_gate,
        "hidden_eligibility": construct_gate and behavior_gate,
        "natural_first_token_gate": construct_gate and behavior_gate and q6_pass,
        "overall_candidate_accuracy": overall_accuracy,
        "tie_count": sum(row["unresolved_tie"] for row in raw),
        "nonfinite_count": sum(not row["all_vocab_logits_finite"] for row in raw),
        "prediction_counts": dict(Counter(str(row["prediction"]) for row in raw)),
    }


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1233 final already exists")
    contract, material, manifest, program_audit, plan = verify_frozen()
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    result_audit = read_json(RESULT_AUDIT_PATH)
    if not result_audit.get("all_checks_passed"):
        raise RuntimeError("independent result audit failed")
    if len(raw) != EXPECTED_ROWS or summary["raw_digest"] != digest(raw):
        raise RuntimeError("raw behavior mismatch")
    if {row["item_id"] for row in raw} != {row["item_id"] for row in manifest}:
        raise RuntimeError("manifest coverage mismatch")
    ledgers = adjudicate(raw, program_audit)
    passed = bool(ledgers["hidden_eligibility"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1233.qwen3.program_identifiable_medal_binding.final.v1",
        "created_at_utc": utc_now(),
        "status": "behavior_and_construct_gate_passed" if passed else "one_shot_behavior_gate_failed",
        "contract_digest": contract["contract_digest"],
        "material_digest": digest(material),
        "manifest_digest": digest(manifest),
        "program_audit_digest": program_audit["program_audit_digest"],
        "batch_plan_digest": plan["plan_digest"],
        "run_summary_digest": summary["summary_digest"],
        "raw_digest": summary["raw_digest"],
        "result_audit_digest": result_audit["audit_digest"],
        "ledgers": ledgers,
        "program_boundary": {
            "registered_alternative_accuracy": program_audit["empirical_bayes_accuracy"],
            "maximum_registered_alternative_accuracy": program_audit["maximum_registered_alternative_accuracy"],
            "unique_neural_algorithm_identified": False,
        },
        "k_item": {
            "identifier": "K208",
            "evidence_grade": "E3-BEHAVIOR-CONSTRUCT" if passed else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                "Qwen3 passed the one-shot behavior and registered program-identifiability gates on the non-bijective medal-binding family."
                if passed
                else "Qwen3 did not pass the one-shot behavior gate on the preregistered non-bijective medal-binding family."
            ),
            "scope": "Qwen3-4B; CUDA FP16; artificial English medal-to-ordinal binding; behavior only",
        },
        "authorization": {
            "behavior_object": passed,
            "registered_program_construct": bool(ledgers["construct_gate"]),
            "natural_first_token_claim": bool(ledgers["natural_first_token_gate"]),
            "unique_neural_algorithm_claim": False,
            "next_experiment": (
                "Phase1234 zero-output future-response tensor protocol over the frozen role spans and collision donors"
                if passed else None
            ),
            "auto_continue": False,
            "hidden_scan_in_this_phase": False,
            "cross_model_run": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    verify_upstream()
    material = generate_material()
    manifest, tokenizer_summary = build_manifest(material)
    program_audit = build_program_audit(material, manifest)
    plan = build_batch_plan(manifest)
    print(
        canonical_json(
            {
                "status": "selftest_passed",
                "rows": len(material),
                "tokenizer_gate": tokenizer_summary["tokenizer_gate"],
                "program_gate": program_audit["program_identifiability_gate"],
                "program_accuracies": program_audit["empirical_bayes_accuracy"],
                "batch_count": plan["batch_count"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("selftest", "preregister", "run", "finalize"))
    stage = parser.parse_args().stage
    {"selftest": selftest, "preregister": preregister, "run": run_qwen3, "finalize": finalize}[stage]()


if __name__ == "__main__":
    main()
