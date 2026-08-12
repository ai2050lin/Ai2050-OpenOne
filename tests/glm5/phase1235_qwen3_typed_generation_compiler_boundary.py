#!/usr/bin/env python3
"""Phase1235: typed selection-to-generation compiler boundary.

The phase keeps the K199 query-object operation fixed while orthogonally
varying object surfaces, marker surfaces, and prompt templates.  Seven readout
contracts are frozen before model execution: choice-conditioned sequence
ranking, bare-prompt sequence ranking, bare-prompt candidate-trie decoding,
bare short generation, field-cued short generation, exact-sentence generation,
and natural-sentence generation.  Teacher-forced token ranks are diagnostic.
No hidden state, attention, or intervention is collected in this phase.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1220_object_relation_value_master_task as p1220
import phase1221_typed_operation_behavior_and_error_fingerprints as p1221
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1235
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1235_qwen3_typed_generation_compiler_boundary_audit.py"
P1220_SCRIPT = TEST_ROOT / "phase1220_object_relation_value_master_task.py"
P1221_SCRIPT = TEST_ROOT / "phase1221_typed_operation_behavior_and_error_fingerprints.py"

UPSTREAM_ROOT = TEST_ROOT / "result/phase1234_qwen3_k199_registry_sealed_confirmation"
UPSTREAM_FINAL_PATH = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_AUDIT_PATH = UPSTREAM_ROOT / "audit/independent_final_audit.json"
UPSTREAM_MATERIAL_PATH = UPSTREAM_ROOT / "material/sealed_query_object_worlds.jsonl"
EXPECTED_UPSTREAM_FINAL = "4d7d28a7c969d145a95adda1fc5cacfc5595fff762c15d2dec914b048ab7d63b"
EXPECTED_UPSTREAM_AUDIT = "90ac15451fbf7239912982419c3c386ce2f49126febaddf9146b28a61ba93708"

OUT_ROOT = TEST_ROOT / "result/phase1235_qwen3_typed_generation_compiler_boundary"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/orthogonal_readout_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
PROGRAM_PATH = OUT_ROOT / "protocol/depth2_nuisance_program_audit.json"
PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

MODEL_PATH = ROOT / "models/hf/qwen3-4b"
AXES = ("object_surface", "value_surface", "template_surface")
PARTITIONS = ("discovery", "confirmation")
LEVELS = (0, 1)
BINDINGS = (0, 1)
QUERY_COUNT = 4
WORLDS_PER_AXIS = 32
ROWS_PER_WORLD = len(LEVELS) * len(BINDINGS) * QUERY_COUNT
EXPECTED_WORLDS = len(AXES) * WORLDS_PER_AXIS
EXPECTED_ROWS = EXPECTED_WORLDS * ROWS_PER_WORLD
VALUE_COUNT = 5
ASSIGNED_COUNT = 4
PROGRAM_DEPTH = 2
PROGRAM_CEILING_MAX = 0.80
MAX_INPUT_LENGTH = 320
GENERATION_BUDGET = 24
GENERATION_BATCH_SIZE = 16
TEACHER_BATCH_SIZE = 8
TIE_TOLERANCE = 1e-7

SYSTEM_PROMPT = "Use only the supplied registry records and follow the requested output contract."

THRESHOLDS = {
    "finite_rate": 1.0,
    "choice_candidate_worst_surface": 0.95,
    "bare_candidate_worst_surface": 0.95,
    "trie_worst_surface": 0.95,
    "candidate_query_quartet": 0.85,
    "candidate_binding_pair": 0.90,
    "candidate_surface_pair": 0.90,
    "bare_exact_worst_surface": 0.90,
    "bare_content_worst_surface": 0.95,
    "cued_exact_worst_surface": 0.90,
    "cued_content_worst_surface": 0.95,
    "short_binding_pair": 0.85,
    "short_surface_pair": 0.85,
    "sentence_content_worst_surface": 0.90,
    "sentence_binding_pair": 0.85,
    "sentence_surface_pair": 0.85,
    "natural_content_worst_surface": 0.90,
    "natural_binding_pair": 0.85,
    "natural_surface_pair": 0.85,
    "program_ceiling": PROGRAM_CEILING_MAX,
}

FIRST_NAMES = (
    "Adrian Alina Amara Ansel Briar Celine Conrad Dahlia Dorian Elara Elias Emilia "
    "Flora Gideon Helena Imani Isadora Julian Leona Lucian Marina Nadia Noemi Orion "
    "Petra Rafael Sabine Selene Silas Talia Tobias Vera Vivian Xavier Yasmin Zora"
).split()
SURNAMES = (
    "Abbott Bellamy Calder Dorsey Ellison Farrow Gresham Hollis Ingram Jarvis Keaton "
    "Langley Mercer Nolan Osborn Prescott Quincy Ramsey Sawyer Thayer Ulrich Varden "
    "Winslow York Arden Blaine Corwin Devlin Everly Fenwick Garland Hadley Iverson "
    "Jensen Kendall Lowell Monroe Northcott Oakley Palmer Reeves Sutton Travers"
).split()
LABEL_LEFT = (
    "bright calm clear cool deep dry fair faint fresh gentle glossy grand light lucid "
    "mellow mild neat pale plain pure quiet rapid rich round sharp sleek soft solid steady "
    "still strong subtle swift warm wide vivid bold crisp dense even firm level matte muted"
).split()
LABEL_RIGHT = (
    "arch beacon bloom bridge cedar circle crest delta ember field flame gate grove harbor "
    "isle jewel key lantern meadow moon oak path peak pearl pine plume quartz reed ridge ring "
    "river seal shore spark spire star stone stream summit torch vale wave wheel wing yard"
).split()
ZONE_WORDS = ("central", "eastern", "northern", "southern")
TEXTURE_WORDS = ("linen", "marble", "oak", "glass")
STATUS_WORDS = ("ready", "stored", "checked", "sealed")

RECORD_TEMPLATES = {
    "base": "Registry record for {object}: marker = {marker}; zone = {zone}; texture = {texture}; status = {status}.",
    "template_a": "The registry lists {object} with marker {marker}, zone {zone}, texture {texture}, and status {status}.",
    "template_b": "For {object}, the dossier gives {marker} as marker, {zone} as zone, {texture} as texture, and {status} as status.",
}
QUERY_TEMPLATES = {
    "base": "Which marker is assigned to {object}?",
    "template_a": "According to the registry, what marker belongs to {object}?",
    "template_b": "Read the exact marker recorded for {object}.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            values[str(row[key])].append(row)
    return values


def lexical_multiset(text: str) -> list[str]:
    return sorted(re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower()))


def verify_upstream() -> tuple[dict[str, Any], dict[str, Any]]:
    final = read_json(UPSTREAM_FINAL_PATH)
    audit = read_json(UPSTREAM_AUDIT_PATH)
    if final.get("final_digest") != EXPECTED_UPSTREAM_FINAL:
        raise RuntimeError("Phase1234 final digest mismatch")
    if audit.get("audit_digest") != EXPECTED_UPSTREAM_AUDIT or audit.get("all_checks_passed") is not True:
        raise RuntimeError("Phase1234 audit mismatch")
    if final.get("authorization", {}).get("future_response_phase") is not False:
        raise RuntimeError("Phase1234 permission boundary changed")
    return final, audit


def direct_token_length(tokenizer: Any, value: str) -> int:
    return len(tokenizer.encode(value, add_special_tokens=False))


def grouped_by_token_length(tokenizer: Any, values: Iterable[str], size: int) -> list[list[str]]:
    buckets: dict[int, list[str]] = defaultdict(list)
    for value in values:
        buckets[direct_token_length(tokenizer, value)].append(value)
    groups: list[list[str]] = []
    for length in sorted(buckets):
        bucket = sorted(set(buckets[length]))
        for start in range(0, len(bucket) - size + 1, size):
            groups.append(bucket[start : start + size])
    return groups


def prior_terms() -> set[str]:
    terms: set[str] = set()
    for row in read_jsonl(UPSTREAM_MATERIAL_PATH):
        terms.update(str(value) for value in row["objects"])
        terms.update(str(value) for value in row["candidates"])
    return terms


def build_lexicon(tokenizer: Any) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    forbidden = prior_terms()
    objects = [f"{first} {surname}" for first in FIRST_NAMES for surname in SURNAMES]
    objects = [value for value in objects if value not in forbidden]
    labels = [f"{left} {right}" for left in LABEL_LEFT for right in LABEL_RIGHT]
    labels = [value for value in labels if value not in forbidden]
    groups = grouped_by_token_length(tokenizer, labels, VALUE_COUNT)
    required_objects = 512
    required_groups = 128
    if len(objects) < required_objects or len(groups) < required_groups:
        raise RuntimeError(f"insufficient new lexicon: {len(objects)}/{required_objects}, {len(groups)}/{required_groups}")
    selected_objects = objects[:required_objects]
    selected_groups = groups[:required_groups]
    audit = {
        "available_new_object_count": len(objects),
        "available_new_label_group_count": len(groups),
        "selected_object_count": len(selected_objects),
        "selected_label_group_count": len(selected_groups),
        "upstream_exact_overlap_count": len((set(selected_objects) | {v for group in selected_groups for v in group}) & forbidden),
        "equal_direct_token_length": all(len({direct_token_length(tokenizer, value) for value in group}) == 1 for group in selected_groups),
    }
    return selected_objects, selected_groups, audit


def render_native(tokenizer: Any, prompt: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def continuation_suffix(tokenizer: Any, rendered: str, continuation: str) -> tuple[list[int], list[int]]:
    prefix = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    appended = [int(value) for value in tokenizer.encode(rendered + continuation, add_special_tokens=False)]
    if appended[: len(prefix)] != prefix or len(appended) == len(prefix):
        raise RuntimeError("continuation does not preserve native prefix")
    return prefix, appended[len(prefix) :]


def candidate_suffixes(tokenizer: Any, rendered: str, candidates: list[str]) -> tuple[list[int], dict[str, list[int]]]:
    prefix: list[int] | None = None
    suffixes: dict[str, list[int]] = {}
    for candidate in candidates:
        current_prefix, suffix = continuation_suffix(tokenizer, rendered, candidate)
        if prefix is None:
            prefix = current_prefix
        if current_prefix != prefix:
            raise RuntimeError("candidate prefixes differ")
        suffixes[candidate] = suffix
    assert prefix is not None
    return prefix, suffixes


def allocate_surfaces(
    axis: str,
    object_cursor: int,
    group_cursor: int,
    objects: list[str],
    groups: list[list[str]],
) -> tuple[list[list[str]], list[list[str]], int, int]:
    if axis == "object_surface":
        object_levels = [objects[object_cursor : object_cursor + 4], objects[object_cursor + 4 : object_cursor + 8]]
        value_levels = [groups[group_cursor], groups[group_cursor]]
        return object_levels, value_levels, object_cursor + 8, group_cursor + 1
    if axis == "value_surface":
        object_level = objects[object_cursor : object_cursor + 4]
        object_levels = [object_level, object_level]
        value_levels = [groups[group_cursor], groups[group_cursor + 1]]
        return object_levels, value_levels, object_cursor + 4, group_cursor + 2
    if axis == "template_surface":
        object_level = objects[object_cursor : object_cursor + 4]
        object_levels = [object_level, object_level]
        value_levels = [groups[group_cursor], groups[group_cursor]]
        return object_levels, value_levels, object_cursor + 4, group_cursor + 1
    raise ValueError(axis)


def style_for(axis: str, level: int) -> str:
    if axis == "template_surface":
        return "template_a" if level == 0 else "template_b"
    return "base"


def candidate_order(candidates: list[str], gold: str, global_row_index: int) -> list[str]:
    position = global_row_index % len(candidates)
    rest = [value for value in candidates if value != gold]
    rng = random.Random(12350000 + global_row_index * 37)
    rng.shuffle(rest)
    rest.insert(position, gold)
    return rest


def render_material_row(
    axis: str,
    world: int,
    level: int,
    binding: int,
    query_index: int,
    object_levels: list[list[str]],
    value_levels: list[list[str]],
    global_row_index: int,
) -> dict[str, Any]:
    objects = object_levels[level]
    candidates = value_levels[level]
    style = style_for(axis, level)
    order = [(index + world) % 4 for index in range(4)]
    displayed_slots = {objects[index]: (index + binding) % 4 for index in range(4)}
    displayed = {obj: candidates[slot] for obj, slot in displayed_slots.items()}
    distractors = {
        obj: {
            "zone": ZONE_WORDS[(world + index) % 4],
            "texture": TEXTURE_WORDS[(world * 3 + index) % 4],
            "status": STATUS_WORDS[(world * 5 + index) % 4],
        }
        for index, obj in enumerate(objects)
    }
    records = " ".join(
        RECORD_TEMPLATES[style].format(
            object=objects[index], marker=displayed[objects[index]], **distractors[objects[index]]
        )
        for index in order
    )
    query_object = objects[query_index]
    query = QUERY_TEMPLATES[style].format(object=query_object)
    base = records + "\n" + query
    gold_slot = displayed_slots[query_object]
    gold = candidates[gold_slot]
    ordered = candidate_order(candidates, gold, global_row_index)
    prompts = {
        "choice": base + f"\nCHOICES: {' / '.join(ordered)}\nAnswer:",
        "bare": base + "\nReturn only the exact two-word marker label.\nAnswer:",
        "cued": base + "\nComplete the field with only the exact two-word label.\nMarker:",
        "sentence": base + (
            "\nAnswer using exactly this sentence form: The marker assigned to the queried object is [two-word marker].\nAnswer:"
        ),
        "natural": base + "\nAnswer the question directly in one concise sentence.\nAnswer:",
    }
    expected_sentence = f"The marker assigned to the queried object is {gold}."
    partition = "discovery" if world < WORLDS_PER_AXIS // 2 else "confirmation"
    identity = {"axis": axis, "world": world, "level": level, "binding": binding, "query": query_index}
    item_id = f"p1235-{digest(identity)[:24]}"
    world_id = f"p1235-{axis}-w{world:03d}"
    row: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1235.typed_readout.row.v1",
        "item_id": item_id,
        "axis": axis,
        "partition": partition,
        "world_id": world_id,
        "world_index": world,
        "surface_level": level,
        "binding_state": binding,
        "query_index": query_index,
        "query_object": query_object,
        "objects": objects,
        "candidates": candidates,
        "displayed_assignments": displayed,
        "displayed_slots": displayed_slots,
        "record_order_indices": order,
        "distractors": distractors,
        "gold": gold,
        "gold_slot": gold_slot,
        "unused_value": candidates[4],
        "candidate_order": ordered,
        "gold_position": ordered.index(gold),
        "template_style": style,
        "prompts": prompts,
        "expected_sentence": expected_sentence,
        "query_group_id": f"query-{digest({'axis': axis, 'world': world, 'level': level, 'binding': binding})[:20]}",
        "surface_pair_id": f"surface-{digest({'axis': axis, 'world': world, 'binding': binding, 'query': query_index})[:20]}",
        "binding_pair_id": f"binding-{digest({'axis': axis, 'world': world, 'level': level, 'query': query_index})[:20]}",
        "prompt_lexical_multiset_digest": digest(lexical_multiset(prompts["bare"])),
    }
    row["row_digest"] = digest(row)
    return row


def build_material(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_terms, value_groups, lexicon_audit = build_lexicon(tokenizer)
    rows: list[dict[str, Any]] = []
    object_cursor = 0
    group_cursor = 0
    row_index = 0
    for axis in AXES:
        for world in range(WORLDS_PER_AXIS):
            object_levels, value_levels, object_cursor, group_cursor = allocate_surfaces(
                axis, object_cursor, group_cursor, object_terms, value_groups
            )
            for level in LEVELS:
                for binding in BINDINGS:
                    for query_index in range(QUERY_COUNT):
                        rows.append(
                            render_material_row(
                                axis, world, level, binding, query_index,
                                object_levels, value_levels, row_index,
                            )
                        )
                        row_index += 1
    if len(rows) != EXPECTED_ROWS or len({row["item_id"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("material cardinality failure")
    lexicon_audit["consumed_object_count"] = object_cursor
    lexicon_audit["consumed_label_group_count"] = group_cursor
    return rows, lexicon_audit


def build_manifest(rows: list[dict[str, Any]], slow: Any, fast: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    slow_fast_mismatch = 0
    candidate_length_mismatch = 0
    max_input = 0
    for execution_index, row in enumerate(rows):
        data: dict[str, Any] = {}
        for readout in ("choice", "bare", "cued", "sentence", "natural"):
            rendered = render_native(slow, row["prompts"][readout])
            if readout in ("choice", "bare"):
                input_ids, suffixes = candidate_suffixes(slow, rendered, row["candidates"])
                data[f"{readout}_candidate_token_ids"] = suffixes
                candidate_length_mismatch += len({len(value) for value in suffixes.values()}) != 1
            else:
                continuation = row["expected_sentence"] if readout == "sentence" else row["gold"]
                input_ids, gold_suffix = continuation_suffix(slow, rendered, continuation)
                data[f"{readout}_gold_token_ids"] = gold_suffix
            fast_ids = [int(value) for value in fast.encode(rendered, add_special_tokens=False)]
            slow_fast_mismatch += input_ids != fast_ids
            data[f"{readout}_input_ids"] = input_ids
            data[f"{readout}_input_token_count"] = len(input_ids)
            data[f"{readout}_rendered_digest"] = digest(rendered)
            max_input = max(max_input, len(input_ids))
        data["bare_gold_token_ids"] = data["bare_candidate_token_ids"][row["gold"]]
        case: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1235.qwen3.manifest.v1",
            "execution_index": execution_index,
            "item_id": row["item_id"],
            "material_row_digest": row["row_digest"],
            "axis": row["axis"],
            "partition": row["partition"],
            "world_id": row["world_id"],
            "surface_level": row["surface_level"],
            "binding_state": row["binding_state"],
            "query_index": row["query_index"],
            "gold": row["gold"],
            "gold_slot": row["gold_slot"],
            "candidates": row["candidates"],
            "candidate_order": row["candidate_order"],
            "query_group_id": row["query_group_id"],
            "surface_pair_id": row["surface_pair_id"],
            "binding_pair_id": row["binding_pair_id"],
            **data,
        }
        case["manifest_row_digest"] = digest(case)
        manifest.append(case)
    summary = {
        "row_count": len(manifest),
        "manifest_digest": digest(manifest),
        "slow_tokenizer_class": type(slow).__name__,
        "fast_tokenizer_class": type(fast).__name__,
        "slow_fast_mismatch_count": slow_fast_mismatch,
        "candidate_suffix_length_mismatch_count": candidate_length_mismatch,
        "maximum_input_length": max_input,
        "model_weights_loaded": False,
    }
    summary["tokenizer_gate"] = bool(
        len(manifest) == EXPECTED_ROWS
        and slow_fast_mismatch == 0
        and candidate_length_mismatch == 0
        and max_input <= MAX_INPUT_LENGTH
    )
    return manifest, summary


def base_program_predictions(row: dict[str, Any]) -> dict[str, str]:
    candidates = row["candidates"]
    displayed = row["displayed_assignments"]
    objects = row["objects"]
    query = int(row["query_index"])
    programs = {f"candidate_position_{index}": row["candidate_order"][index] for index in range(5)}
    programs.update({f"fixed_object_{index}": displayed[objects[index]] for index in range(4)})
    programs.update(
        {
            "first_record": displayed[objects[row["record_order_indices"][0]]],
            "last_record": displayed[objects[row["record_order_indices"][-1]]],
            "next_object": displayed[objects[(query + 1) % 4]],
            "opposite_object": displayed[objects[(query + 2) % 4]],
            "previous_object": displayed[objects[(query + 3) % 4]],
            "unused_value": row["unused_value"],
            "lexical_first": sorted(candidates)[0],
            "lexical_last": sorted(candidates)[-1],
        }
    )
    return programs


def nuisance_features(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "axis": row["axis"],
        "partition": row["partition"],
        "surface_level": row["surface_level"],
        "binding_state": row["binding_state"],
        "record_first_index": row["record_order_indices"][0],
        "unused_candidate_position": row["candidate_order"].index(row["unused_value"]),
    }


def optimal_depth_tree_accuracy(rows: list[dict[str, Any]], depth: int) -> tuple[float, dict[str, Any]]:
    names = sorted(base_program_predictions(rows[0]))
    predictions = [base_program_predictions(row) for row in rows]
    features = [nuisance_features(row) for row in rows]
    conditions = sorted({(name, canonical_json(value)) for feature in features for name, value in feature.items()})
    decoded = {(name, encoded): json.loads(encoded) for name, encoded in conditions}
    memo: dict[tuple[tuple[int, ...], int], tuple[int, dict[str, Any]]] = {}

    def solve(indices: tuple[int, ...], remaining: int) -> tuple[int, dict[str, Any]]:
        key = (indices, remaining)
        if key in memo:
            return memo[key]
        scores = {name: sum(predictions[index][name] == rows[index]["gold"] for index in indices) for name in names}
        maximum = max(scores.values())
        leaf = min(name for name in names if scores[name] == maximum)
        best_score = scores[leaf]
        best_tree: dict[str, Any] = {"leaf_program": leaf, "correct": best_score, "n": len(indices)}
        if remaining:
            for name, encoded in conditions:
                value = decoded[(name, encoded)]
                left = tuple(index for index in indices if features[index][name] == value)
                right = tuple(index for index in indices if features[index][name] != value)
                if not left or not right:
                    continue
                left_score, left_tree = solve(left, remaining - 1)
                right_score, right_tree = solve(right, remaining - 1)
                score = left_score + right_score
                tree = {
                    "condition": {"feature": name, "equals": value},
                    "true": left_tree,
                    "false": right_tree,
                    "correct": score,
                    "n": len(indices),
                }
                if score > best_score or (score == best_score and canonical_json(tree) < canonical_json(best_tree)):
                    best_score, best_tree = score, tree
        memo[key] = (best_score, best_tree)
        return memo[key]

    correct, witness = solve(tuple(range(len(rows))), depth)
    return correct / len(rows), witness


def build_program_audit(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    split_results: dict[str, Any] = {}
    for axis in AXES:
        for partition in PARTITIONS:
            selected = [row for row in rows if row["axis"] == axis and row["partition"] == partition]
            base = {
                name: sum(base_program_predictions(row)[name] == row["gold"] for row in selected) / len(selected)
                for name in sorted(base_program_predictions(selected[0]))
            }
            ceiling, tree = optimal_depth_tree_accuracy(selected, PROGRAM_DEPTH)
            split_results[f"{axis}|{partition}"] = {
                "base_program_accuracies": base,
                "maximum_base_program_accuracy": max(base.values()),
                "depth2_conditional_program_accuracy": ceiling,
                "depth2_witness_tree": tree,
                "construct_gate": ceiling <= PROGRAM_CEILING_MAX,
            }
    manifest_by_id = {row["item_id"]: row for row in manifest}
    query_groups = grouped(rows, "query_group_id")
    surface_groups = grouped(rows, "surface_pair_id")
    binding_groups = grouped(rows, "binding_pair_id")
    collisions = {
        "query_quartets_complete": all(len(cell) == 4 and len({row["gold_slot"] for row in cell}) == 4 for cell in query_groups.values()),
        "surface_pairs_semantically_invariant": all(len(cell) == 2 and len({row["gold_slot"] for row in cell}) == 1 for cell in surface_groups.values()),
        "binding_pairs_discriminating": all(
            len(cell) == 2
            and len({row["gold_slot"] for row in cell}) == 2
            and len({row["prompt_lexical_multiset_digest"] for row in cell}) == 1
            and len({digest(sorted(manifest_by_id[row["item_id"]]["bare_input_ids"])) for row in cell}) == 1
            for cell in binding_groups.values()
        ),
        "unused_never_gold": all(row["unused_value"] != row["gold"] for row in rows),
        "five_candidates_four_assigned": all(len(row["candidates"]) == 5 and len(set(row["displayed_slots"].values())) == 4 for row in rows),
    }
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1235.depth2_nuisance_program.v1",
        "created_at_utc": utc_now(),
        "threat_model": {
            "base_programs": sorted(base_program_predictions(rows[0])),
            "nuisance_features": sorted(nuisance_features(rows[0])),
            "maximum_depth": PROGRAM_DEPTH,
            "ceiling_threshold": PROGRAM_CEILING_MAX,
            "query_identity_branching_excluded_as_target_equivalent": True,
        },
        "split_results": split_results,
        "collision_group_counts": {
            "query_quartets": len(query_groups),
            "surface_pairs": len(surface_groups),
            "binding_pairs": len(binding_groups),
        },
        "collision_checks": collisions,
        "target_equivalent_witness_accuracy": sum(row["displayed_assignments"][row["query_object"]] == row["gold"] for row in rows) / len(rows),
        "claim_boundary": "Construct separation is relative to the frozen depth-2 nuisance grammar and does not identify a unique neural algorithm.",
    }
    value["program_construct_gate"] = bool(
        all(cell["construct_gate"] for cell in split_results.values())
        and all(collisions.values())
        and value["target_equivalent_witness_accuracy"] == 1.0
    )
    value["program_audit_digest"] = digest(value)
    return value


def build_batch_plan(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for readout in ("choice", "bare", "cued", "sentence", "natural"):
        counts = Counter(row[f"{readout}_input_token_count"] for row in manifest)
        views[readout] = {"case_count": len(manifest), "length_bucket_count": len(counts), "length_counts": dict(sorted(counts.items()))}
    value = {
        "phase": PHASE,
        "schema_version": "phase1235.batch_plan.v1",
        "candidate_batch_contract": "phase1220 shared-prefix cache plus phase1221 low-margin direct-prefill fallback",
        "generation_budget_each_free_readout": GENERATION_BUDGET,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "teacher_batch_size": TEACHER_BATCH_SIZE,
        "adaptive_semantic_filtering": False,
        "views": views,
        "execution_item_ids": [row["item_id"] for row in manifest],
    }
    value["plan_digest"] = digest(value)
    return value


def source_hashes() -> dict[str, str]:
    return {
        "main": file_sha256(SCRIPT),
        "audit": file_sha256(AUDIT_SCRIPT),
        "phase1220_scorer": file_sha256(P1220_SCRIPT),
        "phase1221_scorer": file_sha256(P1221_SCRIPT),
    }


def preregister() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("Phase1235 output directory already exists")
    upstream, upstream_audit = verify_upstream()
    from transformers import AutoTokenizer, __version__ as transformers_version

    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    rows, lexicon_audit = build_material(slow)
    manifest, tokenizer_summary = build_manifest(rows, slow, fast)
    program = build_program_audit(rows, manifest)
    if not tokenizer_summary["tokenizer_gate"] or not program["program_construct_gate"]:
        raise RuntimeError("zero-model qualification failed")
    plan = build_batch_plan(manifest)
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1235.typed_generation_compiler_boundary.v1",
        "created_at_utc": utc_now(),
        "objective": "Hold query-object binding fixed and locate the first failed transition from complete-sequence selection to unconstrained string realization across orthogonal surface axes.",
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1234_final_digest": upstream["final_digest"],
            "phase1234_final_audit_digest": upstream_audit["audit_digest"],
            "phase1234_failure_not_reclassified": True,
        },
        "material": {
            "axes": list(AXES),
            "partitions": list(PARTITIONS),
            "worlds_per_axis": WORLDS_PER_AXIS,
            "independent_cluster_unit": "axis-world",
            "surface_levels": list(LEVELS),
            "binding_states": list(BINDINGS),
            "queries_per_cell": QUERY_COUNT,
            "row_count": len(rows),
            "material_digest": digest(rows),
            "lexicon_audit": lexicon_audit,
        },
        "readout_contracts": {
            "choice_sequence_ranking": "complete five-candidate continuation ranking with choices shown",
            "bare_sequence_ranking": "complete five-candidate continuation ranking with no choices shown",
            "bare_candidate_trie": "local greedy decoding constrained only by the five candidate token tries; candidates are not shown",
            "bare_short_generation": "full-vocabulary greedy generation from bare exact-label prompt",
            "field_cued_short_generation": "full-vocabulary greedy generation after Marker: cue",
            "exact_sentence_generation": "full-vocabulary greedy generation under a frozen exact sentence contract",
            "natural_sentence_generation": "full-vocabulary greedy generation under a concise natural answer contract",
            "teacher_forced_diagnostics": "gold-token full-vocabulary ranks and margins under bare, cued, and sentence prefixes",
        },
        "interface": {
            "model": "qwen3",
            "device": "cuda",
            "precision": "float16",
            "quantization": "none",
            "native_chat_template": True,
            "enable_thinking": False,
            "greedy_generation_budget": GENERATION_BUDGET,
            "same_budget_across_free_readouts": True,
            "tokenizer_summary": tokenizer_summary,
            "transformers_version": transformers_version,
        },
        "program_construct": {
            "program_audit_digest": program["program_audit_digest"],
            "maximum_depth": PROGRAM_DEPTH,
            "maximum_ceiling": PROGRAM_CEILING_MAX,
            "gate": program["program_construct_gate"],
        },
        "thresholds": THRESHOLDS,
        "typed_authorization": {
            "candidate_selection_only": "choice + bare sequence ranking + candidate-trie gates; never authorizes hidden state",
            "short_string": "bare + field-cued short-string gates; cannot claim sentence generation",
            "sentence": "exact-sentence content gate; cannot claim general generation",
            "natural": "natural-sentence content gate; cannot claim causal use",
            "future_response": "program construct + candidate selection + short string + sentence + natural gates",
        },
        "execution": {
            "batch_plan_digest": plan["plan_digest"],
            "hidden_states": False,
            "attentions": False,
            "interventions": False,
            "cross_model": False,
        },
        "forbidden": [
            "modify prompts, parser, generation budget, grammar, thresholds, or denominators after preregistration",
            "reuse Phase1234 outputs as Phase1235 evidence",
            "treat teacher-forced rank as free-generation success",
            "inspect hidden states or attentions",
            "run interventions or cross-model rescue",
            "claim separate neural modules from typed behavior differences",
        ],
        "claim_boundary": [
            "All materials are generated English registries, not natural knowledge domains.",
            "The phase identifies behavior-level readout boundaries, not hidden content states or modules.",
            "Candidate-trie decoding is an external decoder diagnostic, not a natural model capability.",
            "Only a fully preregistered cross-readout pass can authorize a separate future-response phase.",
        ],
    }
    contract["contract_digest"] = digest(contract)
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(PROGRAM_PATH, program)
    write_json(PLAN_PATH, plan)
    write_json(CONTRACT_PATH, contract)
    print(canonical_json({
        "status": "phase1235_preregistered",
        "rows": len(rows),
        "worlds": EXPECTED_WORLDS,
        "program_gate": program["program_construct_gate"],
        "contract_digest": contract["contract_digest"],
    }))


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    verify_upstream()
    contract = read_json(CONTRACT_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_PATH)
    plan = read_json(PLAN_PATH)
    if contract["contract_digest"] != digest(strip_digest(contract, "contract_digest")):
        raise RuntimeError("contract drift")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("source drift")
    if contract["material"]["material_digest"] != digest(rows):
        raise RuntimeError("material drift")
    if contract["interface"]["tokenizer_summary"]["manifest_digest"] != digest(manifest):
        raise RuntimeError("manifest drift")
    if program["program_audit_digest"] != digest(strip_digest(program, "program_audit_digest")):
        raise RuntimeError("program drift")
    if plan["plan_digest"] != digest(strip_digest(plan, "plan_digest")):
        raise RuntimeError("plan drift")
    if read_json(PREAUDIT_PATH).get("all_checks_passed") is not True:
        raise RuntimeError("independent preaudit missing or failed")
    return contract, rows, manifest, program, plan


def score_manifest_view(manifest: list[dict[str, Any]], readout: str) -> list[dict[str, Any]]:
    return [
        {
            "item_id": row["item_id"],
            "input_ids": row[f"{readout}_input_ids"],
            "input_token_count": row[f"{readout}_input_token_count"],
            "candidates": row["candidates"],
            "candidate_order": row["candidate_order"],
            "candidate_token_ids": row[f"{readout}_candidate_token_ids"],
            "generation_required": False,
        }
        for row in manifest
    ]


def fixed_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    manifest: list[dict[str, Any]],
    readout: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    entries = [
        {
            "item_id": row["item_id"],
            "input_ids": row[f"{readout}_input_ids"],
            "input_token_count": row[f"{readout}_input_token_count"],
        }
        for row in manifest
    ]
    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    eos = tokenizer.eos_token_id
    batches = 0
    for batch in p1220.homogeneous_batches(entries, GENERATION_BATCH_SIZE, "input_token_count"):
        input_ids = torch.tensor([entry["input_ids"] for entry in batch], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=GENERATION_BUDGET,
                eos_token_id=eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        suffixes = generated.sequences[:, input_ids.shape[1] :].detach().cpu().tolist()
        for index, entry in enumerate(batch):
            raw_suffix = [int(value) for value in suffixes[index]]
            terminated = eos is not None and eos in raw_suffix
            suffix = raw_suffix[: raw_suffix.index(eos)] if terminated else raw_suffix
            results[entry["item_id"]] = {
                "generated_token_ids": suffix,
                "raw_generated_token_count": len(raw_suffix),
                "terminated_by_eos": terminated,
                "reached_budget": not terminated and len(raw_suffix) >= GENERATION_BUDGET,
                "generated_text": tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False),
            }
        del generated, input_ids
        batches += 1
        if batches % 100 == 0:
            print(f"[phase1235/{readout}] batches={batches}", flush=True)
    return results, {"readout": readout, "case_count": len(entries), "batch_count": batches, "elapsed_seconds": time.time() - started}


def trie_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    manifest: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    entries = [
        {
            "item_id": row["item_id"],
            "input_ids": row["bare_input_ids"],
            "input_token_count": row["bare_input_token_count"],
            "candidate_token_ids": row["bare_candidate_token_ids"],
        }
        for row in manifest
    ]
    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    eos = int(tokenizer.eos_token_id)
    batches = 0
    for batch in p1220.homogeneous_batches(entries, GENERATION_BATCH_SIZE, "input_token_count"):
        prompt_length = int(batch[0]["input_token_count"])
        input_ids = torch.tensor([entry["input_ids"] for entry in batch], dtype=torch.long, device=device)
        candidate_sequences = [
            {candidate: tuple(int(token) for token in tokens) for candidate, tokens in entry["candidate_token_ids"].items()}
            for entry in batch
        ]

        def allowed(batch_id: int, sequence: torch.Tensor) -> list[int]:
            prefix = tuple(int(value) for value in sequence[prompt_length:].tolist())
            sequences = candidate_sequences[batch_id]
            if any(prefix == candidate for candidate in sequences.values()):
                return [eos]
            options = sorted({tokens[len(prefix)] for tokens in sequences.values() if len(tokens) > len(prefix) and tokens[: len(prefix)] == prefix})
            return options or [eos]

        max_candidate = max(len(tokens) for sequences in candidate_sequences for tokens in sequences.values())
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=min(GENERATION_BUDGET, max_candidate + 1),
                eos_token_id=eos,
                pad_token_id=int(tokenizer.pad_token_id),
                prefix_allowed_tokens_fn=allowed,
                return_dict_in_generate=True,
            )
        suffixes = generated.sequences[:, prompt_length:].detach().cpu().tolist()
        for index, entry in enumerate(batch):
            raw_suffix = [int(value) for value in suffixes[index]]
            suffix = raw_suffix[: raw_suffix.index(eos)] if eos in raw_suffix else raw_suffix
            matches = [candidate for candidate, tokens in candidate_sequences[index].items() if tuple(suffix) == tokens]
            results[entry["item_id"]] = {
                "generated_token_ids": suffix,
                "generated_text": tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                "prediction": matches[0] if len(matches) == 1 else None,
                "terminated_by_eos": eos in raw_suffix,
            }
        del generated, input_ids
        batches += 1
        if batches % 100 == 0:
            print(f"[phase1235/trie] batches={batches}", flush=True)
    return results, {"readout": "bare_candidate_trie", "case_count": len(entries), "batch_count": batches, "elapsed_seconds": time.time() - started}


def teacher_forced_diagnostics(
    model: Any,
    device: torch.device,
    manifest: list[dict[str, Any]],
    readout: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    entries = []
    for row in manifest:
        gold_tokens = row[f"{readout}_gold_token_ids"]
        prompt = row[f"{readout}_input_ids"]
        entries.append({
            "item_id": row["item_id"],
            "prompt": prompt,
            "gold": gold_tokens,
            "total_length": len(prompt) + len(gold_tokens),
        })
    results: dict[str, dict[str, Any]] = {}
    batches = 0
    started = time.time()
    for batch in p1220.homogeneous_batches(entries, TEACHER_BATCH_SIZE, "total_length"):
        sequences = [entry["prompt"] + entry["gold"] for entry in batch]
        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
        max_gold = max(len(entry["gold"]) for entry in batch)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=max_gold + 1,
                return_dict=True,
            )
        output_start = input_ids.shape[1] - output.logits.shape[1]
        for batch_index, entry in enumerate(batch):
            ranks: list[int] = []
            margins: list[float] = []
            top_ids: list[int] = []
            log_probs: list[float] = []
            finite = True
            for offset, token_id in enumerate(entry["gold"]):
                absolute = len(entry["prompt"]) + offset - 1
                logits = output.logits[batch_index, absolute - output_start].float()
                finite = finite and bool(torch.isfinite(logits).all().item())
                gold_logit = logits[int(token_id)]
                top_values, top_indices = torch.topk(logits, k=2)
                top_id = int(top_indices[0].item())
                other = top_values[1] if top_id == int(token_id) else top_values[0]
                ranks.append(int((logits > gold_logit).sum().item()) + 1)
                margins.append(float((gold_logit - other).item()))
                top_ids.append(top_id)
                log_probs.append(float((gold_logit - torch.logsumexp(logits, dim=-1)).item()))
            first_not_top1 = next((index for index, rank in enumerate(ranks) if rank != 1), None)
            results[entry["item_id"]] = {
                "gold_token_ids": [int(value) for value in entry["gold"]],
                "gold_token_ranks": ranks,
                "gold_token_margins": margins,
                "gold_token_log_probabilities": log_probs,
                "teacher_top_token_ids": top_ids,
                "all_gold_tokens_top1": all(rank == 1 for rank in ranks),
                "first_not_top1_index": first_not_top1,
                "minimum_gold_margin": min(margins),
                "mean_gold_rank": sum(ranks) / len(ranks),
                "all_vocab_logits_finite": finite,
            }
        del output, input_ids
        batches += 1
        if batches % 100 == 0:
            print(f"[phase1235/teacher-{readout}] batches={batches}", flush=True)
    return results, {"readout": f"teacher_{readout}", "case_count": len(entries), "batch_count": batches, "elapsed_seconds": time.time() - started}


def normalize_text(text: str) -> str:
    value = text.strip() if text.strip() else ""
    value = value.strip().strip(string.whitespace + string.punctuation)
    return re.sub(r"\s+", " ", value.lower())


def phrase_present(normalized: str, phrase: str) -> bool:
    target = normalize_text(phrase)
    return re.search(rf"(?<!\w){re.escape(target)}(?!\w)", normalized) is not None


def parse_free_output(
    generation: dict[str, Any],
    candidates: list[str],
    gold: str,
    query_object: str,
    expected_exact: str,
) -> dict[str, Any]:
    normalized = normalize_text(generation["generated_text"])
    mentions = [candidate for candidate in candidates if phrase_present(normalized, candidate)]
    prediction = mentions[0] if len(mentions) == 1 else None
    exact = normalized == normalize_text(expected_exact)
    content = prediction == gold
    gold_words = normalize_text(gold).split()
    if exact:
        category = "exact"
    elif content:
        category = "gold_with_extra"
    elif len(mentions) > 1:
        category = "multiple_candidates"
    elif prediction is not None:
        category = "wrong_complete_candidate"
    elif gold_words and gold_words[0] in normalized.split():
        category = "gold_prefix_fragment"
    elif len(gold_words) > 1 and gold_words[-1] in normalized.split():
        category = "gold_suffix_fragment"
    elif phrase_present(normalized, query_object):
        category = "query_object_restatement"
    elif "marker" in normalized.split():
        category = "relation_restatement"
    elif generation["reached_budget"]:
        category = "budget_without_candidate"
    else:
        category = "other_unparsed"
    return {
        "normalized_text": normalized,
        "mentioned_candidates": mentions,
        "prediction": prediction,
        "exact": exact,
        "content_correct": content,
        "error_category": category,
    }


def argmax_set(scores: dict[str, float]) -> list[str]:
    maximum = max(scores.values())
    return sorted(name for name, value in scores.items() if maximum - value <= TIE_TOLERANCE)


def run_qwen3() -> None:
    if RAW_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("Phase1235 outputs already exist")
    contract, rows, manifest, _program, plan = verify_frozen()
    material_by_id = {row["item_id"]: row for row in rows}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    if device.type != "cuda" or precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
        release_fp16(model)
        raise RuntimeError("numerical contract failed")
    try:
        choice_scores, choice_runtime = p1221.score_candidates_with_fallback(model, device, score_manifest_view(manifest, "choice"))
        bare_scores, bare_runtime = p1221.score_candidates_with_fallback(model, device, score_manifest_view(manifest, "bare"))
        trie, trie_runtime = trie_generation(model, tokenizer, device, manifest)
        bare_gen, bare_gen_runtime = fixed_generation(model, tokenizer, device, manifest, "bare")
        cued_gen, cued_gen_runtime = fixed_generation(model, tokenizer, device, manifest, "cued")
        sentence_gen, sentence_gen_runtime = fixed_generation(model, tokenizer, device, manifest, "sentence")
        natural_gen, natural_gen_runtime = fixed_generation(model, tokenizer, device, manifest, "natural")
        bare_teacher, bare_teacher_runtime = teacher_forced_diagnostics(model, device, manifest, "bare")
        cued_teacher, cued_teacher_runtime = teacher_forced_diagnostics(model, device, manifest, "cued")
        sentence_teacher, sentence_teacher_runtime = teacher_forced_diagnostics(model, device, manifest, "sentence")

        raw: list[dict[str, Any]] = []
        for manifest_row in manifest:
            item_id = manifest_row["item_id"]
            material = material_by_id[item_id]
            choice_sum = {candidate: float(value["sum_log_probability"]) for candidate, value in choice_scores[item_id].items()}
            bare_sum = {candidate: float(value["sum_log_probability"]) for candidate, value in bare_scores[item_id].items()}
            choice_set = argmax_set(choice_sum)
            bare_set = argmax_set(bare_sum)
            choice_prediction = choice_set[0] if len(choice_set) == 1 else None
            bare_prediction = bare_set[0] if len(bare_set) == 1 else None
            trie_value = trie[item_id]
            bare_parse = parse_free_output(bare_gen[item_id], material["candidates"], material["gold"], material["query_object"], material["gold"])
            cued_parse = parse_free_output(cued_gen[item_id], material["candidates"], material["gold"], material["query_object"], material["gold"])
            sentence_parse = parse_free_output(sentence_gen[item_id], material["candidates"], material["gold"], material["query_object"], material["expected_sentence"])
            natural_parse = parse_free_output(natural_gen[item_id], material["candidates"], material["gold"], material["query_object"], material["gold"])

            def slot(prediction: str | None) -> int | None:
                return material["candidates"].index(prediction) if prediction in material["candidates"] else None

            row: dict[str, Any] = {
                "phase": PHASE,
                "schema_version": "phase1235.qwen3.behavior.row.v1",
                "contract_digest": contract["contract_digest"],
                "item_id": item_id,
                "manifest_row_digest": manifest_row["manifest_row_digest"],
                "execution_index": manifest_row["execution_index"],
                "axis": material["axis"],
                "partition": material["partition"],
                "world_id": material["world_id"],
                "surface_level": material["surface_level"],
                "binding_state": material["binding_state"],
                "query_index": material["query_index"],
                "gold": material["gold"],
                "gold_slot": material["gold_slot"],
                "query_group_id": material["query_group_id"],
                "surface_pair_id": material["surface_pair_id"],
                "binding_pair_id": material["binding_pair_id"],
                "choice_candidate_scores": choice_scores[item_id],
                "bare_candidate_scores": bare_scores[item_id],
                "choice_argmax_set": choice_set,
                "bare_argmax_set": bare_set,
                "choice_prediction": choice_prediction,
                "bare_candidate_prediction": bare_prediction,
                "choice_prediction_slot": slot(choice_prediction),
                "bare_candidate_prediction_slot": slot(bare_prediction),
                "choice_correct": choice_prediction == material["gold"],
                "bare_candidate_correct": bare_prediction == material["gold"],
                "choice_finite": all(value["all_vocab_logits_finite"] for value in choice_scores[item_id].values()),
                "bare_candidate_finite": all(value["all_vocab_logits_finite"] for value in bare_scores[item_id].values()),
                "trie_generation": trie_value,
                "trie_prediction_slot": slot(trie_value["prediction"]),
                "trie_correct": trie_value["prediction"] == material["gold"],
                "bare_generation": bare_gen[item_id],
                "bare_parse": bare_parse,
                "bare_prediction_slot": slot(bare_parse["prediction"]),
                "bare_exact": bare_parse["exact"],
                "bare_content_correct": bare_parse["content_correct"],
                "cued_generation": cued_gen[item_id],
                "cued_parse": cued_parse,
                "cued_prediction_slot": slot(cued_parse["prediction"]),
                "cued_exact": cued_parse["exact"],
                "cued_content_correct": cued_parse["content_correct"],
                "sentence_generation": sentence_gen[item_id],
                "sentence_parse": sentence_parse,
                "sentence_prediction_slot": slot(sentence_parse["prediction"]),
                "sentence_exact": sentence_parse["exact"],
                "sentence_content_correct": sentence_parse["content_correct"],
                "natural_generation": natural_gen[item_id],
                "natural_parse": natural_parse,
                "natural_prediction_slot": slot(natural_parse["prediction"]),
                "natural_exact": natural_parse["exact"],
                "natural_content_correct": natural_parse["content_correct"],
                "bare_teacher": bare_teacher[item_id],
                "cued_teacher": cued_teacher[item_id],
                "sentence_teacher": sentence_teacher[item_id],
            }
            row["behavior_row_digest"] = digest(row)
            raw.append(row)
        raw.sort(key=lambda row: row["execution_index"])
        write_jsonl(RAW_PATH, raw)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1235.qwen3.run_summary.v1",
            "created_at_utc": utc_now(),
            "model": "qwen3",
            "contract_digest": contract["contract_digest"],
            "case_count": len(raw),
            "raw_digest": digest(raw),
            "batch_plan_digest": plan["plan_digest"],
            "runtimes": {
                "choice_scores": choice_runtime,
                "bare_scores": bare_runtime,
                "trie": trie_runtime,
                "bare_generation": bare_gen_runtime,
                "cued_generation": cued_gen_runtime,
                "sentence_generation": sentence_gen_runtime,
                "natural_generation": natural_gen_runtime,
                "bare_teacher": bare_teacher_runtime,
                "cued_teacher": cued_teacher_runtime,
                "sentence_teacher": sentence_teacher_runtime,
            },
            "precision_audit": precision,
            "placement": placement,
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
        write_json(SUMMARY_PATH, summary)
        print(canonical_json({"status": "phase1235_behavior_complete", "rows": len(raw), "summary_digest": summary["summary_digest"]}))
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


READOUT_FIELDS = {
    "choice": ("choice_correct", "choice_prediction_slot"),
    "bare_candidate": ("bare_candidate_correct", "bare_candidate_prediction_slot"),
    "trie": ("trie_correct", "trie_prediction_slot"),
    "bare": ("bare_content_correct", "bare_prediction_slot"),
    "cued": ("cued_content_correct", "cued_prediction_slot"),
    "sentence": ("sentence_content_correct", "sentence_prediction_slot"),
    "natural": ("natural_content_correct", "natural_prediction_slot"),
}


def group_success(rows: list[dict[str, Any]], key: str, readout: str, size: int, mode: str) -> float:
    correct_field, slot_field = READOUT_FIELDS[readout]
    outcomes: list[bool] = []
    for cell in grouped(rows, key).values():
        slots = [row[slot_field] for row in cell]
        success = len(cell) == size and all(row[correct_field] for row in cell)
        if mode == "distinct":
            success = success and len(set(slots)) == size
        elif mode == "same_slot":
            success = success and len(set(slots)) == 1
        elif mode == "different_slot":
            success = success and len(set(slots)) == size
        else:
            raise ValueError(mode)
        outcomes.append(success)
    return sum(outcomes) / len(outcomes) if outcomes else float("nan")


def worst_surface(rows: list[dict[str, Any]], field: str) -> tuple[float, dict[str, float]]:
    cells = {
        f"level{level}|binding{binding}": rate(
            [row for row in rows if row["surface_level"] == level and row["binding_state"] == binding], field
        )
        for level in LEVELS for binding in BINDINGS
    }
    return min(cells.values()), cells


def adjudicate(raw: list[dict[str, Any]], program: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    candidate_passes: list[bool] = []
    short_passes: list[bool] = []
    sentence_passes: list[bool] = []
    natural_passes: list[bool] = []
    for axis in AXES:
        for partition in PARTITIONS:
            key = f"{axis}|{partition}"
            selected = [row for row in raw if row["axis"] == axis and row["partition"] == partition]
            worst: dict[str, float] = {}
            by_surface: dict[str, Any] = {}
            for name, field in (
                ("choice", "choice_correct"),
                ("bare_candidate", "bare_candidate_correct"),
                ("trie", "trie_correct"),
                ("bare_exact", "bare_exact"),
                ("bare_content", "bare_content_correct"),
                ("cued_exact", "cued_exact"),
                ("cued_content", "cued_content_correct"),
                ("sentence_exact", "sentence_exact"),
                ("sentence_content", "sentence_content_correct"),
                ("natural_exact", "natural_exact"),
                ("natural_content", "natural_content_correct"),
            ):
                worst[name], by_surface[name] = worst_surface(selected, field)
            metrics = {
                "case_count": len(selected),
                "finite_rate": sum(row["choice_finite"] and row["bare_candidate_finite"] for row in selected) / len(selected),
                "overall": {name: rate(selected, field) for name, field in (
                    ("choice", "choice_correct"), ("bare_candidate", "bare_candidate_correct"),
                    ("trie", "trie_correct"), ("bare_exact", "bare_exact"),
                    ("bare_content", "bare_content_correct"), ("cued_exact", "cued_exact"),
                    ("cued_content", "cued_content_correct"), ("sentence_exact", "sentence_exact"),
                    ("sentence_content", "sentence_content_correct"), ("natural_exact", "natural_exact"),
                    ("natural_content", "natural_content_correct"),
                )},
                "worst_surface": worst,
                "by_surface": by_surface,
                "query_quartet": {name: group_success(selected, "query_group_id", name, 4, "distinct") for name in READOUT_FIELDS},
                "binding_pair": {name: group_success(selected, "binding_pair_id", name, 2, "different_slot") for name in READOUT_FIELDS},
                "surface_pair": {name: group_success(selected, "surface_pair_id", name, 2, "same_slot") for name in READOUT_FIELDS},
                "teacher_all_top1": {
                    readout: sum(row[f"{readout}_teacher"]["all_gold_tokens_top1"] for row in selected) / len(selected)
                    for readout in ("bare", "cued", "sentence")
                },
                "teacher_first_not_top1": {
                    readout: dict(Counter(
                        str(row[f"{readout}_teacher"]["first_not_top1_index"]) for row in selected
                    ))
                    for readout in ("bare", "cued", "sentence")
                },
                "error_categories": {
                    readout: dict(Counter(row[f"{readout}_parse"]["error_category"] for row in selected))
                    for readout in ("bare", "cued", "sentence", "natural")
                },
                "depth2_program_ceiling": program["split_results"][key]["depth2_conditional_program_accuracy"],
            }
            candidate_gates = {
                "finite": metrics["finite_rate"] >= THRESHOLDS["finite_rate"],
                "choice_surface": worst["choice"] >= THRESHOLDS["choice_candidate_worst_surface"],
                "bare_candidate_surface": worst["bare_candidate"] >= THRESHOLDS["bare_candidate_worst_surface"],
                "trie_surface": worst["trie"] >= THRESHOLDS["trie_worst_surface"],
                "query_quartet": min(metrics["query_quartet"][name] for name in ("choice", "bare_candidate", "trie")) >= THRESHOLDS["candidate_query_quartet"],
                "binding_pair": min(metrics["binding_pair"][name] for name in ("choice", "bare_candidate", "trie")) >= THRESHOLDS["candidate_binding_pair"],
                "surface_pair": min(metrics["surface_pair"][name] for name in ("choice", "bare_candidate", "trie")) >= THRESHOLDS["candidate_surface_pair"],
                "program": metrics["depth2_program_ceiling"] <= THRESHOLDS["program_ceiling"],
            }
            short_gates = {
                "bare_exact": worst["bare_exact"] >= THRESHOLDS["bare_exact_worst_surface"],
                "bare_content": worst["bare_content"] >= THRESHOLDS["bare_content_worst_surface"],
                "cued_exact": worst["cued_exact"] >= THRESHOLDS["cued_exact_worst_surface"],
                "cued_content": worst["cued_content"] >= THRESHOLDS["cued_content_worst_surface"],
                "binding_pair": min(metrics["binding_pair"][name] for name in ("bare", "cued")) >= THRESHOLDS["short_binding_pair"],
                "surface_pair": min(metrics["surface_pair"][name] for name in ("bare", "cued")) >= THRESHOLDS["short_surface_pair"],
            }
            sentence_gates = {
                "content_surface": worst["sentence_content"] >= THRESHOLDS["sentence_content_worst_surface"],
                "binding_pair": metrics["binding_pair"]["sentence"] >= THRESHOLDS["sentence_binding_pair"],
                "surface_pair": metrics["surface_pair"]["sentence"] >= THRESHOLDS["sentence_surface_pair"],
            }
            natural_gates = {
                "content_surface": worst["natural_content"] >= THRESHOLDS["natural_content_worst_surface"],
                "binding_pair": metrics["binding_pair"]["natural"] >= THRESHOLDS["natural_binding_pair"],
                "surface_pair": metrics["surface_pair"]["natural"] >= THRESHOLDS["natural_surface_pair"],
            }
            typed = {
                "candidate_selection": all(candidate_gates.values()),
                "short_string": all(short_gates.values()),
                "sentence": all(sentence_gates.values()),
                "natural": all(natural_gates.values()),
            }
            cells[key] = {
                "metrics": metrics,
                "gates": {
                    "candidate_selection": candidate_gates,
                    "short_string": short_gates,
                    "sentence": sentence_gates,
                    "natural": natural_gates,
                },
                "typed_pass": typed,
            }
            candidate_passes.append(typed["candidate_selection"])
            short_passes.append(typed["short_string"])
            sentence_passes.append(typed["sentence"])
            natural_passes.append(typed["natural"])
    typed_global = {
        "program_construct": bool(program["program_construct_gate"]),
        "candidate_selection": all(candidate_passes),
        "short_string": all(short_passes),
        "sentence": all(sentence_passes),
        "natural": all(natural_passes),
    }
    cross_readout = all(typed_global.values())
    return {
        "axis_partition_cells": cells,
        "typed_global_gates": typed_global,
        "cross_readout_gate": cross_readout,
        "future_response_eligibility": cross_readout,
        "overall": {name: rate(raw, field) for name, field in (
            ("choice", "choice_correct"), ("bare_candidate", "bare_candidate_correct"),
            ("trie", "trie_correct"), ("bare_exact", "bare_exact"),
            ("bare_content", "bare_content_correct"), ("cued_exact", "cued_exact"),
            ("cued_content", "cued_content_correct"), ("sentence_exact", "sentence_exact"),
            ("sentence_content", "sentence_content_correct"), ("natural_exact", "natural_exact"),
            ("natural_content", "natural_content_correct"),
        )},
        "nonfinite_count": sum(not (row["choice_finite"] and row["bare_candidate_finite"]) for row in raw),
        "choice_tie_count": sum(len(row["choice_argmax_set"]) != 1 for row in raw),
        "bare_candidate_tie_count": sum(len(row["bare_argmax_set"]) != 1 for row in raw),
    }


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1235 final exists")
    contract, rows, manifest, program, plan = verify_frozen()
    raw = read_jsonl(RAW_PATH)
    summary = read_json(SUMMARY_PATH)
    result_audit = read_json(RESULT_AUDIT_PATH)
    if result_audit.get("all_checks_passed") is not True:
        raise RuntimeError("independent result audit failed")
    if len(raw) != len(rows) or summary["raw_digest"] != digest(raw):
        raise RuntimeError("raw output mismatch")
    ledgers = adjudicate(raw, program)
    passed = bool(ledgers["future_response_eligibility"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1235.typed_generation_compiler_boundary.final.v1",
        "created_at_utc": utc_now(),
        "status": "cross_readout_object_qualified" if passed else "typed_generation_boundary_localized",
        "contract_digest": contract["contract_digest"],
        "material_digest": digest(rows),
        "manifest_digest": digest(manifest),
        "program_audit_digest": program["program_audit_digest"],
        "batch_plan_digest": plan["plan_digest"],
        "run_summary_digest": summary["summary_digest"],
        "raw_digest": summary["raw_digest"],
        "result_audit_digest": result_audit["audit_digest"],
        "ledgers": ledgers,
        "k_item": {
            "identifier": "K210",
            "evidence_grade": "E3-BEHAVIOR-CROSS-READOUT" if passed else "E3-TYPED-BOUNDARY",
            "statement": (
                "Qwen3 query-object binding passed the frozen candidate, short-string, exact-sentence, and natural-sentence readout gates across all orthogonal axes."
                if passed
                else "Qwen3 query-object binding exhibited a preregistered typed boundary between sequence selection, constrained decoding, and one or more full-vocabulary generation contracts."
            ),
            "scope": "Qwen3-4B CUDA FP16; generated English registries; 96 axis-world clusters; behavior and token-rank diagnostics only",
        },
        "authorization": {
            "candidate_selection_claim": ledgers["typed_global_gates"]["candidate_selection"],
            "short_string_claim": ledgers["typed_global_gates"]["short_string"],
            "sentence_claim": ledgers["typed_global_gates"]["sentence"],
            "natural_sentence_claim": ledgers["typed_global_gates"]["natural"],
            "future_response_phase": passed,
            "next_experiment": "Phase1236 typed future-response tensor on the exact frozen Phase1235 object" if passed else None,
            "auto_continue": passed,
            "hidden_scan_in_this_phase": False,
            "cross_model_run": False,
            "separate_neural_module_claim": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    upstream, _audit = verify_upstream()
    from transformers import AutoTokenizer

    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    rows, lexicon = build_material(slow)
    manifest, tokenizer_summary = build_manifest(rows, slow, fast)
    program = build_program_audit(rows, manifest)
    print(canonical_json({
        "status": "phase1235_selftest_passed",
        "upstream_status": upstream["status"],
        "rows": len(rows),
        "worlds": len({row["world_id"] for row in rows}),
        "lexicon": lexicon,
        "tokenizer_gate": tokenizer_summary["tokenizer_gate"],
        "program_gate": program["program_construct_gate"],
        "program_ceilings": {key: value["depth2_conditional_program_accuracy"] for key, value in program["split_results"].items()},
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("selftest", "preregister", "run", "finalize"))
    stage = parser.parse_args().stage
    {"selftest": selftest, "preregister": preregister, "run": run_qwen3, "finalize": finalize}[stage]()


if __name__ == "__main__":
    main()
