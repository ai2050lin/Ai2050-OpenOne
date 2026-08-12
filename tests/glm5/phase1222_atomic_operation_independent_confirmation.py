#!/usr/bin/env python3
"""Phase 1222: independent confirmation contracts for atomic operations.

This phase is behavior-only.  It re-earns, on a fifth material system, the
right to study each operation/track separately.  Failed diagnostic programs
cannot veto a passed atomic operation and cannot inherit its authorization.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import platform
import random
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
from model_utils import MODEL_CONFIGS
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1222
ENGINEERING_REVISION = 3
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1222_atomic_operation_independent_confirmation_audit.py"
UPSTREAM_FINAL = (
    TEST_ROOT
    / "result/phase1221_typed_operation_behavior_and_error_fingerprints/analysis/final.json"
)
EXPECTED_UPSTREAM_FINAL_DIGEST = (
    "752bde3c2f7a37ed5033b55de9ee371c480b3f0ec70bae2baefd7f01bc29aa01"
)

OUT_ROOT = TEST_ROOT / "result/phase1222_atomic_operation_independent_confirmation"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/atomic_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
TOKEN_AUDIT_PATH = OUT_ROOT / "audit/tokenizer_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use", "sealed")
TRACKS = ("natural", "symbolic")
PANELS = ("canonical", "record_order", "paraphrase", "binding_permutation")
TARGET_OPERATIONS = ("direct", "query_object", "query_relation", "inverse_lookup")
DIAGNOSTIC_OPERATIONS = ("binding_swap", "link_then_value", "inverse_link_compose")
OPERATIONS = TARGET_OPERATIONS + DIAGNOSTIC_OPERATIONS
RELATIONS = ("badge", "sector", "texture", "status")
WORLDS_PER_SPLIT = 64
ROWS_PER_WORLD = len(OPERATIONS) * len(PANELS)
EXPECTED_WORLDS = len(SPLITS) * WORLDS_PER_SPLIT
EXPECTED_ROWS = EXPECTED_WORLDS * ROWS_PER_WORLD
TIE_TOLERANCE = 1e-7

FINITE_MIN = 1.0
CANDIDATE_MIN = 0.90
OPTION_GENERATION_MIN = 0.85
OPEN_GENERATION_MIN = 0.80
CONTEXT_ADJUSTED_MIN = 0.85
WORST_PANEL_MIN = 0.80
ALL_PANEL_WORLD_MIN = 0.75
ARGMAX_SET_AGREEMENT_MIN = 1.0

CANDIDATE_SYSTEM = (
    "Use only the current-world records. Return exactly one option from the "
    "final option list and no other text."
)
OPEN_SYSTEM = (
    "Use only the current-world records. Return only the exact requested object "
    "or value, with no explanation."
)

OBJECT_STEMS = (
    "Alder Birch Cedar Dogwood Elm Fir Hawthorn Juniper Larch Maple Oak Pine Rowan Spruce "
    "Willow Acacia Aspen Beech Cypress Hazel Linden Magnolia Olive Poplar"
).split()
OBJECT_NOUNS = (
    "Annex Archive Depot Gallery Hall Lodge Office Pavilion Studio Tower Workshop Yard "
    "Atrium Cabin Chamber Courtyard Hangar Kiosk Laboratory Observatory"
).split()

NATURAL_VALUE_PARTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "badge": (
        tuple("amber azure coral crimson indigo jade maroon ochre scarlet topaz vermilion zinc".split()),
        tuple("anchor bell crown feather lantern moon quill shield star torch wheel wing".split()),
    ),
    "sector": (
        tuple("alpine civic eastern garden inland lunar northern river southern western woodland zenith".split()),
        tuple("arc belt branch circle lane quarter reach ring route tract ward zone".split()),
    ),
    "texture": (
        tuple("brushed carved flecked grooved layered mottled polished scored speckled woven rippled stippled".split()),
        tuple("bamboo ceramic cork granite leather linen porcelain slate timber vellum wicker zinc".split()),
    ),
    "status": (
        tuple("awaiting cleared dormant enrolled guarded inspected loaded mapped queued sealed tracked validated".split()),
        tuple("arrival dispatch review storage testing transfer pickup release routing service sorting update".split()),
    ),
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def symbolic_candidates(prefix: str, count: int = 224) -> list[str]:
    syllables = (
        "bo", "ca", "de", "fi", "ga", "hu", "je", "ki", "lo", "mu", "na", "pe", "qo", "ra", "si", "vu"
    )
    values: list[str] = []
    for first in syllables:
        for second in syllables:
            values.append(f"{prefix} {first}{second}")
            if len(values) >= count:
                return values
    return values


def spaced_token_length(tokenizer: Any, value: str) -> int:
    return len(tokenizer.encode(" " + value, add_special_tokens=False))


def groups_of_four_same_length(tokenizer: Any, values: Iterable[str]) -> list[list[str]]:
    buckets: dict[int, list[str]] = defaultdict(list)
    for value in values:
        buckets[spaced_token_length(tokenizer, value)].append(value)
    groups: list[list[str]] = []
    for token_length in sorted(buckets):
        bucket = buckets[token_length]
        for start in range(0, len(bucket) - 3, 4):
            groups.append(bucket[start : start + 4])
    return groups


def build_lexicon(tokenizer: Any) -> dict[str, Any]:
    natural_objects = [f"{stem} {noun}" for stem in OBJECT_STEMS for noun in OBJECT_NOUNS]
    natural_values = {
        relation: [f"{left} {right}" for left in parts[0] for right in parts[1]]
        for relation, parts in NATURAL_VALUE_PARTS.items()
    }
    lexicon = {
        "natural_objects": groups_of_four_same_length(tokenizer, natural_objects),
        "symbolic_objects": groups_of_four_same_length(tokenizer, symbolic_candidates("node")),
        "natural_values": {
            relation: groups_of_four_same_length(tokenizer, values)
            for relation, values in natural_values.items()
        },
        "symbolic_values": {
            relation: groups_of_four_same_length(tokenizer, symbolic_candidates(f"{relation}tag"))
            for relation in RELATIONS
        },
    }
    for key in ("natural_objects", "symbolic_objects"):
        if len(lexicon[key]) < 32:
            raise RuntimeError(f"insufficient split-partitioned groups for {key}: {len(lexicon[key])}")
    for track in TRACKS:
        for relation in RELATIONS:
            count = len(lexicon[f"{track}_values"][relation])
            if count < 32:
                raise RuntimeError(f"insufficient groups for {track}/{relation}: {count}")
    return lexicon


def split_group(groups: list[list[str]], split: str, index: int, salt: int) -> list[str]:
    split_index = SPLITS.index(split)
    usable = len(groups) // len(SPLITS)
    if usable < 8:
        raise RuntimeError("lexicon cannot be partitioned across four splits")
    start = split_index * usable
    return list(groups[start + ((index * 7 + salt * 5) % usable)])


def make_links(objects: list[str], seed: int) -> dict[str, str]:
    cycle = list(objects)
    random.Random(seed).shuffle(cycle)
    return {cycle[index]: cycle[(index + 1) % len(cycle)] for index in range(len(cycle))}


def make_world(lexicon: dict[str, Any], split: str, local_index: int) -> dict[str, Any]:
    split_index = SPLITS.index(split)
    world_index = split_index * WORLDS_PER_SPLIT + local_index
    track = TRACKS[local_index % 2]
    rng = random.Random(1222000 + world_index * 131)
    objects = split_group(lexicon[f"{track}_objects"], split, local_index, 1)
    rng.shuffle(objects)
    values: dict[str, list[str]] = {}
    for relation_index, relation in enumerate(RELATIONS):
        group = split_group(
            lexicon[f"{track}_values"][relation], split, local_index, relation_index + 3
        )
        rng.shuffle(group)
        values[relation] = group
    assignments: dict[str, dict[str, str]] = {obj: {} for obj in objects}
    for relation_index, relation in enumerate(RELATIONS):
        shift = (world_index + relation_index * 3) % len(objects)
        for object_index, obj in enumerate(objects):
            assignments[obj][relation] = values[relation][(object_index + shift) % len(objects)]
    return {
        "world_id": f"p1222-{split}-w{local_index:03d}-{track}",
        "split": split,
        "world_index": world_index,
        "local_index": local_index,
        "track": track,
        "objects": objects,
        "relations": list(RELATIONS),
        "values": values,
        "assignments": assignments,
        "links": make_links(objects, 1222 + world_index * 17),
    }


def clone_assignments(assignments: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {obj: dict(fields) for obj, fields in assignments.items()}


def binding_permutation_shifts(world: dict[str, Any]) -> dict[str, int]:
    """Choose a nonzero column rotation that also changes the composed answer.

    Rotating every column by the same amount can accidentally cancel an inverse
    lookup followed by a link.  This deterministic finite search prevents that
    algebraic shortcut before material freeze.
    """
    objects = world["objects"]
    base = world["assignments"]
    target_value = base[objects[2]]["sector"]
    canonical_gold = base[world["links"][objects[2]]]["badge"]
    for badge_shift in (1, 2, 3):
        for sector_shift in (1, 2, 3):
            trial = clone_assignments(base)
            shifts = {
                "badge": badge_shift,
                "sector": sector_shift,
                "texture": 1,
                "status": 2,
            }
            for relation, shift in shifts.items():
                old = [base[obj][relation] for obj in objects]
                for index, obj in enumerate(objects):
                    trial[obj][relation] = old[(index + shift) % len(objects)]
            found = owner_of(trial, "sector", target_value)
            composed_gold = trial[world["links"][found]]["badge"]
            if composed_gold != canonical_gold:
                return shifts
    raise RuntimeError("no binding permutation changes the composed answer")


def permuted_assignments(world: dict[str, Any], panel: str) -> dict[str, dict[str, str]]:
    assignments = clone_assignments(world["assignments"])
    if panel != "binding_permutation":
        return assignments
    objects = world["objects"]
    for relation, shift in binding_permutation_shifts(world).items():
        old = [assignments[obj][relation] for obj in objects]
        for index, obj in enumerate(objects):
            assignments[obj][relation] = old[(index + shift) % len(objects)]
    return assignments


def inverse_link(links: dict[str, str], target: str) -> str:
    matches = [source for source, value in links.items() if value == target]
    if len(matches) != 1:
        raise RuntimeError("non-unique inverse link")
    return matches[0]


def owner_of(assignments: dict[str, dict[str, str]], relation: str, value: str) -> str:
    matches = [obj for obj, fields in assignments.items() if fields[relation] == value]
    if len(matches) != 1:
        raise RuntimeError("non-unique inverse value lookup")
    return matches[0]


def operation_state(world: dict[str, Any], operation: str, panel: str) -> dict[str, Any]:
    objects = world["objects"]
    links = world["links"]
    source, second, third, _fourth = objects
    relation = world["relations"][0]
    other_relation = world["relations"][1]
    assignments = permuted_assignments(world, panel)
    displayed = clone_assignments(assignments)
    target = links[source]
    transformation: str | None = None
    candidate_kind = "value"
    fingerprint: dict[str, str] = {}
    shortcuts: dict[str, str] = {}

    if operation == "direct":
        query = f"Read the {relation} recorded for {source}."
        gold = assignments[source][relation]
        candidates = world["values"][relation]
        derivation = [source, relation, gold]
        target_relation = relation
    elif operation == "query_object":
        query = f"Read the {relation} recorded for {second}."
        gold = assignments[second][relation]
        candidates = world["values"][relation]
        derivation = [second, relation, gold]
        target_relation = relation
    elif operation == "query_relation":
        query = f"Read the {other_relation} recorded for {source}."
        gold = assignments[source][other_relation]
        candidates = world["values"][other_relation]
        derivation = [source, other_relation, gold]
        target_relation = other_relation
    elif operation == "inverse_lookup":
        target_value = world["assignments"][third][relation]
        gold = owner_of(assignments, relation, target_value)
        query = f"Which object is recorded with {relation} {target_value}?"
        candidates = objects
        candidate_kind = "object"
        derivation = [relation, target_value, gold]
        target_relation = relation
    elif operation == "binding_swap":
        pre_swap = assignments[source][relation]
        assignments[source][relation], assignments[second][relation] = (
            assignments[second][relation], assignments[source][relation]
        )
        transformation = (
            f"Exchange only the {relation} entries of {source} and {second}; "
            "leave all other entries fixed."
        )
        query = f"After the exchange, read the {relation} for {source}."
        gold = assignments[source][relation]
        candidates = world["values"][relation]
        derivation = ["swap", source, second, relation, source, gold]
        target_relation = relation
        shortcuts[pre_swap] = "pre_swap_value"
    elif operation == "link_then_value":
        query = f"Move from {source} along one route, then read the reached object's {relation}."
        gold = assignments[target][relation]
        candidates = world["values"][relation]
        derivation = [source, "route", target, relation, gold]
        target_relation = relation
        shortcuts[assignments[source][relation]] = "source_value_without_link"
    elif operation == "inverse_link_compose":
        target_value = world["assignments"][third][other_relation]
        found = owner_of(assignments, other_relation, target_value)
        reached = links[found]
        query = (
            f"Find the object with {other_relation} {target_value}, move along its route once, "
            f"then read the reached object's {relation}."
        )
        gold = assignments[reached][relation]
        candidates = world["values"][relation]
        derivation = [other_relation, target_value, found, "route", reached, relation, gold]
        target_relation = relation
        shortcuts[assignments[found][relation]] = "stopped_after_inverse_lookup"
    else:
        raise KeyError(operation)

    fingerprint.update(shortcuts)
    for index, obj in enumerate(objects):
        if candidate_kind == "value":
            fingerprint.setdefault(assignments[obj][target_relation], f"value_of_object_{index}")
        else:
            fingerprint.setdefault(obj, f"object_{index}")
    fingerprint[gold] = "gold"
    return {
        "display_assignments": displayed,
        "computed_assignments": assignments,
        "transformation_instruction": transformation,
        "query": query,
        "gold": gold,
        "candidates": list(candidates),
        "candidate_kind": candidate_kind,
        "target_relation": target_relation,
        "derivation": derivation,
        "fingerprint_by_candidate": fingerprint,
        "shorter_path_candidates": shortcuts,
    }


def record_lines(
    assignments: dict[str, dict[str, str]],
    objects: list[str],
    relations: list[str],
    paraphrase: bool,
) -> list[str]:
    lines: list[str] = []
    for obj in objects:
        fields = assignments[obj]
        if paraphrase:
            body = ", ".join(f"a {relation} of {fields[relation]}" for relation in relations)
            lines.append(f"{obj} has {body}.")
        else:
            body = " | ".join(f"{relation}: {fields[relation]}" for relation in relations)
            lines.append(f"{obj} || {body}")
    return lines


def route_lines(links: dict[str, str], objects: list[str], paraphrase: bool) -> list[str]:
    if paraphrase:
        return [f"A route from {source} reaches {links[source]}." for source in objects]
    return [f"route({source}) = {links[source]}" for source in objects]


def prompt_bodies(world: dict[str, Any], state: dict[str, Any], panel: str) -> tuple[str, str]:
    paraphrase = panel == "paraphrase" or world["split"] == "natural_use"
    records = record_lines(
        state["display_assignments"], world["objects"], world["relations"], paraphrase
    )
    routes = route_lines(world["links"], world["objects"], paraphrase)
    if panel == "record_order":
        records = list(reversed(records))
        routes = routes[2:] + routes[:2]
    instruction = [state["transformation_instruction"]] if state["transformation_instruction"] else []
    body_lines = records + routes + instruction
    split = world["split"]
    if split == "discovery":
        body = "CURRENT DOSSIER\n" + "\n".join(body_lines)
        question = f"TASK: {state['query']}"
    elif split == "confirmation":
        body = "REFERENCE REGISTER\n" + "\n".join(f"- {line}" for line in body_lines)
        question = f"REQUEST: {state['query']}"
    elif split == "natural_use":
        body = "For this account, " + " ".join(body_lines)
        question = f"On that account, {state['query']}"
    else:
        body = "SEALED SNAPSHOT\n" + "\n".join(
            f"<{index + 1}> {line}" for index, line in enumerate(body_lines)
        )
        question = f"SEALED QUERY: {state['query']}"
    return body, question


def render_prompts(
    world: dict[str, Any], state: dict[str, Any], panel: str, candidate_order: list[str]
) -> dict[str, str]:
    body, question = prompt_bodies(world, state, panel)
    options = " / ".join(candidate_order)
    candidate_prompt = f"{body}\n{question}\nCHOICES: {options}\nRESPONSE:"
    open_prompt = f"{body}\n{question}\nGive the exact object or value.\nRESPONSE:"
    null_prompt = (
        "NO CURRENT-WORLD RECORDS ARE SUPPLIED.\n"
        f"{question}\nCHOICES: {options}\nRESPONSE:"
    )
    return {"candidate": candidate_prompt, "open": open_prompt, "null": null_prompt}


def build_materials(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lexicon = build_lexicon(tokenizer)
    permutations = list(itertools.permutations(range(3)))
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for local_index in range(WORLDS_PER_SPLIT):
            world = make_world(lexicon, split, local_index)
            for operation_index, operation in enumerate(OPERATIONS):
                group_id = f"{world['world_id']}::{operation}"
                for panel_index, panel in enumerate(PANELS):
                    state = operation_state(world, operation, panel)
                    other_candidates = [
                        candidate for candidate in state["candidates"] if candidate != state["gold"]
                    ]
                    other_order = [
                        other_candidates[index]
                        for index in permutations[
                            (world["world_index"] * 5 + operation_index * 3 + panel_index) % 6
                        ]
                    ]
                    gold_position = (local_index // 2 + operation_index + panel_index) % 4
                    candidate_order = list(other_order)
                    candidate_order.insert(gold_position, state["gold"])
                    prompts = render_prompts(world, state, panel, candidate_order)
                    row: dict[str, Any] = {
                        "schema_version": "phase1222.atomic.row.v1",
                        "phase": PHASE,
                        "item_id": f"p1222-{digest([group_id, panel])[:20]}",
                        "world_id": world["world_id"],
                        "world_index": world["world_index"],
                        "local_index": local_index,
                        "group_id": group_id,
                        "split": split,
                        "track": world["track"],
                        "operation": operation,
                        "operation_role": "target" if operation in TARGET_OPERATIONS else "diagnostic",
                        "panel": panel,
                        "objects": world["objects"],
                        "relations": world["relations"],
                        "values": world["values"],
                        "base_assignments": world["assignments"],
                        "display_assignments": state["display_assignments"],
                        "computed_assignments": state["computed_assignments"],
                        "links": world["links"],
                        "target_relation": state["target_relation"],
                        "candidate_kind": state["candidate_kind"],
                        "transformation_instruction": state["transformation_instruction"],
                        "derivation": state["derivation"],
                        "query": state["query"],
                        "candidate_prompt": prompts["candidate"],
                        "open_prompt": prompts["open"],
                        "null_prompt": prompts["null"],
                        "gold": state["gold"],
                        "candidates": state["candidates"],
                        "candidate_order": candidate_order,
                        "gold_position": candidate_order.index(state["gold"]),
                        "fingerprint_by_candidate": state["fingerprint_by_candidate"],
                        "shorter_path_candidates": state["shorter_path_candidates"],
                    }
                    row["row_digest"] = digest(row)
                    rows.append(row)
    lexicon_audit = {
        "natural_object_group_count": len(lexicon["natural_objects"]),
        "symbolic_object_group_count": len(lexicon["symbolic_objects"]),
        "natural_value_group_counts": {
            relation: len(groups) for relation, groups in lexicon["natural_values"].items()
        },
        "symbolic_value_group_counts": {
            relation: len(groups) for relation, groups in lexicon["symbolic_values"].items()
        },
    }
    return rows, lexicon_audit


def render_native(tokenizer: Any, prompt: str, system: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def tokenize_candidate_prompt(
    tokenizer: Any, rendered: str, candidates: list[str]
) -> tuple[list[int], dict[str, list[int]]]:
    input_ids: list[int] | None = None
    candidate_token_ids: dict[str, list[int]] = {}
    for candidate in candidates:
        base, suffix = p1220.continuation_ids(tokenizer, rendered, candidate)
        if input_ids is None:
            input_ids = base
        elif input_ids != base:
            raise RuntimeError("candidate retokenized prompt prefix")
        candidate_token_ids[candidate] = suffix
    assert input_ids is not None
    return input_ids, candidate_token_ids


def build_manifest(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    prompt_lengths: dict[str, list[int]] = defaultdict(list)
    candidate_lengths: list[int] = []
    equal_length_count = 0
    for index, row in enumerate(rows):
        world_rendered = render_native(tokenizer, row["candidate_prompt"], CANDIDATE_SYSTEM)
        null_rendered = render_native(tokenizer, row["null_prompt"], CANDIDATE_SYSTEM)
        open_rendered = render_native(tokenizer, row["open_prompt"], OPEN_SYSTEM)
        world_ids, world_candidate_ids = tokenize_candidate_prompt(
            tokenizer, world_rendered, row["candidate_order"]
        )
        null_ids, null_candidate_ids = tokenize_candidate_prompt(
            tokenizer, null_rendered, row["candidate_order"]
        )
        open_ids = [int(value) for value in tokenizer.encode(open_rendered, add_special_tokens=False)]
        world_lengths = [len(world_candidate_ids[value]) for value in row["candidate_order"]]
        null_lengths = [len(null_candidate_ids[value]) for value in row["candidate_order"]]
        if world_lengths != null_lengths:
            raise RuntimeError("world and null candidate continuation lengths differ")
        equal_length_count += len(set(world_lengths)) == 1
        candidate_lengths.extend(world_lengths)
        prompt_lengths["world"].append(len(world_ids))
        prompt_lengths["null"].append(len(null_ids))
        prompt_lengths["open"].append(len(open_ids))
        item: dict[str, Any] = {
            "schema_version": "phase1222.qwen3.manifest.v1",
            "item_id": row["item_id"],
            "row_digest": row["row_digest"],
            "split": row["split"],
            "track": row["track"],
            "operation": row["operation"],
            "operation_role": row["operation_role"],
            "panel": row["panel"],
            "gold": row["gold"],
            "candidates": row["candidate_order"],
            "candidate_order": row["candidate_order"],
            "world_input_ids": world_ids,
            "world_input_token_count": len(world_ids),
            "world_candidate_token_ids": world_candidate_ids,
            "null_input_ids": null_ids,
            "null_input_token_count": len(null_ids),
            "null_candidate_token_ids": null_candidate_ids,
            "open_input_ids": open_ids,
            "open_input_token_count": len(open_ids),
            "world_rendered_prompt_digest": digest(world_rendered),
            "null_rendered_prompt_digest": digest(null_rendered),
            "open_rendered_prompt_digest": digest(open_rendered),
        }
        item["manifest_row_digest"] = digest(item)
        manifest.append(item)
        if (index + 1) % 2048 == 0:
            print(f"[phase1222/tokenize] {index + 1}/{len(rows)}", flush=True)
    audit: dict[str, Any] = {
        "phase": PHASE,
        "row_count": len(manifest),
        "prompt_token_counts": {
            key: {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
            for key, values in prompt_lengths.items()
        },
        "candidate_token_count_min": min(candidate_lengths),
        "candidate_token_count_max": max(candidate_lengths),
        "candidate_token_count_mean": sum(candidate_lengths) / len(candidate_lengths),
        "equal_candidate_token_length_row_rate": equal_length_count / len(rows),
        "manifest_digest": digest(manifest),
    }
    audit["tokenizer_audit_digest"] = digest(audit)
    return manifest, audit


def model_artifact_fingerprint() -> dict[str, Any]:
    root = Path(MODEL_CONFIGS["qwen3"]["path"])
    files = sorted(path for path in root.iterdir() if path.is_file())
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "config_sha256": file_sha256(root / "config.json"),
        "tokenizer_config_sha256": file_sha256(root / "tokenizer_config.json"),
    }


def verify_upstream() -> dict[str, Any]:
    upstream = read_json(UPSTREAM_FINAL)
    if upstream.get("final_digest") != EXPECTED_UPSTREAM_FINAL_DIGEST:
        raise RuntimeError("Phase1221 final digest changed")
    if upstream.get("authorized_next", {}).get("automatic_execution"):
        raise RuntimeError("Phase1221 unexpectedly authorized physical work")
    return upstream


def build_protocol(
    rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    token_audit: dict[str, Any],
    lexicon_audit: dict[str, Any],
) -> dict[str, Any]:
    upstream = verify_upstream()
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1222.atomic.protocol.v1",
        "engineering_revision": ENGINEERING_REVISION,
        "engineering_history": (
            "revision 0 fixed a pre-reveal audit schema error; revision 1 was archived before "
            "model execution because serialized mapping order broke prompt replay and the first "
            "candidate permutation schedule did not balance gold positions; revision 2 fixed records "
            "but exposed the same serialization-order issue for links; revision 3 renders all mappings "
            "through explicit object order and adds JSON round-trip self-testing"
        ),
        "created_at": utc_now(),
        "purpose": (
            "independently re-earn behavior and physical-study permission for each atomic "
            "operation/track on a fifth material system"
        ),
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1221_main": file_sha256(TEST_ROOT / p1221.SCRIPT.name),
            "phase1221_final": file_sha256(UPSTREAM_FINAL),
        },
        "upstream": {
            "phase1221_final_digest": upstream["final_digest"],
            "phase1221_k198_grade": upstream["k_item"]["evidence_grade"],
            "old_operation_passes_do_not_authorize_this_phase": True,
            "explicit_user_restart_received": True,
        },
        "design": {
            "splits": list(SPLITS),
            "tracks": list(TRACKS),
            "panels": list(PANELS),
            "target_operations": list(TARGET_OPERATIONS),
            "diagnostic_operations": list(DIAGNOSTIC_OPERATIONS),
            "worlds_per_split": WORLDS_PER_SPLIT,
            "world_count": EXPECTED_WORLDS,
            "row_count": EXPECTED_ROWS,
            "independent_cluster_unit": "world",
            "split_interpretation": "disjoint generated items from one generator, not four independent domains",
            "binding_permutation": "same object/value multiset; all relation bindings rotate by one object",
            "candidate_order_uses_all_24_permutations": True,
            "same_row_candidate_token_lengths_equal": True,
            "fifth_material_vocabulary_disjoint_from_phase1221_objects_values_relations": True,
        },
        "material": {
            "material_digest": digest(rows),
            "manifest_digest": digest(manifest),
            "tokenizer_audit_digest": token_audit["tokenizer_audit_digest"],
            "lexicon_audit": lexicon_audit,
        },
        "interface": {
            "model": "qwen3",
            "model_artifact": model_artifact_fingerprint(),
            "precision": "FP16",
            "quantization": "none",
            "native_chat_template": True,
            "enable_thinking": False,
            "candidate_system": CANDIDATE_SYSTEM,
            "open_system": OPEN_SYSTEM,
            "candidate_primary": "sum full-continuation log probability",
            "candidate_secondary": "mean full-continuation log probability",
            "tie_rule": "sum and mean maximizer sets at tolerance 1e-7 must be equal",
            "context_adjustment": "world sum log probability minus matched no-record sum log probability",
            "option_generation": "greedy generation with listed candidates for every row",
            "no_option_generation": "greedy generation without listed candidates for every row",
            "hidden_states": False,
            "attentions": False,
            "hooks": False,
            "interventions": False,
        },
        "atomic_gate": {
            "finite_min_each_split": FINITE_MIN,
            "candidate_accuracy_min_each_split": CANDIDATE_MIN,
            "option_generation_min_each_split": OPTION_GENERATION_MIN,
            "no_option_generation_min_each_split": OPEN_GENERATION_MIN,
            "context_adjusted_accuracy_min_each_split": CONTEXT_ADJUSTED_MIN,
            "worst_panel_candidate_min_each_split": WORST_PANEL_MIN,
            "all_panel_candidate_correct_world_rate_min_each_split": ALL_PANEL_WORLD_MIN,
            "sum_mean_argmax_set_agreement_min_each_split": ARGMAX_SET_AGREEMENT_MIN,
            "operation_track_gate": "conjunction across all metrics and four splits",
            "family_or_global_conjunction": False,
            "failure_closes_only_exact_operation_track": True,
            "world_cluster_wilson_interval": "descriptive only; never upgrades or downgrades a frozen gate",
        },
        "diagnostic_policy": {
            "never_blocks_target": True,
            "never_inherits_target_authorization": True,
            "error_outputs_are_behavioral_fingerprints_not_internal_program proofs": True,
            "shorter_path_candidates_must_be_distinct_from_gold": True,
        },
        "authorization": {
            "any_target_operation_track_pass": (
                "authorize a newly frozen physical-trajectory phase for exactly the passed scopes"
            ),
            "no_target_operation_track_pass": "stop before hidden-state work",
            "negative_diagnostics": "behavioral evidence only",
            "head_or_neuron_search": False,
            "cross_model": False,
        },
        "forbidden_after_freeze": [
            "change material, prompts, operations, panels, thresholds, parser, or scoring",
            "retroactively authorize from Phase1221 passes",
            "merge operations into a family gate or split an operation after reveal",
            "let a diagnostic arm veto or borrow atomic authorization",
            "drop binding-permutation, no-record, no-option, or hard split cells",
            "inspect hidden states, attentions, components, or interventions in Phase1222",
            "call generated natural-track text organic language",
            "interpret a shortcut output fingerprint as proof of an internal algorithm",
        ],
        "claim_boundary": {
            "behavior_only": True,
            "qwen3_only": True,
            "generated_worlds": True,
            "organic_language": False,
            "atomic_operation_and_track_typed": True,
            "hidden_state": False,
            "causal": False,
            "cross_model": False,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    return protocol


def materialize() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"formal output already exists: {OUT_ROOT}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, lexicon_audit = build_materials(tokenizer)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected material count: {len(rows)}")
    manifest, token_audit = build_manifest(rows, tokenizer)
    protocol = build_protocol(rows, manifest, token_audit, lexicon_audit)
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(TOKEN_AUDIT_PATH, token_audit)
    write_json(PROTOCOL_PATH, protocol)
    print(
        canonical_json(
            {
                "status": "materialized",
                "worlds": EXPECTED_WORLDS,
                "rows": len(rows),
                "protocol_digest": protocol["protocol_digest"],
            }
        )
    )


def verify_formal_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    embedded = dict(protocol)
    claimed_digest = embedded.pop("protocol_digest")
    if claimed_digest != digest(embedded):
        raise RuntimeError("protocol embedded digest mismatch")
    if protocol["source_hashes"]["main"] != file_sha256(SCRIPT):
        raise RuntimeError("main script changed after freeze")
    if protocol["source_hashes"]["audit"] != file_sha256(AUDIT_SCRIPT):
        raise RuntimeError("audit script changed after freeze")
    if protocol["material"]["material_digest"] != digest(rows):
        raise RuntimeError("material digest mismatch")
    if protocol["material"]["manifest_digest"] != digest(manifest):
        raise RuntimeError("manifest digest mismatch")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    return protocol, rows, manifest


def manifest_view(manifest: list[dict[str, Any]], prefix: str, generation: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest:
        view = {
            "item_id": row["item_id"],
            "input_ids": row[f"{prefix}_input_ids"],
            "input_token_count": row[f"{prefix}_input_token_count"],
            "candidates": row["candidates"],
            "candidate_order": row["candidate_order"],
            "candidate_token_ids": row.get(
                f"{prefix}_candidate_token_ids", row["world_candidate_token_ids"]
            ),
            "generation_required": generation,
        }
        rows.append(view)
    return rows


def argmax_set(scores: dict[str, float], tolerance: float = TIE_TOLERANCE) -> list[str]:
    maximum = max(scores.values())
    return sorted(candidate for candidate, value in scores.items() if maximum - value <= tolerance)


def score_view(
    model: Any, device: torch.device, manifest: list[dict[str, Any]], prefix: str
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    view = manifest_view(manifest, prefix, generation=False)
    # Phase1221's independently audited shared-cache scorer includes direct-prefill fallback.
    return p1221.score_candidates_with_fallback(model, device, view)


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("behavior output already exists")
    protocol, rows, manifest = verify_formal_inputs()
    material_by_id = {row["item_id"]: row for row in rows}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    try:
        world_scores, world_runtime = score_view(model, device, manifest, "world")
        null_scores, null_runtime = score_view(model, device, manifest, "null")
        option_generations, option_runtime = p1220.generation_scores(
            model, tokenizer, device, manifest_view(manifest, "world", generation=True)
        )
        open_generations, open_runtime = p1220.generation_scores(
            model, tokenizer, device, manifest_view(manifest, "open", generation=True)
        )
        raw: list[dict[str, Any]] = []
        for manifest_row in manifest:
            item_id = manifest_row["item_id"]
            material = material_by_id[item_id]
            scores = world_scores[item_id]
            priors = null_scores[item_id]
            sum_map = {key: value["sum_log_probability"] for key, value in scores.items()}
            mean_map = {key: value["mean_log_probability"] for key, value in scores.items()}
            null_map = {key: value["sum_log_probability"] for key, value in priors.items()}
            context_map = {key: sum_map[key] - null_map[key] for key in sum_map}
            sum_set = argmax_set(sum_map)
            mean_set = argmax_set(mean_map)
            null_set = argmax_set(null_map)
            context_set = argmax_set(context_map)
            sum_prediction = sum_set[0] if len(sum_set) == 1 else None
            context_prediction = context_set[0] if len(context_set) == 1 else None
            null_prediction = null_set[0] if len(null_set) == 1 else None
            ordered = sorted(sum_map.values(), reverse=True)
            option = option_generations[item_id]
            opened = open_generations[item_id]
            behavior: dict[str, Any] = {
                "schema_version": "phase1222.qwen3.behavior.row.v1",
                "phase": PHASE,
                "protocol_digest": protocol["protocol_digest"],
                "item_id": item_id,
                "row_digest": manifest_row["row_digest"],
                "world_id": material["world_id"],
                "group_id": material["group_id"],
                "split": material["split"],
                "track": material["track"],
                "operation": material["operation"],
                "operation_role": material["operation_role"],
                "panel": material["panel"],
                "gold": material["gold"],
                "gold_position": material["gold_position"],
                "candidate_scores": scores,
                "null_candidate_scores": priors,
                "context_adjusted_scores": context_map,
                "sum_argmax_set": sum_set,
                "mean_argmax_set": mean_set,
                "null_argmax_set": null_set,
                "context_argmax_set": context_set,
                "sum_prediction": sum_prediction,
                "context_prediction": context_prediction,
                "null_prediction": null_prediction,
                "sum_mean_argmax_set_agreement": sum_set == mean_set,
                "candidate_correct": sum_prediction == material["gold"],
                "context_correct": context_prediction == material["gold"],
                "null_gold_prediction": null_prediction == material["gold"],
                "sum_margin": ordered[0] - ordered[1],
                "all_candidate_scores_finite": all(
                    value["all_vocab_logits_finite"] for value in scores.values()
                ),
                "all_null_scores_finite": all(
                    value["all_vocab_logits_finite"] for value in priors.values()
                ),
                "error_fingerprint": material["fingerprint_by_candidate"].get(
                    sum_prediction, "tie_or_unregistered"
                ),
                "option_generation_prediction": option["generation_prediction"],
                "option_generation_correct": option["generation_prediction"] == material["gold"],
                "option_generation_exact": option["generation_normalized_exact"],
                "option_generated_token_ids": option["generated_token_ids"],
                "option_generated_text": option["generated_text"],
                "open_generation_prediction": opened["generation_prediction"],
                "open_generation_correct": opened["generation_prediction"] == material["gold"],
                "open_generation_exact": opened["generation_normalized_exact"],
                "open_generated_token_ids": opened["generated_token_ids"],
                "open_generated_text": opened["generated_text"],
            }
            behavior["behavior_row_digest"] = digest(behavior)
            raw.append(behavior)
        write_jsonl(RAW_PATH, raw)
        summary: dict[str, Any] = {
            "schema_version": "phase1222.qwen3.run_summary.v1",
            "phase": PHASE,
            "created_at": utc_now(),
            "model": "qwen3",
            "protocol_digest": protocol["protocol_digest"],
            "case_count": len(raw),
            "world_scoring_runtime": world_runtime,
            "null_scoring_runtime": null_runtime,
            "option_generation_runtime": option_runtime,
            "open_generation_runtime": open_runtime,
            "precision_audit": precision,
            "placement": placement,
            "raw_digest": digest(raw),
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "claim_boundary": protocol["claim_boundary"],
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
        print(
            canonical_json(
                {
                    "status": "behavior_complete",
                    "case_count": len(raw),
                    "summary_digest": summary["summary_digest"],
                }
            )
        )
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def operation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_panel = {
        panel: rate([row for row in rows if row["panel"] == panel], "candidate_correct")
        for panel in PANELS
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    all_panel = [
        len(values) == len(PANELS) and all(row["candidate_correct"] for row in values)
        for values in groups.values()
    ]
    successes = sum(all_panel)
    errors = Counter(row["error_fingerprint"] for row in rows if not row["candidate_correct"])
    gold_frequency = Counter(row["gold"] for row in rows)
    null_predictions = Counter(
        row["null_prediction"] if row["null_prediction"] is not None else "<tie>" for row in rows
    )
    prediction_positions = Counter(
        row["sum_argmax_set"][0] if len(row["sum_argmax_set"]) == 1 else "<tie>"
        for row in rows
    )
    return {
        "case_count": len(rows),
        "world_count": len(groups),
        "finite_rate": sum(
            row["all_candidate_scores_finite"] and row["all_null_scores_finite"] for row in rows
        )
        / len(rows),
        "candidate_accuracy": rate(rows, "candidate_correct"),
        "option_generation_accuracy": rate(rows, "option_generation_correct"),
        "open_generation_accuracy": rate(rows, "open_generation_correct"),
        "context_adjusted_accuracy": rate(rows, "context_correct"),
        "null_gold_prediction_rate": rate(rows, "null_gold_prediction"),
        "sum_mean_argmax_set_agreement": rate(rows, "sum_mean_argmax_set_agreement"),
        "candidate_by_panel": by_panel,
        "worst_panel_candidate": min(by_panel.values()),
        "all_panel_world_success_count": successes,
        "all_panel_world_rate": successes / len(all_panel),
        "all_panel_world_wilson_95": wilson_interval(successes, len(all_panel)),
        "mean_sum_margin": sum(row["sum_margin"] for row in rows) / len(rows),
        "error_fingerprints": dict(sorted(errors.items())),
        "gold_answer_frequency": dict(sorted(gold_frequency.items())),
        "null_prediction_frequency": dict(sorted(null_predictions.items())),
        "world_prediction_frequency": dict(sorted(prediction_positions.items())),
    }


def metric_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "finite": metrics["finite_rate"] >= FINITE_MIN,
        "candidate": metrics["candidate_accuracy"] >= CANDIDATE_MIN,
        "option_generation": metrics["option_generation_accuracy"] >= OPTION_GENERATION_MIN,
        "open_generation": metrics["open_generation_accuracy"] >= OPEN_GENERATION_MIN,
        "context_adjusted": metrics["context_adjusted_accuracy"] >= CONTEXT_ADJUSTED_MIN,
        "worst_panel": metrics["worst_panel_candidate"] >= WORST_PANEL_MIN,
        "all_panel_world": metrics["all_panel_world_rate"] >= ALL_PANEL_WORLD_MIN,
        "argmax_set_agreement": (
            metrics["sum_mean_argmax_set_agreement"] >= ARGMAX_SET_AGREEMENT_MIN
        ),
    }


def summarize_behavior(raw: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    authorization: dict[str, bool] = {}
    for operation in OPERATIONS:
        for track in TRACKS:
            split_results: dict[str, Any] = {}
            split_passes: list[bool] = []
            for split in SPLITS:
                selected = [
                    row
                    for row in raw
                    if row["operation"] == operation
                    and row["track"] == track
                    and row["split"] == split
                ]
                metrics = operation_metrics(selected)
                gates = metric_gates(metrics)
                passed = all(gates.values())
                split_results[split] = {"metrics": metrics, "gates": gates, "passed": passed}
                split_passes.append(passed)
            key = f"{operation}|{track}"
            authorized = all(split_passes) if operation in TARGET_OPERATIONS else False
            results[key] = {
                "operation_role": "target" if operation in TARGET_OPERATIONS else "diagnostic",
                "splits": split_results,
                "all_splits_pass_behavior_metrics": all(split_passes),
                "authorized": authorized,
            }
            authorization[key] = authorized
    authorized = sorted(key for key, value in authorization.items() if value)
    return {
        "operation_track_results": results,
        "operation_track_authorization": authorization,
        "authorized_target_operation_tracks": authorized,
        "any_target_operation_track_authorized": bool(authorized),
        "family_gate_exists": False,
        "diagnostics_can_authorize": False,
        "diagnostics_can_veto": False,
    }


def finalize() -> None:
    protocol, rows, _manifest = verify_formal_inputs()
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    if len(raw) != len(rows) or summary["raw_digest"] != digest(raw):
        raise RuntimeError("raw output mismatch")
    behavior = summarize_behavior(raw)
    authorized = behavior["authorized_target_operation_tracks"]
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": (
            "atomic_operation_tracks_authorized" if authorized else "no_atomic_operation_track_authorized"
        ),
        "protocol_digest": protocol["protocol_digest"],
        "material_digest": protocol["material"]["material_digest"],
        "manifest_digest": protocol["material"]["manifest_digest"],
        "run_summary_digest": summary["summary_digest"],
        "behavior": behavior,
        "k_item": {
            "identifier": "K199",
            "evidence_grade": "E3-BEHAVIOR" if authorized else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                f"Independent fifth-material atomic authorization passed for {authorized}."
                if authorized
                else "No target operation/track passed every independent frozen behavior gate."
            ),
            "scope": "Qwen3 FP16; generated fifth material; operation/track typed; behavior only",
        },
        "authorized_next": {
            "automatic_execution": bool(authorized),
            "experiment": "Phase1223 passed-atom physical trajectory" if authorized else None,
            "authorized_operation_tracks": authorized,
            "hidden_state_scan": bool(authorized),
            "head_or_neuron_search": False,
            "cross_model_run": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    rows, lexicon = build_materials(tokenizer)
    assert len(rows) == EXPECTED_ROWS
    assert len({row["item_id"] for row in rows}) == EXPECTED_ROWS
    assert len({row["world_id"] for row in rows}) == EXPECTED_WORLDS
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
        assert row["gold"] not in row["shorter_path_candidates"]
    assert all(len(group) == len(PANELS) for group in groups.values())
    assert all(
        next(row for row in group if row["panel"] == "canonical")["gold"]
        != next(row for row in group if row["panel"] == "binding_permutation")["gold"]
        for group in groups.values()
    )
    position_cells = {
        key: Counter(
            row["gold_position"]
            for row in rows
            if (row["split"], row["track"], row["operation"]) == key
        )
        for key in {(row["split"], row["track"], row["operation"]) for row in rows}
    }
    assert all(set(counts.values()) == {32} for counts in position_cells.values())
    permutation_shapes = {
        tuple(row["candidates"].index(value) for value in row["candidate_order"])
        for row in rows
    }
    assert len(permutation_shapes) == 24
    for row in rows:
        restored = json.loads(canonical_json(row))
        world = {
            "world_id": restored["world_id"],
            "split": restored["split"],
            "world_index": restored["world_index"],
            "local_index": restored["local_index"],
            "track": restored["track"],
            "objects": restored["objects"],
            "relations": restored["relations"],
            "values": restored["values"],
            "assignments": restored["base_assignments"],
            "links": restored["links"],
        }
        state = operation_state(world, restored["operation"], restored["panel"])
        prompts = render_prompts(world, state, restored["panel"], restored["candidate_order"])
        assert restored["candidate_prompt"] == prompts["candidate"]
        assert restored["open_prompt"] == prompts["open"]
        assert restored["null_prompt"] == prompts["null"]
    print(
        canonical_json(
            {
                "status": "selftest_passed",
                "rows": len(rows),
                "worlds": EXPECTED_WORLDS,
                "lexicon": lexicon,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("selftest", "materialize", "run", "finalize"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "materialize":
        materialize()
    elif args.stage == "run":
        run_qwen3()
    elif args.stage == "finalize":
        finalize()


if __name__ == "__main__":
    main()
