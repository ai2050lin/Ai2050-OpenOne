#!/usr/bin/env python3
"""Phase 1221: typed operation behavior gates and error fingerprints.

The phase is behavior-only. It uses new worlds and vocabulary, freezes
operation-by-interface authorization before Qwen3 output, scores matched-token
candidate strings with both sum and mean continuation log probability, and
records program-level error fingerprints. Hidden states and interventions are
forbidden here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import random
import re
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
from model_utils import MODEL_CONFIGS
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1221
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1221_typed_operation_behavior_and_error_fingerprints_audit.py"
UPSTREAM_FINAL = TEST_ROOT / "result/phase1220_object_relation_value_master_task/analysis/final.json"
EXPECTED_UPSTREAM_DIGEST = "3d5c64062189f8523786d7bae32b80af0f753f6918d5bb80e574f68cc83f9342"

OUT_ROOT = TEST_ROOT / "result/phase1221_typed_operation_behavior_and_error_fingerprints"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/typed_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
TOKEN_AUDIT_PATH = OUT_ROOT / "audit/tokenizer_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use", "sealed")
TRACKS = ("natural", "symbolic")
PANELS = ("canonical", "record_order", "paraphrase", "matched_carrier")
FAMILIES: dict[str, tuple[str, ...]] = {
    "core": ("direct", "query_object", "query_relation", "binding_swap", "inverse_lookup"),
    "link": ("link_marker_control", "link_object", "link_then_value", "reverse_link_object", "reverse_link_then_value"),
    "compose": ("object_relation_compose", "binding_query_compose", "double_link_relation", "inverse_link_compose", "link_binding_compose"),
}
WORLDS_PER_FAMILY_SPLIT = 64
ROWS_PER_WORLD = 5 * len(PANELS)
EXPECTED_WORLDS = len(SPLITS) * len(FAMILIES) * WORLDS_PER_FAMILY_SPLIT
EXPECTED_ROWS = EXPECTED_WORLDS * ROWS_PER_WORLD
LOW_MARGIN_PREFILL_THRESHOLD = 0.5
TIE_TOLERANCE = 1e-7

FINITE_MIN = 1.0
CANDIDATE_MIN = 0.90
GENERATION_MIN = 0.85
WORST_PANEL_MIN = 0.80
SURFACE_GROUP_MIN = 0.75
SUM_MEAN_AGREEMENT_MIN = 1.0

SYSTEM_PROMPT = (
    "Answer only from the world in the user message. Return exactly one option "
    "from the final option list and no other text."
)

NATURAL_NAMES = (
    "Arlo Beatrix Cedric Dahlia Emmett Fiona Gideon Helena Ingrid Jasper Keira Malcolm "
    "Nadine Orson Petra Roland Selene Tobias Ursula Vaughn Winona Yasmin Zephyr Alina "
    "Bastian Corinne Desmond Elara Florian Greta Holden Imani Jorah Kalista Leander "
    "Maribel Nolan Octavia Pascal Ramona Sterling Thalia Ulric Viola Wallace Xenia "
    "Yvette Alistair Brielle Cormac Daphne Evander Felicity Gareth Heloise Isolde "
    "Jericho Kendra Lucian Mireille Neville Odelia Peregrine Rosalie Stellan Tabitha "
    "Valentin Wilhelmina Zinnia Ansel Briony Caspian Delphine Ezekiel Francesca Griffin "
    "Honora Iliana Julius Lorelei Matthias Noemi Orlando Philippa Roderick Sabrina "
    "Tristan Verena Whitney Zoraida"
).split()

RELATIONS = ("emblem", "district", "finish", "readiness")
VALUE_POOLS: dict[str, tuple[str, ...]] = {
    "emblem": (
        "violet lilac", "pearl ivory", "charcoal gray", "copper russet", "silver mauve", "cobalt teal",
        "umber sienna", "lemon chartreuse", "plum magenta", "chalk alabaster", "bronze sepia", "aqua turquoise",
        "purple amethyst", "white opal", "black onyx", "orange tangerine", "gray graphite", "blue periwinkle",
    ),
    "district": (
        "lakeside cabin", "market atrium", "valley observatory", "island library", "forest station", "canyon terminal",
        "prairie museum", "harbor academy", "mountain clinic", "village theater", "coastal laboratory", "central conservatory",
        "wetland cottage", "suburban hangar", "downtown chapel", "plateau warehouse", "tundra hostel", "campus pavilion",
    ),
    "finish": (
        "coarse velvet", "glazed marble", "matte resin", "etched brass", "knotted wool", "lacquered clay",
        "ribbed rubber", "satin nickel", "pebbled paper", "waxed canvas", "textured plaster", "beaded acrylic",
        "rough suede", "glossy quartz", "dimpled silicone", "pleated fabric", "hammered silver", "varnished oak",
    ),
    "readiness": (
        "available soon", "stored securely", "tested recently", "packed carefully", "verified manually", "scheduled tomorrow",
        "prepared indoors", "checked twice", "reserved quietly", "sorted correctly", "documented fully", "approved yesterday",
        "operating normally", "resting temporarily", "shipping shortly", "measured precisely", "cataloged separately", "assigned permanently",
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
    rows = []
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


def verify_upstream() -> dict[str, Any]:
    upstream = read_json(UPSTREAM_FINAL)
    if upstream.get("final_digest") != EXPECTED_UPSTREAM_DIGEST:
        raise RuntimeError("Phase 1220 final digest changed")
    if upstream.get("status") != "qwen3_master_task_behavior_gate_failed":
        raise RuntimeError("unexpected Phase 1220 status")
    return upstream


def spaced_token_length(tokenizer: Any, value: str) -> int:
    return len(tokenizer.encode(" " + value, add_special_tokens=False))


def groups_of_four_same_length(tokenizer: Any, values: Iterable[str]) -> list[list[str]]:
    buckets: dict[int, list[str]] = defaultdict(list)
    for value in values:
        buckets[spaced_token_length(tokenizer, value)].append(value)
    groups = []
    for token_length in sorted(buckets):
        bucket = buckets[token_length]
        for start in range(0, len(bucket) - 3, 4):
            groups.append(bucket[start : start + 4])
    return groups


def symbolic_candidates(prefix: str, count: int = 160) -> list[str]:
    syllables = ("ba", "ce", "di", "fo", "gu", "ha", "ji", "ko", "lu", "me", "no", "pa", "ri", "so", "tu", "ve")
    values = []
    for first in syllables:
        for second in syllables:
            values.append(f"{prefix} {first}{second}")
            if len(values) >= count:
                return values
    return values


def build_lexicon(tokenizer: Any) -> dict[str, Any]:
    natural_object_groups = groups_of_four_same_length(tokenizer, NATURAL_NAMES)
    symbolic_object_groups = groups_of_four_same_length(tokenizer, symbolic_candidates("unit"))
    natural_values = {
        relation: groups_of_four_same_length(tokenizer, values)
        for relation, values in VALUE_POOLS.items()
    }
    symbolic_values = {
        relation: groups_of_four_same_length(tokenizer, symbolic_candidates(f"{relation}code"))
        for relation in RELATIONS
    }
    if len(natural_object_groups) < 8 or len(symbolic_object_groups) < 8:
        raise RuntimeError("insufficient matched object groups")
    if any(len(groups) < 3 for groups in natural_values.values()):
        raise RuntimeError("insufficient matched natural value groups")
    if any(len(groups) < 8 for groups in symbolic_values.values()):
        raise RuntimeError("insufficient matched symbolic value groups")
    return {
        "natural_objects": natural_object_groups,
        "symbolic_objects": symbolic_object_groups,
        "natural_values": natural_values,
        "symbolic_values": symbolic_values,
    }


def choose_group(groups: list[list[str]], index: int, salt: int) -> list[str]:
    return list(groups[(index * 7 + salt * 11) % len(groups)])


def make_world(
    lexicon: dict[str, Any], split: str, family: str, local_index: int
) -> dict[str, Any]:
    split_index = SPLITS.index(split)
    family_index = list(FAMILIES).index(family)
    global_index = (split_index * len(FAMILIES) + family_index) * WORLDS_PER_FAMILY_SPLIT + local_index
    track = TRACKS[local_index % len(TRACKS)]
    rng = random.Random(1221000 + global_index * 97)
    object_groups = lexicon[f"{track}_objects"]
    objects = choose_group(object_groups, global_index, 1)
    rng.shuffle(objects)
    value_registry = lexicon[f"{track}_values"]
    values: dict[str, list[str]] = {}
    for relation_index, relation in enumerate(RELATIONS):
        group = choose_group(value_registry[relation], global_index, relation_index + 3)
        rng.shuffle(group)
        values[relation] = group
    assignments: dict[str, dict[str, str]] = {obj: {} for obj in objects}
    for relation_index, relation in enumerate(RELATIONS):
        shift = (global_index + relation_index * 3) % 4
        for object_index, obj in enumerate(objects):
            assignments[obj][relation] = values[relation][(object_index + shift) % 4]
    cycle = list(objects)
    rng.shuffle(cycle)
    if global_index % 2:
        cycle = list(reversed(cycle))
    links = {cycle[index]: cycle[(index + 1) % 4] for index in range(4)}
    return {
        "world_id": f"p1221-{split[:2]}-{family[:2]}-{local_index:03d}-{digest([split, family, local_index])[:8]}",
        "split": split,
        "family": family,
        "track": track,
        "world_index": global_index,
        "objects": objects,
        "relations": list(RELATIONS),
        "values": values,
        "assignments": assignments,
        "links": links,
        "carrier_fact": f"The catalog seal is {['triangle', 'hexagon', 'crescent', 'spiral'][global_index % 4]}.",
    }


def cloned_assignments(world: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {obj: dict(values) for obj, values in world["assignments"].items()}


def inverse_link(links: dict[str, str], target: str) -> str:
    matches = [source for source, value in links.items() if value == target]
    if len(matches) != 1:
        raise RuntimeError("link predecessor is not unique")
    return matches[0]


def operation_state(world: dict[str, Any], operation: str) -> dict[str, Any]:
    objects = world["objects"]
    relations = world["relations"]
    links = world["links"]
    assignments = cloned_assignments(world)
    source = objects[0]
    second = objects[1]
    third = objects[2]
    relation = relations[0]
    other_relation = relations[1]
    target = links[source]
    reverse = inverse_link(links, source)
    fingerprint: dict[str, str] = {}
    candidate_kind = "value"
    transformation_instruction: str | None = None
    display_assignments = cloned_assignments(world)
    derivation: list[str] = []

    if operation == "direct":
        query = f"What is {source}'s {relation}?"
        gold = assignments[source][relation]
        candidates = world["values"][relation]
        derivation = [source, relation, gold]
    elif operation == "query_object":
        query = f"For {second}, what is the {relation}?"
        gold = assignments[second][relation]
        candidates = world["values"][relation]
        derivation = [second, relation, gold]
    elif operation == "query_relation":
        query = f"What is {source}'s {other_relation}?"
        gold = assignments[source][other_relation]
        candidates = world["values"][other_relation]
        relation = other_relation
        derivation = [source, relation, gold]
    elif operation == "binding_swap":
        pre_swap = assignments[source][relation]
        assignments[source][relation], assignments[second][relation] = (
            assignments[second][relation], assignments[source][relation]
        )
        transformation_instruction = (
            f"Exchange only the {relation} values of {source} and {second}; keep every other field unchanged."
        )
        query = f"After that exchange, what is {source}'s {relation}?"
        gold = assignments[source][relation]
        candidates = world["values"][relation]
        fingerprint[pre_swap] = "pre_swap_value"
        derivation = ["swap", source, second, relation, source, gold]
    elif operation == "inverse_lookup":
        target_value = assignments[third][relation]
        query = f"Which object has {relation} {target_value}?"
        gold = third
        candidates = objects
        candidate_kind = "object"
        derivation = [relation, target_value, gold]
    elif operation == "link_marker_control":
        query = f"The links are shown, but read {source} itself: what is its {relation}?"
        gold = assignments[source][relation]
        candidates = world["values"][relation]
        derivation = [source, relation, gold]
    elif operation == "link_object":
        query = f"Which object is reached by following the link from {source} once?"
        gold = target
        candidates = objects
        candidate_kind = "object"
        derivation = [source, "link", gold]
    elif operation == "link_then_value":
        query = f"Follow the link from {source} once. What is the reached object's {relation}?"
        gold = assignments[target][relation]
        candidates = world["values"][relation]
        derivation = [source, "link", target, relation, gold]
    elif operation == "reverse_link_object":
        query = f"Which object points directly to {source}?"
        gold = reverse
        candidates = objects
        candidate_kind = "object"
        derivation = ["inverse_link", source, gold]
    elif operation == "reverse_link_then_value":
        query = f"Find the object that points directly to {source}. What is that object's {relation}?"
        gold = assignments[reverse][relation]
        candidates = world["values"][relation]
        derivation = ["inverse_link", source, reverse, relation, gold]
    elif operation == "object_relation_compose":
        query = f"Use object {second} and relation {other_relation}. What value do they select?"
        gold = assignments[second][other_relation]
        candidates = world["values"][other_relation]
        relation = other_relation
        derivation = [second, relation, gold]
    elif operation == "binding_query_compose":
        pre_swap = assignments[second][other_relation]
        assignments[second][other_relation], assignments[third][other_relation] = (
            assignments[third][other_relation], assignments[second][other_relation]
        )
        transformation_instruction = (
            f"Exchange only the {other_relation} values of {second} and {third}; keep every other field unchanged."
        )
        query = f"After that exchange, use object {second} and relation {other_relation}. What value results?"
        gold = assignments[second][other_relation]
        candidates = world["values"][other_relation]
        relation = other_relation
        fingerprint[pre_swap] = "pre_swap_value"
        derivation = ["swap", second, third, relation, "query", second, gold]
    elif operation == "double_link_relation":
        double_target = links[target]
        query = f"Starting at {source}, follow two links. What is the final object's {relation}?"
        gold = assignments[double_target][relation]
        candidates = world["values"][relation]
        fingerprint[assignments[target][relation]] = "stopped_after_one_link"
        derivation = [source, "link", target, "link", double_target, relation, gold]
    elif operation == "inverse_link_compose":
        target_value = assignments[third][other_relation]
        found = third
        composed_target = links[found]
        query = (
            f"Find the object whose {other_relation} is {target_value}, follow its link once, "
            f"then report the reached object's {relation}."
        )
        gold = assignments[composed_target][relation]
        candidates = world["values"][relation]
        fingerprint[assignments[found][relation]] = "stopped_after_inverse_lookup"
        derivation = [other_relation, target_value, found, "link", composed_target, relation, gold]
    elif operation == "link_binding_compose":
        pre_target = assignments[target][relation]
        swap_partner = next(obj for obj in objects if obj != target)
        assignments[target][relation], assignments[swap_partner][relation] = (
            assignments[swap_partner][relation], assignments[target][relation]
        )
        transformation_instruction = (
            f"Exchange only the {relation} values of {target} and {swap_partner}; keep every other field unchanged."
        )
        query = f"After that exchange, follow the link from {source}. What is the reached object's {relation}?"
        gold = assignments[target][relation]
        candidates = world["values"][relation]
        fingerprint[pre_target] = "ignored_binding_exchange"
        derivation = ["swap", target, swap_partner, relation, source, "link", target, gold]
    else:
        raise KeyError(operation)

    if candidate_kind == "value":
        fingerprint.setdefault(assignments[source][relation], "source_object_value")
        fingerprint.setdefault(assignments[target][relation], "forward_link_value")
        fingerprint.setdefault(assignments[links[target]][relation], "double_link_value")
        fingerprint.setdefault(assignments[reverse][relation], "reverse_link_value")
        for obj in objects:
            fingerprint.setdefault(assignments[obj][relation], f"value_of_{objects.index(obj)}")
    else:
        fingerprint.setdefault(source, "source_object")
        fingerprint.setdefault(target, "forward_link_object")
        fingerprint.setdefault(links[target], "double_link_object")
        fingerprint.setdefault(reverse, "reverse_link_object")
    fingerprint[gold] = "gold"
    return {
        "display_assignments": display_assignments,
        "rendered_assignments": assignments,
        "transformation_instruction": transformation_instruction,
        "query": query,
        "gold": gold,
        "candidates": list(candidates),
        "candidate_kind": candidate_kind,
        "relation": relation,
        "derivation": derivation,
        "fingerprint_by_candidate": fingerprint,
    }


def record_lines(assignments: dict[str, dict[str, str]], relations: list[str]) -> list[str]:
    lines = []
    for obj, values in assignments.items():
        fields = "; ".join(f"{relation}={values[relation]}" for relation in relations)
        lines.append(f"{obj}: {fields}.")
    return lines


def link_lines(links: dict[str, str]) -> list[str]:
    return [f"{source} -> {target}." for source, target in links.items()]


def render_prompt(world: dict[str, Any], state: dict[str, Any], panel: str, candidate_order: list[str]) -> str:
    records = record_lines(state["display_assignments"], world["relations"])
    links = link_lines(world["links"])
    index = world["world_index"]
    if panel == "record_order":
        records = list(reversed(records))
        links = links[1:] + links[:1]
    instructions = [state["transformation_instruction"]] if state["transformation_instruction"] else []
    carrier = [world["carrier_fact"]] if panel == "matched_carrier" else []
    split = world["split"]
    if panel == "paraphrase" or split == "natural_use":
        body = " ".join(records + links + instructions + carrier)
        lead = "In the current account," if split == "natural_use" else "The current registry says:"
        prompt = f"{lead} {body}\nQuestion: {state['query']}"
    elif split == "sealed":
        body = "\n".join(f"[{position + 1}] {line}" for position, line in enumerate(records + links + instructions + carrier))
        prompt = f"CURRENT TABLE\n{body}\nREQUEST: {state['query']}"
    else:
        body = "\n".join(records + links + instructions + carrier)
        prompt = f"Current world records:\n{body}\nQuery: {state['query']}"
    options = " | ".join(candidate_order)
    return f"{prompt}\nOptions: {options}\nAnswer:"


def build_materials(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lexicon = build_lexicon(tokenizer)
    permutations = list(itertools.permutations(range(4)))
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for family, operations in FAMILIES.items():
            for local_index in range(WORLDS_PER_FAMILY_SPLIT):
                world = make_world(lexicon, split, family, local_index)
                for operation_index, operation in enumerate(operations):
                    state = operation_state(world, operation)
                    group_id = f"{world['world_id']}::{operation}"
                    for panel_index, panel in enumerate(PANELS):
                        permutation = permutations[(world["world_index"] * 13 + operation_index * 5 + panel_index) % len(permutations)]
                        candidate_order = [state["candidates"][position] for position in permutation]
                        prompt = render_prompt(world, state, panel, candidate_order)
                        row = {
                            "schema_version": "phase1221.typed.row.v1",
                            "phase": PHASE,
                            "item_id": f"p1221-{digest([group_id, panel])[:20]}",
                            "world_id": world["world_id"],
                            "group_id": group_id,
                            "split": split,
                            "family": family,
                            "track": world["track"],
                            "operation": operation,
                            "panel": panel,
                            "objects": world["objects"],
                            "relations": world["relations"],
                            "base_assignments": world["assignments"],
                            "display_assignments": state["display_assignments"],
                            "rendered_assignments": state["rendered_assignments"],
                            "links": world["links"],
                            "target_relation": state["relation"],
                            "candidate_kind": state["candidate_kind"],
                            "transformation_instruction": state["transformation_instruction"],
                            "derivation": state["derivation"],
                            "query": state["query"],
                            "prompt": prompt,
                            "gold": state["gold"],
                            "candidates": state["candidates"],
                            "candidate_order": candidate_order,
                            "gold_position": candidate_order.index(state["gold"]),
                            "fingerprint_by_candidate": state["fingerprint_by_candidate"],
                            "generation_required": True,
                        }
                        row["row_digest"] = digest(row)
                        rows.append(row)
    lexicon_audit = {
        "natural_object_group_count": len(lexicon["natural_objects"]),
        "symbolic_object_group_count": len(lexicon["symbolic_objects"]),
        "natural_value_group_counts": {key: len(value) for key, value in lexicon["natural_values"].items()},
        "symbolic_value_group_counts": {key: len(value) for key, value in lexicon["symbolic_values"].items()},
    }
    return rows, lexicon_audit


def render_native(tokenizer: Any, prompt: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def build_manifest(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = []
    prompt_lengths = []
    candidate_lengths = []
    equal_length_rows = 0
    for index, row in enumerate(rows):
        rendered = render_native(tokenizer, row["prompt"])
        input_ids: list[int] | None = None
        candidate_token_ids: dict[str, list[int]] = {}
        for candidate in row["candidate_order"]:
            base, suffix = p1220.continuation_ids(tokenizer, rendered, candidate)
            if input_ids is None:
                input_ids = base
            elif input_ids != base:
                raise RuntimeError("candidate changed prompt prefix")
            candidate_token_ids[candidate] = suffix
            candidate_lengths.append(len(suffix))
        assert input_ids is not None
        lengths = {len(value) for value in candidate_token_ids.values()}
        equal_length_rows += len(lengths) == 1
        prompt_lengths.append(len(input_ids))
        item = {
            "schema_version": "phase1221.qwen3.manifest.v1",
            "item_id": row["item_id"],
            "row_digest": row["row_digest"],
            "split": row["split"],
            "family": row["family"],
            "track": row["track"],
            "operation": row["operation"],
            "panel": row["panel"],
            "generation_required": True,
            "gold": row["gold"],
            "candidates": row["candidate_order"],
            "candidate_order": row["candidate_order"],
            "input_ids": input_ids,
            "input_token_count": len(input_ids),
            "candidate_token_ids": candidate_token_ids,
            "rendered_prompt_digest": digest(rendered),
        }
        item["manifest_row_digest"] = digest(item)
        manifest.append(item)
        if (index + 1) % 2048 == 0:
            print(f"[phase1221/tokenize] {index + 1}/{len(rows)}", flush=True)
    audit = {
        "phase": PHASE,
        "row_count": len(manifest),
        "prompt_token_count_min": min(prompt_lengths),
        "prompt_token_count_max": max(prompt_lengths),
        "prompt_token_count_mean": sum(prompt_lengths) / len(prompt_lengths),
        "candidate_token_count_min": min(candidate_lengths),
        "candidate_token_count_max": max(candidate_lengths),
        "candidate_token_count_mean": sum(candidate_lengths) / len(candidate_lengths),
        "equal_candidate_token_length_row_rate": equal_length_rows / len(rows),
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


def build_protocol(rows: list[dict[str, Any]], manifest: list[dict[str, Any]], token_audit: dict[str, Any], lexicon_audit: dict[str, Any]) -> dict[str, Any]:
    upstream = verify_upstream()
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1221.typed.protocol.v1",
        "created_at": utc_now(),
        "purpose": "independently authorize operation-by-interface families before any physical scan",
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1220_main": file_sha256(TEST_ROOT / "phase1220_object_relation_value_master_task.py"),
            "phase1220_final": file_sha256(UPSTREAM_FINAL),
        },
        "upstream": {
            "phase1220_final_digest": upstream["final_digest"],
            "phase1220_status": upstream["status"],
            "explicit_user_restart_received": True,
        },
        "design": {
            "splits": list(SPLITS),
            "tracks": list(TRACKS),
            "panels": list(PANELS),
            "families": {key: list(value) for key, value in FAMILIES.items()},
            "worlds_per_family_split": WORLDS_PER_FAMILY_SPLIT,
            "world_count": EXPECTED_WORLDS,
            "row_count": EXPECTED_ROWS,
            "independent_unit": "world",
            "all_panels_generated": True,
            "candidate_order_uses_all_24_permutations": True,
            "same_row_candidate_token_lengths_equal": True,
            "carrier_has_no_ignore/latest/obsolete control words": True,
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
            "system_prompt": SYSTEM_PROMPT,
            "precision": "FP16",
            "quantization": "none",
            "native_chat_template": True,
            "enable_thinking": False,
            "candidate_primary_score": "sum full-continuation log probability",
            "candidate_secondary_score": "mean full-continuation log probability",
            "low_margin_prefill_threshold": LOW_MARGIN_PREFILL_THRESHOLD,
            "low_margin_rule": "cache top-two sum margin <= threshold triggers direct full-prefix prefill for all four candidates",
            "free_generation": "greedy generation on all panels",
            "hidden_states": False,
            "attentions": False,
            "hooks": False,
            "interventions": False,
        },
        "operation_authorization": {
            "finite_min_each_split": FINITE_MIN,
            "candidate_accuracy_min_each_split": CANDIDATE_MIN,
            "generation_accuracy_min_each_split": GENERATION_MIN,
            "worst_panel_candidate_min_each_split": WORST_PANEL_MIN,
            "all_four_panels_candidate_correct_world_rate_min_each_split": SURFACE_GROUP_MIN,
            "sum_mean_winner_agreement_min_each_split": SUM_MEAN_AGREEMENT_MIN,
            "operation_track_gate": "conjunction across discovery, confirmation, natural_use, and sealed",
            "family_track_gate": "conjunction of all five predeclared operations in that family and track",
            "unified_gate": "conjunction of every family-track gate; used only for unified claims",
        },
        "authorization": {
            "any_family_track_pass": "authorize a separately frozen Phase1222 trajectory and causal-response stage for exactly those family-track scopes",
            "no_family_track_pass": "stop before hidden state",
            "phase1220_success_cells_not_used_for_authorization": True,
        },
        "forbidden_after_freeze": [
            "change worlds, vocabulary, prompts, operations, tracks, panels, thresholds, parser, or scoring rule",
            "merge or split families after behavior reveal",
            "authorize an operation from Phase1220 results",
            "drop hard panels, worlds, operations, or tracks",
            "inspect hidden states, attentions, heads, neurons, or interventions in Phase1221",
            "call generated natural-word worlds organic natural language",
        ],
        "claim_boundary": {
            "behavior_only": True,
            "qwen3_only": True,
            "generated_worlds": True,
            "organic_language": False,
            "operation_and_interface_typed": True,
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
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, lexicon_audit = build_materials(tokenizer)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected material count {len(rows)}")
    manifest, token_audit = build_manifest(rows, tokenizer)
    protocol = build_protocol(rows, manifest, token_audit, lexicon_audit)
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(TOKEN_AUDIT_PATH, token_audit)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"status": "materialized", "rows": len(rows), "worlds": EXPECTED_WORLDS, "protocol_digest": protocol["protocol_digest"]}))


def verify_formal_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    if protocol["source_hashes"]["main"] != file_sha256(SCRIPT):
        raise RuntimeError("main script changed after freeze")
    if protocol["source_hashes"]["audit"] != file_sha256(AUDIT_SCRIPT):
        raise RuntimeError("audit script changed after freeze")
    if digest({key: value for key, value in protocol.items() if key != "protocol_digest"}) != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest invalid")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit failed")
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    if digest(rows) != protocol["material"]["material_digest"] or digest(manifest) != protocol["material"]["manifest_digest"]:
        raise RuntimeError("formal material changed")
    return protocol, rows, manifest


def direct_prefill_scores(model: Any, device: torch.device, row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = []
    prompt = [int(value) for value in row["input_ids"]]
    for candidate in row["candidates"]:
        continuation = [int(value) for value in row["candidate_token_ids"][candidate]]
        entries.append((candidate, continuation, prompt + continuation))
    lengths = {len(sequence) for _, _, sequence in entries}
    if len(lengths) != 1:
        raise RuntimeError("direct prefill requires matched candidate lengths")
    input_ids = torch.tensor([entry[2] for entry in entries], dtype=torch.long, device=device)
    continuation_length = len(entries[0][1])
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=continuation_length + 1,
            return_dict=True,
        )
    output_start = input_ids.shape[1] - output.logits.shape[1]
    result = {}
    for index, (candidate, continuation, _) in enumerate(entries):
        token_scores = []
        finite = True
        for offset, token_id in enumerate(continuation):
            absolute_position = len(prompt) + offset - 1
            logits = output.logits[index, absolute_position - output_start].float()
            finite = finite and bool(torch.isfinite(logits).all().item())
            score = logits[token_id] - torch.logsumexp(logits, dim=-1)
            token_scores.append(float(score.item()))
        result[candidate] = {
            "sum_log_probability": sum(token_scores),
            "mean_log_probability": sum(token_scores) / len(token_scores),
            "token_count": len(token_scores),
            "all_vocab_logits_finite": finite and all(math.isfinite(value) for value in token_scores),
            "scoring_path": "direct_prefill_low_margin",
        }
    del output, input_ids
    return result


def score_candidates_with_fallback(model: Any, device: torch.device, manifest: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    started = time.time()
    scores, runtime = p1220.candidate_scores(model, device, manifest)
    low_margin_rows = []
    cache_sum_mean_disagreement = 0
    for row in manifest:
        values = scores[row["item_id"]]
        for value in values.values():
            value["scoring_path"] = "shared_prefix_cache"
        sum_order = sorted(values, key=lambda candidate: values[candidate]["sum_log_probability"], reverse=True)
        mean_order = sorted(values, key=lambda candidate: values[candidate]["mean_log_probability"], reverse=True)
        cache_sum_mean_disagreement += sum_order[0] != mean_order[0]
        margin = values[sum_order[0]]["sum_log_probability"] - values[sum_order[1]]["sum_log_probability"]
        if margin <= LOW_MARGIN_PREFILL_THRESHOLD:
            low_margin_rows.append(row)
    for index, row in enumerate(low_margin_rows):
        scores[row["item_id"]] = direct_prefill_scores(model, device, row)
        if (index + 1) % 250 == 0:
            print(f"[phase1221/prefill-fallback] {index + 1}/{len(low_margin_rows)}", flush=True)
    return scores, {
        "cache_runtime": runtime,
        "low_margin_prefill_count": len(low_margin_rows),
        "cache_sum_mean_disagreement_count": cache_sum_mean_disagreement,
        "elapsed_seconds": time.time() - started,
    }


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("behavior output already exists")
    protocol, rows, manifest = verify_formal_inputs()
    material_by_id = {row["item_id"]: row for row in rows}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    try:
        candidate_scores, candidate_runtime = score_candidates_with_fallback(model, device, manifest)
        generations, generation_runtime = p1220.generation_scores(model, tokenizer, device, manifest)
        raw = []
        for manifest_row in manifest:
            item_id = manifest_row["item_id"]
            material = material_by_id[item_id]
            scores = candidate_scores[item_id]
            sum_order = sorted(scores, key=lambda candidate: scores[candidate]["sum_log_probability"], reverse=True)
            mean_order = sorted(scores, key=lambda candidate: scores[candidate]["mean_log_probability"], reverse=True)
            sum_margin = scores[sum_order[0]]["sum_log_probability"] - scores[sum_order[1]]["sum_log_probability"]
            sum_prediction = None if abs(sum_margin) <= TIE_TOLERANCE else sum_order[0]
            mean_prediction = mean_order[0]
            generation = generations[item_id]
            fingerprint = material["fingerprint_by_candidate"].get(sum_prediction, "unregistered_candidate") if sum_prediction else "tie"
            prediction_position = (
                material["candidate_order"].index(sum_prediction)
                if sum_prediction in material["candidate_order"]
                else None
            )
            row = {
                "schema_version": "phase1221.qwen3.behavior.row.v1",
                "phase": PHASE,
                "protocol_digest": protocol["protocol_digest"],
                "item_id": item_id,
                "row_digest": manifest_row["row_digest"],
                "world_id": material["world_id"],
                "group_id": material["group_id"],
                "split": material["split"],
                "family": material["family"],
                "track": material["track"],
                "operation": material["operation"],
                "panel": material["panel"],
                "gold": material["gold"],
                "gold_position": material["gold_position"],
                "prediction_position": prediction_position,
                "candidate_scores": scores,
                "sum_prediction": sum_prediction,
                "mean_prediction": mean_prediction,
                "sum_mean_winner_agreement": sum_prediction == mean_prediction,
                "candidate_correct": sum_prediction == material["gold"],
                "sum_margin": sum_margin,
                "scoring_path": next(iter(scores.values()))["scoring_path"],
                "all_candidate_scores_finite": all(value["all_vocab_logits_finite"] for value in scores.values()),
                "error_fingerprint": fingerprint,
                "generation_prediction": generation["generation_prediction"],
                "generation_correct": generation["generation_prediction"] == material["gold"],
                "generation_normalized_exact": generation["generation_normalized_exact"],
                "generated_token_ids": generation["generated_token_ids"],
                "generated_text": generation["generated_text"],
            }
            row["behavior_row_digest"] = digest(row)
            raw.append(row)
        write_jsonl(RAW_PATH, raw)
        summary = {
            "schema_version": "phase1221.qwen3.run_summary.v1",
            "phase": PHASE,
            "created_at": utc_now(),
            "model": "qwen3",
            "protocol_digest": protocol["protocol_digest"],
            "case_count": len(raw),
            "candidate_runtime": candidate_runtime,
            "generation_runtime": generation_runtime,
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
        print(canonical_json({"status": "behavior_complete", "case_count": len(raw), "summary_digest": summary["summary_digest"]}))
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def operation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_panel = {panel: rate([row for row in rows if row["panel"] == panel], "candidate_correct") for panel in PANELS}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    surface_success = [len(values) == len(PANELS) and all(row["candidate_correct"] for row in values) for values in groups.values()]
    errors = Counter(row["error_fingerprint"] for row in rows if not row["candidate_correct"])
    position_predictions = Counter(
        row["prediction_position"] for row in rows if row["prediction_position"] is not None
    )
    correct_positions = Counter(row["gold_position"] for row in rows if row["candidate_correct"])
    return {
        "case_count": len(rows),
        "world_count": len({row["world_id"] for row in rows}),
        "finite_rate": rate(rows, "all_candidate_scores_finite"),
        "candidate_accuracy": rate(rows, "candidate_correct"),
        "generation_accuracy": rate(rows, "generation_correct"),
        "generation_exact_option_rate": rate(rows, "generation_normalized_exact"),
        "sum_mean_winner_agreement": rate(rows, "sum_mean_winner_agreement"),
        "candidate_by_panel": by_panel,
        "worst_panel_candidate": min(by_panel.values()),
        "surface_group_rate": sum(surface_success) / len(surface_success),
        "direct_prefill_fraction": sum(row["scoring_path"] == "direct_prefill_low_margin" for row in rows) / len(rows),
        "mean_sum_margin": sum(row["sum_margin"] for row in rows) / len(rows),
        "error_fingerprints": dict(sorted(errors.items())),
        "prediction_position_counts": {str(key): value for key, value in sorted(position_predictions.items())},
        "correct_gold_position_counts": {str(key): value for key, value in sorted(correct_positions.items())},
    }


def metric_gate(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "finite": metrics["finite_rate"] >= FINITE_MIN,
        "candidate": metrics["candidate_accuracy"] >= CANDIDATE_MIN,
        "generation": metrics["generation_accuracy"] >= GENERATION_MIN,
        "worst_panel": metrics["worst_panel_candidate"] >= WORST_PANEL_MIN,
        "surface_group": metrics["surface_group_rate"] >= SURFACE_GROUP_MIN,
        "sum_mean_agreement": metrics["sum_mean_winner_agreement"] >= SUM_MEAN_AGREEMENT_MIN,
    }


def summarize_behavior(raw: list[dict[str, Any]]) -> dict[str, Any]:
    operation_results: dict[str, Any] = {}
    operation_authorization: dict[str, bool] = {}
    family_authorization: dict[str, bool] = {}
    for family, operations in FAMILIES.items():
        for track in TRACKS:
            family_passes = []
            for operation in operations:
                split_results = {}
                split_passes = []
                for split in SPLITS:
                    selected = [
                        row for row in raw
                        if row["family"] == family and row["track"] == track
                        and row["operation"] == operation and row["split"] == split
                    ]
                    metrics = operation_metrics(selected)
                    gates = metric_gate(metrics)
                    passed = all(gates.values())
                    split_results[split] = {"metrics": metrics, "gates": gates, "passed": passed}
                    split_passes.append(passed)
                key = f"{family}|{track}|{operation}"
                authorized = all(split_passes)
                operation_results[key] = {"splits": split_results, "authorized": authorized}
                operation_authorization[key] = authorized
                family_passes.append(authorized)
            family_authorization[f"{family}|{track}"] = all(family_passes)
    authorized_scopes = sorted(key for key, value in family_authorization.items() if value)
    return {
        "operation_results": operation_results,
        "operation_authorization": operation_authorization,
        "family_authorization": family_authorization,
        "authorized_family_tracks": authorized_scopes,
        "any_family_track_authorized": bool(authorized_scopes),
        "unified_authorized": all(family_authorization.values()),
    }


def finalize() -> None:
    protocol, rows, _manifest = verify_formal_inputs()
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    if len(raw) != len(rows) or summary["raw_digest"] != digest(raw):
        raise RuntimeError("raw output mismatch")
    behavior = summarize_behavior(raw)
    authorized = behavior["authorized_family_tracks"]
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "typed_behavior_authorized" if authorized else "typed_behavior_no_family_authorized",
        "protocol_digest": protocol["protocol_digest"],
        "material_digest": protocol["material"]["material_digest"],
        "manifest_digest": protocol["material"]["manifest_digest"],
        "run_summary_digest": summary["summary_digest"],
        "behavior": behavior,
        "k_item": {
            "identifier": "K198",
            "evidence_grade": "E3-BEHAVIOR" if authorized else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                f"Operation-by-interface authorization passed for: {authorized}."
                if authorized
                else "No predeclared operation family and interface passed all four independent split gates."
            ),
            "scope": "Qwen3 FP16; generated worlds; behavior only",
        },
        "authorized_next": {
            "automatic_execution": bool(authorized),
            "experiment": "Phase1222 physical trajectory and causal-response tensor" if authorized else None,
            "authorized_family_tracks": authorized,
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
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    rows, _ = build_materials(tokenizer)
    assert len(rows) == EXPECTED_ROWS
    assert len({row["item_id"] for row in rows}) == EXPECTED_ROWS
    assert len({row["world_id"] for row in rows}) == EXPECTED_WORLDS
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    assert all(len(values) == len(PANELS) and len({row["gold"] for row in values}) == 1 for values in groups.values())
    print(canonical_json({"status": "selftest_passed", "rows": len(rows), "worlds": EXPECTED_WORLDS}))


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
