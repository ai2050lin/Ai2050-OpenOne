#!/usr/bin/env python3
"""Phase 1220: freeze and execute one object-relation-value master task.

This phase is deliberately behavior-only.  It creates split-disjoint abstract
worlds, exact semantic operations, same-answer surface controls, multi-token
candidate continuations, and a sealed Qwen3 FP16 behavior test.  It never
requests hidden states, attentions, hooks, or interventions.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import re
import string
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1220
ENGINEERING_REVISION = 2
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1220_object_relation_value_master_task_audit.py"
UPSTREAM_FINAL = (
    TEST_ROOT
    / "result/phase1219_target_typed_prerule_prediction/analysis/final.json"
)
EXPECTED_UPSTREAM_DIGEST = (
    "d082328f2b47da8a824f34e3e668c28da0d02d4922b90a160659c80824e74eea"
)

OUT_ROOT = TEST_ROOT / "result/phase1220_object_relation_value_master_task"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/master_task.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
TOKEN_AUDIT_PATH = OUT_ROOT / "audit/tokenizer_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = ("discovery", "confirmation", "composition", "sealed")
TRACKS = ("natural", "symbolic")
PANELS = ("canonical", "record_order", "paraphrase", "same_answer_carrier")
WORLDS_PER_SPLIT = 128
OPERATIONS_PER_WORLD = 6
ROWS_PER_WORLD = OPERATIONS_PER_WORLD * len(PANELS)
EXPECTED_WORLDS = len(SPLITS) * WORLDS_PER_SPLIT
EXPECTED_ROWS = EXPECTED_WORLDS * ROWS_PER_WORLD
GENERATION_PANEL = "canonical"

CANDIDATE_BATCH_SIZE = 12
GENERATION_BATCH_SIZE = 16
TIE_TOLERANCE = 1e-7

FINITE_RATE_MIN = 1.0
CANDIDATE_OVERALL_MIN = 0.90
CANDIDATE_WORST_CELL_MIN = 0.80
SURFACE_GROUP_MIN = 0.80
SEMANTIC_SET_MIN = 0.75
GENERATION_OVERALL_MIN = 0.80
GENERATION_WORST_CELL_MIN = 0.65
GENERATION_EXACT_MIN = 0.70
COMPOSITION_SEALED_CANDIDATE_MIN = 0.85
COMPOSITION_SEALED_GENERATION_MIN = 0.75

SYSTEM_PROMPT = (
    "Use only the current world described by the user. Ignore archived or "
    "obsolete notes. Return exactly one listed option and no explanation."
)

PRIMITIVE_OPERATIONS = (
    "direct",
    "query_object",
    "query_relation",
    "binding_swap",
    "inverse_lookup",
    "link_then_value",
)

COMPOSITION_OPERATIONS = (
    "object_relation_compose",
    "binding_query_compose",
    "link_relation_compose",
    "double_link_relation",
    "inverse_link_compose",
    "link_binding_compose",
)

SEALED_OPERATIONS = (
    "direct",
    "inverse_lookup",
    "link_then_value",
    "object_relation_compose",
    "double_link_relation",
    "inverse_link_compose",
)

RELATIONS = {
    "natural": ("color", "material", "location", "status"),
    "symbolic": ("marker", "texture", "station", "mode"),
}

NAME_POOLS = {
    "discovery": {
        "natural": (
            "Alice", "Bruno", "Carla", "Diego", "Elena", "Felix", "Grace", "Henry",
            "Irene", "Jacob", "Kiara", "Liam", "Maya", "Noah", "Olive", "Peter",
        ),
        "symbolic": (
            "Alder", "Beryl", "Cedar", "Dorian", "Ember", "Fable", "Garnet", "Haven",
            "Indigo", "Jasper", "Kestrel", "Lumen", "Morrow", "Nectar", "Onyx", "Pollen",
        ),
    },
    "confirmation": {
        "natural": (
            "Quinn", "Rosa", "Simon", "Talia", "Uma", "Victor", "Wendy", "Xavier",
            "Yara", "Zane", "Adele", "Boris", "Celine", "Damon", "Eva", "Frank",
        ),
        "symbolic": (
            "Quartz", "Riven", "Sable", "Tundra", "Umber", "Vesper", "Willow", "Xenon",
            "Yonder", "Zephyr", "Arbor", "Brisk", "Cipher", "Delta", "Echo", "Fjord",
        ),
    },
    "composition": {
        "natural": (
            "Gina", "Hugo", "Isla", "Jonas", "Kara", "Leon", "Mona", "Niko",
            "Opal", "Pablo", "Rina", "Sven", "Tina", "Ugo", "Vera", "Wade",
        ),
        "symbolic": (
            "Grove", "Harbor", "Ivory", "Junction", "Kernel", "Lagoon", "Meadow", "Nimbus",
            "Orbit", "Prairie", "Reef", "Summit", "Timber", "Unity", "Vale", "Warden",
        ),
    },
    "sealed": {
        "natural": (
            "Aiden", "Bella", "Cyrus", "Delia", "Ethan", "Freya", "Gavin", "Hazel",
            "Ivan", "Julia", "Kian", "Luna", "Marco", "Nina", "Owen", "Priya",
        ),
        "symbolic": (
            "Anchor", "Beacon", "Canyon", "Drift", "Estuary", "Flint", "Glade", "Horizon",
            "Inlet", "Jetty", "Knoll", "Lantern", "Marsh", "Nova", "Oasis", "Pillar",
        ),
    },
}

NATURAL_VALUE_POOLS = {
    "discovery": (
        ("crimson red", "navy blue", "forest green", "golden yellow"),
        ("polished steel", "woven cotton", "clear glass", "dark wood"),
        ("north hall", "south garden", "east tower", "west harbor"),
        ("ready now", "locked shut", "under repair", "awaiting review"),
    ),
    "confirmation": (
        ("ruby scarlet", "ocean azure", "moss olive", "sunlit amber"),
        ("brushed copper", "smooth leather", "frosted crystal", "carved stone"),
        ("upper gallery", "lower courtyard", "river annex", "hill depot"),
        ("cleared today", "sealed firmly", "being serviced", "pending approval"),
    ),
    "composition": (
        ("coral vermilion", "midnight indigo", "fern emerald", "honey ochre"),
        ("tempered bronze", "spun linen", "shaped ceramic", "pressed bamboo"),
        ("cedar chamber", "willow terrace", "granite loft", "marina office"),
        ("active currently", "paused safely", "in inspection", "queued remotely"),
    ),
    "sealed": (
        ("rose carmine", "storm cobalt", "pine jade", "desert saffron"),
        ("forged titanium", "braided hemp", "molded porcelain", "laminated cork"),
        ("orchard studio", "meadow archive", "cliff workshop", "canal pavilion"),
        ("enabled locally", "disabled briefly", "under calibration", "waiting dispatch"),
    ),
}

SYMBOLIC_ADJECTIVES = {
    "discovery": ("amber", "silver", "violet", "cobalt", "scarlet", "ivory", "teal", "golden"),
    "confirmation": ("quartz", "umber", "indigo", "copper", "jade", "pearl", "sienna", "azure"),
    "composition": ("coral", "bronze", "lilac", "cerulean", "crimson", "opal", "moss", "saffron"),
    "sealed": ("ruby", "platinum", "plum", "navy", "rose", "alabaster", "pine", "ochre"),
}

SYMBOLIC_NOUNS = {
    "discovery": ("crest", "arch", "field", "gate", "rune", "spire", "harbor", "grove"),
    "confirmation": ("beacon", "ridge", "cipher", "vault", "signal", "plaza", "inlet", "crown"),
    "composition": ("compass", "bridge", "garden", "portal", "tablet", "summit", "lagoon", "banner"),
    "sealed": ("anchor", "lantern", "meadow", "pavilion", "emblem", "canyon", "island", "column"),
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
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number} in {path} is not an object")
            rows.append(item)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists():
        pending.unlink()
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(pending, path)


def rotate(values: list[str], shift: int) -> list[str]:
    offset = shift % len(values)
    return values[offset:] + values[:offset]


def operations_for_split(split: str) -> tuple[str, ...]:
    if split in {"discovery", "confirmation"}:
        return PRIMITIVE_OPERATIONS
    if split == "composition":
        return COMPOSITION_OPERATIONS
    if split == "sealed":
        return SEALED_OPERATIONS
    raise KeyError(split)


def choose_entities(split: str, track: str, world_index: int) -> list[str]:
    pool = list(NAME_POOLS[split][track])
    block = (world_index // 2) % 4
    selected = pool[4 * block : 4 * block + 4]
    permutation = (world_index * 3 + (1 if track == "symbolic" else 0)) % 4
    return rotate(selected, permutation)


def symbolic_values(split: str, world_index: int) -> list[str]:
    adjectives = SYMBOLIC_ADJECTIVES[split]
    nouns = SYMBOLIC_NOUNS[split]
    return [
        f"{adjectives[(world_index + 2 * index) % len(adjectives)]} "
        f"{nouns[(3 * world_index + index) % len(nouns)]}"
        for index in range(4)
    ]


def value_sets(split: str, track: str, world_index: int) -> list[list[str]]:
    if track == "natural":
        base = [list(values) for values in NATURAL_VALUE_POOLS[split]]
        return [rotate(values, world_index + relation_index) for relation_index, values in enumerate(base)]
    shared = symbolic_values(split, world_index)
    return [list(shared) for _ in range(4)]


def make_assignments(
    entities: list[str],
    relations: list[str],
    values: list[list[str]],
    world_index: int,
) -> dict[str, dict[str, str]]:
    multiplier = 1 if world_index % 2 == 0 else 3
    result: dict[str, dict[str, str]] = {}
    for entity_index, entity in enumerate(entities):
        result[entity] = {}
        for relation_index, relation in enumerate(relations):
            shift = (world_index + 2 * relation_index + world_index // 7) % 4
            value_index = (multiplier * entity_index + shift) % 4
            result[entity][relation] = values[relation_index][value_index]
    return result


def make_links(entities: list[str], world_index: int) -> dict[str, str]:
    step = 1 if world_index % 2 == 0 else 3
    return {
        entity: entities[(index + step) % 4]
        for index, entity in enumerate(entities)
    }


def clone_assignments(value: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {entity: dict(fields) for entity, fields in value.items()}


def swap_binding(
    assignments: dict[str, dict[str, str]],
    left: str,
    right: str,
    relation: str,
) -> None:
    assignments[left][relation], assignments[right][relation] = (
        assignments[right][relation],
        assignments[left][relation],
    )


def inverse_entity(
    assignments: dict[str, dict[str, str]],
    entities: list[str],
    relation: str,
    target: str,
) -> str:
    matches = [entity for entity in entities if assignments[entity][relation] == target]
    if len(matches) != 1:
        raise RuntimeError(f"inverse query is not unique: {relation=} {target=} {matches=}")
    return matches[0]


def operation_state(world: dict[str, Any], operation: str) -> dict[str, Any]:
    entities = list(world["entities"])
    relations = list(world["relations"])
    assignments = clone_assignments(world["assignments"])
    links = dict(world["links"])
    e0, e1, e2, e3 = entities
    r0, r1, r2, r3 = relations

    query_kind = "value"
    target_relation = r0
    query_text = ""
    derivation: list[str] = []

    if operation == "direct":
        gold = assignments[e0][r0]
        candidates = list(world["value_sets"][0])
        query_text = f"What is the {r0} of {e0}?"
        derivation = [e0, r0, gold]
    elif operation == "query_object":
        gold = assignments[e1][r0]
        candidates = list(world["value_sets"][0])
        query_text = f"What is the {r0} of {e1}?"
        derivation = [e1, r0, gold]
    elif operation == "query_relation":
        target_relation = r1
        gold = assignments[e0][r1]
        candidates = list(world["value_sets"][1])
        query_text = f"What is the {r1} of {e0}?"
        derivation = [e0, r1, gold]
    elif operation == "binding_swap":
        swap_binding(assignments, e0, e1, r0)
        gold = assignments[e0][r0]
        candidates = list(world["value_sets"][0])
        query_text = f"After using the current records, what is the {r0} of {e0}?"
        derivation = ["swap", e0, e1, r0, e0, gold]
    elif operation == "inverse_lookup":
        query_kind = "entity"
        target_relation = r2
        target = assignments[e2][r2]
        gold = inverse_entity(assignments, entities, r2, target)
        candidates = list(entities)
        query_text = f"Which unit has {r2} {target}?"
        derivation = [r2, target, gold]
    elif operation == "link_then_value":
        target_relation = r3
        linked = links[e0]
        gold = assignments[linked][r3]
        candidates = list(world["value_sets"][3])
        query_text = f"What is the {r3} of the unit linked from {e0}?"
        derivation = [e0, "link", linked, r3, gold]
    elif operation == "object_relation_compose":
        target_relation = r1
        gold = assignments[e1][r1]
        candidates = list(world["value_sets"][1])
        query_text = f"For {e1}, report its {r1}."
        derivation = ["object", e1, "relation", r1, gold]
    elif operation == "binding_query_compose":
        swap_binding(assignments, e0, e2, r0)
        gold = assignments[e2][r0]
        candidates = list(world["value_sets"][0])
        query_text = f"Using the updated bindings in this world, report the {r0} of {e2}."
        derivation = ["swap", e0, e2, r0, "query", e2, gold]
    elif operation == "link_relation_compose":
        target_relation = r2
        linked = links[e1]
        gold = assignments[linked][r2]
        candidates = list(world["value_sets"][2])
        query_text = f"Follow the link from {e1}; what {r2} does the reached unit have?"
        derivation = [e1, "link", linked, r2, gold]
    elif operation == "double_link_relation":
        target_relation = r3
        linked1 = links[e0]
        linked2 = links[linked1]
        gold = assignments[linked2][r3]
        candidates = list(world["value_sets"][3])
        query_text = f"Follow two links starting at {e0}; what is the reached unit's {r3}?"
        derivation = [e0, "link", linked1, "link", linked2, r3, gold]
    elif operation == "inverse_link_compose":
        target_relation = r1
        target = assignments[e2][r0]
        found = inverse_entity(assignments, entities, r0, target)
        linked = links[found]
        gold = assignments[linked][r1]
        candidates = list(world["value_sets"][1])
        query_text = (
            f"Find the unit whose {r0} is {target}, follow its link, "
            f"and report the reached unit's {r1}."
        )
        derivation = [r0, target, found, "link", linked, r1, gold]
    elif operation == "link_binding_compose":
        target_relation = r2
        linked = links[e0]
        other = e3 if linked != e3 else e1
        swap_binding(assignments, linked, other, r2)
        gold = assignments[linked][r2]
        candidates = list(world["value_sets"][2])
        query_text = f"Follow the link from {e0} and report the reached unit's current {r2}."
        derivation = ["swap", linked, other, r2, e0, "link", linked, gold]
    else:
        raise KeyError(operation)

    if len(set(candidates)) != 4 or gold not in candidates:
        raise RuntimeError(f"invalid candidate set for {operation}: {gold=} {candidates=}")
    return {
        "assignments": assignments,
        "links": links,
        "query_kind": query_kind,
        "target_relation": target_relation,
        "query_text": query_text,
        "derivation": derivation,
        "gold": gold,
        "candidates": candidates,
    }


def apply_same_answer_carrier(
    state: dict[str, Any], world: dict[str, Any], operation_index: int
) -> tuple[dict[str, dict[str, str]], str]:
    assignments = clone_assignments(state["assignments"])
    relations = list(world["relations"])
    entities = list(world["entities"])
    target_relation = str(state["target_relation"])
    nuisance_relations = [relation for relation in relations if relation != target_relation]
    nuisance = nuisance_relations[operation_index % len(nuisance_relations)]
    left = entities[(operation_index + 1) % 4]
    right = entities[(operation_index + 3) % 4]
    if left == right:
        right = entities[(operation_index + 2) % 4]
    swap_binding(assignments, left, right, nuisance)
    wrong = next(value for value in state["candidates"] if value != state["gold"])
    note = (
        f'An archived note says "{wrong}", but that note is obsolete and not part '
        "of the current world."
    )
    return assignments, note


def render_records(
    entities: list[str],
    relations: list[str],
    assignments: dict[str, dict[str, str]],
    links: dict[str, str],
    order: list[str],
    style: str,
) -> str:
    if style == "ledger":
        rows = []
        for entity in order:
            fields = "; ".join(f"{relation}={assignments[entity][relation]}" for relation in relations)
            rows.append(f"{entity} :: {fields}; link={links[entity]}")
        return "\n".join(rows)
    sentences = []
    for entity in order:
        fields = ", ".join(f"{relation} {assignments[entity][relation]}" for relation in relations)
        sentences.append(f"{entity} has {fields}, and links to {links[entity]}.")
    return " ".join(sentences)


def render_prompt(
    world: dict[str, Any],
    operation: str,
    operation_index: int,
    panel: str,
    state: dict[str, Any],
) -> tuple[str, list[str], dict[str, dict[str, str]], str | None]:
    entities = list(world["entities"])
    relations = list(world["relations"])
    assignments = clone_assignments(state["assignments"])
    order = list(entities)
    style = "prose"
    note: str | None = None

    if panel == "record_order":
        order = rotate(order, operation_index + 1)
    elif panel == "paraphrase":
        style = "ledger"
    elif panel == "same_answer_carrier":
        assignments, note = apply_same_answer_carrier(state, world, operation_index)
    elif panel != "canonical":
        raise KeyError(panel)

    candidate_order = rotate(
        list(state["candidates"]),
        int(world["world_index"]) + operation_index,
    )
    records = render_records(entities, relations, assignments, state["links"], order, style)
    if style == "ledger":
        intro = "Current-world ledger:\n"
        bridge = "\nRead the ledger literally."
    else:
        intro = "Current-world dossier: "
        bridge = " Use only these current records."
    note_text = f" {note}" if note else ""
    prompt = (
        f"{intro}{records}{bridge}{note_text} "
        f"Question: {state['query_text']} Options: {', '.join(candidate_order)}. "
        "Answer with exactly one listed option."
    )
    return prompt, candidate_order, assignments, note


def make_world(split: str, world_index: int) -> dict[str, Any]:
    track = TRACKS[world_index % 2]
    entities = choose_entities(split, track, world_index)
    relations = list(RELATIONS[track])
    values = value_sets(split, track, world_index)
    assignments = make_assignments(entities, relations, values, world_index)
    links = make_links(entities, world_index)
    world_id = f"p1220-{split}-w{world_index:03d}-{track}"
    return {
        "world_id": world_id,
        "split": split,
        "world_index": world_index,
        "track": track,
        "entities": entities,
        "relations": relations,
        "value_sets": values,
        "assignments": assignments,
        "links": links,
    }


def build_materials() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        operations = operations_for_split(split)
        for world_index in range(WORLDS_PER_SPLIT):
            world = make_world(split, world_index)
            for operation_index, operation in enumerate(operations):
                state = operation_state(world, operation)
                group_id = f"{world['world_id']}|{operation}"
                for panel in PANELS:
                    prompt, candidate_order, rendered_assignments, note = render_prompt(
                        world, operation, operation_index, panel, state
                    )
                    item_id = "p1220-" + digest(
                        [world["world_id"], operation, panel]
                    )[:20]
                    row = {
                        "schema_version": "phase1220.master_task.row.v1",
                        "phase": PHASE,
                        "item_id": item_id,
                        "group_id": group_id,
                        "world_id": world["world_id"],
                        "split": split,
                        "world_index": world_index,
                        "track": world["track"],
                        "operation": operation,
                        "operation_index": operation_index,
                        "operation_class": (
                            "primitive" if operation in PRIMITIVE_OPERATIONS else "composition"
                        ),
                        "panel": panel,
                        "generation_required": panel == GENERATION_PANEL,
                        "entities": world["entities"],
                        "relations": world["relations"],
                        "base_assignments": world["assignments"],
                        "rendered_assignments": rendered_assignments,
                        "links": state["links"],
                        "query_kind": state["query_kind"],
                        "target_relation": state["target_relation"],
                        "query_text": state["query_text"],
                        "derivation": state["derivation"],
                        "candidates": state["candidates"],
                        "candidate_order": candidate_order,
                        "gold": state["gold"],
                        "gold_position": candidate_order.index(state["gold"]),
                        "carrier_note": note,
                        "prompt": prompt,
                    }
                    row["row_digest"] = digest(row)
                    rows.append(row)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    return rows


def render_native(tokenizer: Any, prompt: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def continuation_ids(tokenizer: Any, rendered: str, candidate: str) -> tuple[list[int], list[int]]:
    base = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    extended = [
        int(value)
        for value in tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)
    ]
    if extended[: len(base)] != base:
        raise RuntimeError(f"candidate {candidate!r} retokenized the prompt")
    suffix = extended[len(base) :]
    if not suffix:
        raise RuntimeError(f"candidate {candidate!r} has empty continuation")
    return base, suffix


def build_manifest(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    manifest: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    candidate_lengths: list[int] = []
    for index, row in enumerate(rows):
        rendered = render_native(tokenizer, row["prompt"])
        input_ids: list[int] | None = None
        candidate_token_ids: dict[str, list[int]] = {}
        for candidate in row["candidates"]:
            candidate_base, suffix = continuation_ids(tokenizer, rendered, candidate)
            if input_ids is None:
                input_ids = candidate_base
            elif candidate_base != input_ids:
                raise RuntimeError("candidate tokenization changed the prompt prefix")
            candidate_token_ids[candidate] = suffix
            candidate_lengths.append(len(suffix))
        assert input_ids is not None
        prompt_lengths.append(len(input_ids))
        item = {
            "schema_version": "phase1220.qwen3.manifest.v1",
            "item_id": row["item_id"],
            "row_digest": row["row_digest"],
            "split": row["split"],
            "track": row["track"],
            "operation": row["operation"],
            "panel": row["panel"],
            "generation_required": row["generation_required"],
            "gold": row["gold"],
            "candidates": row["candidates"],
            "candidate_order": row["candidate_order"],
            "input_ids": input_ids,
            "input_token_count": len(input_ids),
            "candidate_token_ids": candidate_token_ids,
            "rendered_prompt_digest": digest(rendered),
        }
        item["manifest_row_digest"] = digest(item)
        manifest.append(item)
        if (index + 1) % 2048 == 0:
            print(f"[phase1220/tokenize] {index + 1}/{len(rows)}", flush=True)

    audit = {
        "phase": PHASE,
        "model": "qwen3",
        "row_count": len(manifest),
        "prompt_token_count_min": min(prompt_lengths),
        "prompt_token_count_max": max(prompt_lengths),
        "prompt_token_count_mean": sum(prompt_lengths) / len(prompt_lengths),
        "candidate_token_count_min": min(candidate_lengths),
        "candidate_token_count_max": max(candidate_lengths),
        "candidate_token_count_mean": sum(candidate_lengths) / len(candidate_lengths),
        "multi_token_candidate_fraction": sum(length > 1 for length in candidate_lengths) / len(candidate_lengths),
        "empty_candidate_count": sum(length == 0 for length in candidate_lengths),
        "manifest_digest": digest(manifest),
    }
    audit["tokenizer_audit_digest"] = digest(audit)
    del tokenizer
    gc.collect()
    return manifest, audit


def model_artifact_fingerprint() -> dict[str, Any]:
    root = Path(MODEL_CONFIGS["qwen3"]["path"])
    files = sorted(path for path in root.iterdir() if path.is_file())
    small = {
        path.name: file_sha256(path)
        for path in files
        if path.name in {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
        }
    }
    weights = [path for path in files if path.suffix in {".safetensors", ".bin"}]
    return {
        "model_path": str(root),
        "small_metadata_hashes": small,
        "weight_file_names_and_sizes": [[path.name, path.stat().st_size] for path in weights],
        "total_weight_bytes": sum(path.stat().st_size for path in weights),
    }


def verify_upstream() -> dict[str, Any]:
    upstream = read_json(UPSTREAM_FINAL)
    if upstream.get("final_digest") != EXPECTED_UPSTREAM_DIGEST:
        raise RuntimeError("Phase1219 final digest mismatch")
    if upstream.get("authorized_next", {}).get("automatic_execution"):
        raise RuntimeError("Phase1219 unexpectedly authorized automatic execution")
    return upstream


def build_protocol(
    rows: list[dict[str, Any]], manifest: list[dict[str, Any]], token_audit: dict[str, Any]
) -> dict[str, Any]:
    upstream = verify_upstream()
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1220.master_task.protocol.v2",
        "engineering_revision": ENGINEERING_REVISION,
        "created_at": utc_now(),
        "purpose": (
            "freeze one persistent object-relation-value behavior world before any new Qwen3 output, "
            "without hidden-state or intervention access"
        ),
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1219_final": file_sha256(UPSTREAM_FINAL),
        },
        "upstream": {
            "phase1219_final_digest": upstream["final_digest"],
            "phase1219_status": upstream["status"],
            "phase1219_k_item": upstream["k_item"],
            "explicit_user_restart_required_and_received": True,
        },
        "engineering_history": {
            "revision1": (
                "aborted before any behavior output after a frozen runtime audit showed that "
                "four full-prefix forwards per item would exceed the execution window"
            ),
            "revision2": (
                "shares one exact prompt KV cache across the four candidate continuations; "
                "materials, scores, parsers, ledgers, and thresholds are unchanged"
            ),
        },
        "abstract_world": {
            "definition": "W=(O,R,V,B,L,Q), with B:O x R -> V and L:O -> O",
            "objects_per_world": 4,
            "relations_per_world": 4,
            "values_per_relation": 4,
            "world_count": EXPECTED_WORLDS,
            "worlds_per_split": WORLDS_PER_SPLIT,
            "tracks": list(TRACKS),
            "splits": list(SPLITS),
            "split_disjoint_world_ids": True,
        },
        "operation_registry": {
            "primitive": list(PRIMITIVE_OPERATIONS),
            "composition": list(COMPOSITION_OPERATIONS),
            "sealed": list(SEALED_OPERATIONS),
            "surface_controls": list(PANELS),
            "states_per_world": ROWS_PER_WORLD,
            "same_answer_surface_groups": True,
        },
        "material": {
            "row_count": len(rows),
            "material_digest": digest(rows),
            "manifest_digest": digest(manifest),
            "tokenizer_audit_digest": token_audit["tokenizer_audit_digest"],
            "generation_case_count": sum(row["generation_required"] for row in rows),
        },
        "interface": {
            "model": "qwen3",
            "model_artifact": model_artifact_fingerprint(),
            "precision": "FP16",
            "quantization": "none",
            "cuda_required": True,
            "native_chat_template": True,
            "enable_thinking": False,
            "system_prompt": SYSTEM_PROMPT,
            "candidate_scoring": (
                "mean continuation log probability over every token; no first-token substitution"
            ),
            "candidate_scoring_implementation": (
                "one exact prompt forward per item batch followed by four teacher-forced "
                "continuation branches sharing the prompt KV cache"
            ),
            "free_generation": (
                "greedy generation on canonical panel only; semantic extraction and normalized exact option both recorded"
            ),
            "candidate_batch_size": CANDIDATE_BATCH_SIZE,
            "generation_batch_size": GENERATION_BATCH_SIZE,
            "hidden_states": False,
            "attentions": False,
            "hooks": False,
            "interventions": False,
        },
        "behavior_ledgers": {
            "L1_numerical": {"finite_rate_min": FINITE_RATE_MIN},
            "L2_candidate": {
                "overall_each_split_min": CANDIDATE_OVERALL_MIN,
                "worst_track_operation_cell_each_split_min": CANDIDATE_WORST_CELL_MIN,
            },
            "L3_surface": {"all_four_panels_correct_group_rate_each_split_min": SURFACE_GROUP_MIN},
            "L4_semantic_set": {"all_six_canonical_operations_correct_world_rate_each_split_min": SEMANTIC_SET_MIN},
            "L5_generation": {
                "semantic_each_split_min": GENERATION_OVERALL_MIN,
                "worst_track_operation_cell_each_split_min": GENERATION_WORST_CELL_MIN,
                "normalized_exact_each_split_min": GENERATION_EXACT_MIN,
            },
            "L6_heldout": {
                "composition_and_sealed_candidate_min": COMPOSITION_SEALED_CANDIDATE_MIN,
                "composition_and_sealed_generation_min": COMPOSITION_SEALED_GENERATION_MIN,
            },
        },
        "authorization": {
            "pass": (
                "authorize a separately frozen Phase1221 physical trajectory and causal-response tensor; "
                "no neuron search"
            ),
            "fail": (
                "stop the current interface at behavior; do not inspect hidden state or simplify after reveal"
            ),
        },
        "forbidden_after_freeze": [
            "change prompts, candidates, operations, panels, splits, thresholds, score normalization, or generation parser",
            "drop hard worlds or cells",
            "replace full-sequence score with first-token score",
            "inspect hidden states, attentions, heads, neurons, or interventions in Phase1220",
            "refit the sealed split",
            "claim natural-language mechanism from behavior qualification",
            "run GLM4 or DS7B before Qwen3 single-model closure",
        ],
        "claim_boundary": {
            "behavior_only": True,
            "qwen3_only": True,
            "controlled_and_naturalized_generated_worlds": True,
            "organic_corpus": False,
            "hidden_state": False,
            "causal": False,
            "cross_model": False,
            "language_mechanism": False,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    return protocol


def materialize() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"formal output already exists: {OUT_ROOT}")
    rows = build_materials()
    manifest, token_audit = build_manifest(rows)
    protocol = build_protocol(rows, manifest, token_audit)
    write_jsonl_atomic(MATERIAL_PATH, rows)
    write_jsonl_atomic(MANIFEST_PATH, manifest)
    write_json(TOKEN_AUDIT_PATH, token_audit)
    write_json(PROTOCOL_PATH, protocol)
    print(
        canonical_json(
            {
                "status": "phase1220_protocol_materialized",
                "world_count": EXPECTED_WORLDS,
                "row_count": len(rows),
                "generation_count": protocol["material"]["generation_case_count"],
                "protocol_digest": protocol["protocol_digest"],
            }
        )
    )


def verify_formal_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    if protocol.get("protocol_digest") != digest(
        {key: value for key, value in protocol.items() if key != "protocol_digest"}
    ):
        raise RuntimeError("protocol embedded digest mismatch")
    if protocol["source_hashes"]["main"] != file_sha256(SCRIPT):
        raise RuntimeError("main script changed after protocol freeze")
    if protocol["source_hashes"]["audit"] != file_sha256(AUDIT_SCRIPT):
        raise RuntimeError("audit script changed after protocol freeze")
    if protocol["material"]["material_digest"] != digest(rows):
        raise RuntimeError("material digest mismatch")
    if protocol["material"]["manifest_digest"] != digest(manifest):
        raise RuntimeError("manifest digest mismatch")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    return protocol, rows, manifest


def homogeneous_batches(entries: list[dict[str, Any]], batch_size: int, key: str) -> Iterable[list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[int(entry[key])].append(entry)
    for length in sorted(groups):
        values = groups[length]
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]


def candidate_scores(
    model: Any, device: torch.device, manifest: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    scores: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    prompt_batch_count = 0
    continuation_batch_count = 0
    all_finite = True
    started = time.time()
    for batch in homogeneous_batches(manifest, CANDIDATE_BATCH_SIZE, "input_token_count"):
        candidate_count = len(batch[0]["candidates"])
        if any(len(row["candidates"]) != candidate_count for row in batch):
            raise RuntimeError("candidate count changed within a prompt batch")
        input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        first_logits = output.logits[:, -1].float()
        first_log_norm = torch.logsumexp(first_logits, dim=-1)
        first_finite = torch.isfinite(first_logits).all(dim=-1)
        past_key_values = output.past_key_values

        branches: list[dict[str, Any]] = []
        for row_index, row in enumerate(batch):
            for candidate in row["candidates"]:
                continuation = [int(value) for value in row["candidate_token_ids"][candidate]]
                first_score = first_logits[row_index, continuation[0]] - first_log_norm[row_index]
                branches.append(
                    {
                        "row_index": row_index,
                        "item_id": row["item_id"],
                        "candidate": candidate,
                        "continuation": continuation,
                        "token_log_probs": [float(first_score.item())],
                        "finite": bool(first_finite[row_index].item()) and math.isfinite(float(first_score.item())),
                    }
                )

        for candidate_slot in range(candidate_count):
            slot_branches = [
                branches[row_index * candidate_count + candidate_slot]
                for row_index in range(len(batch))
            ]
            max_prefix = max(len(branch["continuation"]) - 1 for branch in slot_branches)
            if max_prefix == 0:
                continue
            branch_cache = copy.deepcopy(past_key_values)
            continuation_ids = torch.zeros(
                (len(slot_branches), max_prefix), dtype=torch.long, device=device
            )
            continuation_mask = torch.zeros_like(continuation_ids)
            for branch_index, branch in enumerate(slot_branches):
                prefix = branch["continuation"][:-1]
                if prefix:
                    continuation_ids[branch_index, : len(prefix)] = torch.tensor(
                        prefix, dtype=torch.long, device=device
                    )
                    continuation_mask[branch_index, : len(prefix)] = 1
            full_attention_mask = torch.cat(
                [
                    torch.ones(
                        (len(slot_branches), input_ids.shape[1]), dtype=torch.long, device=device
                    ),
                    continuation_mask,
                ],
                dim=1,
            )
            with torch.inference_mode():
                continuation_output = model(
                    input_ids=continuation_ids,
                    attention_mask=full_attention_mask,
                    past_key_values=branch_cache,
                    use_cache=False,
                    logits_to_keep=max_prefix,
                    return_dict=True,
                )
            continuation_logits = continuation_output.logits
            continuation_batch_count += 1
            for branch_index, branch in enumerate(slot_branches):
                for offset, token_id in enumerate(branch["continuation"][1:]):
                    position_logits = continuation_logits[branch_index, offset].float()
                    finite = bool(torch.isfinite(position_logits).all().item())
                    score = position_logits[token_id] - torch.logsumexp(position_logits, dim=-1)
                    branch["token_log_probs"].append(float(score.item()))
                    branch["finite"] = branch["finite"] and finite and math.isfinite(float(score.item()))
            del continuation_output, continuation_logits, continuation_ids
            del continuation_mask, full_attention_mask, branch_cache

        for branch in branches:
            token_log_probs = branch["token_log_probs"]
            finite = bool(branch["finite"])
            all_finite = all_finite and finite
            scores[branch["item_id"]][branch["candidate"]] = {
                "mean_log_probability": sum(token_log_probs) / len(token_log_probs),
                "sum_log_probability": sum(token_log_probs),
                "token_count": len(token_log_probs),
                "all_vocab_logits_finite": finite,
            }
        del output, first_logits, first_log_norm, first_finite, past_key_values
        del input_ids, attention_mask
        prompt_batch_count += 1
        if prompt_batch_count % 100 == 0:
            print(f"[phase1220/candidate] prompt_batches={prompt_batch_count}", flush=True)
    return scores, {
        "entry_count": sum(len(row["candidates"]) for row in manifest),
        "prompt_batch_count": prompt_batch_count,
        "continuation_batch_count": continuation_batch_count,
        "batch_count": prompt_batch_count + continuation_batch_count,
        "all_finite": all_finite,
        "elapsed_seconds": time.time() - started,
    }


def normalize_generated(text: str) -> str:
    value = text.strip().splitlines()[0] if text.strip() else ""
    value = value.strip().strip(string.whitespace + string.punctuation)
    value = re.sub(r"\s+", " ", value.lower())
    return value


def parse_generated(text: str, candidates: list[str]) -> tuple[str | None, bool]:
    normalized = normalize_generated(text)
    matches = []
    for candidate in candidates:
        candidate_norm = normalize_generated(candidate)
        if normalized == candidate_norm:
            return candidate, True
        if normalized.startswith(candidate_norm):
            suffix = normalized[len(candidate_norm) :]
            if not suffix or suffix[0] in " .,:;!?)]}\"'":
                matches.append(candidate)
    return (matches[0], False) if len(matches) == 1 else (None, False)


def generation_scores(
    model: Any, tokenizer: Any, device: torch.device, manifest: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    entries = [
        {
            "item_id": row["item_id"],
            "input_ids": [int(value) for value in row["input_ids"]],
            "input_token_count": int(row["input_token_count"]),
            "candidates": list(row["candidates"]),
            "max_candidate_tokens": max(len(row["candidate_token_ids"][candidate]) for candidate in row["candidates"]),
        }
        for row in manifest
        if row["generation_required"]
    ]
    results: dict[str, dict[str, Any]] = {}
    batch_count = 0
    started = time.time()
    eos = tokenizer.eos_token_id
    for batch in homogeneous_batches(entries, GENERATION_BATCH_SIZE, "input_token_count"):
        input_ids = torch.tensor([entry["input_ids"] for entry in batch], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        max_new = max(entry["max_candidate_tokens"] for entry in batch) + 3
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=max_new,
                eos_token_id=eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        suffixes = generated.sequences[:, input_ids.shape[1] :].detach().cpu().tolist()
        for index, entry in enumerate(batch):
            suffix = [int(value) for value in suffixes[index]]
            if eos is not None and eos in suffix:
                suffix = suffix[: suffix.index(eos)]
            text = tokenizer.decode(
                suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            prediction, exact = parse_generated(text, entry["candidates"])
            results[entry["item_id"]] = {
                "generated_token_ids": suffix,
                "generated_text": text,
                "generation_prediction": prediction,
                "generation_normalized_exact": exact,
            }
        del generated, input_ids, attention_mask
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"[phase1220/generation] batches={batch_count}", flush=True)
    return results, {
        "entry_count": len(entries),
        "batch_count": batch_count,
        "elapsed_seconds": time.time() - started,
    }


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Qwen3 behavior output already exists")
    protocol, rows, manifest = verify_formal_inputs()
    material_by_id = {row["item_id"]: row for row in rows}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    try:
        scores, candidate_runtime = candidate_scores(model, device, manifest)
        generations, generation_runtime = generation_scores(model, tokenizer, device, manifest)
        raw: list[dict[str, Any]] = []
        for manifest_row in manifest:
            item_id = manifest_row["item_id"]
            material = material_by_id[item_id]
            candidate = scores[item_id]
            ordered = sorted(
                candidate,
                key=lambda name: candidate[name]["mean_log_probability"],
                reverse=True,
            )
            tie = (
                len(ordered) > 1
                and abs(
                    candidate[ordered[0]]["mean_log_probability"]
                    - candidate[ordered[1]]["mean_log_probability"]
                )
                <= TIE_TOLERANCE
            )
            prediction = None if tie else ordered[0]
            generation = generations.get(item_id)
            row = {
                "schema_version": "phase1220.qwen3.behavior.row.v1",
                "phase": PHASE,
                "protocol_digest": protocol["protocol_digest"],
                "item_id": item_id,
                "row_digest": manifest_row["row_digest"],
                "world_id": material["world_id"],
                "group_id": material["group_id"],
                "split": material["split"],
                "track": material["track"],
                "operation": material["operation"],
                "operation_class": material["operation_class"],
                "panel": material["panel"],
                "generation_required": material["generation_required"],
                "gold": material["gold"],
                "candidate_scores": candidate,
                "candidate_prediction": prediction,
                "candidate_tie": tie,
                "candidate_correct": prediction == material["gold"],
                "all_candidate_scores_finite": all(
                    value["all_vocab_logits_finite"]
                    and math.isfinite(value["mean_log_probability"])
                    for value in candidate.values()
                ),
                "gold_margin": (
                    candidate[material["gold"]]["mean_log_probability"]
                    - max(
                        value["mean_log_probability"]
                        for name, value in candidate.items()
                        if name != material["gold"]
                    )
                ),
                "generation_prediction": generation["generation_prediction"] if generation else None,
                "generation_correct": (
                    generation["generation_prediction"] == material["gold"] if generation else None
                ),
                "generation_normalized_exact": (
                    generation["generation_normalized_exact"] if generation else None
                ),
                "generated_token_ids": generation["generated_token_ids"] if generation else None,
                "generated_text": generation["generated_text"] if generation else None,
            }
            row["behavior_row_digest"] = digest(row)
            raw.append(row)
    finally:
        release_fp16(model)
        del model, tokenizer
        gc.collect()

    write_jsonl_atomic(RAW_PATH, raw)
    summary = {
        "phase": PHASE,
        "schema_version": "phase1220.qwen3.run_summary.v1",
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
    print(canonical_json({"status": "qwen3_behavior_complete", "case_count": len(raw), "summary_digest": summary["summary_digest"]}))


def rate(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return float("nan")
    return sum(bool(row[field]) for row in rows) / len(rows)


def grouped_rates(
    rows: list[dict[str, Any]], fields: tuple[str, ...], metric: str
) -> dict[str, float]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    return {
        "|".join(key): rate(values, metric)
        for key, values in sorted(groups.items())
    }


def summarize_behavior(raw: list[dict[str, Any]]) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    all_gates: dict[str, bool] = {}
    for split in SPLITS:
        selected = [row for row in raw if row["split"] == split]
        generated = [row for row in selected if row["generation_required"]]
        candidate_cells = grouped_rates(selected, ("track", "operation"), "candidate_correct")
        generation_cells = grouped_rates(generated, ("track", "operation"), "generation_correct")

        surface_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        semantic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            surface_groups[row["group_id"]].append(row)
            if row["panel"] == GENERATION_PANEL:
                semantic_groups[row["world_id"]].append(row)
        surface_success = [
            len(values) == len(PANELS)
            and all(row["candidate_correct"] for row in values)
            and len({row["candidate_prediction"] for row in values}) == 1
            for values in surface_groups.values()
        ]
        semantic_success = [
            len(values) == OPERATIONS_PER_WORLD
            and all(row["candidate_correct"] for row in values)
            for values in semantic_groups.values()
        ]

        metrics = {
            "case_count": len(selected),
            "generation_case_count": len(generated),
            "finite_rate": rate(selected, "all_candidate_scores_finite"),
            "candidate_accuracy": rate(selected, "candidate_correct"),
            "candidate_worst_track_operation_cell": min(candidate_cells.values()),
            "candidate_cells": candidate_cells,
            "surface_group_rate": sum(surface_success) / len(surface_success),
            "semantic_set_rate": sum(semantic_success) / len(semantic_success),
            "generation_semantic_accuracy": rate(generated, "generation_correct"),
            "generation_normalized_exact_rate": rate(generated, "generation_normalized_exact"),
            "generation_worst_track_operation_cell": min(generation_cells.values()),
            "generation_cells": generation_cells,
            "candidate_by_panel": grouped_rates(selected, ("panel",), "candidate_correct"),
            "candidate_by_track": grouped_rates(selected, ("track",), "candidate_correct"),
            "generation_by_track": grouped_rates(generated, ("track",), "generation_correct"),
            "mean_gold_margin": sum(float(row["gold_margin"]) for row in selected) / len(selected),
        }
        gates = {
            "finite": metrics["finite_rate"] >= FINITE_RATE_MIN,
            "candidate_overall": metrics["candidate_accuracy"] >= CANDIDATE_OVERALL_MIN,
            "candidate_worst_cell": metrics["candidate_worst_track_operation_cell"] >= CANDIDATE_WORST_CELL_MIN,
            "surface": metrics["surface_group_rate"] >= SURFACE_GROUP_MIN,
            "semantic_set": metrics["semantic_set_rate"] >= SEMANTIC_SET_MIN,
            "generation_overall": metrics["generation_semantic_accuracy"] >= GENERATION_OVERALL_MIN,
            "generation_worst_cell": metrics["generation_worst_track_operation_cell"] >= GENERATION_WORST_CELL_MIN,
            "generation_exact": metrics["generation_normalized_exact_rate"] >= GENERATION_EXACT_MIN,
        }
        if split in {"composition", "sealed"}:
            gates["heldout_candidate"] = metrics["candidate_accuracy"] >= COMPOSITION_SEALED_CANDIDATE_MIN
            gates["heldout_generation"] = metrics["generation_semantic_accuracy"] >= COMPOSITION_SEALED_GENERATION_MIN
        split_summary[split] = {"metrics": metrics, "gates": gates, "passed": all(gates.values())}
        all_gates[f"split_{split}"] = all(gates.values())
    return {
        "splits": split_summary,
        "all_split_gates": all_gates,
        "passed": all(all_gates.values()),
    }


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("final result already exists")
    protocol, rows, manifest = verify_formal_inputs()
    raw = read_jsonl(RAW_PATH)
    run_summary = read_json(RUN_SUMMARY_PATH)
    if len(raw) != len(rows) or run_summary["raw_digest"] != digest(raw):
        raise RuntimeError("raw behavior completeness or digest mismatch")
    summary = summarize_behavior(raw)
    passed = bool(summary["passed"])
    k_item = {
        "identifier": "K197",
        "evidence_grade": "E3-BEHAVIOR" if passed else "E3-NEGATIVE-BOUNDARY",
        "statement": (
            "Qwen3 satisfied the frozen object-relation-value master-task behavior, surface, composition, sealed, "
            "and multi-token generation ledgers."
            if passed
            else "Qwen3 did not satisfy every frozen object-relation-value master-task behavior ledger; hidden-state access is denied under this interface."
        ),
        "scope": "Qwen3 FP16; 512 generated worlds; behavior only",
    }
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "qwen3_master_task_behavior_qualified" if passed else "qwen3_master_task_behavior_gate_failed",
        "protocol_digest": protocol["protocol_digest"],
        "material_digest": protocol["material"]["material_digest"],
        "manifest_digest": protocol["material"]["manifest_digest"],
        "run_summary_digest": run_summary["summary_digest"],
        "behavior": summary,
        "k_item": k_item,
        "evidence_scope": protocol["claim_boundary"],
        "authorized_next": {
            "automatic_execution": passed,
            "experiment": "Phase1221 physical trajectory and causal-response tensor preregistration" if passed else None,
            "hidden_state_scan": passed,
            "head_or_neuron_search": False,
            "cross_model_run": False,
            "reason": (
                "all frozen behavior ledgers passed"
                if passed
                else "at least one frozen behavior ledger failed"
            ),
        },
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    rows = build_materials()
    assert len(rows) == EXPECTED_ROWS
    assert len({row["item_id"] for row in rows}) == EXPECTED_ROWS
    assert len({row["world_id"] for row in rows}) == EXPECTED_WORLDS
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        assert len(selected) == WORLDS_PER_SPLIT * ROWS_PER_WORLD
        assert sum(row["generation_required"] for row in selected) == WORLDS_PER_SPLIT * OPERATIONS_PER_WORLD
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["group_id"]].append(row)
    for values in grouped_rows.values():
        assert len(values) == len(PANELS)
        assert len({row["gold"] for row in values}) == 1
    example = rows[0]
    assert example["gold"] in example["candidates"]
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
