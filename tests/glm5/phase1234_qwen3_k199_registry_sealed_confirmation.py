#!/usr/bin/env python3
"""Phase1234: deterministic K199 registry selection and sealed confirmation.

The historical K199 behavior registry is used only to select one previously
authorized atomic scope.  Before any Phase1234 model output, this script freezes
three disjoint confirmation splits, a depth-2 conditional shortcut grammar,
query quartets, same-bag binding rotations, surface invariances, exact Qwen3
tokenization, batches, thresholds, and claim permissions.  The phase is
behavior-only: no hidden states, attentions, or interventions are collected.
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
import re
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

import phase1220_object_relation_value_master_task as p1220
import phase1221_typed_operation_behavior_and_error_fingerprints as p1221
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1234
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1234_qwen3_k199_registry_sealed_confirmation_audit.py"
P1220_SCRIPT = TEST_ROOT / "phase1220_object_relation_value_master_task.py"
P1221_SCRIPT = TEST_ROOT / "phase1221_typed_operation_behavior_and_error_fingerprints.py"

K199_ROOT = TEST_ROOT / "result/phase1222_atomic_operation_independent_confirmation"
K199_FINAL_PATH = K199_ROOT / "analysis/final.json"
K199_FP16_AUDIT_PATH = K199_ROOT / "audit/fp16_schema_resolution.json"
EXPECTED_K199_FINAL = "a6be67cce38afa78aef432c8d01b1c8007cd40039dc4cc66c190a360753a65e2"
EXPECTED_K199_FP16_AUDIT = "0599a6d47b67add164bbbc3937fefa4e57f46919b1d98767c24f06ba498ce4a0"

UPSTREAM_ROOT = TEST_ROOT / "result/phase1233_qwen3_program_identifiable_medal_binding"
UPSTREAM_FINAL_PATH = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_AUDIT_PATH = UPSTREAM_ROOT / "audit/independent_final_audit.json"
EXPECTED_UPSTREAM_FINAL = "692faa033e949733dbd1900f89d26fdc140f386e8891309a3746b502437321c2"
EXPECTED_UPSTREAM_AUDIT = "2e61a16d092dade8a65c1aa9cd414f643f96ad2030b8073d0648794f4735354b"

OUT_ROOT = TEST_ROOT / "result/phase1234_qwen3_k199_registry_sealed_confirmation"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/sealed_query_object_worlds.jsonl"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
PROGRAM_AUDIT_PATH = OUT_ROOT / "protocol/depth2_program_grammar_audit.json"
BATCH_PLAN_PATH = OUT_ROOT / "protocol/frozen_batch_plan.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RAW_PATH = OUT_ROOT / "behavior/qwen3/raw_behavior.jsonl"
RUN_SUMMARY_PATH = OUT_ROOT / "behavior/qwen3/run_summary.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

MODEL_PATH = ROOT / "models/hf/qwen3-4b"
SPLITS = ("sealed_alpha", "sealed_beta", "sealed_gamma")
PANELS = ("canonical", "record_order", "paraphrase", "binding_rotation")
WORLDS_PER_SPLIT = 48
QUERY_COUNT = 4
ROWS_PER_WORLD = len(PANELS) * QUERY_COUNT
EXPECTED_WORLDS = len(SPLITS) * WORLDS_PER_SPLIT
EXPECTED_ROWS = EXPECTED_WORLDS * ROWS_PER_WORLD
VALUE_COUNT = 5
ASSIGNED_VALUE_COUNT = 4
TIE_TOLERANCE = 1e-7
PROGRAM_DEPTH = 2
PROGRAM_CEILING_MAX = 0.80
MAX_INPUT_LENGTH = 320

SYSTEM_PROMPT = (
    "Use only the supplied registry records. Return exactly the requested marker label "
    "and no explanation."
)

THRESHOLDS = {
    "Q0_finite_rate": 1.0,
    "Q1_candidate_accuracy": 0.95,
    "Q1_context_adjusted_accuracy": 0.90,
    "Q1_open_generation_accuracy": 0.90,
    "Q2_query_quartet_success": 0.80,
    "Q3_binding_rotation_pair_success": 0.85,
    "Q4_order_pair_success": 0.90,
    "Q4_paraphrase_pair_success": 0.90,
    "Q5_worst_panel_candidate": 0.90,
    "Q5_reliable_world_rate": 0.80,
    "Q6_sum_mean_argmax_agreement": 1.0,
    "Q7_program_ceiling": PROGRAM_CEILING_MAX,
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

TEMPLATES = {
    "sealed_alpha": {
        "canonical": "Profile for {object}: marker = {marker}; zone = {zone}; texture = {texture}; status = {status}.",
        "paraphrase": "The profile assigns {object} marker {marker}, zone {zone}, texture {texture}, and status {status}.",
        "query": "According to the profiles, which marker is assigned to {object}?",
    },
    "sealed_beta": {
        "canonical": "Registry entry for {object} lists marker {marker}, zone {zone}, texture {texture}, status {status}.",
        "paraphrase": "For {object}, the registry gives {marker} as marker, {zone} as zone, {texture} as texture, and {status} as status.",
        "query": "Read the marker belonging to {object} from the registry.",
    },
    "sealed_gamma": {
        "canonical": "{object}'s dossier reports marker {marker}; zone {zone}; texture {texture}; status {status}.",
        "paraphrase": "In the dossier, {object} has the marker {marker}, the zone {zone}, the texture {texture}, and the status {status}.",
        "query": "What exact marker does the dossier give for {object}?",
    },
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


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


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


def lexical_multiset(text: str) -> list[str]:
    return sorted(re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower()))


def verify_upstreams() -> tuple[dict[str, Any], dict[str, Any]]:
    upstream = read_json(UPSTREAM_FINAL_PATH)
    upstream_audit = read_json(UPSTREAM_AUDIT_PATH)
    k199 = read_json(K199_FINAL_PATH)
    k199_fp16 = read_json(K199_FP16_AUDIT_PATH)
    if upstream.get("final_digest") != EXPECTED_UPSTREAM_FINAL:
        raise RuntimeError("Phase1233 final digest mismatch")
    if upstream_audit.get("audit_digest") != EXPECTED_UPSTREAM_AUDIT or not upstream_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1233 final audit mismatch")
    if upstream.get("authorization", {}).get("hidden_scan_in_this_phase") is not False:
        raise RuntimeError("Phase1233 evidence boundary drift")
    if k199.get("final_digest") != EXPECTED_K199_FINAL:
        raise RuntimeError("K199 final digest mismatch")
    if k199_fp16.get("audit_digest") != EXPECTED_K199_FP16_AUDIT or not k199_fp16.get("all_checks_passed"):
        raise RuntimeError("K199 FP16 audit mismatch")
    return upstream, k199


def historical_registry_selection(k199: dict[str, Any]) -> dict[str, Any]:
    scopes = list(k199["behavior"]["authorized_target_operation_tracks"])
    expected = {
        "direct|natural",
        "inverse_lookup|symbolic",
        "query_object|natural",
        "query_relation|natural",
        "query_relation|symbolic",
    }
    if set(scopes) != expected:
        raise RuntimeError("K199 authorized scope registry drift")
    registry: dict[str, Any] = {}
    for scope in sorted(scopes):
        splits = k199["behavior"]["operation_track_results"][scope]["splits"]
        minima = {
            metric: min(float(cell["metrics"][metric]) for cell in splits.values())
            for metric in (
                "candidate_accuracy",
                "open_generation_accuracy",
                "context_adjusted_accuracy",
                "worst_panel_candidate",
                "all_panel_world_rate",
            )
        }
        score = min(minima.values())
        registry[scope] = {"minimum_metrics": minima, "selection_score": score}
    ranking = sorted(registry, key=lambda scope: (-registry[scope]["selection_score"], scope))
    selected = ranking[0]
    if selected != "query_object|natural":
        raise RuntimeError(f"unexpected historical selection: {selected}")
    value = {
        "selection_source": "K199 Phase1222 frozen behavior registry; historical discovery evidence only",
        "selection_rule": "maximize the minimum across candidate, open-generation, context-adjusted, worst-panel, and all-panel-world metrics; lexical scope tie-break",
        "registry": registry,
        "ranking": ranking,
        "selected_scope": selected,
        "selection_is_new_evidence": False,
    }
    value["selection_digest"] = digest(value)
    return value


def continuation_length(tokenizer: Any, value: str) -> int:
    return len(tokenizer.encode(value, add_special_tokens=False))


def grouped_by_token_length(tokenizer: Any, values: Iterable[str], group_size: int) -> list[list[str]]:
    buckets: dict[int, list[str]] = defaultdict(list)
    for value in values:
        buckets[continuation_length(tokenizer, value)].append(value)
    groups: list[list[str]] = []
    for length in sorted(buckets):
        bucket = sorted(set(buckets[length]))
        for start in range(0, len(bucket) - group_size + 1, group_size):
            groups.append(bucket[start : start + group_size])
    return groups


def prior_vocabulary() -> set[str]:
    old = read_jsonl(K199_ROOT / "material/atomic_worlds.jsonl")
    terms: set[str] = set()
    for row in old:
        terms.update(str(value) for value in row.get("objects", []))
        for values in row.get("values", {}).values():
            terms.update(str(value) for value in values)
    terms.update({"gold", "silver", "bronze", "first", "second", "third"})
    return terms


def build_lexicon(tokenizer: Any) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    forbidden = prior_vocabulary()
    objects = [f"{first} {surname}" for first in FIRST_NAMES for surname in SURNAMES]
    objects = [value for value in objects if value not in forbidden]
    labels = [f"{left} {right}" for left in LABEL_LEFT for right in LABEL_RIGHT]
    labels = [value for value in labels if value not in forbidden]
    label_groups = grouped_by_token_length(tokenizer, labels, VALUE_COUNT)
    required_objects = EXPECTED_WORLDS * ASSIGNED_VALUE_COUNT
    required_groups = EXPECTED_WORLDS
    if len(objects) < required_objects or len(label_groups) < required_groups:
        raise RuntimeError(
            f"insufficient lexicon: objects={len(objects)}/{required_objects}, groups={len(label_groups)}/{required_groups}"
        )
    audit = {
        "natural_object_count": len(objects),
        "natural_value_group_count": len(label_groups),
        "required_object_count": required_objects,
        "required_value_group_count": required_groups,
        "old_exact_term_overlap_count": len((set(objects) | {item for group in label_groups for item in group}) & forbidden),
        "all_value_groups_have_equal_direct_token_length": all(
            len({continuation_length(tokenizer, value) for value in group}) == 1 for group in label_groups
        ),
    }
    return objects[:required_objects], label_groups[:required_groups], audit


def add_span(spans: dict[str, list[list[int]]], role: str, start: int, end: int) -> None:
    spans.setdefault(role, []).append([start, end])


def render_records(
    split: str,
    objects: list[str],
    displayed: dict[str, str],
    record_order: list[int],
    panel: str,
    distractors: dict[str, dict[str, str]],
) -> tuple[str, dict[str, list[list[int]]]]:
    style = "paraphrase" if panel == "paraphrase" else "canonical"
    template = TEMPLATES[split][style]
    pieces: list[str] = []
    spans: dict[str, list[list[int]]] = {}
    cursor = 0
    for position, object_index in enumerate(record_order):
        if position:
            pieces.append(" ")
            cursor += 1
        obj = objects[object_index]
        marker = displayed[obj]
        record = template.format(object=obj, marker=marker, **distractors[obj])
        start = cursor
        pieces.append(record)
        cursor += len(record)
        add_span(spans, "record_full", start, cursor)
        object_start = start + record.index(obj)
        marker_start = start + record.index(marker)
        relation_start = start + record.lower().index("marker")
        add_span(spans, "record_object", object_start, object_start + len(obj))
        add_span(spans, "record_relation", relation_start, relation_start + len("marker"))
        add_span(spans, "record_value", marker_start, marker_start + len(marker))
    return "".join(pieces), spans


def candidate_order(candidates: list[str], gold: str, split_index: int, world_index: int, panel_index: int, query_index: int) -> list[str]:
    gold_position = (split_index * WORLDS_PER_SPLIT * ROWS_PER_WORLD + world_index * ROWS_PER_WORLD + panel_index * QUERY_COUNT + query_index) % len(candidates)
    others = [value for value in candidates if value != gold]
    rng = random.Random(1234000 + split_index * 100000 + world_index * 101 + panel_index * 17 + query_index)
    rng.shuffle(others)
    ordered = list(others)
    ordered.insert(gold_position, gold)
    return ordered


def render_row(
    split: str,
    local_world: int,
    objects: list[str],
    marker_values: list[str],
    panel: str,
    query_index: int,
) -> dict[str, Any]:
    split_index = SPLITS.index(split)
    panel_index = PANELS.index(panel)
    world_global = split_index * WORLDS_PER_SPLIT + local_world
    shift = (world_global * 3 + 1) % VALUE_COUNT
    assigned_values = [marker_values[(index + shift) % VALUE_COUNT] for index in range(ASSIGNED_VALUE_COUNT)]
    base = {obj: assigned_values[index] for index, obj in enumerate(objects)}
    displayed = dict(base)
    if panel == "binding_rotation":
        displayed = {obj: base[objects[(index + 1) % len(objects)]] for index, obj in enumerate(objects)}
    record_order = list(range(len(objects)))
    if panel == "record_order":
        record_order = [2, 0, 3, 1]
    distractors = {
        obj: {
            "zone": ZONE_WORDS[(world_global + index) % len(ZONE_WORDS)],
            "texture": TEXTURE_WORDS[(world_global * 3 + index) % len(TEXTURE_WORDS)],
            "status": STATUS_WORDS[(world_global * 5 + index) % len(STATUS_WORDS)],
        }
        for index, obj in enumerate(objects)
    }
    records, spans = render_records(split, objects, displayed, record_order, panel, distractors)
    query_object = objects[query_index]
    query = TEMPLATES[split]["query"].format(object=query_object)
    gold = displayed[query_object]
    ordered_candidates = candidate_order(marker_values, gold, split_index, local_world, panel_index, query_index)
    candidate_tail = f"\nCHOICES: {' / '.join(ordered_candidates)}\nAnswer:"
    open_tail = "\nReturn the exact marker label.\nAnswer:"
    query_start = len(records) + 1
    prompt_prefix = records + "\n" + query
    query_object_start = query_start + query.index(query_object)
    query_relation_start = query_start + query.lower().index("marker")
    add_span(spans, "query_full", query_start, query_start + len(query))
    add_span(spans, "query_object", query_object_start, query_object_start + len(query_object))
    add_span(spans, "query_relation", query_relation_start, query_relation_start + len("marker"))
    candidate_prompt = prompt_prefix + candidate_tail
    open_prompt = prompt_prefix + open_tail
    null_prompt = (
        "No registry records are supplied.\n"
        + query
        + candidate_tail
    )
    candidate_answer_start = len(candidate_prompt) - len("Answer:")
    add_span(spans, "answer_boundary", candidate_answer_start, len(candidate_prompt))
    identity = {
        "split": split,
        "world": local_world,
        "panel": panel,
        "query_index": query_index,
    }
    item_id = f"p1234-{digest(identity)[:24]}"
    world_id = f"p1234-{split}-w{local_world:03d}"
    query_group_id = f"query-{digest({'split': split, 'world': local_world, 'panel': panel})[:20]}"
    pair_base = {"split": split, "world": local_world, "query_index": query_index}
    row: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1234.query_object.row.v1",
        "item_id": item_id,
        "split": split,
        "world_id": world_id,
        "world_index": world_global,
        "local_world_index": local_world,
        "panel": panel,
        "panel_index": panel_index,
        "query_index": query_index,
        "query_object": query_object,
        "objects": objects,
        "marker_values": marker_values,
        "base_assignments": base,
        "display_assignments": displayed,
        "record_order_indices": record_order,
        "distractors": distractors,
        "gold": gold,
        "unused_value": next(value for value in marker_values if value not in set(base.values())),
        "candidates": marker_values,
        "candidate_order": ordered_candidates,
        "gold_position": ordered_candidates.index(gold),
        "candidate_prompt": candidate_prompt,
        "null_prompt": null_prompt,
        "open_prompt": open_prompt,
        "role_char_spans": spans,
        "query_group_id": query_group_id,
        "order_pair_id": f"order-{digest(pair_base)[:20]}" if panel in ("canonical", "record_order") else None,
        "paraphrase_pair_id": f"paraphrase-{digest(pair_base)[:20]}" if panel in ("canonical", "paraphrase") else None,
        "binding_pair_id": f"binding-{digest(pair_base)[:20]}" if panel in ("canonical", "binding_rotation") else None,
        "prompt_lexical_multiset_digest": digest(lexical_multiset(candidate_prompt)),
        "target_record_position": record_order.index(query_index),
    }
    row["row_digest"] = digest(row)
    return row


def generate_material(tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_terms, value_groups, lexicon_audit = build_lexicon(tokenizer)
    rows: list[dict[str, Any]] = []
    object_cursor = 0
    value_cursor = 0
    for split in SPLITS:
        for local_world in range(WORLDS_PER_SPLIT):
            objects = object_terms[object_cursor : object_cursor + ASSIGNED_VALUE_COUNT]
            marker_values = value_groups[value_cursor]
            object_cursor += ASSIGNED_VALUE_COUNT
            value_cursor += 1
            for panel in PANELS:
                for query_index in range(QUERY_COUNT):
                    rows.append(render_row(split, local_world, objects, marker_values, panel, query_index))
    if len(rows) != EXPECTED_ROWS or len({row["item_id"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("Phase1234 material cardinality failure")
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


def candidate_suffixes(tokenizer: Any, rendered: str, candidates: list[str]) -> tuple[list[int], dict[str, list[int]]]:
    input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    suffixes: dict[str, list[int]] = {}
    for candidate in candidates:
        appended = [int(value) for value in tokenizer.encode(rendered + candidate, add_special_tokens=False)]
        if appended[: len(input_ids)] != input_ids:
            raise RuntimeError("candidate continuation does not preserve frozen prefix")
        suffix = appended[len(input_ids) :]
        if not suffix:
            raise RuntimeError("empty candidate suffix")
        suffixes[candidate] = suffix
    return input_ids, suffixes


def token_span_for_chars(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    indices = [index for index, (left, right) in enumerate(offsets) if right > left and right > start and left < end]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise RuntimeError("invalid role token span")
    return [indices[0], indices[-1] + 1]


def build_manifest(rows: list[dict[str, Any]], slow: Any, fast: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    slow_fast_mismatch = 0
    suffix_length_mismatch = 0
    span_failures = 0
    max_length = 0
    for execution_index, row in enumerate(rows):
        prefix_data: dict[str, Any] = {}
        for prefix, prompt in (
            ("world", row["candidate_prompt"]),
            ("null", row["null_prompt"]),
            ("open", row["open_prompt"]),
        ):
            rendered = render_native(slow, prompt)
            input_ids, suffixes = candidate_suffixes(slow, rendered, row["candidates"])
            fast_ids = [int(value) for value in fast.encode(rendered, add_special_tokens=False)]
            slow_fast_mismatch += input_ids != fast_ids
            suffix_length_mismatch += len({len(value) for value in suffixes.values()}) != 1
            max_length = max(max_length, len(input_ids))
            prefix_data[f"{prefix}_input_ids"] = input_ids
            prefix_data[f"{prefix}_input_token_count"] = len(input_ids)
            prefix_data[f"{prefix}_candidate_token_ids"] = suffixes
            prefix_data[f"{prefix}_rendered_prompt_digest"] = digest(rendered)
            if prefix == "world":
                encoded = fast(rendered, add_special_tokens=False, return_offsets_mapping=True)
                offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
                prompt_start = rendered.find(prompt)
                try:
                    roles = {
                        role: [
                            token_span_for_chars(offsets, prompt_start + int(start), prompt_start + int(end))
                            for start, end in spans
                        ]
                        for role, spans in row["role_char_spans"].items()
                    }
                except RuntimeError:
                    span_failures += 1
                    roles = {}
                prefix_data["world_role_token_spans"] = roles
        case: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1234.qwen3.manifest.v1",
            "execution_index": execution_index,
            "item_id": row["item_id"],
            "material_row_digest": row["row_digest"],
            "split": row["split"],
            "world_id": row["world_id"],
            "panel": row["panel"],
            "query_index": row["query_index"],
            "gold": row["gold"],
            "candidates": row["candidates"],
            "candidate_order": row["candidate_order"],
            "query_group_id": row["query_group_id"],
            "order_pair_id": row["order_pair_id"],
            "paraphrase_pair_id": row["paraphrase_pair_id"],
            "binding_pair_id": row["binding_pair_id"],
            **prefix_data,
        }
        case["manifest_row_digest"] = digest(case)
        manifest.append(case)
    summary = {
        "row_count": len(manifest),
        "manifest_digest": digest(manifest),
        "slow_tokenizer_class": type(slow).__name__,
        "fast_tokenizer_class": type(fast).__name__,
        "slow_fast_mismatch_count": slow_fast_mismatch,
        "candidate_suffix_length_mismatch_count": suffix_length_mismatch,
        "role_span_failure_count": span_failures,
        "maximum_input_length": max_length,
        "model_weights_loaded": False,
    }
    summary["tokenizer_gate"] = bool(
        len(manifest) == EXPECTED_ROWS
        and slow_fast_mismatch == 0
        and suffix_length_mismatch == 0
        and span_failures == 0
        and max_length <= MAX_INPUT_LENGTH
    )
    return manifest, summary


def base_program_predictions(row: dict[str, Any]) -> dict[str, str]:
    candidates = row["candidates"]
    displayed = row["display_assignments"]
    objects = row["objects"]
    query_index = int(row["query_index"])
    programs: dict[str, str] = {
        f"candidate_position_{index}": row["candidate_order"][index]
        for index in range(len(candidates))
    }
    programs.update({f"fixed_object_{index}": displayed[objects[index]] for index in range(len(objects))})
    programs.update(
        {
            "first_record": displayed[objects[row["record_order_indices"][0]]],
            "last_record": displayed[objects[row["record_order_indices"][-1]]],
            "next_object": displayed[objects[(query_index + 1) % len(objects)]],
            "opposite_object": displayed[objects[(query_index + 2) % len(objects)]],
            "previous_object": displayed[objects[(query_index + 3) % len(objects)]],
            "unused_value": row["unused_value"],
            "lexical_first": sorted(candidates)[0],
            "lexical_last": sorted(candidates)[-1],
        }
    )
    return programs


def nuisance_condition_features(row: dict[str, Any]) -> dict[str, Any]:
    """Registered branch conditions that do not encode the target query identity.

    Query identity and target-record position are deliberately excluded: routing
    on either and then reading the matching object record is extensionally the
    intended query-object operation, not a competing shortcut.  That
    target-equivalent program is audited separately below.
    """
    return {
        "panel": row["panel"],
        "record_first_index": row["record_order_indices"][0],
        "unused_candidate_position": row["candidate_order"].index(row["unused_value"]),
    }


def optimal_depth_tree_accuracy(rows: list[dict[str, Any]], depth: int) -> tuple[float, dict[str, Any]]:
    program_names = sorted(base_program_predictions(rows[0]))
    predictions = [base_program_predictions(row) for row in rows]
    features = [nuisance_condition_features(row) for row in rows]
    conditions = sorted(
        {
            (name, canonical_json(value))
            for feature in features
            for name, value in feature.items()
        }
    )
    decoded = {(name, value_json): json.loads(value_json) for name, value_json in conditions}
    memo: dict[tuple[tuple[int, ...], int], tuple[int, dict[str, Any]]] = {}

    def solve(indices: tuple[int, ...], remaining: int) -> tuple[int, dict[str, Any]]:
        key = (indices, remaining)
        if key in memo:
            return memo[key]
        leaf_scores = {
            program: sum(predictions[index][program] == rows[index]["gold"] for index in indices)
            for program in program_names
        }
        best_program = min(
            (program for program in program_names if leaf_scores[program] == max(leaf_scores.values())),
            key=str,
        )
        best = leaf_scores[best_program]
        tree: dict[str, Any] = {"leaf_program": best_program, "correct": best, "n": len(indices)}
        if remaining > 0 and len(indices) > 1:
            for name, value_json in conditions:
                value = decoded[(name, value_json)]
                left = tuple(index for index in indices if features[index][name] == value)
                right = tuple(index for index in indices if features[index][name] != value)
                if not left or not right:
                    continue
                left_score, left_tree = solve(left, remaining - 1)
                right_score, right_tree = solve(right, remaining - 1)
                score = left_score + right_score
                candidate_tree = {
                    "condition": {"feature": name, "equals": value},
                    "true": left_tree,
                    "false": right_tree,
                    "correct": score,
                    "n": len(indices),
                }
                if score > best or (score == best and canonical_json(candidate_tree) < canonical_json(tree)):
                    best, tree = score, candidate_tree
        memo[key] = (best, tree)
        return memo[key]

    correct, tree = solve(tuple(range(len(rows))), depth)
    return correct / len(rows), tree


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            result[str(row[key])].append(row)
    return result


def build_program_audit(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_by_id = {row["item_id"]: row for row in manifest}
    split_results: dict[str, Any] = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        base_accuracies = {
            name: sum(base_program_predictions(row)[name] == row["gold"] for row in selected) / len(selected)
            for name in sorted(base_program_predictions(selected[0]))
        }
        depth_accuracy, tree = optimal_depth_tree_accuracy(selected, PROGRAM_DEPTH)
        split_results[split] = {
            "base_program_accuracies": base_accuracies,
            "maximum_base_program_accuracy": max(base_accuracies.values()),
            "depth2_conditional_program_accuracy": depth_accuracy,
            "depth2_witness_tree": tree,
            "construct_gate": depth_accuracy <= PROGRAM_CEILING_MAX,
        }
    query_groups = grouped(rows, "query_group_id")
    order_groups = grouped(rows, "order_pair_id")
    paraphrase_groups = grouped(rows, "paraphrase_pair_id")
    binding_groups = grouped(rows, "binding_pair_id")
    collision_checks = {
        "query_quartets_complete": all(
            len(cell) == 4 and len({row["gold"] for row in cell}) == 4 for cell in query_groups.values()
        ),
        "order_pairs_invariant": all(len(cell) == 2 and len({row["gold"] for row in cell}) == 1 for cell in order_groups.values()),
        "paraphrase_pairs_invariant": all(len(cell) == 2 and len({row["gold"] for row in cell}) == 1 for cell in paraphrase_groups.values()),
        "binding_pairs_discriminating": all(
            len(cell) == 2
            and len({row["gold"] for row in cell}) == 2
            and len({row["prompt_lexical_multiset_digest"] for row in cell}) == 1
            and len({digest(sorted(manifest_by_id[row["item_id"]]["world_input_ids"])) for row in cell}) == 1
            for cell in binding_groups.values()
        ),
        "unused_value_never_gold": all(row["unused_value"] != row["gold"] for row in rows),
        "five_candidates_four_assigned": all(
            len(row["candidates"]) == 5 and len(set(row["base_assignments"].values())) == 4 for row in rows
        ),
    }
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1234.depth2_program_grammar.v1",
        "created_at_utc": utc_now(),
        "threat_model": {
            "base_programs": sorted(base_program_predictions(rows[0])),
            "nuisance_branch_features": sorted(nuisance_condition_features(rows[0])),
            "maximum_depth": PROGRAM_DEPTH,
            "ceiling_threshold": PROGRAM_CEILING_MAX,
            "world_identity_and_lexical_identity_forbidden": True,
            "query_identity_branching_excluded_as_target_equivalent": True,
        },
        "target_equivalent_witness": {
            "description": "select the queried object and read its displayed marker",
            "accuracy": sum(
                row["display_assignments"][row["query_object"]] == row["gold"] for row in rows
            )
            / len(rows),
        },
        "split_results": split_results,
        "collision_group_counts": {
            "query_quartets": len(query_groups),
            "order_pairs": len(order_groups),
            "paraphrase_pairs": len(paraphrase_groups),
            "binding_pairs": len(binding_groups),
        },
        "collision_checks": collision_checks,
        "claim_boundary": (
            "The gate distinguishes the intended query-object behavior from the frozen depth-2 grammar only. "
            "It does not uniquely identify the model's neural algorithm."
        ),
    }
    value["program_construct_gate"] = (
        all(result["construct_gate"] for result in split_results.values())
        and all(collision_checks.values())
        and value["target_equivalent_witness"]["accuracy"] == 1.0
    )
    value["program_audit_digest"] = digest(value)
    return value


def build_batch_plan(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    views: dict[str, Any] = {}
    for prefix in ("world", "null", "open"):
        counts = Counter(int(row[f"{prefix}_input_token_count"]) for row in manifest)
        views[prefix] = {
            "length_bucket_count": len(counts),
            "length_counts": dict(sorted(counts.items())),
            "case_count": len(manifest),
        }
    value = {
        "phase": PHASE,
        "schema_version": "phase1234.batch_plan.v1",
        "candidate_scoring_batch_contract": "phase1220 homogeneous shared-prefix cache with phase1221 low-margin direct-prefill fallback",
        "generation_batch_contract": "phase1220 deterministic homogeneous greedy generation",
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
        raise RuntimeError("Phase1234 output directory already exists")
    upstream, k199 = verify_upstreams()
    selection = historical_registry_selection(k199)
    from transformers import AutoTokenizer, __version__ as transformers_version

    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    rows, lexicon_audit = generate_material(slow)
    manifest, tokenizer_summary = build_manifest(rows, slow, fast)
    program_audit = build_program_audit(rows, manifest)
    if not tokenizer_summary["tokenizer_gate"] or not program_audit["program_construct_gate"]:
        raise RuntimeError("Phase1234 zero-model construct qualification failed")
    plan = build_batch_plan(manifest)
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1234.k199_registry_sealed_confirmation.v1",
        "created_at_utc": utc_now(),
        "objective": (
            "Use the frozen K199 registry to select one behavior-authorized atomic scope, then independently confirm it on three new non-bijective query-object materials under a depth-2 shortcut grammar."
        ),
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1233_final_digest": upstream["final_digest"],
            "phase1233_final_audit_digest": EXPECTED_UPSTREAM_AUDIT,
            "k199_final_digest": k199["final_digest"],
            "k199_fp16_audit_digest": EXPECTED_K199_FP16_AUDIT,
            "phase1233_family_not_patched": True,
        },
        "historical_registry_selection": selection,
        "material": {
            "selected_scope": selection["selected_scope"],
            "splits": list(SPLITS),
            "worlds_per_split": WORLDS_PER_SPLIT,
            "independent_cluster_unit": "world",
            "panels": list(PANELS),
            "queries_per_world_panel": QUERY_COUNT,
            "candidate_count": VALUE_COUNT,
            "assigned_value_count": ASSIGNED_VALUE_COUNT,
            "unused_value_count": 1,
            "row_count": len(rows),
            "material_digest": digest(rows),
            "lexicon_audit": lexicon_audit,
        },
        "interface": {
            "model": "qwen3",
            "device": "cuda",
            "precision": "float16",
            "quantization": "none",
            "native_chat_template": True,
            "enable_thinking": False,
            "candidate_score": "complete continuation sum and mean log probability",
            "context_adjustment": "world sum score minus matched no-record null sum score",
            "open_generation": "greedy exact candidate string plus separately recorded normalized-prefix match",
            "tokenizer_summary": tokenizer_summary,
            "transformers_version": transformers_version,
        },
        "program_construct": {
            "program_audit_digest": program_audit["program_audit_digest"],
            "maximum_depth": PROGRAM_DEPTH,
            "maximum_ceiling": PROGRAM_CEILING_MAX,
            "gate": program_audit["program_construct_gate"],
        },
        "thresholds": THRESHOLDS,
        "behavior_gate": "Q0 and Q1 and Q2 and Q3 and Q4 and Q5 and Q6 and Q7",
        "execution": {
            "batch_plan_digest": plan["plan_digest"],
            "hidden_states": False,
            "attentions": False,
            "interventions": False,
            "cross_model": False,
        },
        "forbidden": [
            "change selection, material, prompts, candidates, grammar, thresholds, or denominators after preregistration",
            "inspect hidden states or attentions",
            "perform interventions",
            "run GLM4 or DS7B as a rescue",
            "patch the failed medal family",
            "claim unique internal algorithm identification from this behavior gate",
        ],
        "claim_boundary": [
            "K199 is historical discovery evidence, not a new untouched registry run.",
            "The three Phase1234 splits are disjoint generated materials from one generator, not three natural domains.",
            "Program identifiability is relative to the frozen depth-2 grammar.",
            "A behavior pass authorizes a separate future-response phase but is not itself a neural mechanism result.",
        ],
    }
    contract["contract_digest"] = digest(contract)
    write_jsonl(MATERIAL_PATH, rows)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(PROGRAM_AUDIT_PATH, program_audit)
    write_json(BATCH_PLAN_PATH, plan)
    write_json(CONTRACT_PATH, contract)
    print(
        canonical_json(
            {
                "status": "phase1234_preregistered",
                "selected_scope": selection["selected_scope"],
                "rows": len(rows),
                "program_gate": program_audit["program_construct_gate"],
                "contract_digest": contract["contract_digest"],
            }
        )
    )


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    verify_upstreams()
    contract = read_json(CONTRACT_PATH)
    rows = read_jsonl(MATERIAL_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    program = read_json(PROGRAM_AUDIT_PATH)
    plan = read_json(BATCH_PLAN_PATH)
    if contract["contract_digest"] != digest(strip_digest(contract, "contract_digest")):
        raise RuntimeError("contract drift")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    if contract["material"]["material_digest"] != digest(rows):
        raise RuntimeError("material drift")
    if contract["interface"]["tokenizer_summary"]["manifest_digest"] != digest(manifest):
        raise RuntimeError("manifest drift")
    if program["program_audit_digest"] != digest(strip_digest(program, "program_audit_digest")):
        raise RuntimeError("program audit drift")
    if plan["plan_digest"] != digest(strip_digest(plan, "plan_digest")):
        raise RuntimeError("batch plan drift")
    if not read_json(PREAUDIT_PATH).get("all_checks_passed"):
        raise RuntimeError("independent preaudit failed")
    return contract, rows, manifest, program, plan


def manifest_view(manifest: list[dict[str, Any]], prefix: str, generation: bool) -> list[dict[str, Any]]:
    return [
        {
            "item_id": row["item_id"],
            "input_ids": row[f"{prefix}_input_ids"],
            "input_token_count": row[f"{prefix}_input_token_count"],
            "candidates": row["candidates"],
            "candidate_order": row["candidate_order"],
            "candidate_token_ids": row[f"{prefix}_candidate_token_ids"],
            "generation_required": generation,
        }
        for row in manifest
    ]


def argmax_set(scores: dict[str, float]) -> list[str]:
    maximum = max(scores.values())
    return sorted(candidate for candidate, value in scores.items() if maximum - value <= TIE_TOLERANCE)


def run_qwen3() -> None:
    if RAW_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1234 behavior outputs already exist")
    contract, rows, manifest, _program, plan = verify_frozen()
    material_by_id = {row["item_id"]: row for row in rows}
    started = time.time()
    model, tokenizer, device, placement = load_fp16("qwen3")
    precision = quantization_audit(model)
    if device.type != "cuda" or precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
        release_fp16(model)
        raise RuntimeError("Phase1234 numerical contract failed")
    try:
        world_scores, world_runtime = p1221.score_candidates_with_fallback(
            model, device, manifest_view(manifest, "world", generation=False)
        )
        null_scores, null_runtime = p1221.score_candidates_with_fallback(
            model, device, manifest_view(manifest, "null", generation=False)
        )
        open_generations, generation_runtime = p1220.generation_scores(
            model, tokenizer, device, manifest_view(manifest, "open", generation=True)
        )
        raw: list[dict[str, Any]] = []
        for manifest_row in manifest:
            item_id = manifest_row["item_id"]
            material = material_by_id[item_id]
            scores = world_scores[item_id]
            priors = null_scores[item_id]
            sum_map = {candidate: float(value["sum_log_probability"]) for candidate, value in scores.items()}
            mean_map = {candidate: float(value["mean_log_probability"]) for candidate, value in scores.items()}
            null_map = {candidate: float(value["sum_log_probability"]) for candidate, value in priors.items()}
            context_map = {candidate: sum_map[candidate] - null_map[candidate] for candidate in sum_map}
            sum_set = argmax_set(sum_map)
            mean_set = argmax_set(mean_map)
            context_set = argmax_set(context_map)
            sum_prediction = sum_set[0] if len(sum_set) == 1 else None
            context_prediction = context_set[0] if len(context_set) == 1 else None
            ordered = sorted(sum_map.values(), reverse=True)
            generated = open_generations[item_id]
            row: dict[str, Any] = {
                "phase": PHASE,
                "schema_version": "phase1234.qwen3.behavior.row.v1",
                "contract_digest": contract["contract_digest"],
                "item_id": item_id,
                "manifest_row_digest": manifest_row["manifest_row_digest"],
                "execution_index": manifest_row["execution_index"],
                "split": material["split"],
                "world_id": material["world_id"],
                "panel": material["panel"],
                "query_index": material["query_index"],
                "gold": material["gold"],
                "query_group_id": material["query_group_id"],
                "order_pair_id": material["order_pair_id"],
                "paraphrase_pair_id": material["paraphrase_pair_id"],
                "binding_pair_id": material["binding_pair_id"],
                "candidate_scores": scores,
                "null_candidate_scores": priors,
                "context_adjusted_scores": context_map,
                "sum_argmax_set": sum_set,
                "mean_argmax_set": mean_set,
                "context_argmax_set": context_set,
                "sum_prediction": sum_prediction,
                "context_prediction": context_prediction,
                "candidate_correct": sum_prediction == material["gold"],
                "context_correct": context_prediction == material["gold"],
                "sum_mean_argmax_set_agreement": sum_set == mean_set,
                "sum_margin": ordered[0] - ordered[1],
                "all_candidate_scores_finite": all(value["all_vocab_logits_finite"] for value in scores.values()),
                "all_null_scores_finite": all(value["all_vocab_logits_finite"] for value in priors.values()),
                "open_generation_prediction": generated["generation_prediction"],
                "open_generation_correct": generated["generation_prediction"] == material["gold"],
                "open_generation_exact": generated["generation_normalized_exact"],
                "open_generated_token_ids": generated["generated_token_ids"],
                "open_generated_text": generated["generated_text"],
            }
            row["behavior_row_digest"] = digest(row)
            raw.append(row)
        raw.sort(key=lambda row: row["execution_index"])
        write_jsonl(RAW_PATH, raw)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1234.qwen3.run_summary.v1",
            "created_at_utc": utc_now(),
            "model": "qwen3",
            "contract_digest": contract["contract_digest"],
            "case_count": len(raw),
            "raw_digest": digest(raw),
            "batch_plan_digest": plan["plan_digest"],
            "world_scoring_runtime": world_runtime,
            "null_scoring_runtime": null_runtime,
            "open_generation_runtime": generation_runtime,
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
        write_json(RUN_SUMMARY_PATH, summary)
        print(canonical_json({"status": "phase1234_behavior_complete", "rows": len(raw), "summary_digest": summary["summary_digest"]}))
    finally:
        release_fp16(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else float("nan")


def group_success_rate(rows: list[dict[str, Any]], key: str, expected_size: int, mode: str) -> float:
    outcomes: list[bool] = []
    for cell in grouped(rows, key).values():
        success = len(cell) == expected_size and all(row["candidate_correct"] for row in cell)
        predictions = [row["sum_prediction"] for row in cell]
        if mode == "distinct":
            success = success and len(set(predictions)) == expected_size
        elif mode == "invariant":
            success = success and len(set(predictions)) == 1
        elif mode == "different":
            success = success and len(set(predictions)) == 2
        else:
            raise ValueError(mode)
        outcomes.append(success)
    return sum(outcomes) / len(outcomes) if outcomes else float("nan")


def adjudicate(raw: list[dict[str, Any]], program: dict[str, Any]) -> dict[str, Any]:
    split_ledgers: dict[str, Any] = {}
    split_passes: list[bool] = []
    for split in SPLITS:
        selected = [row for row in raw if row["split"] == split]
        by_panel = {
            panel: rate([row for row in selected if row["panel"] == panel], "candidate_correct")
            for panel in PANELS
        }
        world_groups = grouped(selected, "world_id")
        reliable_world_rate = sum(
            sum(row["candidate_correct"] for row in cell) / len(cell) >= 0.875
            for cell in world_groups.values()
        ) / len(world_groups)
        metrics = {
            "finite_rate": sum(
                row["all_candidate_scores_finite"] and row["all_null_scores_finite"] for row in selected
            ) / len(selected),
            "candidate_accuracy": rate(selected, "candidate_correct"),
            "context_adjusted_accuracy": rate(selected, "context_correct"),
            "open_generation_accuracy": rate(selected, "open_generation_correct"),
            "open_generation_exact_rate": rate(selected, "open_generation_exact"),
            "sum_mean_argmax_set_agreement": rate(selected, "sum_mean_argmax_set_agreement"),
            "candidate_by_panel": by_panel,
            "worst_panel_candidate": min(by_panel.values()),
            "query_quartet_success": group_success_rate(selected, "query_group_id", 4, "distinct"),
            "binding_rotation_pair_success": group_success_rate(selected, "binding_pair_id", 2, "different"),
            "order_pair_success": group_success_rate(selected, "order_pair_id", 2, "invariant"),
            "paraphrase_pair_success": group_success_rate(selected, "paraphrase_pair_id", 2, "invariant"),
            "reliable_world_rate": reliable_world_rate,
            "depth2_program_ceiling": program["split_results"][split]["depth2_conditional_program_accuracy"],
        }
        gates = {
            "Q0_finite": metrics["finite_rate"] >= THRESHOLDS["Q0_finite_rate"],
            "Q1_candidate": metrics["candidate_accuracy"] >= THRESHOLDS["Q1_candidate_accuracy"],
            "Q1_context": metrics["context_adjusted_accuracy"] >= THRESHOLDS["Q1_context_adjusted_accuracy"],
            "Q1_open_generation": metrics["open_generation_accuracy"] >= THRESHOLDS["Q1_open_generation_accuracy"],
            "Q2_query_quartet": metrics["query_quartet_success"] >= THRESHOLDS["Q2_query_quartet_success"],
            "Q3_binding_rotation": metrics["binding_rotation_pair_success"] >= THRESHOLDS["Q3_binding_rotation_pair_success"],
            "Q4_order": metrics["order_pair_success"] >= THRESHOLDS["Q4_order_pair_success"],
            "Q4_paraphrase": metrics["paraphrase_pair_success"] >= THRESHOLDS["Q4_paraphrase_pair_success"],
            "Q5_worst_panel": metrics["worst_panel_candidate"] >= THRESHOLDS["Q5_worst_panel_candidate"],
            "Q5_reliable_world": metrics["reliable_world_rate"] >= THRESHOLDS["Q5_reliable_world_rate"],
            "Q6_sum_mean": metrics["sum_mean_argmax_set_agreement"] >= THRESHOLDS["Q6_sum_mean_argmax_agreement"],
            "Q7_program_ceiling": metrics["depth2_program_ceiling"] <= THRESHOLDS["Q7_program_ceiling"],
        }
        passed = all(gates.values())
        split_ledgers[split] = {"metrics": metrics, "gates": gates, "passed": passed}
        split_passes.append(passed)
    behavior_gate = all(split_passes)
    return {
        "split_ledgers": split_ledgers,
        "program_construct_gate": bool(program["program_construct_gate"]),
        "behavior_gate": behavior_gate,
        "future_response_eligibility": bool(program["program_construct_gate"] and behavior_gate),
        "overall_candidate_accuracy": rate(raw, "candidate_correct"),
        "overall_context_adjusted_accuracy": rate(raw, "context_correct"),
        "overall_open_generation_accuracy": rate(raw, "open_generation_correct"),
        "tie_count": sum(len(row["sum_argmax_set"]) != 1 for row in raw),
        "nonfinite_count": sum(
            not (row["all_candidate_scores_finite"] and row["all_null_scores_finite"]) for row in raw
        ),
    }


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1234 final already exists")
    contract, rows, manifest, program, plan = verify_frozen()
    raw = read_jsonl(RAW_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    result_audit = read_json(RESULT_AUDIT_PATH)
    if not result_audit.get("all_checks_passed"):
        raise RuntimeError("independent result audit failed")
    if len(raw) != len(rows) or summary["raw_digest"] != digest(raw):
        raise RuntimeError("raw behavior mismatch")
    if {row["item_id"] for row in raw} != {row["item_id"] for row in manifest}:
        raise RuntimeError("behavior coverage mismatch")
    ledgers = adjudicate(raw, program)
    passed = bool(ledgers["future_response_eligibility"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1234.k199_registry_sealed_confirmation.final.v1",
        "created_at_utc": utc_now(),
        "status": "sealed_atomic_object_confirmed" if passed else "sealed_atomic_object_confirmation_failed",
        "contract_digest": contract["contract_digest"],
        "material_digest": digest(rows),
        "manifest_digest": digest(manifest),
        "program_audit_digest": program["program_audit_digest"],
        "batch_plan_digest": plan["plan_digest"],
        "run_summary_digest": summary["summary_digest"],
        "raw_digest": summary["raw_digest"],
        "result_audit_digest": result_audit["audit_digest"],
        "historical_registry_selection": contract["historical_registry_selection"],
        "ledgers": ledgers,
        "k_item": {
            "identifier": "K209",
            "evidence_grade": "E3-BEHAVIOR-CONSTRUCT" if passed else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                "K199-selected query_object|natural passed three new sealed materials under the frozen depth-2 program grammar."
                if passed
                else "K199-selected query_object|natural did not pass all three new sealed materials under the frozen depth-2 program grammar."
            ),
            "scope": "Qwen3-4B; CUDA FP16; generated English registry records; query-object marker selection; behavior only",
        },
        "authorization": {
            "selected_behavior_object": passed,
            "future_response_phase": passed,
            "next_experiment": (
                "Phase1235 typed future-response tensor and donor interchange on the exact frozen Phase1234 object"
                if passed
                else None
            ),
            "auto_continue": passed,
            "hidden_scan_in_this_phase": False,
            "cross_model_run": False,
            "unique_neural_algorithm_claim": False,
        },
        "claim_boundary": contract["claim_boundary"],
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    _upstream, k199 = verify_upstreams()
    selection = historical_registry_selection(k199)
    from transformers import AutoTokenizer

    slow = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False)
    fast = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
    rows, lexicon_audit = generate_material(slow)
    manifest, tokenizer_summary = build_manifest(rows, slow, fast)
    program = build_program_audit(rows, manifest)
    print(
        canonical_json(
            {
                "status": "selftest_passed",
                "selected_scope": selection["selected_scope"],
                "rows": len(rows),
                "lexicon": lexicon_audit,
                "tokenizer_gate": tokenizer_summary["tokenizer_gate"],
                "program_gate": program["program_construct_gate"],
                "program_ceilings": {
                    split: value["depth2_conditional_program_accuracy"]
                    for split, value in program["split_results"].items()
                },
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
