#!/usr/bin/env python3
"""Qwen3-only object-attribute vertical-closure hidden specificity gate.

This is a new, model-specific research question.  It does not reopen the
Phase1204 cross-model registry.  The primary selection object is restricted to
the whole residual stream at the actual generation boundary.  Other roles and
attention/MLP outputs are descriptive and cannot select a causal target.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402
import phase1203_object_attribute_behavior_protocol as phase1203  # noqa: E402


PHASE = 1205
MODEL = "qwen3"
MODEL_PATH = ROOT / "models/hf/qwen3-4b"
SOURCE1202 = ROOT / "tests/glm5/result/phase1202_object_attribute_mother_contract"
SOURCE1203 = ROOT / "tests/glm5/result/phase1203_object_attribute_behavior_protocol"
SOURCE1204 = ROOT / "tests/glm5/result/phase1204_object_attribute_behavior_execution"
SOURCE_ROWS = SOURCE1202 / "material/object_attribute_binding.jsonl"
SOURCE_MANIFEST = SOURCE1203 / "protocol/model_manifests/qwen3.jsonl"
SOURCE_BEHAVIOR = SOURCE1204 / "behavior/qwen3/raw_scores.jsonl"
SOURCE_FINAL = SOURCE1204 / "analysis/final.json"
SOURCE_AUDIT = SOURCE1204 / "audit/independent_result_audit.json"

OUT_ROOT = ROOT / "tests/glm5/result/phase1205_qwen3_object_attribute_vertical_closure"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PAIR_MANIFEST_PATH = OUT_ROOT / "protocol/pair_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/preexecution_audit.json"
ARRAY_PATH = OUT_ROOT / "runs/hidden_response_arrays.npz"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
VERDICT_PATH = OUT_ROOT / "analysis/hidden_specificity_verdict.json"
TRAJECTORY_PATH = OUT_ROOT / "analysis/role_component_trajectories.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

AUDIT_SCRIPT = TEST_ROOT / "phase1205_qwen3_object_attribute_vertical_closure_audit.py"
RUNNER_SCRIPT = TEST_ROOT / "phase1205_run_sequential.py"

EXPECTED_PHASE1204_FINAL_DIGEST = "5f35f53486123e4aa04806fec0a2ccf3633486a127ae72e9b6afb2c1a72c81dd"
EXPECTED_PHASE1204_AUDIT_DIGEST = "a0c87e4426ce3d56cd7af6405d776cf4c9113bb836b07fc8e3696a0ea165bbf8"
EXPECTED_QWEN_MANIFEST_DIGEST = "892b6a5b8904090d849f4b4cd85e8307b7f7a555d727d0d273db17741430b590"

PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SPLITS = ("discovery", "confirmation", "unseen_composition")
ROLES = (
    "record_entity0",
    "record_value0",
    "record_entity1",
    "record_value1",
    "record_anchor_value",
    "query_attribute",
    "query_value",
    "answer_prefix",
    "generation_boundary",
)
PREQUERY_ROLES = (
    "record_entity0",
    "record_value0",
    "record_entity1",
    "record_value1",
    "record_anchor_value",
)
COMPONENTS = ("residual", "attention_output", "mlp_output")
LAYER_COUNT = 36
HIDDEN_SIZE = 2560
PROJECTION_DIM = 64
PROJECTION_SEED = 12050017
BATCH_PAIRS = 8
EPSILON = 1e-8

THRESHOLDS = {
    "finite_fraction": 1.0,
    "minimum_active_relative_distance": 0.001,
    "active_to_max_control_median_ratio": 1.25,
    "active_over_all_controls_fraction": 0.75,
    "minimum_contiguous_discovery_depths": 2,
    "prequery_active_null_max_abs_difference": 1e-4,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def validate_embedded_digest(value: dict[str, Any], key: str) -> None:
    candidate = {name: item for name, item in value.items() if name != key}
    if digest(candidate) != value.get(key):
        raise RuntimeError(f"embedded digest mismatch: {key}")


def source_hashes() -> dict[str, str]:
    return {
        "main": sha256_file(Path(__file__).resolve()),
        "audit": sha256_file(AUDIT_SCRIPT),
        "runner": sha256_file(RUNNER_SCRIPT),
    }


def event_registry() -> list[dict[str, Any]]:
    events = [{"event_id": "residual_d00", "component": "residual", "depth": 0}]
    for depth in range(1, LAYER_COUNT + 1):
        events.append({"event_id": f"residual_d{depth:02d}", "component": "residual", "depth": depth})
    for component in ("attention_output", "mlp_output"):
        for depth in range(1, LAYER_COUNT + 1):
            events.append({
                "event_id": f"{component}_d{depth:02d}",
                "component": component,
                "depth": depth,
            })
    return events


def token_positions_for_span(
    offsets: list[tuple[int, int]], start: int, end: int
) -> list[int]:
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end and right > left
    ]
    if not positions:
        raise RuntimeError(f"no token overlaps character span {start}:{end}")
    return positions


def find_unique(container: str, needle: str, start: int = 0) -> int:
    position = container.find(needle, start)
    if position < 0:
        raise RuntimeError(f"substring not found: {needle!r}")
    if container.find(needle, position + 1) >= 0:
        raise RuntimeError(f"substring is not unique in scoped text: {needle!r}")
    return position


def role_positions(
    tokenizer: Any,
    source_row: dict[str, Any],
    manifest_row: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    rendered = phase1203.render_native(tokenizer, MODEL, str(source_row["prompt"]))
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    if input_ids != [int(value) for value in manifest_row["input_ids"]]:
        raise RuntimeError(f"native rendering drift for {manifest_row['item_id']}")

    prompt = str(source_row["prompt"])
    prompt_start = find_unique(rendered, prompt)
    entities = [str(value) for value in source_row["entities"]]
    record_order = [str(value) for value in source_row["record_order"]]
    records = [str(value) for value in source_row["rendered_records"]]
    records_by_entity = dict(zip(record_order, records))
    assignments = source_row["assignments"]
    attribute = str(source_row["attribute"])

    spans: dict[str, tuple[int, int, str]] = {}
    cursor = prompt_start
    record_bounds: dict[str, tuple[int, int]] = {}
    for entity in record_order:
        record = records_by_entity[entity]
        record_start = rendered.find(record, cursor)
        if record_start < 0:
            raise RuntimeError(f"record not found for {entity}")
        record_bounds[entity] = (record_start, record_start + len(record))
        cursor = record_start + len(record)

    for entity_index, role_entity, role_value in (
        (0, "record_entity0", "record_value0"),
        (1, "record_entity1", "record_value1"),
    ):
        entity = entities[entity_index]
        record = records_by_entity[entity]
        record_start, _ = record_bounds[entity]
        local_entity = record.find(entity)
        if local_entity < 0:
            raise RuntimeError(f"entity not found in record: {entity}")
        spans[role_entity] = (
            record_start + local_entity,
            record_start + local_entity + len(entity),
            entity,
        )
        value = str(assignments[entity][attribute])
        phrase = f"{attribute} {value}"
        local_phrase = record.find(phrase)
        if local_phrase < 0:
            raise RuntimeError(f"binding phrase not found: {phrase}")
        value_start = record_start + local_phrase + len(attribute) + 1
        spans[role_value] = (value_start, value_start + len(value), value)

    anchor = entities[2]
    anchor_record = records_by_entity[anchor]
    anchor_start, _ = record_bounds[anchor]
    anchor_value = str(assignments[anchor][attribute])
    anchor_phrase = f"{attribute} {anchor_value}"
    anchor_local = anchor_record.find(anchor_phrase)
    if anchor_local < 0:
        raise RuntimeError(f"anchor phrase not found: {anchor_phrase}")
    anchor_value_start = anchor_start + anchor_local + len(attribute) + 1
    spans["record_anchor_value"] = (
        anchor_value_start,
        anchor_value_start + len(anchor_value),
        anchor_value,
    )

    query = str(source_row["query"])
    query_local = prompt.find(query)
    if query_local < 0:
        raise RuntimeError("query not found in prompt")
    query_start = prompt_start + query_local
    attribute_local = query.find(attribute)
    if attribute_local < 0:
        raise RuntimeError("attribute not found in query")
    spans["query_attribute"] = (
        query_start + attribute_local,
        query_start + attribute_local + len(attribute),
        attribute,
    )
    target_value = str(source_row["target_value"])
    value_local = query.find(target_value, attribute_local + len(attribute))
    if value_local < 0:
        raise RuntimeError("target value not found in query")
    spans["query_value"] = (
        query_start + value_local,
        query_start + value_local + len(target_value),
        target_value,
    )
    answer_local = prompt.rfind(str(source_row["answer_prefix"]))
    if answer_local < 0:
        raise RuntimeError("answer prefix not found")
    answer_start = prompt_start + answer_local
    spans["answer_prefix"] = (
        answer_start,
        answer_start + len(str(source_row["answer_prefix"])),
        str(source_row["answer_prefix"]),
    )

    positions: dict[str, int] = {}
    span_audit: dict[str, Any] = {}
    for role, (left, right, text) in spans.items():
        token_span = token_positions_for_span(offsets, left, right)
        positions[role] = token_span[-1]
        span_audit[role] = {
            "text": text,
            "character_span": [left, right],
            "token_span": token_span,
            "selected_position": token_span[-1],
            "selected_token_id": input_ids[token_span[-1]],
        }
    positions["generation_boundary"] = len(input_ids) - 1
    span_audit["generation_boundary"] = {
        "text": tokenizer.decode([input_ids[-1]]),
        "character_span": list(offsets[-1]),
        "token_span": [len(input_ids) - 1],
        "selected_position": len(input_ids) - 1,
        "selected_token_id": input_ids[-1],
    }
    if tuple(positions) != ROLES:
        raise RuntimeError(f"role ordering drift: {tuple(positions)}")
    if not all(0 <= value < len(input_ids) for value in positions.values()):
        raise RuntimeError("role position out of range")
    return positions, span_audit


def build_pair_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_jsonl(SOURCE_MANIFEST)
    source_rows = read_jsonl(SOURCE_ROWS)
    behavior = read_jsonl(SOURCE_BEHAVIOR)
    manifest_index = {str(row["item_id"]): row for row in manifest}
    source_index = {str(row["item_id"]): row for row in source_rows}
    behavior_index = {str(row["item_id"]): row for row in behavior}
    if set(manifest_index) != set(source_index) or set(manifest_index) != set(behavior_index):
        raise RuntimeError("Qwen source item sets do not match")
    if not all(bool(row["correct"]) for row in behavior):
        raise RuntimeError("Qwen behavior qualification drifted")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    materialized: dict[str, dict[str, Any]] = {}
    span_length_counts: dict[str, list[int]] = defaultdict(list)
    for item_id, manifest_row in manifest_index.items():
        positions, span_audit = role_positions(
            tokenizer,
            source_index[item_id],
            manifest_row,
        )
        materialized[item_id] = {
            "positions": positions,
            "span_audit": span_audit,
        }
        for role in ROLES:
            span_length_counts[role].append(len(span_audit[role]["token_span"]))

    groups: dict[str, dict[str, dict[int, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    group_meta: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        group_id = (
            f"{row['combination_id']}|{row['template']}|order{row['candidate_order']}"
        )
        groups[group_id][str(row["panel"])][int(row["binding_state"])] = str(row["item_id"])
        group_meta[group_id] = {
            "group_id": group_id,
            "combination_id": str(row["combination_id"]),
            "split": str(row["split"]),
            "world": str(row["world"]),
            "profile_index": int(row["profile_index"]),
            "attribute": str(row["attribute"]),
            "template": str(row["template"]),
            "candidate_order": int(row["candidate_order"]),
            "entities": list(row["entities"]),
        }

    pair_rows: list[dict[str, Any]] = []
    excluded_groups: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        panel_map = groups[group_id]
        complete = (
            set(panel_map) == set(PANELS)
            and all(set(panel_map[panel]) == {0, 1} for panel in PANELS)
        )
        if not complete:
            raise RuntimeError(f"incomplete quartet group: {group_id}")
        mismatched = []
        for panel in PANELS:
            item0 = panel_map[panel][0]
            item1 = panel_map[panel][1]
            length0 = int(manifest_index[item0]["input_length"])
            length1 = int(manifest_index[item1]["input_length"])
            if length0 != length1:
                mismatched.append({
                    "panel": panel,
                    "state0_length": length0,
                    "state1_length": length1,
                })
        if mismatched:
            excluded_groups.append({"group_id": group_id, "mismatches": mismatched})
            continue
        for panel in PANELS:
            item0 = panel_map[panel][0]
            item1 = panel_map[panel][1]
            source0 = source_index[item0]
            source1 = source_index[item1]
            entities = list(source0["entities"])
            if panel == "active":
                if source0["gold_candidate"] == entities[0] and source1["gold_candidate"] == entities[1]:
                    orientation = 1
                elif source0["gold_candidate"] == entities[1] and source1["gold_candidate"] == entities[0]:
                    orientation = -1
                else:
                    raise RuntimeError(f"active orientation drift: {group_id}")
            else:
                orientation = 0
            pair_rows.append({
                **group_meta[group_id],
                "panel": panel,
                "pair_id": f"{group_id}|{panel}",
                "state0_item_id": item0,
                "state1_item_id": item1,
                "input_length": int(manifest_index[item0]["input_length"]),
                "state0_positions": materialized[item0]["positions"],
                "state1_positions": materialized[item1]["positions"],
                "state0_input_ids_digest": str(manifest_index[item0]["input_ids_digest"]),
                "state1_input_ids_digest": str(manifest_index[item1]["input_ids_digest"]),
                "state0_gold": str(source0["gold_candidate"]),
                "state1_gold": str(source1["gold_candidate"]),
                "active_entity0_to_entity1_orientation": orientation,
            })
    pair_rows.sort(key=lambda row: (int(row["input_length"]), str(row["pair_id"])))
    for index, row in enumerate(pair_rows):
        row["pair_index"] = index

    eligible_groups = sorted({str(row["group_id"]) for row in pair_rows})
    split_group_counts = {
        split: len({str(row["group_id"]) for row in pair_rows if row["split"] == split})
        for split in SPLITS
    }
    audit = {
        "source_case_count": len(manifest),
        "source_group_count": len(groups),
        "eligible_group_count": len(eligible_groups),
        "excluded_group_count": len(excluded_groups),
        "excluded_groups": excluded_groups,
        "eligible_pair_count": len(pair_rows),
        "split_group_counts": split_group_counts,
        "split_pair_counts": {
            split: sum(row["split"] == split for row in pair_rows)
            for split in SPLITS
        },
        "panels_per_group": len(PANELS),
        "all_qwen_behavior_correct": all(bool(row["correct"]) for row in behavior),
        "role_span_token_lengths": {
            role: {
                "minimum": min(values),
                "maximum": max(values),
            }
            for role, values in span_length_counts.items()
        },
        "selection_uses_hidden_output": False,
        "exclusion_rule": "exclude an entire quartet iff any panel has unequal state input lengths",
    }
    return pair_rows, audit


def projection_matrix() -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(PROJECTION_SEED)
    values = rng.integers(
        0,
        2,
        size=(PROJECTION_DIM, HIDDEN_SIZE),
        dtype=np.int8,
    )
    matrix = (values.astype(np.float32) * 2.0 - 1.0) / math.sqrt(PROJECTION_DIM)
    return matrix, {
        "seed": PROJECTION_SEED,
        "shape": list(matrix.shape),
        "matrix_digest": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    }


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists() or PROTOCOL_PATH.exists():
        raise RuntimeError("refusing to rewrite Phase1205 after protocol or hidden output exists")
    final1204 = read_json(SOURCE_FINAL)
    audit1204 = read_json(SOURCE_AUDIT)
    validate_embedded_digest(final1204, "final_digest")
    validate_embedded_digest(audit1204, "audit_digest")
    checks = {
        "phase1204_final_digest": final1204["final_digest"] == EXPECTED_PHASE1204_FINAL_DIGEST,
        "phase1204_audit_digest": audit1204["audit_digest"] == EXPECTED_PHASE1204_AUDIT_DIGEST,
        "phase1204_audit_passed": bool(audit1204["gate_pass"]),
        "qwen3_only_passed": final1204["passing_models"] == ["qwen3"],
        "cross_model_gate_failed": final1204["cross_model_behavior_pass"] is False,
        "cross_model_hidden_stays_denied": final1204["authorized_next"]["cross_model_hidden_claim"] is False,
        "no_prior_phase1205_output": not (OUT_ROOT / "runs").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1205 upstream checks failed: {checks}")

    pairs, material_audit = build_pair_manifest()
    matrix, projection = projection_matrix()
    del matrix
    events = event_registry()
    write_jsonl(PAIR_MANIFEST_PATH, pairs)

    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1205.qwen3_object_attribute_vertical_closure.v1",
        "created_at": utc_now(),
        "objective": (
            "Test a new Qwen3-specific question: whether the frozen object-attribute task contains an "
            "active-specific, control-exceeding, split-repeating residual transition band at the actual "
            "generation boundary. This does not reopen or weaken K184."
        ),
        "scope": {
            "model": MODEL,
            "model_specific_only": True,
            "cross_model_claim": False,
            "natural_use_claim": False,
            "causal_claim_in_this_phase": False,
            "phase1204_registry_reopened": False,
            "explicit_user_requested_new_question": True,
        },
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1204_final_digest": final1204["final_digest"],
            "phase1204_audit_digest": audit1204["audit_digest"],
            "qwen_manifest_digest": EXPECTED_QWEN_MANIFEST_DIGEST,
            "files": {
                "phase1202_rows": sha256_file(SOURCE_ROWS),
                "phase1203_qwen_manifest": sha256_file(SOURCE_MANIFEST),
                "phase1204_qwen_behavior": sha256_file(SOURCE_BEHAVIOR),
                "phase1204_final": sha256_file(SOURCE_FINAL),
                "phase1204_audit": sha256_file(SOURCE_AUDIT),
            },
        },
        "model": {
            "path": str(MODEL_PATH.resolve()),
            "precision": "FP16",
            "quantization": "none",
            "placement": "full_cuda",
            "expected_layer_count": LAYER_COUNT,
            "expected_hidden_size": HIDDEN_SIZE,
            "batch_pairs": BATCH_PAIRS,
            "logits_to_keep": 1,
        },
        "material": {
            "pair_manifest": str(PAIR_MANIFEST_PATH.relative_to(ROOT)),
            "pair_manifest_digest": digest(pairs),
            "pair_manifest_file_sha256": sha256_file(PAIR_MANIFEST_PATH),
            "roles": list(ROLES),
            "prequery_roles": list(PREQUERY_ROLES),
            "panels": list(PANELS),
            "splits": list(SPLITS),
            "material_audit": material_audit,
            "tokenization_residue_policy": (
                "Exclude the entire eight-case quartet before hidden output iff any within-panel state pair "
                "has unequal exact input length; retain and report every excluded quartet."
            ),
        },
        "event_registry": events,
        "primary_gate": {
            "selection_scope": "residual component at generation_boundary only",
            "selection_split": "discovery",
            "candidate_depths": list(range(0, LAYER_COUNT + 1)),
            "controls": ["matched_null", "surface_only", "semantic_neighbor"],
            "distance": (
                "RMS(h_state1-h_state0) divided by the mean RMS state magnitude plus epsilon"
            ),
            "group_ratio": "active distance / (max of the three matched-control distances + epsilon)",
            "thresholds": THRESHOLDS,
            "band_rule": (
                "Require at least two adjacent passing residual depths on discovery; freeze the earliest "
                "depth in the earliest qualifying run."
            ),
            "confirmation_rule": "the frozen depth must pass the same thresholds without refitting",
            "unseen_rule": "the frozen depth must pass the same thresholds without refitting",
            "no_hotspot_rule": (
                "attention/MLP outputs and all non-generation roles are descriptive only and cannot select "
                "a depth or authorize causality"
            ),
        },
        "descriptive_map": {
            "roles": list(ROLES),
            "components": list(COMPONENTS),
            "events": len(events),
            "signed_projection": {
                **projection,
                "scope": "generation-boundary residual state differences only",
                "evidence_role": "descriptive geometry; never a primary gate",
            },
        },
        "instrument_audit": {
            "causal_order_test": (
                "At record roles, active and matched-null have identical record prefixes. Their state-pair "
                "distance trajectories must agree within the frozen tolerance."
            ),
            "maximum_abs_difference": THRESHOLDS["prequery_active_null_max_abs_difference"],
        },
        "authorization": {
            "run_hidden_response_map_after_preexecution_audit": True,
            "phase1206_qwen3_causal_preregistration_if_all_hidden_gates_pass": True,
            "automatic_causal_execution": False,
            "hidden_neuron_or_head_search": False,
            "cross_model_upgrade": False,
            "natural_use_upgrade": False,
        },
        "stop_rules": [
            "If the preexecution audit fails, do not load Qwen3.",
            "If no discovery contiguous band exists, stop without causal target selection.",
            "If the frozen depth fails confirmation or unseen composition, stop without causal work.",
            "A positive result authorizes only a separate Qwen3-specific causal preregistration.",
            "No result may weaken or reinterpret K184 as a cross-model claim.",
        ],
        "claim_boundary": {
            "behavior_evidence": True,
            "hidden_specificity_evidence": "pending",
            "causal_evidence": False,
            "natural_use_evidence": False,
            "cross_model_evidence": False,
            "mechanism_closure": False,
        },
        "upstream_checks": checks,
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "eligible_groups": material_audit["eligible_group_count"],
        "eligible_pairs": material_audit["eligible_pair_count"],
        "excluded_groups": material_audit["excluded_group_count"],
        "split_group_counts": material_audit["split_group_counts"],
    }, ensure_ascii=False, indent=2))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_embedded_digest(protocol, "protocol_digest")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("Phase1205 source hash drift")
    if sha256_file(PAIR_MANIFEST_PATH) != protocol["material"]["pair_manifest_file_sha256"]:
        raise RuntimeError("Phase1205 pair manifest file drift")
    pairs = read_jsonl(PAIR_MANIFEST_PATH)
    if digest(pairs) != protocol["material"]["pair_manifest_digest"]:
        raise RuntimeError("Phase1205 pair manifest digest drift")
    return protocol


class RoleEventCapture:
    def __init__(self, model: Any, layers: list[Any], events: list[dict[str, Any]]):
        self.model = model
        self.layers = layers
        self.events = events
        self.positions: torch.Tensor | None = None
        self.values: dict[str, torch.Tensor] = {}
        self.calls: dict[str, int] = defaultdict(int)
        self.handles: list[Any] = []

    def _hook(self, event_id: str):
        def hook(module: Any, args: Any, output: Any):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None or not isinstance(value, torch.Tensor):
                raise RuntimeError(f"capture not initialized for {event_id}")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[event_id] = value[batch, positions, :].detach()
            self.calls[event_id] += 1
            return output
        return hook

    def register(self) -> None:
        self.handles.append(
            self.model.get_input_embeddings().register_forward_hook(
                self._hook("residual_d00")
            )
        )
        for depth, layer in enumerate(self.layers, 1):
            self.handles.append(
                layer.register_forward_hook(self._hook(f"residual_d{depth:02d}"))
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook(f"attention_output_d{depth:02d}")
                )
            )
            self.handles.append(
                layer.mlp.register_forward_hook(
                    self._hook(f"mlp_output_d{depth:02d}")
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.calls = defaultdict(int)

    def validate(self) -> None:
        expected = {str(row["event_id"]) for row in self.events}
        if set(self.values) != expected:
            raise RuntimeError(
                f"capture event drift missing={sorted(expected-set(self.values))[:5]} "
                f"extra={sorted(set(self.values)-expected)[:5]}"
            )
        repeated = {key: value for key, value in self.calls.items() if value != 1}
        if repeated:
            raise RuntimeError(f"capture call drift: {repeated}")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


def run_command() -> None:
    protocol = verify_protocol()
    preaudit = read_json(PREAUDIT_PATH)
    validate_embedded_digest(preaudit, "audit_digest")
    if not preaudit.get("gate_pass"):
        raise RuntimeError("Phase1205 preexecution audit did not pass")
    if ARRAY_PATH.exists() or RUN_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1205 hidden output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    pairs = read_jsonl(PAIR_MANIFEST_PATH)
    qwen_manifest = read_jsonl(SOURCE_MANIFEST)
    item_index = {str(row["item_id"]): row for row in qwen_manifest}
    events = list(protocol["event_registry"])
    event_ids = [str(row["event_id"]) for row in events]
    role_index = {role: index for index, role in enumerate(ROLES)}
    residual_event_indices = [
        index for index, row in enumerate(events) if row["component"] == "residual"
    ]
    pair_count = len(pairs)
    relative = np.empty((pair_count, len(events), len(ROLES)), dtype=np.float32)
    absolute = np.empty_like(relative)
    signed_projection = np.empty(
        (pair_count, len(residual_event_indices), PROJECTION_DIM), dtype=np.float32
    )
    behavior_correct = np.zeros((pair_count, 2), dtype=np.bool_)
    behavior_finite = np.zeros((pair_count, 2), dtype=np.bool_)
    gold_margins = np.empty((pair_count, 2), dtype=np.float32)
    projection_np, projection_audit = projection_matrix()

    started = time.time()
    model = None
    capture = None
    completed = 0
    try:
        model, _tokenizer, device, placement = load_fp16(MODEL)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
            or set(precision["parameter_dtypes"]) != {"float16"}
        ):
            raise RuntimeError("Qwen3 FP16/no-quantization audit failed")
        layers = get_layers(model)
        if len(layers) != LAYER_COUNT:
            raise RuntimeError("Qwen3 layer count drift")
        capture = RoleEventCapture(model, layers, events)
        capture.register()
        projection_t = torch.tensor(projection_np, dtype=torch.float32, device=device)

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            by_length[int(pair["input_length"])].append(pair)

        with torch.inference_mode():
            for input_length in sorted(by_length):
                bucket = by_length[input_length]
                for start in range(0, len(bucket), BATCH_PAIRS):
                    batch_pairs = bucket[start : start + BATCH_PAIRS]
                    cases: list[dict[str, Any]] = []
                    position_rows: list[list[int]] = []
                    for pair in batch_pairs:
                        for state in (0, 1):
                            item_id = str(pair[f"state{state}_item_id"])
                            case = item_index[item_id]
                            if int(case["input_length"]) != input_length:
                                raise RuntimeError("pair input length drift")
                            cases.append(case)
                            position_rows.append([
                                int(pair[f"state{state}_positions"][role])
                                for role in ROLES
                            ])
                    input_ids = torch.tensor(
                        [row["input_ids"] for row in cases],
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.ones_like(input_ids)
                    positions = torch.tensor(
                        position_rows,
                        dtype=torch.long,
                        device=device,
                    )
                    capture.begin(positions)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                        output_hidden_states=False,
                        output_attentions=False,
                        logits_to_keep=1,
                    )
                    capture.validate()
                    final_logits = output.logits[:, -1, :]
                    vocab_finite = torch.isfinite(final_logits).all(dim=-1)
                    log_probs = torch.log_softmax(final_logits.float(), dim=-1)

                    for event_index, event_id in enumerate(event_ids):
                        values = capture.values[event_id].float()
                        state0 = values[0::2]
                        state1 = values[1::2]
                        difference = state1 - state0
                        absolute_batch = torch.sqrt(torch.mean(difference * difference, dim=-1))
                        magnitude0 = torch.sqrt(torch.mean(state0 * state0, dim=-1))
                        magnitude1 = torch.sqrt(torch.mean(state1 * state1, dim=-1))
                        relative_batch = absolute_batch / (
                            0.5 * (magnitude0 + magnitude1) + EPSILON
                        )
                        indices = [int(pair["pair_index"]) for pair in batch_pairs]
                        absolute[indices, event_index, :] = absolute_batch.cpu().numpy()
                        relative[indices, event_index, :] = relative_batch.cpu().numpy()
                        if event_index in residual_event_indices:
                            residual_slot = residual_event_indices.index(event_index)
                            generation_difference = difference[
                                :, role_index["generation_boundary"], :
                            ]
                            projected = generation_difference @ projection_t.T
                            projected = projected / (
                                torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
                                + EPSILON
                            )
                            signed_projection[
                                indices, residual_slot, :
                            ] = projected.cpu().numpy()

                    for local_pair, pair in enumerate(batch_pairs):
                        pair_index = int(pair["pair_index"])
                        for state in (0, 1):
                            slot = local_pair * 2 + state
                            case = cases[slot]
                            scores = {
                                label: float(
                                    log_probs[
                                        slot,
                                        case["candidate_token_ids"][label][0],
                                    ].item()
                                )
                                for label in case["candidate_labels"]
                            }
                            finite = bool(vocab_finite[slot].item()) and all(
                                math.isfinite(value) for value in scores.values()
                            )
                            ranked = sorted(scores, key=lambda label: (-scores[label], label))
                            gold = str(case["gold_candidate"])
                            margin = scores[gold] - max(
                                value for label, value in scores.items() if label != gold
                            )
                            behavior_finite[pair_index, state] = finite
                            behavior_correct[pair_index, state] = finite and ranked[0] == gold
                            gold_margins[pair_index, state] = float(margin)

                    completed += len(batch_pairs)
                    del (
                        output,
                        final_logits,
                        vocab_finite,
                        log_probs,
                        input_ids,
                        attention_mask,
                        positions,
                    )
                    capture.values = {}
                print(canonical({
                    "phase": PHASE,
                    "completed_input_length": input_length,
                    "completed_pairs": completed,
                }), flush=True)

        if completed != pair_count:
            raise RuntimeError("Phase1205 pair count drift")
        if not (
            np.isfinite(relative).all()
            and np.isfinite(absolute).all()
            and np.isfinite(signed_projection).all()
            and np.isfinite(gold_margins).all()
        ):
            raise RuntimeError("Phase1205 nonfinite captured array")
        write_npz_atomic(
            ARRAY_PATH,
            relative_distance=relative,
            absolute_rms_distance=absolute,
            signed_generation_residual_projection=signed_projection,
            behavior_correct=behavior_correct,
            behavior_finite=behavior_finite,
            gold_margins=gold_margins,
        )
        summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1205.qwen3_hidden_run.v1",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "preexecution_audit_digest": preaudit["audit_digest"],
            "pair_count": pair_count,
            "event_count": len(events),
            "role_count": len(ROLES),
            "array_shapes": {
                "relative_distance": list(relative.shape),
                "absolute_rms_distance": list(absolute.shape),
                "signed_generation_residual_projection": list(signed_projection.shape),
                "behavior_correct": list(behavior_correct.shape),
                "behavior_finite": list(behavior_finite.shape),
                "gold_margins": list(gold_margins.shape),
            },
            "array_file_sha256": sha256_file(ARRAY_PATH),
            "all_arrays_finite": True,
            "repeat_behavior_finite_rate": float(behavior_finite.mean()),
            "repeat_behavior_accuracy": float(behavior_correct.mean()),
            "precision_audit": precision,
            "placement": placement,
            "projection_audit": projection_audit,
            "runtime": {
                "elapsed_seconds": time.time() - started,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "claim_boundary": protocol["claim_boundary"],
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
        print(json.dumps({
            "phase": PHASE,
            "pair_count": pair_count,
            "repeat_behavior_accuracy": summary["repeat_behavior_accuracy"],
            "elapsed_seconds": summary["runtime"]["elapsed_seconds"],
            "summary_digest": summary["summary_digest"],
        }, ensure_ascii=False, indent=2))
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)
        gc.collect()


def median(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return statistics.median(materialized) if materialized else 0.0


def event_metrics(
    relative: np.ndarray,
    pairs: list[dict[str, Any]],
    event_index: int,
    role_index: int,
    split: str,
) -> dict[str, Any]:
    by_group: dict[str, dict[str, float]] = defaultdict(dict)
    for pair in pairs:
        if pair["split"] != split:
            continue
        by_group[str(pair["group_id"])][str(pair["panel"])] = float(
            relative[int(pair["pair_index"]), event_index, role_index]
        )
    ratios: list[float] = []
    advantages: list[float] = []
    active_values: list[float] = []
    controls: dict[str, list[float]] = {
        panel: [] for panel in PANELS if panel != "active"
    }
    for group_id, values in by_group.items():
        if set(values) != set(PANELS):
            raise RuntimeError(f"incomplete analysis quartet: {group_id}")
        active = values["active"]
        maximum_control = max(values[panel] for panel in controls)
        active_values.append(active)
        for panel in controls:
            controls[panel].append(values[panel])
        ratios.append(active / (maximum_control + EPSILON))
        advantages.append(active - maximum_control)
    finite_fraction = float(np.isfinite(np.asarray(
        active_values + ratios + advantages + [v for values in controls.values() for v in values]
    )).mean())
    result = {
        "split": split,
        "group_count": len(by_group),
        "finite_fraction": finite_fraction,
        "active_median_relative_distance": median(active_values),
        "control_median_relative_distance": {
            panel: median(values) for panel, values in controls.items()
        },
        "active_to_max_control_median_ratio": median(ratios),
        "active_over_all_controls_fraction": sum(value > 0 for value in advantages) / max(len(advantages), 1),
        "median_active_minus_max_control": median(advantages),
    }
    result["pass"] = bool(
        result["finite_fraction"] >= THRESHOLDS["finite_fraction"]
        and result["active_median_relative_distance"]
        >= THRESHOLDS["minimum_active_relative_distance"]
        and result["active_to_max_control_median_ratio"]
        >= THRESHOLDS["active_to_max_control_median_ratio"]
        and result["active_over_all_controls_fraction"]
        >= THRESHOLDS["active_over_all_controls_fraction"]
    )
    return result


def contiguous_runs(depths: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for depth in sorted(depths):
        if not runs or depth != runs[-1][-1] + 1:
            runs.append([depth])
        else:
            runs[-1].append(depth)
    return runs


def analyze_command() -> None:
    protocol = verify_protocol()
    if VERDICT_PATH.exists() or TRAJECTORY_PATH.exists() or RESULT_AUDIT_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("Phase1205 analysis output already exists")
    summary = read_json(RUN_SUMMARY_PATH)
    validate_embedded_digest(summary, "summary_digest")
    if sha256_file(ARRAY_PATH) != summary["array_file_sha256"]:
        raise RuntimeError("Phase1205 array file drift")
    pairs = read_jsonl(PAIR_MANIFEST_PATH)
    events = list(protocol["event_registry"])
    event_lookup = {str(row["event_id"]): index for index, row in enumerate(events)}
    role_lookup = {role: index for index, role in enumerate(ROLES)}
    with np.load(ARRAY_PATH, allow_pickle=False) as arrays:
        relative = arrays["relative_distance"]
        projections = arrays["signed_generation_residual_projection"]

        trajectories: dict[str, Any] = {
            "phase": PHASE,
            "protocol_digest": protocol["protocol_digest"],
            "evidence_scope": "descriptive_only_except_primary_generation_residual_gate",
            "splits": {},
        }
        for split in SPLITS:
            trajectories["splits"][split] = {}
            for role in ROLES:
                trajectories["splits"][split][role] = {}
                for component in COMPONENTS:
                    component_events = [row for row in events if row["component"] == component]
                    trajectories["splits"][split][role][component] = [
                        {
                            "depth": int(event["depth"]),
                            **event_metrics(
                                relative,
                                pairs,
                                event_lookup[str(event["event_id"])],
                                role_lookup[role],
                                split,
                            ),
                        }
                        for event in component_events
                    ]
        trajectories["trajectory_digest"] = digest(trajectories)
        write_json(TRAJECTORY_PATH, trajectories)

        primary_discovery: dict[int, dict[str, Any]] = {}
        for depth in range(0, LAYER_COUNT + 1):
            event_index = event_lookup[f"residual_d{depth:02d}"]
            primary_discovery[depth] = event_metrics(
                relative,
                pairs,
                event_index,
                role_lookup["generation_boundary"],
                "discovery",
            )
        passing_depths = [depth for depth, metrics in primary_discovery.items() if metrics["pass"]]
        runs = [
            run for run in contiguous_runs(passing_depths)
            if len(run) >= THRESHOLDS["minimum_contiguous_discovery_depths"]
        ]
        selected_depth = runs[0][0] if runs else None
        selected_metrics = None
        if selected_depth is not None:
            event_index = event_lookup[f"residual_d{selected_depth:02d}"]
            selected_metrics = {
                split: event_metrics(
                    relative,
                    pairs,
                    event_index,
                    role_lookup["generation_boundary"],
                    split,
                )
                for split in SPLITS
            }

        prequery_differences: list[float] = []
        by_group_panel = {
            (str(pair["group_id"]), str(pair["panel"])): int(pair["pair_index"])
            for pair in pairs
        }
        for group_id in sorted({str(pair["group_id"]) for pair in pairs}):
            active_index = by_group_panel[(group_id, "active")]
            null_index = by_group_panel[(group_id, "matched_null")]
            for event_index in range(len(events)):
                for role in PREQUERY_ROLES:
                    role_index = role_lookup[role]
                    prequery_differences.append(abs(float(
                        relative[active_index, event_index, role_index]
                        - relative[null_index, event_index, role_index]
                    )))
        prequery_max_abs = max(prequery_differences) if prequery_differences else math.inf
        instrument_pass = prequery_max_abs <= THRESHOLDS["prequery_active_null_max_abs_difference"]

        projection_summary: dict[str, Any] = {"descriptive_only": True}
        if selected_depth is not None:
            active_pairs = [pair for pair in pairs if pair["panel"] == "active"]
            residual_slot = selected_depth
            oriented: dict[str, list[np.ndarray]] = defaultdict(list)
            for pair in active_pairs:
                vector = projections[int(pair["pair_index"]), residual_slot].astype(np.float64)
                vector *= int(pair["active_entity0_to_entity1_orientation"])
                norm = np.linalg.norm(vector)
                if norm > 0:
                    oriented[str(pair["split"])].append(vector / norm)
            prototypes = {
                split: np.mean(np.stack(values), axis=0)
                for split, values in oriented.items() if values
            }
            for split, vector in prototypes.items():
                norm = np.linalg.norm(vector)
                prototypes[split] = vector / norm if norm > 0 else vector
            projection_summary.update({
                "selected_depth": selected_depth,
                "prototype_cosines": {
                    f"{left}_vs_{right}": float(np.dot(prototypes[left], prototypes[right]))
                    for left, right in (
                        ("discovery", "confirmation"),
                        ("discovery", "unseen_composition"),
                        ("confirmation", "unseen_composition"),
                    )
                    if left in prototypes and right in prototypes
                },
                "split_counts": {split: len(values) for split, values in oriented.items()},
            })

    discovery_band_pass = bool(runs)
    confirmation_pass = bool(
        selected_metrics is not None and selected_metrics["confirmation"]["pass"]
    )
    unseen_pass = bool(
        selected_metrics is not None and selected_metrics["unseen_composition"]["pass"]
    )
    hidden_gate = bool(
        instrument_pass and discovery_band_pass and confirmation_pass and unseen_pass
    )
    if hidden_gate:
        status = "qwen3_hidden_specificity_qualified"
        proposed_k = {
            "id": "K185",
            "scope": "Qwen3-specific controlled hidden transition",
            "statement": (
                "A preregistered active-specific generation-boundary residual band exceeds all matched controls "
                "and repeats at the frozen depth in confirmation and unseen composition; causality remains untested."
            ),
        }
    else:
        status = "qwen3_hidden_specificity_not_qualified"
        proposed_k = {
            "id": "K185",
            "scope": "Qwen3-specific hidden-specificity boundary",
            "statement": (
                "The preregistered generation-boundary residual criterion did not complete discovery-band, "
                "confirmation, unseen-composition, and instrument gates; causal target selection is denied."
            ),
        }
    verdict: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1205.qwen3_hidden_specificity_verdict.v1",
        "created_at": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "run_summary_digest": summary["summary_digest"],
        "trajectory_digest": read_json(TRAJECTORY_PATH)["trajectory_digest"],
        "primary_gate": {
            "role": "generation_boundary",
            "component": "residual",
            "discovery_passing_depths": passing_depths,
            "discovery_qualifying_runs": runs,
            "selected_depth": selected_depth,
            "selected_metrics": selected_metrics,
            "discovery_metrics_by_depth": {
                str(depth): metrics for depth, metrics in primary_discovery.items()
            },
            "instrument_prequery_max_abs_difference": prequery_max_abs,
            "instrument_pass": instrument_pass,
            "discovery_band_pass": discovery_band_pass,
            "confirmation_pass": confirmation_pass,
            "unseen_composition_pass": unseen_pass,
            "hidden_specificity_gate": hidden_gate,
        },
        "signed_projection_summary": projection_summary,
        "status": status,
        "proposed_k_item_pending_independent_audit": proposed_k,
        "claim_boundary": {
            "qwen3_model_specific": True,
            "hidden_specificity_evidence": hidden_gate,
            "causal_evidence": False,
            "natural_use_evidence": False,
            "cross_model_evidence": False,
            "mechanism_closure": False,
        },
    }
    verdict["verdict_digest"] = digest(verdict)
    write_json(VERDICT_PATH, verdict)
    print(json.dumps({
        "status": status,
        "selected_depth": selected_depth,
        "discovery_runs": runs,
        "confirmation_pass": confirmation_pass,
        "unseen_pass": unseen_pass,
        "instrument_pass": instrument_pass,
        "hidden_specificity_gate": hidden_gate,
        "verdict_digest": verdict["verdict_digest"],
    }, ensure_ascii=False, indent=2))


def finalize_command() -> None:
    protocol = verify_protocol()
    verdict = read_json(VERDICT_PATH)
    audit = read_json(RESULT_AUDIT_PATH)
    validate_embedded_digest(verdict, "verdict_digest")
    validate_embedded_digest(audit, "audit_digest")
    if not audit.get("gate_pass") or audit.get("verdict_digest") != verdict["verdict_digest"]:
        raise RuntimeError("Phase1205 result audit failed")
    hidden_gate = bool(verdict["primary_gate"]["hidden_specificity_gate"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "status": verdict["status"],
        "protocol_digest": protocol["protocol_digest"],
        "verdict_digest": verdict["verdict_digest"],
        "independent_result_audit_digest": audit["audit_digest"],
        "selected_depth": verdict["primary_gate"]["selected_depth"],
        "hidden_specificity_gate": hidden_gate,
        "new_k_item": verdict["proposed_k_item_pending_independent_audit"],
        "evidence_scope": verdict["claim_boundary"],
        "authorized_next": {
            "phase1206_qwen3_causal_preregistration": hidden_gate,
            "automatic_causal_execution": False,
            "head_or_neuron_search": False,
            "cross_model_claim": False,
            "natural_use_claim": False,
            "new_mechanism_algebra": False,
        },
        "stop_rule": (
            "Hidden specificity passed; only a separate zero-output Qwen3 causal preregistration is authorized."
            if hidden_gate
            else "Hidden specificity did not pass; no causal target, component, head, or neuron may be selected."
        ),
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "run", "analyze", "finalize"))
    args = parser.parse_args()
    {
        "protocol": protocol_command,
        "run": run_command,
        "analyze": analyze_command,
        "finalize": finalize_command,
    }[args.command]()


if __name__ == "__main__":
    main()
