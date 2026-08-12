#!/usr/bin/env python3
"""Phase 1223: full physical trajectory and causal handoff for passed atoms.

Only Phase1222-authorized Qwen3 operation/track scopes enter this phase.  The
camera records residual, attention, MLP, Q, K, and V events at six semantic
roles.  A separately frozen donor-patch test searches discovery data for the
earliest residual depth whose binding-state replacement transfers the future
answer response, then tests that single depth on three held-out splits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1222_atomic_operation_independent_confirmation as p1222
from model_utils import MODEL_CONFIGS, get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1223
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1223_passed_atom_physical_trajectory_audit.py"
SOURCE_ROOT = TEST_ROOT / "result/phase1222_atomic_operation_independent_confirmation"
SOURCE_FINAL = SOURCE_ROOT / "analysis/final.json"
SOURCE_RESULT_AUDIT = SOURCE_ROOT / "audit/independent_result_audit.json"
SOURCE_PRECISION_AUDIT = SOURCE_ROOT / "audit/fp16_schema_resolution.json"
SOURCE_MATERIAL = SOURCE_ROOT / "material/atomic_worlds.jsonl"
SOURCE_MANIFEST = SOURCE_ROOT / "protocol/qwen3_manifest.jsonl"
SOURCE_RAW = SOURCE_ROOT / "behavior/qwen3/raw_behavior.jsonl"
EXPECTED_SOURCE_FINAL_DIGEST = "a6be67cce38afa78aef432c8d01b1c8007cd40039dc4cc66c190a360753a65e2"
EXPECTED_SOURCE_PRECISION_AUDIT_DIGEST = "0599a6d47b67add164bbbc3937fefa4e57f46919b1d98767c24f06ba498ce4a0"

OUT_ROOT = TEST_ROOT / "result/phase1223_passed_atom_physical_trajectory"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PAIR_PATH = OUT_ROOT / "protocol/pair_manifest.jsonl"
STATE_PATH = OUT_ROOT / "protocol/state_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
ARRAY_PATH = OUT_ROOT / "runs/physical_trajectory_arrays.npz"
PATCH_PATH = OUT_ROOT / "runs/patch_records.jsonl"
SELECTION_PATH = OUT_ROOT / "protocol/discovery_selection.json"
RUN_SUMMARY_PATH = OUT_ROOT / "runs/run_summary.json"
TRAJECTORY_PATH = OUT_ROOT / "analysis/trajectory_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use", "sealed")
HOLDOUT_SPLITS = ("confirmation", "natural_use", "sealed")
PANELS = ("canonical", "record_order", "paraphrase", "binding_permutation")
CONTRASTS = {
    "binding_content": ("canonical", "binding_permutation"),
    "record_order_surface": ("canonical", "record_order"),
    "paraphrase_surface": ("canonical", "paraphrase"),
}
ROLES = (
    "record_object",
    "record_relation",
    "record_value",
    "query_subject",
    "query_relation",
    "generation_boundary",
)
PAIRS_PER_SCOPE_SPLIT = 8
LAYER_COUNT = 36
HIDDEN_SIZE = 2560
PROJECTION_DIM = 32
PROJECTION_SEED = 12230019
BATCH_STATES = 4
EPSILON = 1e-8

DISCOVERY_THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "positive_target_shift_fraction_min": 0.875,
    "median_correct_completion_min": 0.50,
    "positive_correct_completion_fraction_min": 0.75,
    "median_correct_over_wrong_completion_min": 0.25,
    "contiguous_depths_min": 2,
}
HOLDOUT_THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "positive_target_shift_fraction_min": 0.75,
    "median_correct_completion_min": 0.50,
    "positive_correct_completion_fraction_min": 0.75,
    "median_correct_over_wrong_completion_min": 0.25,
    "correct_patch_donor_choice_fraction_min": 0.75,
    "zero_patch_max_abs_score_drift_max": 1e-4,
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def event_registry() -> list[dict[str, Any]]:
    events = [{"event_id": "residual_d00", "component": "residual", "depth": 0}]
    for depth in range(1, LAYER_COUNT + 1):
        for component in (
            "residual",
            "attention_output",
            "mlp_output",
            "q_output",
            "k_output",
            "v_output",
        ):
            events.append(
                {
                    "event_id": f"{component}_d{depth:02d}",
                    "component": component,
                    "depth": depth,
                }
            )
    return events


def verify_source() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final = read_json(SOURCE_FINAL)
    frozen_audit = read_json(SOURCE_RESULT_AUDIT)
    precision_audit = read_json(SOURCE_PRECISION_AUDIT)
    if final.get("final_digest") != EXPECTED_SOURCE_FINAL_DIGEST:
        raise RuntimeError("Phase1222 final digest changed")
    if precision_audit.get("audit_digest") != EXPECTED_SOURCE_PRECISION_AUDIT_DIGEST:
        raise RuntimeError("Phase1222 precision supplement changed")
    if not precision_audit.get("all_checks_passed"):
        raise RuntimeError("Phase1222 precision supplement failed")
    failed = [row["name"] for row in frozen_audit["checks"] if not row["passed"]]
    if failed != ["fp16_nonquantized_execution"]:
        raise RuntimeError(f"unexpected Phase1222 frozen audit failures: {failed}")
    if not final.get("authorized_next", {}).get("automatic_execution"):
        raise RuntimeError("Phase1222 did not authorize automatic physical work")
    return final, read_jsonl(SOURCE_MATERIAL), read_jsonl(SOURCE_MANIFEST), read_jsonl(SOURCE_RAW)


def token_positions_for_span(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    values = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end and right > left
    ]
    if not values:
        raise RuntimeError(f"no token overlaps span {start}:{end}")
    return values


def unique_find(container: str, needle: str, start: int = 0) -> int:
    position = container.find(needle, start)
    if position < 0:
        raise RuntimeError(f"substring missing: {needle!r}")
    return position


def role_strings(row: dict[str, Any]) -> dict[str, str]:
    operation = row["operation"]
    if operation == "direct":
        record_object = row["objects"][0]
        query_subject = record_object
    elif operation == "query_object":
        record_object = row["objects"][1]
        query_subject = record_object
    elif operation == "query_relation":
        record_object = row["objects"][0]
        query_subject = record_object
    elif operation == "inverse_lookup":
        record_object = row["gold"]
        query_subject = row["derivation"][1]
    else:
        raise KeyError(operation)
    relation = row["target_relation"]
    record_value = row["computed_assignments"][record_object][relation]
    return {
        "record_object": record_object,
        "record_relation": relation,
        "record_value": record_value,
        "query_subject": query_subject,
        "query_relation": relation,
    }


def role_positions(
    tokenizer: Any,
    row: dict[str, Any],
    source_manifest: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any], str, dict[str, list[int]]]:
    open_prompt = row["open_prompt"]
    rendered = p1222.render_native(tokenizer, open_prompt, p1222.OPEN_SYSTEM)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    if input_ids != [int(value) for value in source_manifest["open_input_ids"]]:
        raise RuntimeError("Phase1223 tokenizer IDs differ from frozen Phase1222 open prompt")
    prompt_start = unique_find(rendered, open_prompt)
    strings = role_strings(row)
    paraphrase = row["panel"] == "paraphrase" or row["split"] == "natural_use"
    record_candidates = p1222.record_lines(
        row["display_assignments"], row["objects"], row["relations"], paraphrase
    )
    record_line = next(line for line in record_candidates if line.startswith(strings["record_object"]))
    record_local = unique_find(open_prompt, record_line)
    query_local = unique_find(open_prompt, row["query"])

    spans: dict[str, tuple[int, int]] = {}
    for role in ("record_object", "record_relation", "record_value"):
        value = strings[role]
        local = record_local + unique_find(record_line, value)
        spans[role] = (prompt_start + local, prompt_start + local + len(value))
    query_text = row["query"]
    for role in ("query_subject", "query_relation"):
        value = strings[role]
        local = query_local + unique_find(query_text, value)
        spans[role] = (prompt_start + local, prompt_start + local + len(value))
    positions = {
        role: token_positions_for_span(offsets, left, right)[-1]
        for role, (left, right) in spans.items()
    }
    positions["generation_boundary"] = len(input_ids) - 1

    candidate_token_ids: dict[str, list[int]] = {}
    for candidate in row["candidate_order"]:
        base, suffix = p1222.p1220.continuation_ids(tokenizer, rendered, candidate)
        if base != input_ids:
            raise RuntimeError("candidate changed open-prompt prefix")
        candidate_token_ids[candidate] = suffix
    audit = {
        "rendered_digest": digest(rendered),
        "input_token_count": len(input_ids),
        "role_strings": strings,
        "role_spans": {key: list(value) for key, value in spans.items()},
        "role_positions": positions,
        "candidate_token_lengths": {
            key: len(value) for key, value in candidate_token_ids.items()
        },
    }
    return positions, audit, rendered, candidate_token_ids


def build_material() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    final, material, source_manifest, raw = verify_source()
    material_by_id = {row["item_id"]: row for row in material}
    manifest_by_id = {row["item_id"]: row for row in source_manifest}
    raw_by_id = {row["item_id"]: row for row in raw}
    authorized = list(final["behavior"]["authorized_target_operation_tracks"])
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in material:
        scope = f"{row['operation']}|{row['track']}"
        if scope in authorized:
            grouped[row["group_id"]][row["panel"]] = row

    pair_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    state_seen: set[str] = set()
    eligible_counts: dict[str, int] = {}
    for scope in authorized:
        operation, track = scope.split("|")
        for split in SPLITS:
            eligible: list[dict[str, dict[str, Any]]] = []
            for panels in grouped.values():
                if set(panels) != set(PANELS):
                    continue
                canonical = panels["canonical"]
                if canonical["split"] != split or canonical["operation"] != operation or canonical["track"] != track:
                    continue
                if all(
                    raw_by_id[panels[panel]["item_id"]]["candidate_correct"]
                    and raw_by_id[panels[panel]["item_id"]]["context_correct"]
                    and raw_by_id[panels[panel]["item_id"]]["open_generation_correct"]
                    for panel in PANELS
                ):
                    eligible.append(panels)
            eligible.sort(key=lambda panels: digest([scope, split, panels["canonical"]["group_id"], 1223]))
            eligible_counts[f"{scope}|{split}"] = len(eligible)
            if len(eligible) < PAIRS_PER_SCOPE_SPLIT:
                raise RuntimeError(f"insufficient behavior-correct pairs for {scope}/{split}: {len(eligible)}")
            for local_pair, panels in enumerate(eligible[:PAIRS_PER_SCOPE_SPLIT]):
                pair_id = f"p1223-{digest([scope, split, panels['canonical']['group_id']])[:20]}"
                panel_states: dict[str, str] = {}
                for panel in PANELS:
                    source = panels[panel]
                    item_id = source["item_id"]
                    state_id = f"{pair_id}::{panel}"
                    panel_states[panel] = state_id
                    if state_id in state_seen:
                        continue
                    positions, position_audit, _rendered, candidate_ids = role_positions(
                        tokenizer, source, manifest_by_id[item_id]
                    )
                    state: dict[str, Any] = {
                        "schema_version": "phase1223.state.v1",
                        "phase": PHASE,
                        "state_index": len(state_rows),
                        "state_id": state_id,
                        "pair_id": pair_id,
                        "scope": scope,
                        "split": split,
                        "operation": operation,
                        "track": track,
                        "panel": panel,
                        "source_item_id": item_id,
                        "source_row_digest": source["row_digest"],
                        "gold": source["gold"],
                        "candidates": source["candidate_order"],
                        "input_ids": manifest_by_id[item_id]["open_input_ids"],
                        "input_token_count": manifest_by_id[item_id]["open_input_token_count"],
                        "candidate_token_ids": candidate_ids,
                        "role_positions": positions,
                        "position_audit": position_audit,
                    }
                    state["state_digest"] = digest(state)
                    state_rows.append(state)
                    state_seen.add(state_id)
                pair: dict[str, Any] = {
                    "schema_version": "phase1223.pair.v1",
                    "phase": PHASE,
                    "pair_index": len(pair_rows),
                    "pair_id": pair_id,
                    "scope": scope,
                    "split": split,
                    "operation": operation,
                    "track": track,
                    "local_pair": local_pair,
                    "source_group_id": panels["canonical"]["group_id"],
                    "panel_states": panel_states,
                    "recipient_gold": panels["canonical"]["gold"],
                    "donor_gold": panels["binding_permutation"]["gold"],
                }
                pair["pair_digest"] = digest(pair)
                pair_rows.append(pair)
    audit = {
        "authorized_scopes": authorized,
        "eligible_counts": eligible_counts,
        "pair_count": len(pair_rows),
        "state_count": len(state_rows),
        "pair_digest": digest(pair_rows),
        "state_digest": digest(state_rows),
    }
    return pair_rows, state_rows, audit


def model_artifact_fingerprint() -> dict[str, Any]:
    root = Path(MODEL_CONFIGS["qwen3"]["path"])
    return {
        "path": str(root),
        "config_sha256": file_sha256(root / "config.json"),
        "tokenizer_config_sha256": file_sha256(root / "tokenizer_config.json"),
    }


def build_protocol(pairs: list[dict[str, Any]], states: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    events = event_registry()
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1223.physical.protocol.v1",
        "created_at": utc_now(),
        "purpose": "measure complete typed physical trajectories and the earliest residual causal handoff for Phase1222-passed atoms",
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1222_main": file_sha256(p1222.SCRIPT),
            "phase1222_final": file_sha256(SOURCE_FINAL),
            "phase1222_raw": file_sha256(SOURCE_RAW),
        },
        "upstream": {
            "phase1222_final_digest": EXPECTED_SOURCE_FINAL_DIGEST,
            "phase1222_precision_resolution_digest": EXPECTED_SOURCE_PRECISION_AUDIT_DIGEST,
            "frozen_result_audit_status": "13/14; only obsolete FP16 schema check failed",
            "automatic_branch": True,
            "authorized_scopes": audit["authorized_scopes"],
        },
        "material": {
            "pairs_per_scope_split": PAIRS_PER_SCOPE_SPLIT,
            "pair_count": len(pairs),
            "state_count": len(states),
            "pair_digest": digest(pairs),
            "state_digest": digest(states),
            "eligible_counts": audit["eligible_counts"],
            "selection_rule": "lowest deterministic hash among groups whose four panels pass candidate, context-adjusted, and open-generation behavior",
            "selection_is_behavior_conditioned": True,
        },
        "camera": {
            "roles": list(ROLES),
            "events": events,
            "event_count": len(events),
            "components": ["residual", "attention_output", "mlp_output", "q_output", "k_output", "v_output"],
            "depths": list(range(LAYER_COUNT + 1)),
            "projection_dimension": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "contrasts": {key: list(value) for key, value in CONTRASTS.items()},
            "trajectory_is_descriptive": True,
            "trajectory_cannot_select_components_or_neurons": True,
        },
        "causal_handoff": {
            "patch_component": "whole residual stream",
            "patch_role": "generation_boundary",
            "discovery_split": "discovery",
            "scan_depths": list(range(LAYER_COUNT + 1)),
            "correct_patch": "recipient plus its matched binding-permutation donor delta",
            "wrong_patch": "recipient plus next-pair donor delta rescaled to the matched delta norm",
            "zero_patch": "exact recipient state replacement",
            "discovery_candidates": "recipient gold and donor gold only",
            "holdout_candidates": "all four candidates",
            "selection": "earliest start of two consecutive discovery-passing depths, independently per scope",
            "discovery_thresholds": DISCOVERY_THRESHOLDS,
            "holdout_splits": list(HOLDOUT_SPLITS),
            "holdout_thresholds": HOLDOUT_THRESHOLDS,
            "selection_file_must_be_written_before_holdout_patch": True,
        },
        "interface": {
            "model": "qwen3",
            "model_artifact": model_artifact_fingerprint(),
            "precision": "FP16",
            "quantization": "none",
            "native_chat_template": True,
            "enable_thinking": False,
            "full_continuation_sum_log_probability": True,
            "single_model_sequential_cuda": True,
        },
        "authorization": {
            "scope_unit": "exact operation|track",
            "physical_gate": "discovery selection and every frozen holdout split gate",
            "failure_closes_only_exact_physical_scope": True,
            "no_head_or_neuron_search": True,
            "no_cross_model_claim": True,
            "next_phase_not_automatic": "minimal-state and semantic compiler contracts are not frozen in Phase1223",
        },
        "forbidden_after_freeze": [
            "add a Phase1222-failed scope",
            "change pair selection, semantic roles, events, depths, controls, thresholds, or candidate scoring",
            "select a trajectory hotspot instead of the frozen residual generation-boundary scan",
            "change discovery depth after holdout responses are seen",
            "drop a holdout split or failed scope",
            "run head, neuron, cross-model, or organic-language searches",
            "call projected trajectory geometry a causal mechanism",
        ],
        "claim_boundary": {
            "qwen3_only": True,
            "generated_behavior_correct_subset": True,
            "organic_language": False,
            "full_component_trajectory": True,
            "whole_residual_causal_patch": True,
            "head_or_neuron": False,
            "cross_model": False,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    return protocol


def materialize() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(f"formal output already exists: {OUT_ROOT}")
    pairs, states, audit = build_material()
    expected_pairs = len(audit["authorized_scopes"]) * len(SPLITS) * PAIRS_PER_SCOPE_SPLIT
    if len(pairs) != expected_pairs or len(states) != expected_pairs * len(PANELS):
        raise RuntimeError("material count drift")
    protocol = build_protocol(pairs, states, audit)
    write_jsonl(PAIR_PATH, pairs)
    write_jsonl(STATE_PATH, states)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"status": "materialized", "pairs": len(pairs), "states": len(states), "protocol_digest": protocol["protocol_digest"]}))


def verify_formal_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    pairs = read_jsonl(PAIR_PATH)
    states = read_jsonl(STATE_PATH)
    claimed = protocol.get("protocol_digest")
    if claimed != digest({key: value for key, value in protocol.items() if key != "protocol_digest"}):
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"]["main"] != file_sha256(SCRIPT) or protocol["source_hashes"]["audit"] != file_sha256(AUDIT_SCRIPT):
        raise RuntimeError("source changed after freeze")
    if protocol["material"]["pair_digest"] != digest(pairs) or protocol["material"]["state_digest"] != digest(states):
        raise RuntimeError("material changed after freeze")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("preaudit did not pass")
    return protocol, pairs, states


class PhysicalCapture:
    def __init__(self, model: Any, layers: list[Any], events: list[dict[str, Any]]):
        self.model = model
        self.layers = layers
        self.events = events
        self.positions: torch.Tensor | None = None
        self.values: dict[str, torch.Tensor] = {}
        self.calls: Counter[str] = Counter()
        self.handles: list[Any] = []

    def _hook(self, event_id: str):
        def hook(_module: Any, _args: Any, output: Any):
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
        self.handles.append(self.model.get_input_embeddings().register_forward_hook(self._hook("residual_d00")))
        for depth, layer in enumerate(self.layers, 1):
            modules = {
                "residual": layer,
                "attention_output": layer.self_attn,
                "mlp_output": layer.mlp,
                "q_output": layer.self_attn.q_proj,
                "k_output": layer.self_attn.k_proj,
                "v_output": layer.self_attn.v_proj,
            }
            for component, module in modules.items():
                self.handles.append(module.register_forward_hook(self._hook(f"{component}_d{depth:02d}")))

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.calls = Counter()

    def validate(self) -> None:
        expected = {row["event_id"] for row in self.events}
        if set(self.values) != expected or any(self.calls[key] != 1 for key in expected):
            raise RuntimeError("physical capture event/call drift")

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.values = {}
        self.positions = None


class BoundaryPatch(AbstractContextManager["BoundaryPatch"]):
    def __init__(self, module: Any, position: int, replacement: torch.Tensor):
        self.module = module
        self.position = int(position)
        self.replacement = replacement
        self.handle: Any | None = None
        self.calls = 0

    def _hook(self, _module: Any, _args: Any, output: Any):
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("patch output is not tensor")
        patched = value.clone()
        replacement = self.replacement.to(value.device, dtype=value.dtype)
        if replacement.ndim == 1:
            replacement = replacement[None, :].expand(value.shape[0], -1)
        patched[:, self.position, :] = replacement
        self.calls += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    def __enter__(self) -> "BoundaryPatch":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def projection_matrix(dimension: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(PROJECTION_SEED + dimension * 17)
    values = rng.integers(0, 2, size=(PROJECTION_DIM, dimension), dtype=np.int8)
    matrix = (values.astype(np.float32) * 2.0 - 1.0) / math.sqrt(float(dimension))
    return torch.tensor(matrix, dtype=torch.float32, device=device)


def grouped_batches(states: list[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        buckets[int(state["input_token_count"])].append(state)
    for length in sorted(buckets):
        values = buckets[length]
        for start in range(0, len(values), BATCH_STATES):
            yield values[start : start + BATCH_STATES]


def score_candidates(
    model: Any,
    device: torch.device,
    state: dict[str, Any],
    candidates: list[str],
    module: Any | None = None,
    replacement: torch.Tensor | None = None,
) -> tuple[dict[str, float], bool, int]:
    prompt = [int(value) for value in state["input_ids"]]
    entries: list[tuple[str, list[int], list[int]]] = []
    for candidate in candidates:
        suffix = [int(value) for value in state["candidate_token_ids"][candidate]]
        entries.append((candidate, suffix, prompt + suffix))
    if len({len(entry[2]) for entry in entries}) != 1:
        raise RuntimeError("patched candidate lengths are not matched")
    input_ids = torch.tensor([entry[2] for entry in entries], dtype=torch.long, device=device)
    context: AbstractContextManager[Any]
    if module is None:
        from contextlib import nullcontext
        context = nullcontext()
    else:
        if replacement is None:
            raise RuntimeError("patch module lacks replacement")
        context = BoundaryPatch(module, state["role_positions"]["generation_boundary"], replacement)
    continuation_length = len(entries[0][1])
    with torch.inference_mode(), context as patch:
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=continuation_length + 1,
            return_dict=True,
        )
    calls = 0 if module is None else int(patch.calls)
    output_start = input_ids.shape[1] - output.logits.shape[1]
    scores: dict[str, float] = {}
    finite = True
    for index, (candidate, suffix, _sequence) in enumerate(entries):
        token_scores: list[float] = []
        for offset, token_id in enumerate(suffix):
            absolute = len(prompt) + offset - 1
            logits = output.logits[index, absolute - output_start].float()
            finite = finite and bool(torch.isfinite(logits).all().item())
            score = logits[token_id] - torch.logsumexp(logits, dim=-1)
            token_scores.append(float(score.item()))
        scores[candidate] = sum(token_scores)
        finite = finite and all(math.isfinite(value) for value in token_scores)
    del output, input_ids
    return scores, finite, calls


def margin(scores: dict[str, float], donor_gold: str, recipient_gold: str) -> float:
    return float(scores[donor_gold] - scores[recipient_gold])


def patch_record(
    pair: dict[str, Any],
    depth: int,
    condition: str,
    recipient_scores: dict[str, float],
    donor_scores: dict[str, float],
    patched_scores: dict[str, float],
    finite: bool,
    patch_calls: int,
) -> dict[str, Any]:
    recipient_gold = pair["recipient_gold"]
    donor_gold = pair["donor_gold"]
    recipient_margin = margin(recipient_scores, donor_gold, recipient_gold)
    donor_margin = margin(donor_scores, donor_gold, recipient_gold)
    patched_margin = margin(patched_scores, donor_gold, recipient_gold)
    target_shift = donor_margin - recipient_margin
    patch_shift = patched_margin - recipient_margin
    completion = patch_shift / target_shift if abs(target_shift) > EPSILON else 0.0
    prediction = max(patched_scores, key=lambda key: (patched_scores[key], key))
    row: dict[str, Any] = {
        "schema_version": "phase1223.patch.v1",
        "phase": PHASE,
        "pair_id": pair["pair_id"],
        "scope": pair["scope"],
        "split": pair["split"],
        "depth": int(depth),
        "condition": condition,
        "recipient_gold": recipient_gold,
        "donor_gold": donor_gold,
        "candidate_count": len(patched_scores),
        "recipient_margin": recipient_margin,
        "donor_margin": donor_margin,
        "target_shift": target_shift,
        "patched_margin": patched_margin,
        "patch_shift": patch_shift,
        "completion": completion,
        "recipient_scores": recipient_scores,
        "donor_scores": donor_scores,
        "patched_scores": patched_scores,
        "patched_prediction": prediction,
        "finite": bool(finite),
        "patch_calls": patch_calls,
    }
    row["patch_digest"] = digest(row)
    return row


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def discovery_depth_metrics(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_depth[int(row["depth"])].append(row)
    result: dict[int, dict[str, Any]] = {}
    for depth in range(LAYER_COUNT + 1):
        values = by_depth[depth]
        correct = {row["pair_id"]: row for row in values if row["condition"] == "correct"}
        wrong = {row["pair_id"]: row for row in values if row["condition"] == "wrong"}
        pair_ids = sorted(set(correct) & set(wrong))
        advantages = [correct[key]["completion"] - wrong[key]["completion"] for key in pair_ids]
        metrics = {
            "pair_count": len(pair_ids),
            "finite_fraction": sum(correct[key]["finite"] and wrong[key]["finite"] for key in pair_ids) / len(pair_ids),
            "positive_target_shift_fraction": sum(correct[key]["target_shift"] > 0 for key in pair_ids) / len(pair_ids),
            "median_correct_completion": median([correct[key]["completion"] for key in pair_ids]),
            "positive_correct_completion_fraction": sum(correct[key]["completion"] > 0 for key in pair_ids) / len(pair_ids),
            "median_correct_over_wrong_completion": median(advantages),
        }
        gates = {
            "finite": metrics["finite_fraction"] >= DISCOVERY_THRESHOLDS["finite_fraction_min"],
            "target_shift": metrics["positive_target_shift_fraction"] >= DISCOVERY_THRESHOLDS["positive_target_shift_fraction_min"],
            "completion": metrics["median_correct_completion"] >= DISCOVERY_THRESHOLDS["median_correct_completion_min"],
            "positive": metrics["positive_correct_completion_fraction"] >= DISCOVERY_THRESHOLDS["positive_correct_completion_fraction_min"],
            "wrong_control": metrics["median_correct_over_wrong_completion"] >= DISCOVERY_THRESHOLDS["median_correct_over_wrong_completion_min"],
        }
        result[depth] = {"metrics": metrics, "gates": gates, "passed": all(gates.values())}
    return result


def select_discovery(records: list[dict[str, Any]], scopes: list[str], protocol_digest: str) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    contiguous = int(DISCOVERY_THRESHOLDS["contiguous_depths_min"])
    for scope in scopes:
        selected = [row for row in records if row["scope"] == scope and row["split"] == "discovery"]
        depths = discovery_depth_metrics(selected)
        onset: int | None = None
        for depth in range(0, LAYER_COUNT - contiguous + 2):
            if all(depths[candidate]["passed"] for candidate in range(depth, depth + contiguous)):
                onset = depth
                break
        selections[scope] = {
            "depth_results": {str(key): value for key, value in depths.items()},
            "selected_depth": onset,
            "discovery_authorized": onset is not None,
        }
    selection: dict[str, Any] = {
        "phase": PHASE,
        "created_at": utc_now(),
        "protocol_digest": protocol_digest,
        "written_before_holdout_patch": True,
        "scope_selections": selections,
    }
    selection["selection_digest"] = digest(selection)
    return selection


def run() -> None:
    if ARRAY_PATH.exists() or PATCH_PATH.exists() or RUN_SUMMARY_PATH.exists() or SELECTION_PATH.exists():
        raise RuntimeError("Phase1223 run output already exists")
    protocol, pairs, states = verify_formal_inputs()
    events = protocol["camera"]["events"]
    event_index = {row["event_id"]: index for index, row in enumerate(events)}
    role_index = {role: index for index, role in enumerate(ROLES)}
    state_by_id = {row["state_id"]: row for row in states}
    state_index = {row["state_id"]: int(row["state_index"]) for row in states}
    pair_by_scope_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pair_by_scope_split[(pair["scope"], pair["split"])].append(pair)
    for values in pair_by_scope_split.values():
        values.sort(key=lambda row: row["local_pair"])

    projections = np.empty((len(states), len(events), len(ROLES), PROJECTION_DIM), dtype=np.float16)
    rms = np.empty((len(states), len(events), len(ROLES)), dtype=np.float32)
    residual_boundary = np.empty((len(states), LAYER_COUNT + 1, HIDDEN_SIZE), dtype=np.float16)
    started = time.time()
    model = tokenizer = capture = None
    patch_records: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 is not pure FP16")
        layers = get_layers(model)
        if len(layers) != LAYER_COUNT:
            raise RuntimeError("layer count drift")
        capture = PhysicalCapture(model, layers, events)
        capture.register()
        projection_cache: dict[int, torch.Tensor] = {}
        completed_states = 0
        with torch.inference_mode():
            for batch in grouped_batches(states):
                input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
                positions = torch.tensor(
                    [[row["role_positions"][role] for role in ROLES] for row in batch],
                    dtype=torch.long,
                    device=device,
                )
                capture.begin(positions)
                output = model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                    logits_to_keep=1,
                )
                capture.validate()
                for event in events:
                    event_id = event["event_id"]
                    index = event_index[event_id]
                    values = capture.values[event_id].float()
                    dimension = values.shape[-1]
                    if dimension not in projection_cache:
                        projection_cache[dimension] = projection_matrix(dimension, device)
                    projected = values @ projection_cache[dimension].T
                    norms = torch.sqrt(torch.mean(values * values, dim=-1))
                    target_indices = [int(row["state_index"]) for row in batch]
                    projections[target_indices, index] = projected.cpu().numpy().astype(np.float16)
                    rms[target_indices, index] = norms.cpu().numpy().astype(np.float32)
                    if event["component"] == "residual":
                        depth = int(event["depth"])
                        boundary = values[:, role_index["generation_boundary"], :]
                        if boundary.shape[-1] != HIDDEN_SIZE:
                            raise RuntimeError("residual hidden size drift")
                        residual_boundary[target_indices, depth] = boundary.cpu().numpy().astype(np.float16)
                del output, input_ids, positions
                completed_states += len(batch)
                if completed_states % 80 == 0:
                    print(f"[phase1223/capture] {completed_states}/{len(states)}", flush=True)
        capture.close()
        capture = None
        gc.collect()
        torch.cuda.empty_cache()

        clean_cache: dict[str, tuple[dict[str, float], bool]] = {}

        def clean(state_id: str) -> tuple[dict[str, float], bool]:
            if state_id not in clean_cache:
                state = state_by_id[state_id]
                scores, finite, calls = score_candidates(model, device, state, list(state["candidates"]))
                if calls != 0:
                    raise RuntimeError("unpatched scorer unexpectedly called a patch")
                clean_cache[state_id] = (scores, finite)
            return clean_cache[state_id]

        modules = [model.get_input_embeddings()] + list(layers)
        discovery_pairs = [pair for pair in pairs if pair["split"] == "discovery"]
        for pair_offset, pair in enumerate(discovery_pairs):
            scope_pairs = pair_by_scope_split[(pair["scope"], "discovery")]
            wrong_pair = scope_pairs[(int(pair["local_pair"]) + 1) % len(scope_pairs)]
            recipient_id = pair["panel_states"]["canonical"]
            donor_id = pair["panel_states"]["binding_permutation"]
            wrong_recipient_id = wrong_pair["panel_states"]["canonical"]
            wrong_donor_id = wrong_pair["panel_states"]["binding_permutation"]
            recipient_state = state_by_id[recipient_id]
            recipient_scores, recipient_finite = clean(recipient_id)
            donor_scores, donor_finite = clean(donor_id)
            candidates = [pair["recipient_gold"], pair["donor_gold"]]
            ri = state_index[recipient_id]
            di = state_index[donor_id]
            wri = state_index[wrong_recipient_id]
            wdi = state_index[wrong_donor_id]
            for depth in range(LAYER_COUNT + 1):
                recipient_vector = torch.tensor(residual_boundary[ri, depth].astype(np.float32), device=device)
                donor_vector = torch.tensor(residual_boundary[di, depth].astype(np.float32), device=device)
                correct_delta = donor_vector - recipient_vector
                wrong_delta = torch.tensor(
                    residual_boundary[wdi, depth].astype(np.float32)
                    - residual_boundary[wri, depth].astype(np.float32),
                    device=device,
                )
                correct_norm = torch.linalg.vector_norm(correct_delta)
                wrong_norm = torch.linalg.vector_norm(wrong_delta)
                wrong_replacement = recipient_vector + wrong_delta * (
                    correct_norm / (wrong_norm + EPSILON)
                )
                for condition, replacement in (
                    ("correct", donor_vector),
                    ("wrong", wrong_replacement),
                ):
                    patched, finite, calls = score_candidates(
                        model,
                        device,
                        recipient_state,
                        candidates,
                        modules[depth],
                        replacement,
                    )
                    if calls != 1:
                        raise RuntimeError("discovery patch call drift")
                    patch_records.append(
                        patch_record(
                            pair,
                            depth,
                            condition,
                            recipient_scores,
                            donor_scores,
                            patched,
                            finite and recipient_finite and donor_finite,
                            calls,
                        )
                    )
            if (pair_offset + 1) % 8 == 0:
                print(f"[phase1223/discovery-patch] {pair_offset + 1}/{len(discovery_pairs)}", flush=True)

        selection = select_discovery(
            patch_records,
            list(protocol["upstream"]["authorized_scopes"]),
            protocol["protocol_digest"],
        )
        write_json(SELECTION_PATH, selection)

        holdout_pairs = [pair for pair in pairs if pair["split"] in HOLDOUT_SPLITS]
        eligible_holdout = [
            pair
            for pair in holdout_pairs
            if selection["scope_selections"][pair["scope"]]["discovery_authorized"]
        ]
        for pair_offset, pair in enumerate(eligible_holdout):
            depth = int(selection["scope_selections"][pair["scope"]]["selected_depth"])
            scope_pairs = pair_by_scope_split[(pair["scope"], pair["split"])]
            wrong_pair = scope_pairs[(int(pair["local_pair"]) + 1) % len(scope_pairs)]
            recipient_id = pair["panel_states"]["canonical"]
            donor_id = pair["panel_states"]["binding_permutation"]
            wrong_recipient_id = wrong_pair["panel_states"]["canonical"]
            wrong_donor_id = wrong_pair["panel_states"]["binding_permutation"]
            recipient_state = state_by_id[recipient_id]
            recipient_scores, recipient_finite = clean(recipient_id)
            donor_scores, donor_finite = clean(donor_id)
            ri, di = state_index[recipient_id], state_index[donor_id]
            wri, wdi = state_index[wrong_recipient_id], state_index[wrong_donor_id]
            recipient_vector = torch.tensor(residual_boundary[ri, depth].astype(np.float32), device=device)
            donor_vector = torch.tensor(residual_boundary[di, depth].astype(np.float32), device=device)
            correct_delta = donor_vector - recipient_vector
            wrong_delta = torch.tensor(
                residual_boundary[wdi, depth].astype(np.float32)
                - residual_boundary[wri, depth].astype(np.float32),
                device=device,
            )
            wrong_replacement = recipient_vector + wrong_delta * (
                torch.linalg.vector_norm(correct_delta)
                / (torch.linalg.vector_norm(wrong_delta) + EPSILON)
            )
            for condition, replacement in (
                ("correct", donor_vector),
                ("wrong", wrong_replacement),
                ("zero", recipient_vector),
            ):
                patched, finite, calls = score_candidates(
                    model,
                    device,
                    recipient_state,
                    list(recipient_state["candidates"]),
                    modules[depth],
                    replacement,
                )
                if calls != 1:
                    raise RuntimeError("holdout patch call drift")
                patch_records.append(
                    patch_record(
                        pair,
                        depth,
                        condition,
                        recipient_scores,
                        donor_scores,
                        patched,
                        finite and recipient_finite and donor_finite,
                        calls,
                    )
                )
            if (pair_offset + 1) % 20 == 0:
                print(f"[phase1223/holdout-patch] {pair_offset + 1}/{len(eligible_holdout)}", flush=True)

        write_npz_atomic(
            ARRAY_PATH,
            projections=projections,
            rms=rms,
            residual_boundary=residual_boundary,
        )
        write_jsonl(PATCH_PATH, patch_records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "selection_digest": selection["selection_digest"],
            "state_count": len(states),
            "pair_count": len(pairs),
            "event_count": len(events),
            "role_count": len(ROLES),
            "patch_record_count": len(patch_records),
            "array_shapes": {
                "projections": list(projections.shape),
                "rms": list(rms.shape),
                "residual_boundary": list(residual_boundary.shape),
            },
            "array_file_sha256": file_sha256(ARRAY_PATH),
            "patch_digest": digest(patch_records),
            "precision_audit": precision,
            "placement": placement,
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        summary["summary_digest"] = digest(summary)
        write_json(RUN_SUMMARY_PATH, summary)
        print(canonical_json({"status": "run_complete", "patch_records": len(patch_records), "summary_digest": summary["summary_digest"]}))
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_fp16(model)
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def holdout_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct = {row["pair_id"]: row for row in records if row["condition"] == "correct"}
    wrong = {row["pair_id"]: row for row in records if row["condition"] == "wrong"}
    zero = {row["pair_id"]: row for row in records if row["condition"] == "zero"}
    pair_ids = sorted(set(correct) & set(wrong) & set(zero))
    advantages = [correct[key]["completion"] - wrong[key]["completion"] for key in pair_ids]
    zero_drift = max(
        abs(
            zero[key]["patched_scores"][candidate]
            - zero[key]["recipient_scores"][candidate]
        )
        for key in pair_ids
        for candidate in zero[key]["patched_scores"]
    )
    zero_margin_drift = max(abs(zero[key]["patch_shift"]) for key in pair_ids)
    metrics = {
        "pair_count": len(pair_ids),
        "finite_fraction": sum(correct[key]["finite"] and wrong[key]["finite"] and zero[key]["finite"] for key in pair_ids) / len(pair_ids),
        "positive_target_shift_fraction": sum(correct[key]["target_shift"] > 0 for key in pair_ids) / len(pair_ids),
        "median_correct_completion": median([correct[key]["completion"] for key in pair_ids]),
        "positive_correct_completion_fraction": sum(correct[key]["completion"] > 0 for key in pair_ids) / len(pair_ids),
        "median_correct_over_wrong_completion": median(advantages),
        "correct_patch_donor_choice_fraction": sum(correct[key]["patched_prediction"] == correct[key]["donor_gold"] for key in pair_ids) / len(pair_ids),
        "zero_patch_max_abs_margin_drift": zero_margin_drift,
        "zero_patch_max_abs_score_drift": zero_drift,
    }
    gates = {
        "finite": metrics["finite_fraction"] >= HOLDOUT_THRESHOLDS["finite_fraction_min"],
        "target_shift": metrics["positive_target_shift_fraction"] >= HOLDOUT_THRESHOLDS["positive_target_shift_fraction_min"],
        "completion": metrics["median_correct_completion"] >= HOLDOUT_THRESHOLDS["median_correct_completion_min"],
        "positive": metrics["positive_correct_completion_fraction"] >= HOLDOUT_THRESHOLDS["positive_correct_completion_fraction_min"],
        "wrong_control": metrics["median_correct_over_wrong_completion"] >= HOLDOUT_THRESHOLDS["median_correct_over_wrong_completion_min"],
        "donor_choice": metrics["correct_patch_donor_choice_fraction"] >= HOLDOUT_THRESHOLDS["correct_patch_donor_choice_fraction_min"],
        "zero_identity": metrics["zero_patch_max_abs_score_drift"] <= HOLDOUT_THRESHOLDS["zero_patch_max_abs_score_drift_max"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}
def trajectory_summary(
    protocol: dict[str, Any],
    pairs: list[dict[str, Any]],
    states: list[dict[str, Any]],
    arrays: Any,
) -> dict[str, Any]:
    projections = arrays["projections"].astype(np.float32)
    rms = arrays["rms"].astype(np.float32)
    state_index = {row["state_id"]: int(row["state_index"]) for row in states}
    events = protocol["camera"]["events"]
    role_index = {role: index for index, role in enumerate(ROLES)}
    summaries: dict[str, Any] = {}
    for scope in protocol["upstream"]["authorized_scopes"]:
        for split in SPLITS:
            selected = [pair for pair in pairs if pair["scope"] == scope and pair["split"] == split]
            contrast_arrays: dict[str, np.ndarray] = {}
            for contrast, (left_panel, right_panel) in CONTRASTS.items():
                values = []
                for pair in selected:
                    left = state_index[pair["panel_states"][left_panel]]
                    right = state_index[pair["panel_states"][right_panel]]
                    difference = projections[right] - projections[left]
                    numerator = np.sqrt(np.mean(difference * difference, axis=-1))
                    denominator = 0.5 * (rms[left] + rms[right]) + EPSILON
                    values.append(numerator / denominator)
                contrast_arrays[contrast] = np.stack(values, axis=0)
            content = np.median(contrast_arrays["binding_content"], axis=0)
            control = np.maximum(
                np.median(contrast_arrays["record_order_surface"], axis=0),
                np.median(contrast_arrays["paraphrase_surface"], axis=0),
            )
            ratio = content / (control + EPSILON)
            top_flat = np.argsort(ratio.reshape(-1))[::-1][:10]
            top = []
            for flat in top_flat:
                event_slot, role_slot = np.unravel_index(flat, ratio.shape)
                top.append(
                    {
                        "event": events[event_slot],
                        "role": ROLES[role_slot],
                        "content_relative": float(content[event_slot, role_slot]),
                        "max_surface_relative": float(control[event_slot, role_slot]),
                        "content_surface_ratio": float(ratio[event_slot, role_slot]),
                    }
                )
            residual_curve = []
            for event_slot, event in enumerate(events):
                if event["component"] == "residual":
                    role_slot = role_index["generation_boundary"]
                    residual_curve.append(
                        {
                            "depth": event["depth"],
                            "content_relative": float(content[event_slot, role_slot]),
                            "max_surface_relative": float(control[event_slot, role_slot]),
                            "ratio": float(ratio[event_slot, role_slot]),
                        }
                    )
            summaries[f"{scope}|{split}"] = {
                "pair_count": len(selected),
                "top_content_surface_events": top,
                "generation_boundary_residual_curve": residual_curve,
            }
    result: dict[str, Any] = {
        "phase": PHASE,
        "projection_is_random_signed_camera_not_raw_coordinate": True,
        "trajectory_is_descriptive_not_causal": True,
        "scope_split_summaries": summaries,
    }
    result["trajectory_digest"] = digest(result)
    return result


def analyze() -> None:
    protocol, pairs, states = verify_formal_inputs()
    patch_records = read_jsonl(PATCH_PATH)
    summary = read_json(RUN_SUMMARY_PATH)
    selection = read_json(SELECTION_PATH)
    if summary["patch_digest"] != digest(patch_records) or summary["selection_digest"] != selection["selection_digest"]:
        raise RuntimeError("run output digest mismatch")
    arrays = np.load(ARRAY_PATH, allow_pickle=False)
    trajectory = trajectory_summary(protocol, pairs, states, arrays)
    write_json(TRAJECTORY_PATH, trajectory)
    scope_results: dict[str, Any] = {}
    physical_scopes: list[str] = []
    for scope in protocol["upstream"]["authorized_scopes"]:
        discovery = selection["scope_selections"][scope]
        holdouts: dict[str, Any] = {}
        for split in HOLDOUT_SPLITS:
            selected = [row for row in patch_records if row["scope"] == scope and row["split"] == split]
            holdouts[split] = holdout_metrics(selected) if selected else None
        physical = bool(discovery["discovery_authorized"]) and all(
            holdouts[split] is not None and holdouts[split]["passed"] for split in HOLDOUT_SPLITS
        )
        if physical:
            physical_scopes.append(scope)
        scope_results[scope] = {
            "selected_depth": discovery["selected_depth"],
            "discovery_authorized": discovery["discovery_authorized"],
            "holdouts": holdouts,
            "physical_scope_closed": physical,
        }
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "physical_scopes_closed" if physical_scopes else "no_physical_scope_closed",
        "protocol_digest": protocol["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "run_summary_digest": summary["summary_digest"],
        "trajectory_digest": trajectory["trajectory_digest"],
        "scope_results": scope_results,
        "physical_scopes": physical_scopes,
        "k_item": {
            "identifier": "K200",
            "evidence_grade": "E3-CAUSAL" if physical_scopes else "E3-NEGATIVE-BOUNDARY",
            "statement": (
                f"Whole-residual generation-boundary handoff closed on {physical_scopes}."
                if physical_scopes
                else "No Phase1222-passed atomic scope met the frozen discovery plus three-holdout causal handoff gate."
            ),
            "scope": "Qwen3 FP16; generated behavior-correct subset; full component trajectory; whole-residual patch only",
        },
        "authorized_next": {
            "automatic_execution": False,
            "reason": "Phase1224 minimal-state and semantic-intervention contracts were not frozen before Phase1223 reveal",
            "candidate_scopes": physical_scopes,
            "head_or_neuron_search": False,
            "cross_model": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def selftest() -> None:
    pairs, states, audit = build_material()
    assert len(pairs) == len(audit["authorized_scopes"]) * len(SPLITS) * PAIRS_PER_SCOPE_SPLIT
    assert len(states) == len(pairs) * len(PANELS)
    assert len(event_registry()) == 1 + LAYER_COUNT * 6
    assert all(pair["recipient_gold"] != pair["donor_gold"] for pair in pairs)
    assert all(set(pair["panel_states"]) == set(PANELS) for pair in pairs)
    print(canonical_json({"status": "selftest_passed", "pairs": len(pairs), "states": len(states), "events": len(event_registry())}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("selftest", "materialize", "run", "analyze"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "materialize":
        materialize()
    elif args.stage == "run":
        run()
    elif args.stage == "analyze":
        analyze()


if __name__ == "__main__":
    main()
