#!/usr/bin/env python3
"""Phase 1296: frozen multi-event residual response path for C030."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
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

from model_utils import MODEL_CONFIGS  # noqa: E402
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16  # noqa: E402


PHASE = 1296
CAMPAIGN = "C030"
SCRIPT = Path(__file__).resolve()
AUDITOR = TEST_ROOT / "phase1296_c030_multievent_response_path_audit.py"
PARENT = TEST_ROOT / "result/phase1295_c030_qwen3_grounded_lookup_behavior"
PARENT_PROTOCOL = PARENT / "protocol/preregistration.json"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
MATERIAL = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract/material/frozen_grounded_lookup_cases.jsonl"
MATERIAL_CONTRACT = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract/protocol/preregistration.json"
OUT = TEST_ROOT / "result/phase1296_c030_multievent_response_path"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_pair_event_manifest.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
ARRAYS = OUT / "raw/residual_response_arrays.npz"
RUN_META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/response_path_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

SYSTEM_PROMPT = "Use only the supplied catalog. Reply exactly as requested and do not explain."
PARTITIONS = ("discovery", "confirmation", "holdout")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SURFACES = ("catalog_prose", "inventory_ledger")
ROLES = ("record_slot0_entity", "record_slot0_value", "query_value", "answer_boundary")
PRIMARY_ROLES = ("query_value", "answer_boundary")
DEPTHS = tuple(range(37))
PAIR_BATCH_SIZE = 4
EPS = 1e-12
THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "behavior_replay_accuracy_min": 0.99,
    "record_entity_active_relative_max": 1e-7,
    "record_value_active_null_max_abs_difference": 1e-7,
    "discovery_active_relative_median_min": 0.001,
    "discovery_active_to_max_control_ratio_min": 1.25,
    "discovery_active_over_controls_fraction_min": 0.75,
    "discovery_adjacent_depths_min": 2,
    "transfer_active_relative_median_min": 0.001,
    "transfer_active_to_max_control_ratio_min": 1.15,
    "transfer_active_over_controls_fraction_min": 0.70,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    os.replace(tmp, path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def token_positions(offsets: list[tuple[int, int]], left: int, right: int) -> list[int]:
    positions = [index for index, (start, end) in enumerate(offsets) if end > left and start < right and end > start]
    if not positions:
        raise RuntimeError(f"no token overlaps {left}:{right}")
    return positions


def role_manifest(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    rendered = render(tokenizer, row["candidate_prompt"])
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    prompt_start = rendered.find(row["candidate_prompt"])
    if prompt_start < 0 or rendered.find(row["candidate_prompt"], prompt_start + 1) >= 0:
        raise RuntimeError(f"prompt embedding not unique: {row['case_id']}")
    first_record = row["typed_spans"]["records"][0]
    query_span = row["typed_spans"]["query"][0]
    query_value_spans = [
        span for span in row["typed_spans"]["query_value"]
        if span[0] >= query_span[0] and span[1] <= query_span[1]
    ]
    if len(first_record["entity_spans"]) != 1 or len(first_record["queried_attribute_value_spans"]) != 1 or len(query_value_spans) != 1:
        raise RuntimeError(f"typed span multiplicity: {row['case_id']}")
    char_spans = {
        "record_slot0_entity": first_record["entity_spans"][0],
        "record_slot0_value": first_record["queried_attribute_value_spans"][0],
        "query_value": query_value_spans[0],
    }
    positions = {}
    span_audit = {}
    for role, span in char_spans.items():
        left, right = prompt_start + span[0], prompt_start + span[1]
        overlap = token_positions(offsets, left, right)
        positions[role] = overlap[-1]
        span_audit[role] = {"character_span": [left, right], "token_span": overlap, "selected_position": overlap[-1]}
    positions["answer_boundary"] = len(ids) - 1
    span_audit["answer_boundary"] = {"token_span": [len(ids) - 1], "selected_position": len(ids) - 1}
    candidate_ids = []
    for candidate in row["candidates"]:
        full = tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            raise RuntimeError(f"candidate token drift: {row['case_id']} {candidate}")
        candidate_ids.append(int(full[-1]))
    return {
        "case_id": row["case_id"],
        "input_length": len(ids),
        "input_ids_digest": digest(ids),
        "positions": positions,
        "span_audit": span_audit,
        "candidate_token_ids": candidate_ids,
    }


def build_manifest(tokenizer: Any) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(MATERIAL) if row["candidate_order"] == 0]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    output = []
    for group_id in sorted(by_group):
        pair = sorted(by_group[group_id], key=lambda row: row["binding_state"])
        if len(pair) != 2 or [row["binding_state"] for row in pair] != [0, 1]:
            raise RuntimeError(f"bad pair: {group_id}")
        output.append({
            "group_id": group_id,
            "partition": pair[0]["partition"],
            "profile_index": pair[0]["profile_index"],
            "attribute": pair[0]["attribute"],
            "panel": pair[0]["panel"],
            "surface": pair[0]["surface"],
            "candidate_order": 0,
            "states": [role_manifest(tokenizer, row) for row in pair],
        })
    if len(output) != 1152:
        raise RuntimeError(f"expected 1152 pairs, got {len(output)}")
    return output


def preregister(force: bool) -> None:
    if OUT.exists() and not force:
        raise RuntimeError(f"{OUT} already exists")
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    parent_final = load(PARENT_FINAL)
    parent_audit = load(PARENT_AUDIT)
    if parent_final.get("authorization") != "phase1296_multievent_response_preregistration_only":
        raise RuntimeError("Phase1295 authorization missing")
    if parent_audit.get("scientific_authorization") != "phase1296_multievent_response_preregistration_only":
        raise RuntimeError("Phase1295 independent authorization missing")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True, local_files_only=True, use_fast=True
    )
    manifest = build_manifest(tokenizer)
    write_jsonl(MANIFEST, manifest)
    timeless = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "experiment_id": "EXP-C030-WP02-001",
        "schema_version": "phase1296.c030.multievent.v1",
        "research_object": "binding-state residual response path conditional on query relevance",
        "model": "qwen3-4b-fp16-cuda-no-quantization",
        "formal_run_budget": 1,
        "material": {
            "source_sha256": sha(MATERIAL),
            "candidate_order": 0,
            "pair_count": len(manifest),
            "case_count": len(manifest) * 2,
            "partition_pair_count": {partition: sum(row["partition"] == partition for row in manifest) for partition in PARTITIONS},
            "manifest_sha256": sha(MANIFEST),
        },
        "events": {
            "roles": list(ROLES),
            "primary_roles": list(PRIMARY_ROLES),
            "depths": list(DEPTHS),
            "state": "whole residual stream at the last token overlapping each frozen role",
            "record_roles_are_fixed_slot_zero_not_selected_by_gold": True,
            "answer_boundary": "last native-chat input token before first answer token",
        },
        "response": {
            "delta": "h(binding_state=1)-h(binding_state=0)",
            "relative_distance": "L2(delta)/(0.5*(L2(h0)+L2(h1))+1e-12)",
            "active_specificity": "active relative distance compared with max of three matched controls for same partition/profile/attribute/surface",
            "no_direction_or_identity_gate": True,
        },
        "selection_and_transfer": {
            "discovery": "for each primary role choose earliest two-adjacent-depth band whose every depth passes all discovery thresholds",
            "role_priority": list(PRIMARY_ROLES),
            "confirmation": "evaluate frozen role-depth bands without reselection",
            "holdout": "evaluate frozen role-depth bands without reselection",
            "both_primary_roles_required": True,
        },
        "instrument_identities": {
            "record_slot0_entity": "active state pair has identical causal prefix through this token and must have near-zero response",
            "record_slot0_value": "active and matched-null prompts are identical through this token, so their response must match",
        },
        "thresholds": THRESHOLDS,
        "batching": {"pairs_per_batch": PAIR_BATCH_SIZE, "output_hidden_states": True, "hidden_dtype": "FP16 model; norms accumulated FP32"},
        "unblinding_order": [
            "save all pair x depth x role response arrays",
            "hash arrays and run metadata",
            "evaluate instrument identities",
            "select discovery bands independently for both primary roles",
            "evaluate frozen bands on confirmation and holdout",
            "write authorization without changing any event, threshold, or branch",
        ],
        "dependencies": {
            "phase1295_protocol": sha(PARENT_PROTOCOL),
            "phase1295_final": sha(PARENT_FINAL),
            "phase1295_audit": sha(PARENT_AUDIT),
            "phase1294_material_contract": sha(MATERIAL_CONTRACT),
        },
        "source_hashes": {"main": sha(SCRIPT), "auditor": sha(AUDITOR)},
        "authorization_if_pass": "phase1297_path_cut_and_independent_rescue_preregistration_only",
        "authorization_if_fail": "close_c030_without_path_claim",
        "claims_forbidden": [
            "A passing norm path is not a fixed semantic direction or minimal circuit.",
            "No causality, necessity, component identity, or cross-model conservation is claimed in Phase1296.",
            "A failure does not imply object-attribute information is absent elsewhere in the model.",
        ],
        "hard_stops": [
            "No event, role alignment, threshold, split, or statistic changes after preregistration.",
            "Either primary role discovery or transfer failure closes C030 without causal continuation.",
            "No head or MLP hotspot scan is authorized.",
        ],
        "model_weights_loaded": False,
    }
    frozen = {**timeless, "created_at_utc": utc_now(), "protocol_digest": digest(timeless)}
    save(PROTOCOL, frozen)
    save(ENVIRONMENT, {
        "created_at_utc": utc_now(), "python": sys.version, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_weights_loaded": False, "tokenizer_only": True,
    })
    print(canonical({"status": "preregistered", "pairs": len(manifest), "protocol_digest": frozen["protocol_digest"]}))


def prepare_pair(tokenizer: Any, row_index: dict[str, dict[str, Any]], pair: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for state in pair["states"]:
        row = row_index[state["case_id"]]
        rendered = render(tokenizer, row["candidate_prompt"])
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if digest(ids) != state["input_ids_digest"] or len(ids) != state["input_length"]:
            raise RuntimeError(f"input drift: {state['case_id']}")
        output.append({"ids": ids, "state": state, "row": row})
    return output


def response_cell(relative: np.ndarray, meta: list[dict[str, Any]], partition: str, role_index: int, depth: int) -> dict[str, float]:
    lookup = {
        (row["profile_index"], row["attribute"], row["surface"], row["panel"]): index
        for index, row in enumerate(meta) if row["partition"] == partition
    }
    active_values = []
    controls = {panel: [] for panel in PANELS if panel != "active"}
    wins = []
    for profile in range(8):
        for attribute in ("color", "material", "location", "size", "shape", "status"):
            for surface in SURFACES:
                active = float(relative[lookup[(profile, attribute, surface, "active")], depth, role_index])
                control_values = {
                    panel: float(relative[lookup[(profile, attribute, surface, panel)], depth, role_index])
                    for panel in controls
                }
                active_values.append(active)
                for panel, value in control_values.items():
                    controls[panel].append(value)
                wins.append(active > max(control_values.values()))
    medians = {panel: float(np.median(values)) for panel, values in controls.items()}
    active_median = float(np.median(active_values))
    max_control = max(medians.values())
    return {
        "active_median": active_median,
        "matched_null_median": medians["matched_null"],
        "surface_only_median": medians["surface_only"],
        "semantic_neighbor_median": medians["semantic_neighbor"],
        "max_control_median": max_control,
        "active_to_max_control_ratio": active_median / (max_control + EPS),
        "active_over_all_controls_fraction": float(np.mean(wins)),
    }


def passes_cell(cell: dict[str, float], discovery: bool) -> bool:
    prefix = "discovery" if discovery else "transfer"
    return (
        cell["active_median"] >= THRESHOLDS[f"{prefix}_active_relative_median_min"]
        and cell["active_to_max_control_ratio"] >= THRESHOLDS[f"{prefix}_active_to_max_control_ratio_min"]
        and cell["active_over_all_controls_fraction"] >= THRESHOLDS[f"{prefix}_active_over_controls_fraction_min"]
    )


def analyze(relative: np.ndarray, meta: list[dict[str, Any]], behavior_correct: np.ndarray) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for partition in PARTITIONS:
        tables[partition] = {}
        for role_index, role in enumerate(ROLES):
            tables[partition][role] = [response_cell(relative, meta, partition, role_index, depth) for depth in DEPTHS]

    selected = {}
    discovery_pass = {}
    for role in PRIMARY_ROLES:
        eligible = [passes_cell(tables["discovery"][role][depth], discovery=True) for depth in DEPTHS]
        start = next((depth for depth in range(len(DEPTHS) - 1) if eligible[depth] and eligible[depth + 1]), None)
        selected[role] = [] if start is None else [start, start + 1]
        discovery_pass[role] = start is not None

    transfer = {}
    for partition in ("confirmation", "holdout"):
        transfer[partition] = {}
        for role in PRIMARY_ROLES:
            depths = selected[role]
            cells = [tables[partition][role][depth] for depth in depths]
            transfer[partition][role] = {
                "depths": depths,
                "cells": cells,
                "passed": bool(depths) and all(passes_cell(cell, discovery=False) for cell in cells),
            }

    active_indices = [index for index, row in enumerate(meta) if row["panel"] == "active"]
    entity_role = ROLES.index("record_slot0_entity")
    entity_max = float(np.max(relative[active_indices, :, entity_role]))
    lookup = {(row["partition"], row["profile_index"], row["attribute"], row["surface"], row["panel"]): index for index, row in enumerate(meta)}
    record_role = ROLES.index("record_slot0_value")
    prefix_differences = []
    for partition in PARTITIONS:
        for profile in range(8):
            for attribute in ("color", "material", "location", "size", "shape", "status"):
                for surface in SURFACES:
                    active = lookup[(partition, profile, attribute, surface, "active")]
                    null = lookup[(partition, profile, attribute, surface, "matched_null")]
                    prefix_differences.extend(np.abs(relative[active, :, record_role] - relative[null, :, record_role]).tolist())
    record_value_diff_max = float(max(prefix_differences))
    finite_fraction = float(np.isfinite(relative).mean())
    behavior_accuracy = float(np.mean(behavior_correct))
    gates = {
        "finite": finite_fraction >= THRESHOLDS["finite_fraction_min"],
        "behavior_replay": behavior_accuracy >= THRESHOLDS["behavior_replay_accuracy_min"],
        "record_entity_identity": entity_max <= THRESHOLDS["record_entity_active_relative_max"],
        "record_value_active_null_identity": record_value_diff_max <= THRESHOLDS["record_value_active_null_max_abs_difference"],
        "discovery_query_value": discovery_pass["query_value"],
        "discovery_answer_boundary": discovery_pass["answer_boundary"],
        "confirmation_query_value": transfer["confirmation"]["query_value"]["passed"],
        "confirmation_answer_boundary": transfer["confirmation"]["answer_boundary"]["passed"],
        "holdout_query_value": transfer["holdout"]["query_value"]["passed"],
        "holdout_answer_boundary": transfer["holdout"]["answer_boundary"]["passed"],
    }
    return {
        "finite_fraction": finite_fraction,
        "behavior_replay_accuracy": behavior_accuracy,
        "instrument_identities": {
            "record_entity_active_relative_max": entity_max,
            "record_value_active_null_max_abs_difference": record_value_diff_max,
        },
        "selected_discovery_bands": selected,
        "discovery_pass": discovery_pass,
        "transfer": transfer,
        "response_tables": tables,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


@torch.inference_mode()
def run() -> None:
    protocol = load(PROTOCOL)
    preaudit = load(PREAUDIT)
    if preaudit.get("authorization") != "run_phase1296_once" or not preaudit.get("all_checks_passed"):
        raise RuntimeError("Phase1296 preaudit authorization missing")
    if any(path.exists() for path in (ARRAYS, RUN_META, SUMMARY, FINAL, COMPLETE)):
        raise RuntimeError("formal run already consumed or partial output exists")
    manifest = read_jsonl(MANIFEST)
    row_index = {row["case_id"]: row for row in read_jsonl(MATERIAL)}
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        qa = quantization_audit(model)
        if qa["has_quantized_modules"] or not qa["has_fp16_parameters"]:
            raise RuntimeError(f"FP16 qualification failed: {qa}")
        relative = np.empty((len(manifest), len(DEPTHS), len(ROLES)), dtype=np.float32)
        delta_norm = np.empty_like(relative)
        state_norm_mean = np.empty_like(relative)
        behavior_correct = np.empty((len(manifest), 2), dtype=np.bool_)
        behavior_margin = np.empty((len(manifest), 2), dtype=np.float32)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        supports_last = "logits_to_keep" in inspect.signature(model.forward).parameters
        for start in range(0, len(manifest), PAIR_BATCH_SIZE):
            batch_pairs = manifest[start:start + PAIR_BATCH_SIZE]
            examples = [item for pair in batch_pairs for item in prepare_pair(tokenizer, row_index, pair)]
            maximum = max(len(item["ids"]) for item in examples)
            ids = torch.full((len(examples), maximum), int(pad_id), dtype=torch.long, device=device)
            mask = torch.zeros((len(examples), maximum), dtype=torch.long, device=device)
            offsets = []
            for index, item in enumerate(examples):
                values = item["ids"]
                offset = maximum - len(values)
                offsets.append(offset)
                ids[index, offset:] = torch.tensor(values, dtype=torch.long, device=device)
                mask[index, offset:] = 1
            kwargs = {
                "input_ids": ids, "attention_mask": mask, "use_cache": False,
                "return_dict": True, "output_hidden_states": True,
            }
            if supports_last:
                kwargs["logits_to_keep"] = 1
            result = model(**kwargs)
            if len(result.hidden_states) != len(DEPTHS):
                raise RuntimeError(f"hidden depth drift: {len(result.hidden_states)}")
            last_logits = result.logits[:, -1, :].float()
            for local_pair, pair in enumerate(batch_pairs):
                global_pair = start + local_pair
                first_index, second_index = 2 * local_pair, 2 * local_pair + 1
                for state_index, example_index in enumerate((first_index, second_index)):
                    item = examples[example_index]
                    candidate_ids = item["state"]["candidate_token_ids"]
                    scores = last_logits[example_index, candidate_ids]
                    predicted = int(torch.argmax(scores).item())
                    gold_position = item["row"]["gold_position"]
                    behavior_correct[global_pair, state_index] = predicted == gold_position
                    gold_score = scores[gold_position]
                    other = torch.max(torch.cat([scores[:gold_position], scores[gold_position + 1:]]))
                    behavior_margin[global_pair, state_index] = float((gold_score - other).item())
                for depth, hidden in enumerate(result.hidden_states):
                    for role_index, role in enumerate(ROLES):
                        p0 = offsets[first_index] + examples[first_index]["state"]["positions"][role]
                        p1 = offsets[second_index] + examples[second_index]["state"]["positions"][role]
                        h0 = hidden[first_index, p0].float()
                        h1 = hidden[second_index, p1].float()
                        dnorm = torch.linalg.vector_norm(h1 - h0)
                        base = 0.5 * (torch.linalg.vector_norm(h0) + torch.linalg.vector_norm(h1))
                        delta_norm[global_pair, depth, role_index] = float(dnorm.item())
                        state_norm_mean[global_pair, depth, role_index] = float(base.item())
                        relative[global_pair, depth, role_index] = float((dnorm / (base + EPS)).item())
            del result
            if (start // PAIR_BATCH_SIZE + 1) % 25 == 0:
                print(canonical({"pairs_processed": min(start + PAIR_BATCH_SIZE, len(manifest)), "total": len(manifest)}), flush=True)
        save_npz(
            ARRAYS,
            relative_distance=relative,
            delta_norm=delta_norm,
            state_norm_mean=state_norm_mean,
            behavior_correct=behavior_correct,
            behavior_margin=behavior_margin,
            depths=np.asarray(DEPTHS, dtype=np.int16),
            roles=np.asarray(ROLES),
        )
        meta = [{key: pair[key] for key in ("group_id", "partition", "profile_index", "attribute", "panel", "surface")} for pair in manifest]
        analysis = analyze(relative, meta, behavior_correct)
        authorization = "phase1297_path_cut_and_independent_rescue_preregistration_only" if analysis["all_gates_passed"] else "close_c030_without_path_claim"
        save(RUN_META, {
            "phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"],
            "array_sha256": sha(ARRAYS), "manifest_sha256": sha(MANIFEST), "model_audit": qa,
            "placement": placement, "runtime_seconds": time.time() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "pair_metadata": meta,
        })
        save(SUMMARY, {**analysis, "phase": PHASE, "campaign": CAMPAIGN, "protocol_digest": protocol["protocol_digest"], "authorization": authorization})
        save(FINAL, {
            "phase": PHASE, "campaign": CAMPAIGN,
            "verdict": "multievent_response_path_qualified" if analysis["all_gates_passed"] else "multievent_response_path_gate_failed",
            "protocol_digest": protocol["protocol_digest"], "array_sha256": sha(ARRAYS),
            "all_gates_passed": analysis["all_gates_passed"], "selected_discovery_bands": analysis["selected_discovery_bands"],
            "authorization": authorization, "causal_intervention_performed": False,
        })
        save(COMPLETE, {"completed_at_utc": utc_now(), "formal_runs_consumed": 1, "protocol_digest": protocol["protocol_digest"]})
        print(canonical({"verdict": load(FINAL)["verdict"], "selected": analysis["selected_discovery_bands"], "authorization": authorization}))
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    preregister(args.force) if args.command == "preregister" else run()


if __name__ == "__main__":
    main()
