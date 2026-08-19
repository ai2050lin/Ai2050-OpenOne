#!/usr/bin/env python3
"""Independent pre/post audit for Phase 1296/C030 multi-event response."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN = TEST_ROOT / "phase1296_c030_multievent_response_path.py"
AUDITOR = Path(__file__).resolve()
PARENT = TEST_ROOT / "result/phase1295_c030_qwen3_grounded_lookup_behavior"
PARENT_PROTOCOL = PARENT / "protocol/preregistration.json"
PARENT_FINAL = PARENT / "analysis/final.json"
PARENT_AUDIT = PARENT / "audit/independent_final_audit.json"
MATERIAL = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract/material/frozen_grounded_lookup_cases.jsonl"
MATERIAL_CONTRACT = TEST_ROOT / "result/phase1294_c030_grounded_lookup_contract/protocol/preregistration.json"
OUT = TEST_ROOT / "result/phase1296_c030_multievent_response_path"
PROTOCOL = OUT / "protocol/preregistration.json"
MANIFEST = OUT / "protocol/frozen_pair_event_manifest.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
POSTAUDIT = OUT / "audit/independent_final_audit.json"
ARRAYS = OUT / "raw/residual_response_arrays.npz"
RUN_META = OUT / "raw/run_metadata.json"
SUMMARY = OUT / "analysis/response_path_summary.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"

SYSTEM_PROMPT = "Use only the supplied catalog. Reply exactly as requested and do not explain."
MODEL_PATH = ROOT / "models/hf/qwen3-4b"
PARTITIONS = ("discovery", "confirmation", "holdout")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
SURFACES = ("catalog_prose", "inventory_ledger")
ATTRIBUTES = ("color", "material", "location", "size", "shape", "status")
ROLES = ("record_slot0_entity", "record_slot0_value", "query_value", "answer_boundary")
PRIMARY_ROLES = ("query_value", "answer_boundary")
DEPTHS = tuple(range(37))
EPS = 1e-12
EXPECTED_THRESHOLDS = {
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def render(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def overlap(offsets: list[tuple[int, int]], left: int, right: int) -> list[int]:
    values = [index for index, (start, end) in enumerate(offsets) if end > left and start < right and end > start]
    if not values:
        raise RuntimeError(f"no token overlap {left}:{right}")
    return values


def recompute_state(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    rendered = render(tokenizer, row["candidate_prompt"])
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(left), int(right)) for left, right in encoded["offset_mapping"]]
    prompt_start = rendered.find(row["candidate_prompt"])
    first = row["typed_spans"]["records"][0]
    query = row["typed_spans"]["query"][0]
    query_values = [span for span in row["typed_spans"]["query_value"] if span[0] >= query[0] and span[1] <= query[1]]
    if prompt_start < 0 or len(first["entity_spans"]) != 1 or len(first["queried_attribute_value_spans"]) != 1 or len(query_values) != 1:
        raise RuntimeError(f"span failure {row['case_id']}")
    spans = {
        "record_slot0_entity": first["entity_spans"][0],
        "record_slot0_value": first["queried_attribute_value_spans"][0],
        "query_value": query_values[0],
    }
    positions = {}
    span_audit = {}
    for role, span in spans.items():
        left, right = prompt_start + span[0], prompt_start + span[1]
        tokens = overlap(offsets, left, right)
        positions[role] = tokens[-1]
        span_audit[role] = {"character_span": [left, right], "token_span": tokens, "selected_position": tokens[-1]}
    positions["answer_boundary"] = len(ids) - 1
    span_audit["answer_boundary"] = {"token_span": [len(ids) - 1], "selected_position": len(ids) - 1}
    candidate_ids = []
    for candidate in row["candidates"]:
        full = tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)
        if full[:len(ids)] != ids or len(full) != len(ids) + 1:
            raise RuntimeError(f"candidate drift {row['case_id']}")
        candidate_ids.append(int(full[-1]))
    return {
        "case_id": row["case_id"], "input_length": len(ids), "input_ids_digest": digest(ids),
        "positions": positions, "span_audit": span_audit, "candidate_token_ids": candidate_ids,
    }


def base_checks(protocol: dict[str, Any], verify_manifest: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    parent_final = load(PARENT_FINAL)
    parent_audit = load(PARENT_AUDIT)
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest_recomputes", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes_match", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(AUDITOR)}, protocol["source_hashes"])
    add(checks, "phase_campaign_exact", (protocol["phase"], protocol["campaign"]) == (1296, "C030"), [protocol["phase"], protocol["campaign"]])
    add(checks, "parent_authorization", parent_final.get("authorization") == "phase1296_multievent_response_preregistration_only" and parent_audit.get("scientific_authorization") == "phase1296_multievent_response_preregistration_only", parent_final)
    add(checks, "dependency_hashes", protocol["dependencies"] == {
        "phase1295_protocol": sha(PARENT_PROTOCOL), "phase1295_final": sha(PARENT_FINAL),
        "phase1295_audit": sha(PARENT_AUDIT), "phase1294_material_contract": sha(MATERIAL_CONTRACT),
    }, protocol["dependencies"])
    add(checks, "material_and_manifest_hashes", protocol["material"]["source_sha256"] == sha(MATERIAL) and protocol["material"]["manifest_sha256"] == sha(MANIFEST), protocol["material"])
    add(checks, "event_registry_exact", protocol["events"]["roles"] == list(ROLES) and protocol["events"]["primary_roles"] == list(PRIMARY_ROLES) and protocol["events"]["depths"] == list(DEPTHS) and protocol["events"]["record_roles_are_fixed_slot_zero_not_selected_by_gold"] is True, protocol["events"])
    add(checks, "thresholds_exact", protocol["thresholds"] == EXPECTED_THRESHOLDS, protocol["thresholds"])
    add(checks, "single_fp16_run", protocol["model"] == "qwen3-4b-fp16-cuda-no-quantization" and protocol["formal_run_budget"] == 1, [protocol["model"], protocol["formal_run_budget"]])
    add(checks, "both_primary_roles_required", protocol["selection_and_transfer"]["both_primary_roles_required"] is True, protocol["selection_and_transfer"])
    add(checks, "failure_closes_and_no_hotspot", protocol["authorization_if_fail"] == "close_c030_without_path_claim" and any("No head or MLP" in rule for rule in protocol["hard_stops"]), protocol["hard_stops"])

    manifest = read_jsonl(MANIFEST)
    add(checks, "manifest_count_and_dimensions", len(manifest) == 1152 and CounterLike(manifest) == {"discovery": 384, "confirmation": 384, "holdout": 384}, CounterLike(manifest))
    add(checks, "manifest_pairs_complete", len({row["group_id"] for row in manifest}) == 1152 and all(len(row["states"]) == 2 for row in manifest), len(manifest))
    if verify_manifest:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=True)
        material = {row["case_id"]: row for row in read_jsonl(MATERIAL)}
        exact = True
        for pair in manifest:
            for frozen in pair["states"]:
                exact &= recompute_state(tokenizer, material[frozen["case_id"]]) == frozen
        add(checks, "independent_event_manifest_replay", exact, "2304 states")
        by_key = {(row["partition"], row["profile_index"], row["attribute"], row["surface"], row["panel"]): row for row in manifest}
        prefix_ok = True
        for partition in PARTITIONS:
            for profile in range(8):
                for attribute in ATTRIBUTES:
                    for surface in SURFACES:
                        active = by_key[(partition, profile, attribute, surface, "active")]
                        null = by_key[(partition, profile, attribute, surface, "matched_null")]
                        for state_index in (0, 1):
                            arow = material[active["states"][state_index]["case_id"]]
                            nrow = material[null["states"][state_index]["case_id"]]
                            aend = arow["typed_spans"]["records"][0]["queried_attribute_value_spans"][0][1]
                            nend = nrow["typed_spans"]["records"][0]["queried_attribute_value_spans"][0][1]
                            prefix_ok &= arow["candidate_prompt"][:aend] == nrow["candidate_prompt"][:nend]
        add(checks, "record_value_causal_prefix_identity", prefix_ok, "active equals matched-null through role")
    return checks


def CounterLike(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {partition: sum(row["partition"] == partition for row in rows) for partition in PARTITIONS}


def preaudit() -> None:
    protocol = load(PROTOCOL)
    checks = base_checks(protocol, verify_manifest=True)
    clear = not any(path.exists() for path in (ARRAYS, RUN_META, SUMMARY, FINAL, COMPLETE))
    add(checks, "formal_run_not_started", clear, clear)
    passed = all(check["passed"] for check in checks)
    result = {
        "phase": 1296, "campaign": "C030", "audit_stage": "pre_model",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
        "checks": checks, "passed_count": sum(check["passed"] for check in checks), "total_count": len(checks),
        "all_checks_passed": passed, "authorization": "run_phase1296_once" if passed else "none",
        "protocol_digest": protocol["protocol_digest"],
    }
    save(PREAUDIT, result)
    print(canonical({"stage": "preaudit", "passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


def response_cell(relative: np.ndarray, meta: list[dict[str, Any]], partition: str, role_index: int, depth: int) -> dict[str, float]:
    lookup = {(row["profile_index"], row["attribute"], row["surface"], row["panel"]): index for index, row in enumerate(meta) if row["partition"] == partition}
    active_values = []
    controls = {panel: [] for panel in PANELS if panel != "active"}
    wins = []
    for profile in range(8):
        for attribute in ATTRIBUTES:
            for surface in SURFACES:
                active = float(relative[lookup[(profile, attribute, surface, "active")], depth, role_index])
                control_values = {panel: float(relative[lookup[(profile, attribute, surface, panel)], depth, role_index]) for panel in controls}
                active_values.append(active)
                for panel, value in control_values.items():
                    controls[panel].append(value)
                wins.append(active > max(control_values.values()))
    medians = {panel: float(np.median(values)) for panel, values in controls.items()}
    active_median = float(np.median(active_values))
    max_control = max(medians.values())
    return {
        "active_median": active_median, "matched_null_median": medians["matched_null"],
        "surface_only_median": medians["surface_only"], "semantic_neighbor_median": medians["semantic_neighbor"],
        "max_control_median": max_control, "active_to_max_control_ratio": active_median / (max_control + EPS),
        "active_over_all_controls_fraction": float(np.mean(wins)),
    }


def passes_cell(cell: dict[str, float], discovery: bool) -> bool:
    prefix = "discovery" if discovery else "transfer"
    return (
        cell["active_median"] >= EXPECTED_THRESHOLDS[f"{prefix}_active_relative_median_min"]
        and cell["active_to_max_control_ratio"] >= EXPECTED_THRESHOLDS[f"{prefix}_active_to_max_control_ratio_min"]
        and cell["active_over_all_controls_fraction"] >= EXPECTED_THRESHOLDS[f"{prefix}_active_over_controls_fraction_min"]
    )


def recompute_analysis(relative: np.ndarray, meta: list[dict[str, Any]], behavior_correct: np.ndarray) -> dict[str, Any]:
    tables = {partition: {role: [response_cell(relative, meta, partition, role_index, depth) for depth in DEPTHS] for role_index, role in enumerate(ROLES)} for partition in PARTITIONS}
    selected = {}
    discovery_pass = {}
    for role in PRIMARY_ROLES:
        eligible = [passes_cell(tables["discovery"][role][depth], True) for depth in DEPTHS]
        start = next((depth for depth in range(36) if eligible[depth] and eligible[depth + 1]), None)
        selected[role] = [] if start is None else [start, start + 1]
        discovery_pass[role] = start is not None
    transfer = {}
    for partition in ("confirmation", "holdout"):
        transfer[partition] = {}
        for role in PRIMARY_ROLES:
            depths = selected[role]
            cells = [tables[partition][role][depth] for depth in depths]
            transfer[partition][role] = {"depths": depths, "cells": cells, "passed": bool(depths) and all(passes_cell(cell, False) for cell in cells)}
    active_indices = [index for index, row in enumerate(meta) if row["panel"] == "active"]
    entity_max = float(np.max(relative[active_indices, :, ROLES.index("record_slot0_entity")]))
    lookup = {(row["partition"], row["profile_index"], row["attribute"], row["surface"], row["panel"]): index for index, row in enumerate(meta)}
    differences = []
    for partition in PARTITIONS:
        for profile in range(8):
            for attribute in ATTRIBUTES:
                for surface in SURFACES:
                    a = lookup[(partition, profile, attribute, surface, "active")]
                    n = lookup[(partition, profile, attribute, surface, "matched_null")]
                    differences.extend(np.abs(relative[a, :, ROLES.index("record_slot0_value")] - relative[n, :, ROLES.index("record_slot0_value")]).tolist())
    value_diff = float(max(differences))
    finite = float(np.isfinite(relative).mean())
    behavior = float(np.mean(behavior_correct))
    gates = {
        "finite": finite >= EXPECTED_THRESHOLDS["finite_fraction_min"],
        "behavior_replay": behavior >= EXPECTED_THRESHOLDS["behavior_replay_accuracy_min"],
        "record_entity_identity": entity_max <= EXPECTED_THRESHOLDS["record_entity_active_relative_max"],
        "record_value_active_null_identity": value_diff <= EXPECTED_THRESHOLDS["record_value_active_null_max_abs_difference"],
        "discovery_query_value": discovery_pass["query_value"],
        "discovery_answer_boundary": discovery_pass["answer_boundary"],
        "confirmation_query_value": transfer["confirmation"]["query_value"]["passed"],
        "confirmation_answer_boundary": transfer["confirmation"]["answer_boundary"]["passed"],
        "holdout_query_value": transfer["holdout"]["query_value"]["passed"],
        "holdout_answer_boundary": transfer["holdout"]["answer_boundary"]["passed"],
    }
    return {
        "finite_fraction": finite, "behavior_replay_accuracy": behavior,
        "instrument_identities": {"record_entity_active_relative_max": entity_max, "record_value_active_null_max_abs_difference": value_diff},
        "selected_discovery_bands": selected, "discovery_pass": discovery_pass, "transfer": transfer,
        "response_tables": tables, "gates": gates, "all_gates_passed": all(gates.values()),
    }


def postaudit() -> None:
    protocol = load(PROTOCOL)
    checks = base_checks(protocol, verify_manifest=False)
    arrays = np.load(ARRAYS, allow_pickle=False)
    meta_doc = load(RUN_META)
    summary = load(SUMMARY)
    final = load(FINAL)
    complete = load(COMPLETE)
    relative = arrays["relative_distance"]
    delta_norm = arrays["delta_norm"]
    base_norm = arrays["state_norm_mean"]
    correct = arrays["behavior_correct"]
    add(checks, "array_hash_matches", meta_doc["array_sha256"] == sha(ARRAYS) and final["array_sha256"] == sha(ARRAYS), sha(ARRAYS))
    add(checks, "array_shapes_exact", relative.shape == (1152, 37, 4) and delta_norm.shape == relative.shape and base_norm.shape == relative.shape and correct.shape == (1152, 2), [relative.shape, correct.shape])
    add(checks, "array_axes_exact", arrays["depths"].tolist() == list(DEPTHS) and arrays["roles"].tolist() == list(ROLES), [arrays["depths"].tolist(), arrays["roles"].tolist()])
    add(checks, "relative_distance_recomputes", np.allclose(relative, delta_norm / (base_norm + EPS), atol=2e-7, rtol=2e-6), float(np.max(np.abs(relative - delta_norm / (base_norm + EPS)))))
    add(checks, "all_arrays_finite", all(np.isfinite(arrays[key]).all() for key in ("relative_distance", "delta_norm", "state_norm_mean", "behavior_margin")), "all numeric arrays")
    manifest = read_jsonl(MANIFEST)
    expected_meta = [{key: pair[key] for key in ("group_id", "partition", "profile_index", "attribute", "panel", "surface")} for pair in manifest]
    add(checks, "pair_metadata_matches_manifest", meta_doc["pair_metadata"] == expected_meta, len(expected_meta))
    analysis = recompute_analysis(relative, expected_meta, correct)
    summarized = {key: value for key, value in summary.items() if key not in {"phase", "campaign", "protocol_digest", "authorization"}}
    add(checks, "full_analysis_recomputes", canonical(analysis) == canonical(summarized), {"selected": analysis["selected_discovery_bands"], "gates": analysis["gates"]})
    expected_auth = "phase1297_path_cut_and_independent_rescue_preregistration_only" if analysis["all_gates_passed"] else "close_c030_without_path_claim"
    add(checks, "verdict_authorization_recomputes", summary["authorization"] == expected_auth and final["authorization"] == expected_auth and final["all_gates_passed"] == analysis["all_gates_passed"], expected_auth)
    qa = meta_doc["model_audit"]
    add(checks, "fp16_no_quantization", qa["has_fp16_parameters"] is True and qa["has_quantized_modules"] is False, qa)
    add(checks, "causal_not_performed", final["causal_intervention_performed"] is False, final["causal_intervention_performed"])
    add(checks, "formal_run_once", complete["formal_runs_consumed"] == 1 and complete["protocol_digest"] == protocol["protocol_digest"], complete)
    passed = all(check["passed"] for check in checks)
    result = {
        "phase": 1296, "campaign": "C030", "audit_stage": "post_model",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
        "checks": checks, "passed_count": sum(check["passed"] for check in checks), "total_count": len(checks),
        "all_checks_passed": passed, "scientific_authorization": expected_auth if passed else "none_due_to_audit_failure",
        "protocol_digest": protocol["protocol_digest"],
    }
    save(POSTAUDIT, result)
    print(canonical({"stage": "postaudit", "passed": result["passed_count"], "total": result["total_count"], "authorization": result["scientific_authorization"]}))
    if not passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()


if __name__ == "__main__":
    main()
