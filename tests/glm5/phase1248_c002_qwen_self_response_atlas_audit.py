#!/usr/bin/env python3
"""Independent material and result audit for Phase1248."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))
import phase1248_c002_qwen_self_response_atlas as main  # noqa: E402


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def embedded_digest(value: dict[str, Any], key: str) -> bool:
    copy = dict(value)
    stored = copy.pop(key, None)
    return stored == digest(copy)


def preaudit() -> None:
    protocol = load_json(main.PROTOCOL_PATH)
    material = main.read_jsonl(main.MATERIAL_PATH)
    tokens = main.read_jsonl(main.TOKEN_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest", embedded_digest(protocol, "protocol_digest"))
    add(checks, "source_hashes", protocol["source_hashes"] == main.source_hashes())
    add(checks, "material_hash", main.file_sha256(main.MATERIAL_PATH) == protocol["material_file_sha256"])
    add(checks, "token_hash", main.file_sha256(main.TOKEN_PATH) == protocol["token_file_sha256"])
    expected_rows = sum(main.PARTITION_WORLDS.values()) * len(main.CONDITIONS) * 2
    add(checks, "row_count", len(material) == len(tokens) == expected_rows, len(material))
    add(checks, "sample_ids_unique", len({row["sample_id"] for row in material}) == expected_rows)
    row_digests = True
    for row in material:
        copy = dict(row)
        stored = copy.pop("row_digest")
        row_digests &= stored == digest(copy)
    add(checks, "row_digests", row_digests)
    token_digests = True
    for row in tokens:
        copy = dict(row)
        stored = copy.pop("token_digest")
        token_digests &= stored == digest(copy)
    add(checks, "token_digests", token_digests)
    partition_counts = Counter(row["partition"] for row in material)
    expected_partition = {key: value * len(main.CONDITIONS) * 2 for key, value in main.PARTITION_WORLDS.items()}
    add(checks, "partition_counts", dict(partition_counts) == expected_partition, dict(partition_counts))
    world_cells: dict[str, Counter[str]] = defaultdict(Counter)
    for row in material:
        world_cells[row["world_id"]][row["condition"]] += 1
    add(checks, "world_cluster_integrity", all(set(values) == set(main.CONDITIONS) and all(count == 2 for count in values.values()) for values in world_cells.values()))
    target_semantics = all(row["golds"]["target"] != row["golds"]["receiver"] for row in material)
    null_semantics = all(row["golds"]["null"] == row["golds"]["receiver"] for row in material)
    add(checks, "target_donor_changes_gold", target_semantics)
    add(checks, "null_donor_preserves_gold", null_semantics)
    direct_invariance = True
    code_relevance = True
    grouped: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for row in material:
        key = (row["world_id"], row["representation"], row["interface"], row["receiver_state"])
        grouped[key][row["mapping"]] = row["golds"]["receiver"]
    for key, values in grouped.items():
        if key[1] == "direct":
            direct_invariance &= values["identity"] == values["permuted"]
        else:
            code_relevance &= values["identity"] != values["permuted"]
    add(checks, "direct_mapping_irrelevance", direct_invariance)
    add(checks, "code_mapping_relevance", code_relevance)
    balanced_surface = True
    for row in material:
        for variant in ("receiver", "target", "null"):
            prompt = row["variants"][variant]["prompt"]
            balanced_surface &= all(len(re.findall(rf"=(?:{label})(?:[;.]|$)", prompt)) == 1 for label in main.LABELS)
    add(checks, "balanced_answer_surface", balanced_surface)
    candidate_ids = protocol["token_summary"]["candidate_token_ids"]
    add(checks, "candidate_tokens_unique", len(candidate_ids) == len(set(candidate_ids.values())) == len(main.LABELS))
    add(checks, "tokenizer_gate", protocol["token_summary"]["gate"] is True)
    positions_valid = all(
        0 <= row["variants"][variant]["positions"][role] < row["variants"][variant]["input_length"]
        for row in tokens for variant in ("receiver", "target", "null") for role in ("source", "boundary")
    )
    add(checks, "positions_valid", positions_valid)
    template_sets = {
        partition: {(row["record_template"], row["codebook_template"]) for row in material if row["partition"] == partition}
        for partition in main.PARTITIONS
    }
    add(checks, "confirmation_template_holdout", template_sets["confirmation"].isdisjoint(template_sets["discovery"] | template_sets["selection"]), {k: sorted(v) for k, v in template_sets.items()})
    add(checks, "camera_partitions_frozen", protocol["camera"]["fit"]["partition"] == "discovery" and protocol["camera"]["selection"]["partition"] == "selection" and protocol["camera"]["confirmation"]["partition"] == "confirmation")
    add(checks, "gold_is_side_ledger", protocol["ledgers"]["primary"] == "model_self_patch_response" and protocol["ledgers"]["side"] == "semantic_gold_correctness")
    add(checks, "typed_abstention", set(protocol["typed_abstention"]) == {"in_domain", "out_of_domain", "nonidentifiable"})
    add(checks, "one_shot_budget", protocol["budget"]["max_formal_runs"] == 1 and protocol["budget"]["max_adaptive_rounds"] == 0)
    add(checks, "no_formal_result_before_run", not main.ARRAY_PATH.exists() and not main.RUN_PATH.exists())
    payload = {
        "phase": main.PHASE,
        "schema_version": "phase1248.c002.qwen_self_response.preaudit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "checks": checks,
        "all_checks_passed": all(row["passed"] for row in checks),
        "claim_boundary": "Preaudit validates the frozen factorial material and response contract only.",
    }
    payload["audit_digest"] = digest(payload)
    write_json(main.PREAUDIT_PATH, payload)
    print(canonical_json({"status": "phase1248_preaudit", "passed": payload["all_checks_passed"], "checks": len(checks)}))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def final_audit() -> None:
    protocol = load_json(main.PROTOCOL_PATH)
    run = load_json(main.RUN_PATH)
    atlas = load_json(main.ATLAS_PATH)
    final = load_json(main.FINAL_PATH)
    pre = load_json(main.PREAUDIT_PATH)
    rows = main.read_jsonl(main.TOKEN_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "preaudit_passed", pre["all_checks_passed"] is True)
    add(checks, "protocol_digest", embedded_digest(protocol, "protocol_digest"))
    add(checks, "run_digest", embedded_digest(run, "run_digest"))
    add(checks, "array_hash", main.file_sha256(main.ARRAY_PATH) == run["array_file_sha256"])
    add(checks, "fp16_no_quantization", set(run["precision"]["parameter_dtypes"]) == {"float16"} and run["precision"]["has_quantized_modules"] is False)
    add(checks, "gpu_budget", run["gpu_hours"] <= protocol["budget"]["max_gpu_hours"], run["gpu_hours"])
    arrays = np.load(main.ARRAY_PATH)
    baseline = arrays["baseline"]
    replay = arrays["replay"]
    finite = bool(np.all(np.isfinite(baseline)) and np.all(np.isfinite(replay)))
    replay_max = float(np.max(np.abs(baseline - replay)))
    replay_top1 = float(np.mean(np.argmax(baseline, axis=1) == np.argmax(replay, axis=1)))
    add(checks, "numerical_recomputed", finite == atlas["numerical"]["finite"] and close(replay_max, atlas["numerical"]["replay_max_abs"]) and close(replay_top1, atlas["numerical"]["replay_top1_agreement"]))
    selection_scores: dict[str, float] = {}
    for event_index, event in enumerate(main.EVENTS):
        result = main.evaluate_event(arrays, rows, event_index, "selection", 0.75)
        selection_scores[event["event_id"]] = main.selection_score(result)
    selected_id = max(selection_scores, key=selection_scores.get)
    add(checks, "selection_recomputed", selected_id == atlas["selected_event"]["event_id"])
    selected_index = next(index for index, event in enumerate(main.EVENTS) if event["event_id"] == selected_id)
    confirmation = main.evaluate_event(arrays, rows, selected_index, "confirmation", 1.0)
    stored_confirmation = atlas["confirmation"]
    metric_ok = True
    for method in ("camera", "constant", "shuffled", "norm_only"):
        for key, value in confirmation[method].items():
            metric_ok &= close(value, stored_confirmation[method][key])
    metric_ok &= close(confirmation["prediction_advantage"], stored_confirmation["prediction_advantage"])
    metric_ok &= close(confirmation["target_to_null_effect_ratio"], stored_confirmation["target_to_null_effect_ratio"])
    add(checks, "confirmation_metrics_recomputed", metric_ok)
    indices = confirmation["indices"]
    actual = confirmation["actual"]
    predicted = confirmation["camera_prediction"]
    interface = main.grouped_metrics(rows, indices, actual, predicted, "interface")
    interface_ok = all(
        all(close(value[key], atlas["interface_metrics"][name][key]) for key in value)
        for name, value in interface.items()
    )
    add(checks, "interface_metrics_recomputed", interface_ok)
    gold_indices = np.asarray([main.LABELS.index(row["gold"]) for row in rows], dtype=np.int64)
    correct = np.argmax(baseline, axis=1) == gold_indices
    strata_gate = True
    stratum_ok = True
    for name, mask in (("correct", correct[indices]), ("error", ~correct[indices])):
        count = int(mask.sum())
        stored = atlas["correct_error_strata"][name]
        if count >= main.THRESHOLDS["stratum_min_count"]:
            value = main.metrics(actual[mask], predicted[mask])
            passed = value["cosine_mean"] >= main.THRESHOLDS["stratum_cosine_min"] and value["cosine_positive_fraction"] >= main.THRESHOLDS["stratum_positive_fraction_min"]
            stratum_ok &= stored["status"] == "eligible" and stored["gate"] == passed
            strata_gate &= passed
        else:
            stratum_ok &= stored["status"] == "abstain_insufficient_count" and stored["count"] == count
    add(checks, "correct_error_strata_recomputed", stratum_ok)
    numerical_gate = finite and replay_top1 == 1.0 and replay_max <= main.THRESHOLDS["replay_max_abs"]
    signal_gate = confirmation["camera"]["actual_effect_norm_mean"] >= main.THRESHOLDS["target_effect_min"] and confirmation["target_to_null_effect_ratio"] >= main.THRESHOLDS["target_null_ratio_min"]
    camera_gate = (
        confirmation["camera"]["cosine_mean"] >= main.THRESHOLDS["confirmation_cosine_min"]
        and confirmation["camera"]["cosine_positive_fraction"] >= main.THRESHOLDS["confirmation_positive_fraction_min"]
        and confirmation["camera"]["relative_error_mean"] <= main.THRESHOLDS["confirmation_relative_error_max"]
        and confirmation["prediction_advantage"] >= main.THRESHOLDS["prediction_advantage_min"]
    )
    interface_gate = all(value["cosine_mean"] >= main.THRESHOLDS["interface_cosine_min"] and value["cosine_positive_fraction"] >= main.THRESHOLDS["interface_positive_fraction_min"] for value in interface.values())
    sentinel_drop = main.metrics(actual, predicted)["cosine_mean"] - main.metrics(actual, -predicted)["cosine_mean"]
    identifiability_gate = sentinel_drop >= 0.5
    gates = {
        "G-NUMERICAL": numerical_gate,
        "G-RESPONSE-SIGNAL": signal_gate,
        "G-CAMERA": camera_gate,
        "G-INTERFACE": interface_gate,
        "G-CORRECT-ERROR": strata_gate,
        "G-IDENTIFIABILITY": identifiability_gate,
    }
    add(checks, "gates_recomputed", gates == atlas["gates"] == final["gates"], gates)
    verdict = "qwen_model_self_response_atlas_qualified" if all(gates.values()) else "bounded_external_validity_failure"
    add(checks, "verdict_recomputed", verdict == atlas["verdict"] == final["verdict"])
    add(checks, "atlas_digest", embedded_digest(atlas, "atlas_digest"))
    add(checks, "final_digest", embedded_digest(final, "final_digest"))
    add(checks, "authorization_typed", final["semantic_mechanism_claim_authorized"] is False and final["phase1249_authorized"] == all(gates.values()))
    payload = {
        "phase": main.PHASE,
        "schema_version": "phase1248.c002.qwen_self_response.final_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks),
        "checks": checks,
        "all_checks_passed": all(row["passed"] for row in checks),
        "recomputed_gates": gates,
        "recomputed_verdict": verdict,
        "claim_boundary": "Audit verifies model-self response prediction only; it does not identify a semantic mechanism.",
    }
    payload["audit_digest"] = digest(payload)
    write_json(main.FINAL_AUDIT_PATH, payload)
    print(canonical_json({"status": "phase1248_final_audit", "passed": payload["all_checks_passed"], "checks": len(checks), "verdict": verdict}))
    if not payload["all_checks_passed"]:
        raise SystemExit(1)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.mode == "preaudit":
        preaudit()
    else:
        final_audit()


if __name__ == "__main__":
    cli()
