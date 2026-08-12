#!/usr/bin/env python3
"""Independent audit for Phase1236.

This file intentionally does not import the Phase1236 implementation.  It
recomputes digests, cardinalities, typed behavior metrics, gate order, and the
final evidence boundary from serialized artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
MAIN_SCRIPT = TEST_ROOT / "phase1236_global_functional_structure_identification.py"
AUDIT_SCRIPT = Path(__file__).resolve()
OUT_ROOT = TEST_ROOT / "result/phase1236_global_functional_structure_identification"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MATERIAL_PATH = OUT_ROOT / "material/frozen_response_worlds.jsonl"
FIXTURE_PATH = OUT_ROOT / "material/evaluator_adversarial_fixtures.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
BEHAVIOR_PATH = OUT_ROOT / "analysis/behavior_adjudication.json"
CAPTURE_ARRAY_PATH = OUT_ROOT / "hidden/qwen3/response_tensor.npz"
CAPTURE_META_PATH = OUT_ROOT / "hidden/qwen3/response_tensor_metadata.json"
STRUCTURE_PATH = OUT_ROOT / "analysis/structure_competition.json"
STRUCTURE_AUDIT_PATH = OUT_ROOT / "audit/independent_structure_audit.json"
CAUSAL_PATH = OUT_ROOT / "causal/qwen3/cross_protocol_interchange.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

PHASE = 1236
MODELS = ("qwen3", "glm4", "deepseek7b")
PROTOCOLS = ("bare", "sentence", "natural")
PARTITIONS = ("discovery", "model_selection", "sealed")
EXPECTED_ROWS = 1152
EXPECTED_WORLDS = 48
EXPECTED_ROWS_PER_WORLD = 24
THRESHOLDS = {
    "finite_rate": 0.99,
    "content_score_worst_partition_protocol": 0.85,
    "content_score_worst_protocol": 0.90,
    "contract_score_worst_partition_protocol": 0.80,
    "generation_content_worst_partition_protocol": 0.75,
    "generation_content_worst_protocol": 0.80,
    "format_valid_worst_bare_sentence": 0.75,
    "structure_improvement_over_mean": 0.10,
    "structure_shuffled_advantage": 0.10,
    "structure_median_cosine": 0.20,
    "structure_positive_cosine_fraction": 0.65,
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def manifest_path(model: str) -> Path:
    return OUT_ROOT / f"protocol/{model}_manifest.jsonl"


def raw_path(model: str) -> Path:
    return OUT_ROOT / f"behavior/{model}/raw_behavior.jsonl"


def summary_path(model: str) -> Path:
    return OUT_ROOT / f"behavior/{model}/run_summary.json"


def behavior_audit_path(model: str) -> Path:
    return OUT_ROOT / f"audit/{model}_behavior_audit.json"


def embedded_digest_ok(value: dict[str, Any], key: str) -> bool:
    return value.get(key) == digest(strip_digest(value, key))


def preaudit() -> None:
    if PREAUDIT_PATH.exists():
        raise RuntimeError("preaudit already exists")
    contract = read_json(CONTRACT_PATH)
    material = read_jsonl(MATERIAL_PATH)
    fixtures = read_jsonl(FIXTURE_PATH)
    checks: dict[str, bool] = {}
    checks["phase"] = contract.get("phase") == PHASE
    checks["contract_digest"] = embedded_digest_ok(contract, "contract_digest")
    checks["main_hash"] = contract.get("source_hashes", {}).get("main") == file_sha256(MAIN_SCRIPT)
    checks["audit_hash"] = contract.get("source_hashes", {}).get("audit") == file_sha256(AUDIT_SCRIPT)
    checks["material_digest"] = contract.get("material", {}).get("material_digest") == digest(material)
    checks["row_count"] = len(material) == EXPECTED_ROWS
    checks["item_unique"] = len({row["item_id"] for row in material}) == EXPECTED_ROWS
    checks["world_count"] = len({row["world_id"] for row in material}) == EXPECTED_WORLDS
    checks["rows_per_world"] = set(Counter(row["world_id"] for row in material).values()) == {EXPECTED_ROWS_PER_WORLD}
    checks["partition_worlds"] = all(
        len({row["world_id"] for row in material if row["partition"] == partition}) == 16 for partition in PARTITIONS
    )
    checks["protocol_balance"] = all(sum(row["protocol"] == protocol for row in material) == 384 for protocol in PROTOCOLS)
    checks["state_balance"] = all(sum(row["binding_state"] == state for row in material) == 576 for state in (0, 1))
    checks["pair_cardinality"] = set(Counter(row["pair_id"] for row in material).values()) == {2}
    checks["base_pair_cardinality"] = set(Counter(row["base_pair_id"] for row in material).values()) == {6}
    checks["prior_disjoint"] = contract.get("material", {}).get("material_audit", {}).get("prior_overlap") == []
    checks["parser_fixtures"] = len(fixtures) == 8 and all(row.get("pass") is True for row in fixtures)
    checks["parser_fixture_digest"] = contract.get("material", {}).get("parser_fixture_digest") == digest(fixtures)
    for model in MODELS:
        manifest = read_jsonl(manifest_path(model))
        summary = contract["manifest_summaries"][model]
        checks[f"{model}_manifest_count"] = len(manifest) == EXPECTED_ROWS
        checks[f"{model}_manifest_digest"] = summary["manifest_digest"] == digest(manifest)
        checks[f"{model}_token_gate"] = summary["gate"] is True and summary["candidate_length_mismatch_count"] == 0
        checks[f"{model}_material_alignment"] = [row["item_id"] for row in manifest] == [row["item_id"] for row in material]
        checks[f"{model}_fp16"] = contract["execution"]["precision"] == "float16" and contract["execution"]["quantization"] == "none"
    checks["sealed_partition_named"] = contract["structure_competition"]["sealed_partition"].startswith("sealed")
    checks["one_shot_causal"] = contract["causal_interchange"]["one_shot"] is True
    checks["stop_rules"] = len(contract.get("stop_rules", [])) >= 5
    checks["phase1235_stop_respected"] = contract["upstream"]["phase1235_stop_respected"] is True
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.independent_preaudit.v1",
        "created_at_utc": utc_now(),
        "contract_digest": contract["contract_digest"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    value["audit_digest"] = digest(value)
    write_json(PREAUDIT_PATH, value)
    if not value["all_checks_passed"]:
        raise RuntimeError(f"preaudit failed: {[key for key, passed in checks.items() if not passed]}")
    print(canonical_json({"status": "preaudit_passed", "passed": value["passed"], "total": value["total"], "audit_digest": value["audit_digest"]}))


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def recompute_behavior_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for partition in PARTITIONS:
        for protocol in PROTOCOLS:
            selected = [row for row in rows if row["partition"] == partition and row["protocol"] == protocol]
            cells[f"{partition}|{protocol}"] = {
                "finite": rate(selected, "candidate_scores_finite"),
                "content_score": rate(selected, "content_score_correct"),
                "contract_score": rate(selected, "contract_score_correct"),
                "generation_content": rate(selected, "generation_content_correct"),
                "generation_exact": rate(selected, "generation_exact"),
            }
    protocols = {}
    for protocol in PROTOCOLS:
        selected = [row for row in rows if row["protocol"] == protocol]
        protocols[protocol] = {
            "content_score": rate(selected, "content_score_correct"),
            "generation_content": rate(selected, "generation_content_correct"),
            "generation_exact": rate(selected, "generation_exact"),
        }
    finite = min(value["finite"] for value in cells.values())
    content_cell = min(value["content_score"] for value in cells.values())
    content_protocol = min(value["content_score"] for value in protocols.values())
    generation_cell = min(value["generation_content"] for value in cells.values())
    generation_protocol = min(value["generation_content"] for value in protocols.values())
    content_gate = finite >= THRESHOLDS["finite_rate"] and content_cell >= THRESHOLDS["content_score_worst_partition_protocol"] and content_protocol >= THRESHOLDS["content_score_worst_protocol"]
    generation_gate = generation_cell >= THRESHOLDS["generation_content_worst_partition_protocol"] and generation_protocol >= THRESHOLDS["generation_content_worst_protocol"]
    return {
        "content_gate": content_gate,
        "generation_gate": generation_gate,
        "hidden_content_lane": content_gate and generation_gate,
        "finite": finite,
        "content_cell": content_cell,
        "content_protocol": content_protocol,
        "generation_cell": generation_cell,
        "generation_protocol": generation_protocol,
    }


def audit_behavior(model: str) -> None:
    path = behavior_audit_path(model)
    if path.exists():
        raise RuntimeError(f"{model} behavior audit already exists")
    contract = read_json(CONTRACT_PATH)
    raw = read_jsonl(raw_path(model))
    summary = read_json(summary_path(model))
    material = {row["item_id"]: row for row in read_jsonl(MATERIAL_PATH)}
    checks: dict[str, bool] = {}
    checks["contract_digest"] = summary.get("contract_digest") == contract["contract_digest"]
    checks["summary_digest"] = embedded_digest_ok(summary, "summary_digest")
    checks["raw_digest"] = summary.get("raw_digest") == digest(raw)
    checks["count"] = len(raw) == EXPECTED_ROWS == summary.get("case_count")
    checks["unique"] = len({row["item_id"] for row in raw}) == EXPECTED_ROWS
    checks["model"] = all(row["model"] == model for row in raw) and summary["model"] == model
    checks["fp16"] = set(summary["precision_audit"]["parameter_dtypes"]) == {"float16"}
    checks["no_quantization"] = summary["precision_audit"]["has_quantized_modules"] is False
    checks["row_digests"] = all(row["behavior_row_digest"] == digest(strip_digest(row, "behavior_row_digest")) for row in raw)
    checks["gold_alignment"] = all(row["gold"] == material[row["item_id"]]["gold"] for row in raw)
    checks["content_flags"] = all(row["content_score_correct"] == (row["content_prediction"] == row["gold"]) for row in raw)
    checks["contract_flags"] = all(row["contract_score_correct"] == (row["contract_prediction"] == row["gold"]) for row in raw)
    checks["generation_flags"] = all(row["generation_content_correct"] == (row["generation_parse"]["prediction"] == row["gold"]) for row in raw)
    checks["format_flags"] = all(row["generation_format_valid"] == row["generation_parse"]["format_valid"] for row in raw)
    checks["finite_denominator"] = all(isinstance(row["candidate_scores_finite"], bool) for row in raw)
    metrics = recompute_behavior_metrics(raw)
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.independent_behavior_audit.v1",
        "created_at_utc": utc_now(),
        "model": model,
        "contract_digest": contract["contract_digest"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed_typed_metrics": metrics,
    }
    value["audit_digest"] = digest(value)
    write_json(path, value)
    if not value["all_checks_passed"]:
        raise RuntimeError(f"{model} behavior audit failed")
    print(canonical_json({"status": "behavior_audit_passed", "model": model, "metrics": metrics, "audit_digest": value["audit_digest"]}))


def structure_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["improvement_over_mean"] >= THRESHOLDS["structure_improvement_over_mean"]
        and metrics["shuffled_advantage"] >= THRESHOLDS["structure_shuffled_advantage"]
        and metrics["median_cosine"] >= THRESHOLDS["structure_median_cosine"]
        and metrics["positive_cosine_fraction"] >= THRESHOLDS["structure_positive_cosine_fraction"]
        and 0.25 <= metrics["prediction_target_norm_ratio"] <= 4.0
    )


def structure_audit() -> None:
    if STRUCTURE_AUDIT_PATH.exists():
        raise RuntimeError("structure audit already exists")
    contract = read_json(CONTRACT_PATH)
    capture = read_json(CAPTURE_META_PATH)
    structure = read_json(STRUCTURE_PATH)
    checks: dict[str, bool] = {}
    checks["structure_digest"] = embedded_digest_ok(structure, "structure_digest")
    checks["contract_alignment"] = structure.get("contract_digest") == contract["contract_digest"]
    checks["capture_gate"] = bool(
        (capture.get("capture_performed") is True and structure.get("status") == "complete")
        or (capture.get("capture_performed") is False and structure.get("status") == "denied_no_response_tensor")
    )
    if capture.get("capture_performed") is True:
        checks["capture_file"] = CAPTURE_ARRAY_PATH.exists() and capture["array_file_sha256"] == file_sha256(CAPTURE_ARRAY_PATH)
        expected_candidates = 6 * int(capture["depth_count"]) * 7
        checks["candidate_count"] = structure.get("candidate_count") == expected_candidates == len(structure.get("selection_records", []))
        passing = [row for row in structure["selection_records"] if row["selection_pass"]]
        winner = max(
            passing,
            key=lambda row: (row["selection_score"], -row["depth"], row["family"], row["source_protocol"], row["target_protocol"]),
        ) if passing else None
        checks["winner_recomputed"] = winner == structure.get("winner")
        checks["passing_count"] = structure.get("selection_passing_count") == len(passing)
        if structure.get("sealed") is not None:
            checks["sealed_gate"] = structure["structure_gate"] == structure_gate(structure["sealed"]["metrics"])
            checks["sealed_once"] = structure.get("sealed_inspected_once") is True and structure.get("sealed_not_used_for_selection") is True
        else:
            checks["sealed_gate"] = structure.get("structure_gate") is False
            checks["sealed_once"] = winner is None
    else:
        checks["capture_file"] = not CAPTURE_ARRAY_PATH.exists()
        checks["candidate_count"] = structure.get("candidate_count") is None
        checks["winner_recomputed"] = structure.get("winner") is None
        checks["passing_count"] = structure.get("selection_passing_count") is None
        checks["sealed_gate"] = structure.get("structure_gate") is False
        checks["sealed_once"] = True
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.independent_structure_audit.v1",
        "created_at_utc": utc_now(),
        "contract_digest": contract["contract_digest"],
        "structure_digest": structure["structure_digest"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    value["audit_digest"] = digest(value)
    write_json(STRUCTURE_AUDIT_PATH, value)
    if not value["all_checks_passed"]:
        raise RuntimeError(f"structure audit failed: {[key for key, passed in checks.items() if not passed]}")
    print(canonical_json({"status": "structure_audit_passed", "passed": value["passed"], "total": value["total"], "audit_digest": value["audit_digest"]}))


def final_audit() -> None:
    if FINAL_AUDIT_PATH.exists():
        raise RuntimeError("final audit already exists")
    contract = read_json(CONTRACT_PATH)
    behavior = read_json(BEHAVIOR_PATH)
    capture = read_json(CAPTURE_META_PATH)
    structure = read_json(STRUCTURE_PATH)
    causal = read_json(CAUSAL_PATH)
    final = read_json(FINAL_PATH)
    checks: dict[str, bool] = {}
    checks["contract_digest"] = embedded_digest_ok(contract, "contract_digest")
    checks["behavior_digest"] = embedded_digest_ok(behavior, "adjudication_digest")
    checks["structure_digest"] = embedded_digest_ok(structure, "structure_digest")
    checks["causal_digest"] = embedded_digest_ok(causal, "causal_digest")
    checks["final_digest"] = embedded_digest_ok(final, "final_digest")
    checks["source_hashes"] = contract["source_hashes"] == {"main": file_sha256(MAIN_SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)}
    checks["behavior_audits"] = all(read_json(behavior_audit_path(model)).get("all_checks_passed") is True for model in MODELS)
    authorized = [model for model in MODELS if behavior["models"][model]["gates"]["hidden_content_lane"]]
    checks["behavior_authorization"] = authorized == behavior["authorized_content_models"]
    checks["cross_model_gate"] = behavior["cross_model_behavior_authorized"] == (len(authorized) >= 2)
    checks["capture_gate_order"] = bool(
        (behavior["qwen_hidden_authorized"] and capture.get("capture_performed") is True)
        or (not behavior["qwen_hidden_authorized"] and capture.get("capture_performed") is False)
    )
    if capture.get("capture_performed") is True:
        checks["capture_file"] = CAPTURE_ARRAY_PATH.exists() and capture["array_file_sha256"] == file_sha256(CAPTURE_ARRAY_PATH)
        checks["capture_shape_metadata"] = capture["base_pair_count"] == 192 and capture["depth_count"] > 0 and len(capture["event_ids"]) == 3 * capture["depth_count"]
    else:
        checks["capture_file"] = not CAPTURE_ARRAY_PATH.exists()
        checks["capture_shape_metadata"] = capture.get("status") == "denied_by_behavior_gate"
    if structure.get("sealed") is not None:
        checks["sealed_gate_recomputed"] = structure["structure_gate"] == structure_gate(structure["sealed"]["metrics"])
        checks["selection_before_sealed"] = structure.get("sealed_not_used_for_selection") is True and structure.get("sealed_inspected_once") is True
    else:
        checks["sealed_gate_recomputed"] = structure.get("structure_gate") is False
        checks["selection_before_sealed"] = structure.get("winner") is None or structure.get("status") == "denied_no_response_tensor"
    checks["causal_gate_order"] = bool(
        (structure["structure_gate"] and causal.get("intervention_performed") is True)
        or (not structure["structure_gate"] and causal.get("intervention_performed") is False)
    )
    checks["typed_final_gates"] = final["typed_gates"] == {
        "qwen_hidden_authorized": behavior["qwen_hidden_authorized"],
        "cross_model_behavior_authorized": behavior["cross_model_behavior_authorized"],
        "sealed_structure_gate": structure["structure_gate"],
        "causal_interchange_gate": causal["causal_gate"],
    }
    checks["no_cross_model_physical_claim"] = final["claims"]["cross_model_physical_identity_proven"] is False
    checks["not_cracked"] = final["claims"]["natural_language_encoding_cracked"] is False
    checks["registry_closed"] = final["registry_status"] == "closed_after_one_sealed_adjudication"
    checks["auto_continue_false"] = final["auto_continue"] is False
    checks["no_new_math_claim"] = final["claims"]["new_mathematics_required"] is False
    value: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1236.independent_final_audit.v1",
        "created_at_utc": utc_now(),
        "contract_digest": contract["contract_digest"],
        "final_digest": final["final_digest"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    value["audit_digest"] = digest(value)
    write_json(FINAL_AUDIT_PATH, value)
    if not value["all_checks_passed"]:
        raise RuntimeError(f"final audit failed: {[key for key, passed in checks.items() if not passed]}")
    print(canonical_json({"status": "final_audit_passed", "passed": value["passed"], "total": value["total"], "audit_digest": value["audit_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("pre", "behavior", "structure", "final"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "pre":
        preaudit()
    elif args.stage == "behavior":
        if args.model is None:
            raise SystemExit("--model required for behavior audit")
        audit_behavior(args.model)
    elif args.stage == "structure":
        structure_audit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
