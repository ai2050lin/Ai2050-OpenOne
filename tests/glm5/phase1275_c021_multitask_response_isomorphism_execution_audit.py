#!/usr/bin/env python3
"""Independent audit for the Phase1275 execution revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1275_c021_multitask_response_isomorphism_execution as phase
import phase1274_c021_multitask_free_response_isomorphism as base


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def preaudit() -> None:
    protocol, rows = phase.verify_protocol()
    predecessor = base.read_json(base.PROTOCOL)
    checks: list[dict[str, Any]] = []
    add(checks, "phase1274_incomplete", not protocol["phase1274_dependency"]["formal_complete"])
    add(checks, "phase1274_zero_scientific_outputs", not protocol["phase1274_dependency"]["scientific_outputs_exist"])
    add(checks, "semantic_digest_equal", phase.semantic_digest(protocol) == phase.semantic_digest(predecessor))
    add(checks, "material_bytes_equal", file_sha256(phase.MATERIAL) == file_sha256(base.MATERIAL))
    add(checks, "model_seeds_equal", protocol["model_seeds"] == predecessor["model_seeds"])
    add(checks, "thresholds_equal", protocol["thresholds"] == predecessor["thresholds"])
    add(checks, "event_registry_equal", protocol["program_registry"] == predecessor["program_registry"])
    add(checks, "roles_readouts_equal", protocol["roles"] == predecessor["roles"] and protocol["readouts"] == predecessor["readouts"])
    add(checks, "batch_only_execution_change", set(protocol["execution_revision"]) == {"batch_by_architecture", "atomic_per_model_checkpoints", "scientific_semantics_changed"} and not protocol["execution_revision"]["scientific_semantics_changed"])
    add(checks, "material_count", len(rows) == 6144, len(rows))
    payload = {"phase": phase.PHASE, "mode": "preaudit", "checks": checks, "passed": all(item["passed"] for item in checks), "passed_count": sum(item["passed"] for item in checks), "check_count": len(checks), "protocol_hash": file_sha256(phase.PROTOCOL)}
    write_json(phase.PREAUDIT, payload)
    print(json.dumps({"passed": payload["passed"], "passed_count": payload["passed_count"], "check_count": payload["check_count"]}))
    if not payload["passed"]: raise SystemExit(1)


def final_audit() -> None:
    protocol, _rows = phase.verify_protocol()
    qualification, models, ledger = base.read_jsonl(phase.QUALIFICATION), base.read_jsonl(phase.MODELS), base.read_jsonl(phase.PAIRS)
    recomputed, rebuilt = base.analyze_results(qualification, models)
    stored, summary = read_json(phase.FINAL), read_json(phase.SUMMARY)
    checks: list[dict[str, Any]] = []
    add(checks, "formal_complete", phase.COMPLETE.exists() and read_json(phase.COMPLETE)["complete"])
    add(checks, "qualification_complete", len(qualification) == 27, len(qualification))
    add(checks, "no_seed_replacement", {row["model_key"] for row in qualification} == set(protocol["model_seeds"]))
    add(checks, "legal_measurement_subset", {row["model_key"] for row in models} == {row["model_key"] for row in qualification if row["behavior_passed"]})
    add(checks, "typed_statuses", all(row["qualification_status"] in {"qualified", "behavior_rejected"} for row in qualification) and all(row["measurement_status"] == "measured" and row["claim_status"] == "abstained" for row in models))
    add(checks, "execution_batches", all(row["response_tensor"]["execution_batch"] == phase.BATCH_BY_ARCHITECTURE[row["architecture"]] for row in models))
    add(checks, "pair_ledger_rebuilt", base.digest(ledger) == base.digest(rebuilt), len(ledger))
    add(checks, "raw_hashes", summary["qualification_hash"] == file_sha256(phase.QUALIFICATION) and summary["models_hash"] == file_sha256(phase.MODELS) and summary["pairs_hash"] == file_sha256(phase.PAIRS))
    add(checks, "decision_recomputed", stored["decision"] == recomputed["decision"])
    add(checks, "camera_recomputed", stored["selected_camera"] == recomputed["selected_camera"] and stored["selected_executable_camera"] == recomputed["selected_executable_camera"])
    add(checks, "gates_recomputed", stored["gates"] == recomputed["gates"])
    add(checks, "selection_recomputed", stored["selection_results"] == recomputed["selection_results"])
    add(checks, "evaluations_recomputed", stored["evaluations"] == recomputed["evaluations"])
    add(checks, "authorization_consistent", stored["rescue_authorized"] == stored["passed"] and not stored["pretrained_authorized"])
    add(checks, "no_pretrained", not summary["pretrained_model_loaded"])
    add(checks, "semantic_digest_preserved", stored["phase1274_semantic_digest"] == phase.semantic_digest(base.read_json(base.PROTOCOL)))
    add(checks, "material_still_equal", file_sha256(phase.MATERIAL) == file_sha256(base.MATERIAL))
    add(checks, "confirmation_not_selection", all(row["selection"] == (row["left_task"] in base.DISCOVERY_TASKS and row["right_task"] in base.DISCOVERY_TASKS and row["left_seed_index"] < 2 and row["right_seed_index"] < 2) for row in ledger))
    payload = {"phase": phase.PHASE, "mode": "final", "checks": checks, "passed": all(item["passed"] for item in checks), "passed_count": sum(item["passed"] for item in checks), "check_count": len(checks), "final_hash": file_sha256(phase.FINAL), "pairs_hash": file_sha256(phase.PAIRS)}
    write_json(phase.FINAL_AUDIT, payload)
    print(json.dumps({"passed": payload["passed"], "passed_count": payload["passed_count"], "check_count": payload["check_count"]}))
    if not payload["passed"]: raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preaudit", "final"))
    args = parser.parse_args()
    preaudit() if args.mode == "preaudit" else final_audit()


if __name__ == "__main__": main()
