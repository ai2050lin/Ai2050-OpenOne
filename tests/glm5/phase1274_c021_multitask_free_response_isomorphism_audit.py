#!/usr/bin/env python3
"""Independent audit for Phase1274 multi-task free response isomorphism."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1274_c021_multitask_free_response_isomorphism as phase


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
    checks: list[dict[str, Any]] = []
    add(checks, "phase1273_passed", protocol["phase1273_dependency"]["passed"] and protocol["phase1273_dependency"]["audit_passed"])
    add(checks, "model_count", protocol["planned_models"] == 27)
    add(checks, "three_tasks", len(protocol["tasks"]) == 3)
    add(checks, "three_depths", len(protocol["architectures"]) == 3)
    add(checks, "three_seeds", protocol["seeds_per_cell"] == 3)
    add(checks, "material_count", len(rows) == len(phase.TASKS) * sum(phase.PARTITION_COUNTS.values()), len(rows))
    add(checks, "orthogonal_roles", set(protocol["roles"]) == set(phase.ROLES))
    add(checks, "camera_registry", set(protocol["cameras"]) == set(phase.CAMERAS))
    add(checks, "selection_sealed", protocol["selection_panel"].startswith("cyclic/xor"))
    add(checks, "pretrained_denied", any("No pretrained" in value for value in protocol["hard_stops"]))
    payload = {"phase": phase.PHASE, "mode": "preaudit", "checks": checks, "passed": all(item["passed"] for item in checks), "passed_count": sum(item["passed"] for item in checks), "check_count": len(checks), "protocol_hash": file_sha256(phase.PROTOCOL)}
    write_json(phase.PREAUDIT, payload)
    print(json.dumps({"passed": payload["passed"], "passed_count": payload["passed_count"], "check_count": payload["check_count"]}))
    if not payload["passed"]: raise SystemExit(1)


def final_audit() -> None:
    protocol, _rows = phase.verify_protocol()
    qualification, models, ledger = phase.read_jsonl(phase.QUALIFICATION), phase.read_jsonl(phase.MODELS), phase.read_jsonl(phase.PAIRS)
    final, rebuilt = phase.analyze_results(qualification, models)
    stored, summary = read_json(phase.FINAL), read_json(phase.SUMMARY)
    checks: list[dict[str, Any]] = []
    add(checks, "formal_complete", phase.COMPLETE.exists() and read_json(phase.COMPLETE)["complete"])
    add(checks, "qualification_complete", len(qualification) == protocol["planned_models"], len(qualification))
    add(checks, "no_seed_replacement", {row["model_key"] for row in qualification} == set(protocol["model_seeds"]))
    add(checks, "legal_measurement_subset", {row["model_key"] for row in models} == {row["model_key"] for row in qualification if row["behavior_passed"]})
    add(checks, "typed_behavior_rejection", all(row["qualification_status"] in {"qualified", "behavior_rejected"} for row in qualification))
    add(checks, "typed_measurement", all(row["measurement_status"] == "measured" and row["claim_status"] == "abstained" for row in models))
    add(checks, "pair_ledger_rebuilt", phase.digest(ledger) == phase.digest(rebuilt), len(ledger))
    add(checks, "qualification_hash", summary["qualification_hash"] == file_sha256(phase.QUALIFICATION))
    add(checks, "model_hash", summary["models_hash"] == file_sha256(phase.MODELS))
    add(checks, "pair_hash", summary["pairs_hash"] == file_sha256(phase.PAIRS))
    add(checks, "decision_recomputed", stored["decision"] == final["decision"])
    add(checks, "selected_camera_recomputed", stored["selected_camera"] == final["selected_camera"] and stored["selected_executable_camera"] == final["selected_executable_camera"])
    add(checks, "gates_recomputed", stored["gates"] == final["gates"])
    add(checks, "selection_recomputed", stored["selection_results"] == final["selection_results"])
    add(checks, "evaluation_recomputed", stored["evaluations"] == final["evaluations"])
    add(checks, "authorization_consistent", stored["rescue_authorized"] == stored["passed"] and not stored["pretrained_authorized"])
    add(checks, "run_without_pretrained", not summary["pretrained_model_loaded"])
    add(checks, "confirmation_not_selection", all(row["selection"] == (row["left_task"] in phase.DISCOVERY_TASKS and row["right_task"] in phase.DISCOVERY_TASKS and row["left_seed_index"] < 2 and row["right_seed_index"] < 2) for row in ledger))
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
