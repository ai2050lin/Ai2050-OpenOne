#!/usr/bin/env python3
"""Independent audit for Phase1276 checkpoint continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1276_c021_checkpoint_continuation_adjudication as phase
import phase1274_c021_multitask_free_response_isomorphism as base
import phase1275_c021_multitask_response_isomorphism_execution as execution


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
    qualification, models, remaining = phase.validate_partial()
    checks: list[dict[str, Any]] = []
    add(checks, "partial_counts", len(qualification) == 22 and len(models) == 17)
    add(checks, "partial_hashes", protocol["partial"]["qualification_hash"] == file_sha256(execution.QUALIFICATION) and protocol["partial"]["models_hash"] == file_sha256(execution.MODELS))
    add(checks, "prefix_order", [row["model_key"] for row in qualification] == phase.expected_order()[:22])
    add(checks, "legal_measured_subset", {row["model_key"] for row in models} == {row["model_key"] for row in qualification if row["behavior_passed"]})
    add(checks, "remaining_exact", remaining == phase.expected_order()[22:] == protocol["remaining_keys"])
    add(checks, "remaining_five", len(remaining) == 5)
    add(checks, "material_equal", file_sha256(phase.MATERIAL) == file_sha256(execution.MATERIAL))
    add(checks, "semantic_digest", protocol["semantic_digest"] == execution.semantic_digest(base.read_json(execution.PROTOCOL)))
    add(checks, "no_camera_outputs_yet", not phase.PAIRS.exists() and not phase.FINAL.exists())
    add(checks, "all_27_original_seeds", set(phase.expected_order()) == set(base.MODEL_SEEDS))
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
    add(checks, "full_order", [row["model_key"] for row in qualification] == phase.expected_order())
    add(checks, "all_27", len(qualification) == 27)
    add(checks, "partial_prefix_unchanged", base.digest(qualification[:22]) == protocol["partial"]["qualification_digest"] and base.digest(models[:17]) == protocol["partial"]["models_digest"])
    add(checks, "no_seed_replacement", {row["model_key"] for row in qualification} == set(base.MODEL_SEEDS))
    add(checks, "legal_measurement_subset", {row["model_key"] for row in models} == {row["model_key"] for row in qualification if row["behavior_passed"]})
    add(checks, "typed_statuses", all(row["qualification_status"] in {"qualified", "behavior_rejected"} for row in qualification) and all(row["measurement_status"] == "measured" and row["claim_status"] == "abstained" for row in models))
    add(checks, "pair_rebuilt", base.digest(ledger) == base.digest(rebuilt), len(ledger))
    add(checks, "raw_hashes", summary["qualification_hash"] == file_sha256(phase.QUALIFICATION) and summary["models_hash"] == file_sha256(phase.MODELS) and summary["pairs_hash"] == file_sha256(phase.PAIRS))
    add(checks, "decision_recomputed", stored["decision"] == recomputed["decision"])
    add(checks, "camera_recomputed", stored["selected_camera"] == recomputed["selected_camera"] and stored["selected_executable_camera"] == recomputed["selected_executable_camera"])
    add(checks, "gates_recomputed", stored["gates"] == recomputed["gates"])
    add(checks, "selection_recomputed", stored["selection_results"] == recomputed["selection_results"])
    add(checks, "evaluation_recomputed", stored["evaluations"] == recomputed["evaluations"])
    add(checks, "authorization_consistent", stored["rescue_authorized"] == stored["passed"] and not stored["pretrained_authorized"])
    add(checks, "no_pretrained", not summary["pretrained_model_loaded"])
    add(checks, "continuation_count", summary["continued_models"] == 5)
    add(checks, "material_equal", file_sha256(phase.MATERIAL) == file_sha256(execution.MATERIAL))
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
