#!/usr/bin/env python3
"""Independent pre/final audit for Phase1280."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tests/glm5/result/phase1279_c023_relation_operation_behavior_contract"
OUT = ROOT / "tests/glm5/result/phase1280_c023_qwen3_relation_operation_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
RAW = OUT / "raw/candidate_scores.jsonl"
FINAL = OUT / "analysis/final.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
OPERATIONS = ("contrast", "addition", "cause", "sequence")
PANELS = ("base", "target", "wrong", "null", "joint", "surface", "implicit")
FACTORIAL = ("base", "target", "wrong", "null", "joint")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def preaudit() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads((INPUT / "analysis/final.json").read_text(encoding="utf-8"))
    parent_audit = json.loads((INPUT / "audit/independent_final_audit.json").read_text(encoding="utf-8"))
    checks = [
        check("phase", protocol["phase"] == 1280),
        check("parent_authorized", parent["authorization"] == "phase1280_qwen3_behavior_only" and parent_audit["all_checks_passed"]),
        check("dependency_hashes", all(protocol["dependencies"][name] == file_sha256(path) for name, path in {
            "phase1279_protocol": INPUT / "protocol/preregistration.json",
            "phase1279_material": INPUT / "material/frozen_relation_worlds.jsonl",
            "phase1279_final": INPUT / "analysis/final.json",
            "phase1279_audit": INPUT / "audit/independent_final_audit.json",
        }.items())),
        check("frozen_dimensions", protocol["row_count"] == 256 and protocol["prompt_count"] == 1792),
        check("frozen_panels", tuple(protocol["panels"]) == PANELS and tuple(protocol["factorial_panels"]) == FACTORIAL),
        check("single_formal_run", protocol["formal_run_budget"] == 1),
        check("fp16_cuda", protocol["model"]["precision"] == "fp16_cuda_no_quantization"),
    ]
    result = {
        "phase": 1280, "audit_type": "independent_preaudit", "checks": checks,
        "passed_count": sum(row["passed"] for row in checks), "check_count": len(checks),
        "all_checks_passed": all(row["passed"] for row in checks),
    }
    atomic_json(PREAUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def final_audit() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    thresholds = protocol["thresholds"]
    cells = {}
    for partition in ("discovery", "selection", "confirmation"):
        for panel in PANELS:
            subset = [row for row in rows if row["partition"] == partition and row["panel"] == panel]
            cells[f"{partition}.{panel}"] = float(np.mean([row["correct"] for row in subset]))
    operation_macro = {
        operation: float(np.mean([row["correct"] for row in rows if row["expected"] == operation]))
        for operation in OPERATIONS
    }
    finite_fraction = float(np.mean([row["finite"] for row in rows]))
    factorial_min = min(value for key, value in cells.items() if key.split(".")[1] in FACTORIAL)
    surface_min = min(value for key, value in cells.items() if key.endswith(".surface"))
    implicit_min = min(value for key, value in cells.items() if key.endswith(".implicit"))
    margin = float(np.median([row["gold_margin"] for row in rows]))
    gates = {
        "finite": finite_fraction >= thresholds["candidate_finite_fraction_min"],
        "factorial": factorial_min >= thresholds["factorial_cell_accuracy_min"],
        "surface": surface_min >= thresholds["surface_cell_accuracy_min"],
        "implicit": implicit_min >= thresholds["implicit_cell_accuracy_min"],
        "operation_macro": min(operation_macro.values()) >= thresholds["operation_macro_accuracy_min"],
        "margin": margin >= thresholds["gold_margin_median_min"],
    }
    expected_pass = all(gates.values())
    precision = final["precision_audit"]
    checks = [
        check("raw_count", len(rows) == 1792, len(rows)),
        check("unique_prompt_keys", len({(row["row_id"], row["panel"]) for row in rows}) == len(rows)),
        check("all_finite", finite_fraction == 1.0),
        check("cell_accuracy_recomputed", cells == final["behavior"]["cell_accuracy"]),
        check("operation_macro_recomputed", operation_macro == final["behavior"]["operation_macro_accuracy"]),
        check("factorial_min_recomputed", factorial_min == final["behavior"]["factorial_minimum_accuracy"]),
        check("surface_min_recomputed", surface_min == final["behavior"]["surface_minimum_accuracy"]),
        check("implicit_min_recomputed", implicit_min == final["behavior"]["implicit_minimum_accuracy"]),
        check("margin_recomputed", margin == final["behavior"]["overall_median_gold_margin"]),
        check("gates_recomputed", gates == final["behavior"]["gates"]),
        check("verdict_consistent", final["behavior"]["passed"] == expected_pass),
        check("authorization_consistent", final["authorization"] == ("phase1281_qwen3_typed_causal_closure" if expected_pass else "stop_c023_at_behavior_object")),
        check("fp16_not_quantized", set(precision["parameter_dtypes"]) == {"float16"} and not precision["has_quantized_modules"] and not precision["has_bf16_parameters"]),
    ]
    result = {
        "phase": 1280, "audit_type": "independent_final_audit", "checks": checks,
        "passed_count": sum(row["passed"] for row in checks), "check_count": len(checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "scientific_gate_passed": expected_pass,
    }
    atomic_json(FINAL_AUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pre", "final"))
    args = parser.parse_args()
    preaudit() if args.action == "pre" else final_audit()
