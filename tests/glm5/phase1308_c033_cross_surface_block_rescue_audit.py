#!/usr/bin/env python3
"""Independent pre/post audit for Phase1308; does not import the execution script."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
OUT = T / "result/phase1308_c033_cross_surface_block_rescue"
PARENT = T / "result/phase1307_c033_bidirectional_answer_boundary_swap"
HIDDEN = T / "result/phase1306_c033_frozen_answer_boundary_hidden"
CONTRACT = T / "result/phase1304_c033_role_typed_causal_graph_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_rescue_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/rescue_arrays.npz"
META = OUT / "raw/run_metadata.json"
S = OUT / "analysis/rescue_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1308_c033_cross_surface_block_rescue.py"
SCRIPT = Path(__file__).resolve()
PARTITIONS = ("confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
ARMS = ("baseline", "block_only", "correct_cross_surface", "matched_null_cross_surface", "wrong_attribute_cross_surface", "self_retention")
EPS = 1e-12


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(checks, "parent_authorization",
        load(PARENT / "analysis/final.json")["authorization"] == "phase1308_cross_surface_block_rescue_only"
        and load(PARENT / "audit/independent_final_audit.json")["all_checks_passed"], "Phase1307")
    frozen = load(CONTRACT / "protocol/preregistration.json")["cross_surface_block_rescue"]
    add(checks, "thresholds", protocol["thresholds"] == frozen["thresholds"], protocol["thresholds"])
    add(checks, "fixed_graph",
        protocol["event"] == "assistant_answer_boundary" and protocol["block_depth"] == 25 and protocol["rescue_depth"] == 26,
        {"event": protocol["event"], "block": protocol["block_depth"], "rescue": protocol["rescue_depth"]})
    manifest = rows(M)
    add(checks, "manifest_count", len(manifest) == 192, len(manifest))
    add(checks, "manifest_hash", protocol["manifest"]["sha256"] == sha(M), protocol["manifest"])
    add(checks, "partition_balance",
        {p: sum(x["partition"] == p for x in manifest) for p in PARTITIONS} == {"confirmation": 96, "holdout": 96},
        protocol["manifest"]["partition_counts"])
    add(checks, "surface_balance",
        {s: sum(x["target_surface"] == s for x in manifest) for s in SURFACES} == {s: 96 for s in SURFACES},
        protocol["manifest"]["surface_counts"])
    add(checks, "unique_cases", len({x["case_key"] for x in manifest}) == 192, "unique")
    add(checks, "opposite_surface",
        all(x["target_surface"] != x["donor_surface"] for x in manifest), "opposite")
    add(checks, "target_pair",
        all(x["target_state0"]["candidate_ids"] == x["target_state1"]["candidate_ids"]
            and x["target_state0"]["gold_position"] == x["identity_positions"][0]
            and x["target_state1"]["gold_position"] == x["identity_positions"][1] for x in manifest), "0->1")
    add(checks, "correct_delta_identity",
        all(x["correct_donor_state0"]["candidate_ids"] == x["target_state0"]["candidate_ids"]
            and x["correct_donor_state1"]["candidate_ids"] == x["target_state0"]["candidate_ids"]
            and x["correct_donor_state0"]["gold_position"] == x["identity_positions"][0]
            and x["correct_donor_state1"]["gold_position"] == x["identity_positions"][1] for x in manifest),
        "opposite surface, same identity transition")
    add(checks, "null_delta_no_gold_change",
        all(x["null_donor_state0"]["gold_position"] == x["null_donor_state1"]["gold_position"]
            for x in manifest), "matched-null")
    add(checks, "wrong_attribute_identity_matched",
        all(x["wrong_attribute_state0"]["candidate_ids"] == x["target_state0"]["candidate_ids"]
            and x["wrong_attribute_state1"]["candidate_ids"] == x["target_state0"]["candidate_ids"]
            and x["wrong_attribute_state0"]["gold_position"] == x["identity_positions"][0]
            and x["wrong_attribute_state1"]["gold_position"] == x["identity_positions"][1] for x in manifest),
        "same identity transition, wrong attribute")
    add(checks, "arms", protocol["arms"] == list(ARMS), protocol["arms"])
    add(checks, "hard_stops", protocol["hard_stops"] == [
        "No discovery partition", "No same-surface rescue fallback", "No new event, depth, donor, component, or threshold",
        "No second formal model run", "C033 closes after this phase regardless of verdict"], protocol["hard_stops"])
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(x["passed"] for x in checks)
    value = {"phase": 1308, "campaign": "C033", "audit_stage": stage,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
             "checks": checks, "passed_count": sum(x["passed"] for x in checks), "total_count": len(checks),
             "all_checks_passed": passed, "authorization": authorization if passed else "none",
             "protocol_digest": load(P)["protocol_digest"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": value["passed_count"], "total": value["total_count"],
                     "authorization": value["authorization"]}))
    if not passed:
        raise SystemExit(1)


def preaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    add(checks, "formal_outputs_absent", not any(x.exists() for x in (A, META, S, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1308_once")


def recompute(margins: np.ndarray, answers: np.ndarray, metadata: list[dict[str, Any]], th: dict[str, float]):
    baseline, blocked, correct, null, wrong = [margins[:, i] for i in range(5)]
    correct_gain = correct - blocked
    null_gain = null - blocked
    wrong_gain = wrong - blocked
    denominator = baseline - blocked
    recovery = correct_gain / np.where(np.abs(denominator) > EPS, denominator, np.nan)
    cells = {}
    for partition in PARTITIONS:
        cells[partition] = {}
        for surface in SURFACES:
            indices = [i for i, x in enumerate(metadata) if x["partition"] == partition and x["target_surface"] == surface]
            cells[partition][surface] = {
                "baseline_accuracy": float(np.mean(answers[indices, 0])),
                "blocked_target_identity_accuracy": float(np.mean(answers[indices, 1])),
                "correct_rescue_accuracy": float(np.mean(answers[indices, 2])),
                "self_retention_accuracy": float(np.mean(answers[indices, 5])),
            }
    metrics = {
        "baseline_accuracy": float(np.mean(answers[:, 0])),
        "blocked_target_identity_accuracy": float(np.mean(answers[:, 1])),
        "correct_rescue_accuracy": float(np.mean(answers[:, 2])),
        "self_retention_accuracy": float(np.mean(answers[:, 5])),
        "correct_rescue_gain_median": float(np.median(correct_gain)),
        "matched_null_gain_median": float(np.median(null_gain)),
        "wrong_attribute_gain_median": float(np.median(wrong_gain)),
        "recovery_fraction_median": float(np.nanmedian(recovery)),
        "valid_recovery_fraction": float(np.mean(np.isfinite(recovery))),
        "cross_surface_over_null_margin_ratio": float(np.median(correct_gain)) / max(abs(float(np.median(null_gain))), EPS),
        "pairwise_rescue_win_fraction": float(np.mean(correct_gain > np.maximum(null_gain, wrong_gain))),
        "natural_retention": float(np.mean(answers[:, [0, 5]])),
    }
    gates = {
        "finite": bool(np.isfinite(margins).all()),
        "recovery_defined": metrics["valid_recovery_fraction"] == 1.0,
        "recovery": metrics["recovery_fraction_median"] >= th["cross_surface_rescue_recovery_fraction_median_min"],
        "null_ratio": metrics["cross_surface_over_null_margin_ratio"] >= th["cross_surface_over_null_margin_ratio_min"],
        "pairwise_win": metrics["pairwise_rescue_win_fraction"] >= th["pairwise_rescue_win_fraction_min"],
        "natural_retention": metrics["natural_retention"] >= th["natural_retention_min"],
    }
    for partition in PARTITIONS:
        for surface in SURFACES:
            cell = cells[partition][surface]
            prefix = f"{partition}_{surface}"
            gates[f"{prefix}_baseline"] = cell["baseline_accuracy"] >= th["baseline_accuracy_min"]
            gates[f"{prefix}_blocked"] = cell["blocked_target_identity_accuracy"] <= th["blocked_target_identity_accuracy_max"]
            gates[f"{prefix}_rescue"] = cell["correct_rescue_accuracy"] >= th["cross_surface_rescue_accuracy_min"]
            gates[f"{prefix}_self"] = cell["self_retention_accuracy"] >= th["natural_retention_min"]
    return {"metrics": metrics, "cells": cells, "gates": gates, "all_gates_passed": all(gates.values())}


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    z = np.load(A, allow_pickle=False)
    margins = z["identity1_minus_identity0_margin"]
    answers = z["target_identity_correct"]
    add(checks, "array_shapes", margins.shape == answers.shape == (192, 6), [margins.shape, answers.shape])
    add(checks, "finite", np.isfinite(margins).all(), "finite")
    metadata = load(META)["case_metadata"]
    result = recompute(margins, answers, metadata, protocol["thresholds"])
    summary = load(S)
    add(checks, "metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "cells", summary["cells"] == result["cells"], result["cells"])
    add(checks, "gates", summary["gates"] == result["gates"], result["gates"])
    authorization = "close_c033_with_cross_surface_rescue_candidate" if result["all_gates_passed"] else "close_c033_at_rescue_boundary"
    final = load(F)
    add(checks, "authorization",
        final["authorization"] == authorization and final["all_gates_passed"] == result["all_gates_passed"]
        and final["c033_closed"] is True, final)
    qa = load(META)["model_audit"]
    add(checks, "fp16", qa["has_fp16_parameters"] and not qa["has_quantized_modules"], qa)
    add(checks, "formal_budget", load(C)["formal_runs_consumed"] == 1, load(C))
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
