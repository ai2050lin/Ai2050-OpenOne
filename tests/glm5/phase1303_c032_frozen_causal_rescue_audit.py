#!/usr/bin/env python3
"""Independent pre/post audit for Phase1303; deliberately does not import the main script."""
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
OUT = T / "result/phase1303_c032_frozen_causal_rescue"
PARENT = T / "result/phase1302_c032_event_identity_path"
CONTRACT = T / "result/phase1299_c032_execution_compiler_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_causal_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/causal_arrays.npz"
META = OUT / "raw/run_metadata.json"
S = OUT / "analysis/causal_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1303_c032_frozen_causal_rescue.py"
SCRIPT = Path(__file__).resolve()
PARTITIONS = ("confirmation", "holdout")
CELLS = (
    ("user_answer_cue_end", 29),
    ("user_answer_cue_end", 30),
    ("assistant_answer_boundary", 25),
    ("assistant_answer_boundary", 26),
)
ARMS = ("neutral", "correct", "matched_null", "wrong_entity", "wrong_attribute")
EPS = 1e-12
TH = {
    "correct_donor_signed_gain_median_min": 0.5,
    "correct_over_wrong_donor_ratio_min": 1.25,
    "correct_over_matched_null_ratio_min": 1.25,
    "pairwise_correct_donor_win_fraction_min": 0.75,
    "confirmation_holdout_each_min": 0.70,
    "natural_behavior_retention_min": 0.99,
}


def canonical(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v: Any) -> str:
    return hashlib.sha256(canonical(v).encode()).hexdigest()


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
    add(
        checks,
        "source_hashes",
        protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)},
        protocol["source_hashes"],
    )
    add(
        checks,
        "parent_authorization",
        load(PARENT / "analysis/final.json")["authorization"] == "phase1303_frozen_causal_rescue"
        and load(PARENT / "audit/independent_final_audit.json")["all_checks_passed"],
        "Phase1302",
    )
    add(checks, "thresholds", protocol["thresholds"] == TH, protocol["thresholds"])
    add(checks, "contract_thresholds", load(CONTRACT / "protocol/preregistration.json")["causal"]["thresholds"] == TH, TH)
    manifest = rows(M)
    add(checks, "manifest_count", len(manifest) == 192, len(manifest))
    add(checks, "manifest_hash", protocol["manifest"]["sha256"] == sha(M), protocol["manifest"])
    add(
        checks,
        "partition_balance",
        {p: sum(x["partition"] == p for x in manifest) for p in PARTITIONS} == {"confirmation": 96, "holdout": 96},
        protocol["manifest"]["partition_counts"],
    )
    add(checks, "unique_cases", len({x["case_key"] for x in manifest}) == 192, "unique")
    add(
        checks,
        "wrong_entity_disjoint",
        all(
            not set(x["target_state0"]["candidate_ids"]).intersection(x["wrong_entity_state1"]["candidate_ids"])
            for x in manifest
        ),
        "next-profile candidates are disjoint",
    )
    add(
        checks,
        "wrong_attribute_identity",
        all(
            x["wrong_attribute_state1"]["candidate_ids"] == x["target_state0"]["candidate_ids"]
            and x["wrong_attribute_state1"]["gold_position"] == x["identity_positions"][0]
            for x in manifest
        ),
        "same roster and state0 identity",
    )
    add(checks, "cells", protocol["cells"] == [{"event": e, "depth": d} for e, d in CELLS], protocol["cells"])
    add(checks, "arms", protocol["arms"] == list(ARMS + ("self_state1",)), protocol["arms"])
    add(
        checks,
        "hard_stops",
        protocol["hard_stops"]
        == [
            "No discovery partition",
            "No new event/depth/component scan",
            "No donor reselection",
            "No threshold modification",
            "No second formal model run",
        ],
        protocol["hard_stops"],
    )
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(x["passed"] for x in checks)
    value = {
        "phase": 1303,
        "campaign": "C032",
        "audit_stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(x["passed"] for x in checks),
        "total_count": len(checks),
        "all_checks_passed": passed,
        "authorization": authorization if passed else "none",
        "protocol_digest": load(P)["protocol_digest"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": value["passed_count"], "total": value["total_count"], "authorization": value["authorization"]}))
    if not passed:
        raise SystemExit(1)


def preaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    add(checks, "formal_outputs_absent", not any(x.exists() for x in (A, META, S, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1303_once")


def recompute(gains: np.ndarray, answers: np.ndarray, retention: np.ndarray, metadata: list[dict[str, Any]]) -> dict[str, Any]:
    correct = gains[:, :, 1].reshape(-1)
    null = gains[:, :, 2].reshape(-1)
    wrong_entity = gains[:, :, 3].reshape(-1)
    wrong_attribute = gains[:, :, 4].reshape(-1)
    correct_median = float(np.median(correct))
    wrong_scale = max(abs(float(np.median(wrong_entity))), abs(float(np.median(wrong_attribute))), EPS)
    null_scale = max(abs(float(np.median(null))), EPS)
    part_accuracy = {}
    for partition in PARTITIONS:
        indices = [i for i, x in enumerate(metadata) if x["partition"] == partition]
        part_accuracy[partition] = float(np.mean(answers[indices, :, 1]))
    metrics = {
        "correct_donor_signed_gain_median": correct_median,
        "matched_null_signed_gain_median": float(np.median(null)),
        "wrong_entity_signed_gain_median": float(np.median(wrong_entity)),
        "wrong_attribute_signed_gain_median": float(np.median(wrong_attribute)),
        "correct_over_wrong_donor_ratio": correct_median / wrong_scale,
        "correct_over_matched_null_ratio": correct_median / null_scale,
        "pairwise_correct_donor_win_fraction": float(
            np.mean(correct > np.maximum.reduce([null, wrong_entity, wrong_attribute]))
        ),
        "correct_identity_accuracy_by_partition": part_accuracy,
        "natural_behavior_retention": float(np.mean(retention)),
    }
    gates = {
        "finite": bool(np.isfinite(gains).all()),
        "signed_gain": metrics["correct_donor_signed_gain_median"] >= TH["correct_donor_signed_gain_median_min"],
        "wrong_ratio": metrics["correct_over_wrong_donor_ratio"] >= TH["correct_over_wrong_donor_ratio_min"],
        "null_ratio": metrics["correct_over_matched_null_ratio"] >= TH["correct_over_matched_null_ratio_min"],
        "pairwise_win": metrics["pairwise_correct_donor_win_fraction"] >= TH["pairwise_correct_donor_win_fraction_min"],
        "confirmation_identity": part_accuracy["confirmation"] >= TH["confirmation_holdout_each_min"],
        "holdout_identity": part_accuracy["holdout"] >= TH["confirmation_holdout_each_min"],
        "natural_retention": metrics["natural_behavior_retention"] >= TH["natural_behavior_retention_min"],
    }
    return {"metrics": metrics, "gates": gates, "all_gates_passed": all(gates.values())}


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    z = np.load(A, allow_pickle=False)
    gains = z["signed_gain"]
    answers = z["target_identity_correct"]
    retention = z["natural_retention"]
    add(
        checks,
        "array_shapes",
        gains.shape == answers.shape == (192, 4, 5) and retention.shape == (192, 4, 2),
        [gains.shape, answers.shape, retention.shape],
    )
    add(checks, "finite", np.isfinite(gains).all(), "finite")
    metadata = load(META)["case_metadata"]
    result = recompute(gains, answers, retention, metadata)
    summary = load(S)
    add(checks, "metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "gates", summary["gates"] == result["gates"], result["gates"])
    authorization = (
        "close_c032_mechanism_stage_with_causal_sufficiency_candidate"
        if result["all_gates_passed"]
        else "close_c032_with_descriptive_path_only"
    )
    final = load(F)
    add(
        checks,
        "authorization",
        final["authorization"] == authorization
        and final["all_gates_passed"] == result["all_gates_passed"]
        and final["c032_closed"] is True,
        final,
    )
    qa = load(META)["model_audit"]
    add(checks, "fp16", qa["has_fp16_parameters"] and not qa["has_quantized_modules"], qa)
    add(checks, "formal_budget", load(C)["formal_runs_consumed"] == 1, load(C))
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
