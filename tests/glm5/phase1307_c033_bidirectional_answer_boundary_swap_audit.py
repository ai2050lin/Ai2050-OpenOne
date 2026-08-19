#!/usr/bin/env python3
"""Independent pre/post audit for Phase1307; does not import the execution script."""
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
OUT = T / "result/phase1307_c033_bidirectional_answer_boundary_swap"
PARENT = T / "result/phase1306_c033_frozen_answer_boundary_hidden"
CONTRACT = T / "result/phase1304_c033_role_typed_causal_graph_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_swap_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
A = OUT / "raw/swap_arrays.npz"
META = OUT / "raw/run_metadata.json"
S = OUT / "analysis/swap_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1307_c033_bidirectional_answer_boundary_swap.py"
SCRIPT = Path(__file__).resolve()
PARTITIONS = ("confirmation", "holdout")
DIRECTIONS = ("state0_to_state1", "state1_to_state0")
ARMS = ("neutral", "correct", "matched_null", "wrong_entity", "wrong_attribute")
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
        load(PARENT / "analysis/final.json")["authorization"] == "phase1307_bidirectional_swap_only"
        and load(PARENT / "audit/independent_final_audit.json")["all_checks_passed"], "Phase1306")
    contract = load(CONTRACT / "protocol/preregistration.json")["bidirectional_swap"]
    add(checks, "thresholds", protocol["thresholds"] == contract["thresholds"], protocol["thresholds"])
    add(checks, "fixed_cell", protocol["event"] == "assistant_answer_boundary" and protocol["depth"] == 26,
        {"event": protocol["event"], "depth": protocol["depth"]})
    manifest = rows(M)
    add(checks, "manifest_count", len(manifest) == 192, len(manifest))
    add(checks, "manifest_hash", protocol["manifest"]["sha256"] == sha(M), protocol["manifest"])
    add(checks, "partition_balance",
        {p: sum(x["partition"] == p for x in manifest) for p in PARTITIONS} == {"confirmation": 96, "holdout": 96},
        protocol["manifest"]["partition_counts"])
    add(checks, "unique_cases", len({x["case_key"] for x in manifest}) == 192, "unique")
    add(checks, "two_directions",
        all([d["name"] for d in x["directions"]] == list(DIRECTIONS) for x in manifest), list(DIRECTIONS))
    add(checks, "direction_reversal",
        all(x["directions"][0]["source_state"] == 0 and x["directions"][0]["destination_state"] == 1
            and x["directions"][1]["source_state"] == 1 and x["directions"][1]["destination_state"] == 0
            for x in manifest), "0->1 and 1->0")
    add(checks, "correct_donor",
        all(d["correct_donor"]["case_id"] == d["destination"]["case_id"]
            for x in manifest for d in x["directions"]), "destination state")
    add(checks, "wrong_entity_disjoint",
        all(not set(d["target"]["candidate_ids"]).intersection(d["wrong_entity_donor"]["candidate_ids"])
            for x in manifest for d in x["directions"]), "disjoint roster")
    add(checks, "wrong_attribute_source_identity",
        all(d["wrong_attribute_donor"]["candidate_ids"] == d["target"]["candidate_ids"]
            and d["wrong_attribute_donor"]["gold_position"] == d["source_identity_position"]
            for x in manifest for d in x["directions"]), "same roster, source identity")
    add(checks, "arms", protocol["arms"] == list(ARMS) + ["self_patch"], protocol["arms"])
    add(checks, "hard_stops", protocol["hard_stops"] == [
        "No discovery partition", "No new event, depth, component, or donor search", "No one-direction fallback",
        "No threshold modification", "No second formal model run"], protocol["hard_stops"])
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(x["passed"] for x in checks)
    value = {"phase": 1307, "campaign": "C033", "audit_stage": stage,
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
    write(PRE, checks, "pre_model", "run_phase1307_once")


def recompute(gains: np.ndarray, answers: np.ndarray, retention: np.ndarray,
              metadata: list[dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    correct = gains[:, :, 1].reshape(-1)
    null = gains[:, :, 2].reshape(-1)
    wrong_entity = gains[:, :, 3].reshape(-1)
    wrong_attribute = gains[:, :, 4].reshape(-1)
    correct_median = float(np.median(correct))
    direction_partition_accuracy = {}
    for di, direction in enumerate(DIRECTIONS):
        direction_partition_accuracy[direction] = {}
        for partition in PARTITIONS:
            indices = [i for i, x in enumerate(metadata) if x["partition"] == partition]
            direction_partition_accuracy[direction][partition] = float(np.mean(answers[indices, di, 1]))
    metrics = {
        "correct_signed_gain_median": correct_median,
        "matched_null_signed_gain_median": float(np.median(null)),
        "wrong_entity_signed_gain_median": float(np.median(wrong_entity)),
        "wrong_attribute_signed_gain_median": float(np.median(wrong_attribute)),
        "correct_over_matched_null_ratio": correct_median / max(abs(float(np.median(null))), EPS),
        "pairwise_correct_win_fraction": float(np.mean(correct > np.maximum.reduce([null, wrong_entity, wrong_attribute]))),
        "direction_partition_accuracy": direction_partition_accuracy,
        "natural_retention": float(np.mean(retention)),
    }
    gates = {
        "finite": bool(np.isfinite(gains).all()),
        "signed_gain": metrics["correct_signed_gain_median"] >= th["signed_margin_gain_median_min"],
        "null_ratio": metrics["correct_over_matched_null_ratio"] >= th["correct_over_matched_null_ratio_min"],
        "pairwise_win": metrics["pairwise_correct_win_fraction"] >= th["pairwise_correct_win_fraction_min"],
        "natural_retention": metrics["natural_retention"] >= th["natural_retention_min"],
    }
    for direction in DIRECTIONS:
        for partition in PARTITIONS:
            gates[f"{direction}_{partition}"] = (
                direction_partition_accuracy[direction][partition] >= th["direction_partition_accuracy_min"]
            )
    return {"metrics": metrics, "gates": gates, "all_gates_passed": all(gates.values())}


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    z = np.load(A, allow_pickle=False)
    gains = z["signed_gain"]
    answers = z["target_identity_correct"]
    retention = z["natural_retention"]
    margins = z["destination_margin"]
    add(checks, "array_shapes",
        gains.shape == answers.shape == margins.shape == (192, 2, 5) and retention.shape == (192, 2, 2),
        [gains.shape, answers.shape, margins.shape, retention.shape])
    add(checks, "finite", np.isfinite(gains).all() and np.isfinite(margins).all(), "finite")
    add(checks, "neutral_gain_zero", np.array_equal(gains[:, :, 0], np.zeros((192, 2), np.float32)),
        float(np.max(np.abs(gains[:, :, 0]))))
    metadata = load(META)["case_metadata"]
    result = recompute(gains, answers, retention, metadata, protocol["thresholds"])
    summary = load(S)
    add(checks, "metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "gates", summary["gates"] == result["gates"], result["gates"])
    authorization = "phase1308_cross_surface_block_rescue_only" if result["all_gates_passed"] else "close_c033_without_rescue"
    final = load(F)
    add(checks, "authorization",
        final["authorization"] == authorization and final["all_gates_passed"] == result["all_gates_passed"], final)
    qa = load(META)["model_audit"]
    add(checks, "fp16", qa["has_fp16_parameters"] and not qa["has_quantized_modules"], qa)
    add(checks, "formal_budget", load(C)["formal_runs_consumed"] == 1, load(C))
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
