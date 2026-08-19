#!/usr/bin/env python3
"""Independent pre/post audit for Phase1310."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PHASE = 1310
CAMPAIGN = "C034"
OUT = T / "result/phase1310_c034_qwen3_typed_behavior"
PARENT = T / "result/phase1309_c034_typed_response_camera_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_generation_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GEN = OUT / "raw/list_free_generations.jsonl"
S = OUT / "analysis/behavior_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1310_c034_qwen3_typed_behavior.py"
SCRIPT = Path(__file__).resolve()
MATERIAL = PARENT / "material/frozen_typed_response_pairs.jsonl"
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")


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


def rate(values: Any) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def recompute(candidate: list[dict[str, Any]], generation: list[dict[str, Any]], th: dict[str, float]):
    partition = {p: rate(x["correct"] for x in candidate if x["partition"] == p) for p in PARTITIONS}
    attribute = {a: rate(x["correct"] for x in candidate if x["attribute"] == a) for a in ATTRS}
    surface = {s: rate(x["correct"] for x in candidate if x["surface"] == s) for s in SURFACES}
    active_groups = defaultdict(list)
    families = defaultdict(list)
    for x in candidate:
        if x["panel"] == "active":
            active_groups[x["pair_key"]].append(x)
            families[(x["partition"], x["profile_index"], x["surface"], x["binding_state"])].append(x)
    gen_groups = defaultdict(list)
    for x in generation:
        gen_groups[x["pair_key"]].append(x)
    metrics = {
        "finite_fraction": rate(x["finite"] for x in candidate),
        "candidate_accuracy": rate(x["correct"] for x in candidate),
        "partition_accuracy": partition, "attribute_accuracy": attribute, "surface_accuracy": surface,
        "active_pair_success": rate(len(v) == 2 and all(x["correct"] for x in v) for v in active_groups.values()),
        "attribute_family_success": rate(len(v) == 6 and all(x["correct"] for x in v) for v in families.values()),
        "generation_coverage": rate(x["covered"] for x in generation),
        "generation_label_accuracy": rate(x["label_correct"] for x in generation),
        "generation_pair_success": rate(len(v) == 2 and all(x["label_correct"] for x in v) for v in gen_groups.values()),
    }
    gates = {
        "finite": metrics["finite_fraction"] >= th["finite_fraction_min"],
        "candidate": metrics["candidate_accuracy"] >= th["candidate_accuracy_min"],
        "partition": min(partition.values()) >= th["partition_accuracy_min"],
        "attribute": min(attribute.values()) >= th["attribute_accuracy_min"],
        "surface": min(surface.values()) >= th["surface_accuracy_min"],
        "active_pair": metrics["active_pair_success"] >= th["active_pair_success_min"],
        "attribute_family": metrics["attribute_family_success"] >= th["attribute_family_success_min"],
        "generation_coverage": metrics["generation_coverage"] >= th["generation_coverage_min"],
        "generation_accuracy": metrics["generation_label_accuracy"] >= th["generation_label_accuracy_min"],
        "generation_pair": metrics["generation_pair_success"] >= th["generation_pair_success_min"],
    }
    return {"metrics": metrics, "gates": gates, "all_gates_passed": all(gates.values())}


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(checks, "parent", load(PARENT / "analysis/final.json")["authorization"] == "phase1310_qwen3_behavior_only"
        and load(PARENT / "audit/independent_final_audit.json")["all_checks_passed"], "Phase1309")
    add(checks, "material", protocol["material"]["sha256"] == sha(MATERIAL) and protocol["material"]["state_count"] == 1152, protocol["material"])
    generation = rows(M)
    add(checks, "generation_manifest", len(generation) == 384 and protocol["material"]["generation_manifest_sha256"] == sha(M), len(generation))
    add(checks, "generation_balance", all(sum(x["partition"] == p and x["attribute"] == a and x["surface"] == s for x in generation) == 16
        for p in ("confirmation", "holdout") for a in ATTRS for s in SURFACES), "16 states per cell")
    add(checks, "hidden_forbidden", protocol["hidden_states_read"] is False, protocol["hidden_states_read"])
    add(checks, "single_run", protocol["formal_run_budget"] == 1, protocol["formal_run_budget"])
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(x["passed"] for x in checks)
    result = {"phase": PHASE, "campaign": CAMPAIGN, "audit_stage": stage,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
              "checks": checks, "passed_count": sum(x["passed"] for x in checks), "total_count": len(checks),
              "all_checks_passed": passed, "authorization": authorization if passed else "none",
              "protocol_digest": load(P)["protocol_digest"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


def preaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    add(checks, "formal_outputs_absent", not any(x.exists() for x in (RAW, GEN, S, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1310_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    candidate = rows(RAW)
    generation = rows(GEN)
    add(checks, "counts", len(candidate) == 1152 and len(generation) == 384, [len(candidate), len(generation)])
    summary = load(S)
    add(checks, "hashes", summary["raw_hashes"] == {"candidate": sha(RAW), "generation": sha(GEN)}, summary["raw_hashes"])
    result = recompute(candidate, generation, protocol["thresholds"])
    add(checks, "metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "gates", summary["gates"] == result["gates"], result["gates"])
    authorization = "phase1311_typed_trajectory_only" if result["all_gates_passed"] else "close_c034_without_hidden"
    final = load(F)
    add(checks, "authorization", final["authorization"] == authorization and final["all_gates_passed"] == result["all_gates_passed"] and final["hidden_states_read"] is False, final)
    qa = summary["model_audit"]
    add(checks, "fp16", qa["has_fp16_parameters"] and not qa["has_quantized_modules"], qa)
    add(checks, "formal_budget", load(C)["formal_runs_consumed"] == 1, load(C))
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
