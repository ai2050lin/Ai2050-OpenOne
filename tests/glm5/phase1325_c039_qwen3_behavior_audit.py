#!/usr/bin/env python3
"""Independent pre/post audit for Phase1325; never imports the executor."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PHASE, CAMPAIGN = 1325, "C039"
OUT = T / "result/phase1325_c039_qwen3_behavior"
PARENT = T / "result/phase1324_c039_exact_truth_scope_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "protocol/frozen_generation_manifest.jsonl"
PRE = OUT / "audit/independent_preaudit.json"
POST = OUT / "audit/independent_final_audit.json"
RAW = OUT / "raw/candidate_scores.jsonl"
GEN = OUT / "raw/free_generations.jsonl"
S = OUT / "analysis/behavior_summary.json"
F = OUT / "analysis/final.json"
C = OUT / "protocol/formal_run_complete.json"
MAIN = T / "phase1325_c039_qwen3_behavior.py"
SCRIPT = Path(__file__).resolve()
MATERIAL = PARENT / "material/frozen_truth_scope_pairs.jsonl"
PARTITIONS = ("discovery", "confirmation", "holdout")
SURFACES = ("prefix_scope", "reported_statement")
PANELS = ("active_single", "active_outer_context_true", "active_outer_context_false",
          "active_inner_context_true", "active_inner_context_false", "wrong_scope", "lexical_null", "self_repeat")
ACTIVE = set(PANELS[:5])


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(values: Any) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def recompute(candidate: list[dict[str, Any]], generation: list[dict[str, Any]], th: dict[str, float]) -> dict[str, Any]:
    partition = {key: rate(x["correct"] for x in candidate if x["partition"] == key) for key in PARTITIONS}
    surface = {key: rate(x["correct"] for x in candidate if x["surface"] == key) for key in SURFACES}
    panel = {key: rate(x["correct"] for x in candidate if x["panel"] == key) for key in PANELS}
    active_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    generated_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidate:
        if item["panel"] in ACTIVE:
            active_pairs[item["pair_key"]].append(item)
    for item in generation:
        generated_pairs[item["pair_key"]].append(item)
    metrics = {
        "finite_fraction": rate(x["finite"] for x in candidate),
        "candidate_accuracy": rate(x["correct"] for x in candidate),
        "partition_accuracy": partition, "surface_accuracy": surface, "panel_accuracy": panel,
        "active_pair_success": rate(len(v) == 2 and all(x["correct"] for x in v) for v in active_pairs.values()),
        "generation_coverage": rate(x["covered"] for x in generation),
        "generation_accuracy": rate(x["label_correct"] for x in generation),
        "generation_pair_success": rate(len(v) == 2 and all(x["label_correct"] for x in v) for v in generated_pairs.values()),
    }
    gates = {
        "finite": metrics["finite_fraction"] >= th["finite_fraction_min"],
        "candidate": metrics["candidate_accuracy"] >= th["candidate_accuracy_min"],
        "partition": min(partition.values()) >= th["partition_accuracy_min"],
        "surface": min(surface.values()) >= th["surface_accuracy_min"],
        "panel": min(panel.values()) >= th["panel_accuracy_min"],
        "active_pair": metrics["active_pair_success"] >= th["active_pair_success_min"],
        "generation_coverage": metrics["generation_coverage"] >= th["generation_coverage_min"],
        "generation_accuracy": metrics["generation_accuracy"] >= th["generation_accuracy_min"],
        "generation_pair": metrics["generation_pair_success"] >= th["generation_pair_success_min"],
    }
    return {"metrics": metrics, "gates": gates, "all_gates_passed": all(gates.values())}


def base(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    parent_ok = load(PARENT / "analysis/final.json").get("authorization") == "phase1325_c039_qwen3_behavior_only"
    parent_ok &= load(PARENT / "audit/independent_final_audit.json").get("all_checks_passed", False)
    add(checks, "parent_authorization", parent_ok, "Phase1324")
    expected = {"parent_protocol": sha(PARENT / "protocol/preregistration.json"),
                "parent_final": sha(PARENT / "analysis/final.json"),
                "parent_audit": sha(PARENT / "audit/independent_final_audit.json"),
                "material": sha(MATERIAL), "generation_manifest": sha(M)}
    add(checks, "dependencies", protocol["dependencies"] == expected, protocol["dependencies"])
    material, generation = rows(MATERIAL), rows(M)
    add(checks, "material_counts", len(material) == 1152 and protocol["material"]["state_count"] == 2304, len(material))
    add(checks, "generation_count_hash", len(generation) == 960 and protocol["material"]["generation_manifest_sha256"] == sha(M), len(generation))
    add(checks, "generation_scope", Counter(x["partition"] for x in generation) == Counter({"confirmation": 480, "holdout": 480})
        and all(x["panel"] in ACTIVE and x["true_boundary"] == len(x["ids"]) - 1 for x in generation),
        dict(Counter(x["partition"] for x in generation)))
    add(checks, "behavior_only", protocol["hidden_states_read"] is False and protocol["formal_run_budget"] == 1
        and protocol["success_authorization"] == "phase1326_c039_composition_field_only", protocol["hard_stops"])
    return checks


def write(path: Path, checks: list[dict[str, Any]], stage: str, authorization: str) -> None:
    passed = all(item["passed"] for item in checks)
    value = {"phase": PHASE, "campaign": CAMPAIGN, "audit_stage": stage,
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "auditor_imports_main": False,
             "checks": checks, "passed_count": sum(item["passed"] for item in checks), "total_count": len(checks),
             "all_checks_passed": passed, "authorization": authorization if passed else "none",
             "protocol_digest": load(P)["protocol_digest"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical({"stage": stage, "passed": value["passed_count"], "total": value["total_count"],
                     "authorization": value["authorization"]}))
    if not passed:
        raise SystemExit(1)


def preaudit() -> None:
    checks = base(load(P))
    add(checks, "formal_outputs_absent", not any(path.exists() for path in (RAW, GEN, S, F, C)), "clear")
    write(PRE, checks, "pre_model", "run_phase1325_once")


def postaudit() -> None:
    protocol = load(P)
    checks = base(protocol)
    candidate, generation = rows(RAW), rows(GEN)
    add(checks, "raw_counts", len(candidate) == 2304 and len(generation) == 960,
        {"candidate": len(candidate), "generation": len(generation)})
    add(checks, "raw_balance", Counter(x["partition"] for x in candidate) == Counter({key: 768 for key in PARTITIONS})
        and all(np.isfinite(x["candidate_logits"]).all() for x in candidate), dict(Counter(x["partition"] for x in candidate)))
    result, summary = recompute(candidate, generation, protocol["thresholds"]), load(S)
    add(checks, "independent_metrics", summary["metrics"] == result["metrics"], result["metrics"])
    add(checks, "independent_gates", summary["gates"] == result["gates"]
        and summary["all_gates_passed"] == result["all_gates_passed"], result["gates"])
    authorization = "phase1326_c039_composition_field_only" if result["all_gates_passed"] else "close_c039_without_hidden"
    final = load(F)
    add(checks, "verdict_authorization", final["authorization"] == authorization
        and final["all_gates_passed"] == result["all_gates_passed"] and final["hidden_states_read"] is False, final)
    add(checks, "raw_hashes", summary["raw_hashes"] == {"candidate": sha(RAW), "generation": sha(GEN)}, summary["raw_hashes"])
    qa = summary["model_audit"]
    add(checks, "fp16_cuda", qa["has_fp16_parameters"] and not qa["has_quantized_modules"]
        and summary["cuda_peak_allocated_bytes"] > 0, {"qa": qa, "peak": summary["cuda_peak_allocated_bytes"]})
    complete = load(C)
    add(checks, "formal_budget_consumed", complete["formal_runs_consumed"] == 1
        and complete["protocol_digest"] == protocol["protocol_digest"], complete)
    write(POST, checks, "post_model", authorization)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "postaudit"))
    args = parser.parse_args()
    preaudit() if args.stage == "preaudit" else postaudit()
