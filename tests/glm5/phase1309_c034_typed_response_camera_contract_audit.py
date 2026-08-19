#!/usr/bin/env python3
"""Independent zero-model audit for Phase1309; does not import the contract builder."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tests/glm5"
PHASE = 1309
CAMPAIGN = "C034"
OUT = T / "result/phase1309_c034_typed_response_camera_contract"
P = OUT / "protocol/preregistration.json"
M = OUT / "material/frozen_typed_response_pairs.jsonl"
N = OUT / "material/semantic_naturalness_review.json"
K = OUT / "analysis/known_truth_camera_calibration.json"
MACHINE = OUT / "audit/tokenizer_semantic_program_audit.json"
AUDIT = OUT / "audit/independent_final_audit.json"
FINAL = OUT / "analysis/final.json"
MAIN = T / "phase1309_c034_typed_response_camera_contract.py"
SCRIPT = Path(__file__).resolve()
SOURCE = T / "result/phase1304_c033_role_typed_causal_graph_contract/material/frozen_role_typed_lookup_cases.jsonl"
PARTITIONS = ("discovery", "confirmation", "holdout")
ATTRS = ("color", "material", "location", "size", "shape", "status")
SURFACES = ("catalog_prose", "inventory_ledger")
PANELS = ("active", "matched_null")


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


def calibration_recompute() -> dict[str, float]:
    examples = []
    for target in range(len(ATTRS)):
        signatures = {
            "generic": [1] * len(ATTRS),
            "typed": [int(a == target) for a in range(len(ATTRS))],
            "mixed": [int(a in {target, (target + 1) % len(ATTRS)}) for a in range(len(ATTRS))],
            "null": [0] * len(ATTRS),
        }
        for label, signature in signatures.items():
            if sum(signature) == len(ATTRS):
                prediction = "generic"
            elif sum(signature) == 1 and signature[target] == 1:
                prediction = "typed"
            elif sum(signature) == 2 and signature[target] == 1:
                prediction = "mixed"
            else:
                prediction = "null"
            examples.append((label, signature[target], prediction))
    return {
        "collision": sum(single == 1 for label, single, _ in examples if label in {"generic", "typed"}) / (2 * len(ATTRS)),
        "accuracy": sum(label == prediction for label, _, prediction in examples) / len(examples),
        "null_fp": sum(prediction != "null" for label, _, prediction in examples if label == "null") / len(ATTRS),
    }


def main() -> None:
    protocol = load(P)
    material = rows(M)
    checks: list[dict[str, Any]] = []
    timeless = {k: v for k, v in protocol.items() if k not in {"created_at_utc", "protocol_digest"}}
    add(checks, "protocol_digest", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == {"main": sha(MAIN), "auditor": sha(SCRIPT)}, protocol["source_hashes"])
    add(checks, "c033_closed", load(T / "result/phase1308_c033_cross_surface_block_rescue/analysis/final.json")["authorization"] == "close_c033_at_rescue_boundary", "closed")
    add(checks, "material_count", len(material) == 576 and sum(len(x["states"]) for x in material) == 1152, len(material))
    add(checks, "material_hash", protocol["material"]["material_sha256"] == sha(M), sha(M))
    add(checks, "unique_pairs", len({x["pair_key"] for x in material}) == 576, "unique")
    add(checks, "factorial_balance",
        all(sum(x["partition"] == p and x["attribute"] == a and x["surface"] == s and x["panel"] == panel for x in material) == 8
            for p in PARTITIONS for a in ATTRS for s in SURFACES for panel in PANELS), "8 profiles per cell")
    add(checks, "candidate_ids", all(len(set(state["candidate_ids"])) == 3 for x in material for state in x["states"]), "unique")
    add(checks, "role_order", all(state["positions"]["query_end"] < state["positions"]["answer_boundary"] < len(state["ids"])
                                   for x in material for state in x["states"]), "query before boundary")
    add(checks, "active_gold_change", all(x["identity_positions"][0] != x["identity_positions"][1] for x in material if x["panel"] == "active"), "active")
    add(checks, "null_gold_stable", all(x["states"][0]["gold_position"] == x["states"][1]["gold_position"] for x in material if x["panel"] == "matched_null"), "null")
    source = rows(SOURCE)
    unique = all(sum(fields[r["attribute"]] == r["target_value"] for fields in r["assignments"].values()) == 1 for r in source)
    naturalness = load(N)
    add(checks, "semantic_unique", unique and naturalness["semantic_uniqueness_recomputed"], unique)
    add(checks, "naturalness", naturalness["all_checks_passed"] and naturalness["source_material_reused_intentionally"], naturalness)
    camera = calibration_recompute()
    recorded = load(K)
    add(checks, "single_readout_collision", camera["collision"] == recorded["single_target_readout_generic_typed_collision_fraction"] == 1.0, camera)
    add(checks, "multi_readout_camera", camera["accuracy"] == recorded["multi_readout_classification_accuracy"] == 1.0 and camera["null_fp"] == 0.0, camera)
    add(checks, "machine_audit", load(MACHINE)["all_machine_checks_passed"], load(MACHINE))
    add(checks, "frozen_trajectory", protocol["trajectory"]["query_depths"] == [8, 14, 20, 26, 32]
        and protocol["trajectory"]["late_comparator"] == {"role": "answer_boundary", "depth": 26}, protocol["trajectory"])
    add(checks, "hard_stops", protocol["hard_stops"] == [
        "No model weights before Phase1309 independent audit passes",
        "No hidden states before Phase1310 behavior gate passes",
        "No causal intervention before Phase1311 trajectory gate passes",
        "No post-unblinding role, depth, metric, donor, threshold, or partition change",
        "No head, MLP, neuron, or subspace search in C034",
        "C034 closes after Phase1312 or at the first failed gate",
    ], protocol["hard_stops"])
    passed = all(x["passed"] for x in checks)
    authorization = "phase1310_qwen3_behavior_only" if passed else "close_c034_before_model"
    final = load(FINAL)
    add(checks, "final_authorization", final["authorization"] == authorization and final["all_gates_passed"] == passed, final)
    passed = all(x["passed"] for x in checks)
    result = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "auditor_imports_main": False, "checks": checks,
              "passed_count": sum(x["passed"] for x in checks), "total_count": len(checks),
              "all_checks_passed": passed, "authorization": authorization if passed else "none",
              "protocol_digest": protocol["protocol_digest"]}
    save = lambda path, value: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))
    save(AUDIT, result)
    print(canonical({"passed": result["passed_count"], "total": result["total_count"], "authorization": result["authorization"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
