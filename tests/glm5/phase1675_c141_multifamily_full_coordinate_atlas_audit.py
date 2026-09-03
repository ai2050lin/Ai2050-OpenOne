#!/usr/bin/env python3
"""Independent contract/final audit for C141."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1675_c141_multifamily_full_coordinate_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def contract() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    checks = {
        "internal": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "rows": len(rows) == 1280,
        "arms": {row["arm"] for row in rows} == set(protocol["arms"]),
        "factorial": all(set(row["factors"]) == {"f1", "f2", "f3"} for row in rows),
        "roles": all(set(row["role_positions"]) == set(protocol["roles"]) for row in rows),
        "code_balance": sum(row["gold_position"] == 0 for row in rows) == 640,
        "tokens": sum(len(row["prompt_ids"]) for row in rows) == protocol["total_actual_tokens"],
        "continue_after_error": "continues" in protocol["behavior_policy"],
        "source_hash": core.sha(TESTS / "result/phase1674_c140_identifiability_and_master_contract/audit/independent_contract_audit.json") == protocol["source_hashes"]["C140"],
    }
    report = {"phase": 1675, "campaign": "C141", "stage": "contract", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "run_authoritative_qwen3_capture"}
    core.save(OUT / "audit/independent_contract_audit.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


def final() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    run = core.load(OUT / "analysis/authoritative_run.json")
    full_path = OUT / "raw/qwen3_all_token_all_checkpoint.bf16.npy"
    role_path = OUT / "raw/qwen3_six_role_field.bf16.npy"
    full = np.load(full_path, mmap_mode="r")
    role = np.load(role_path, mmap_mode="r")
    checks = {
        "contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"],
        "internal": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "full_shape": list(full.shape) == protocol["expected_full_field_shape"],
        "role_shape": list(role.shape) == protocol["expected_role_field_shape"],
        "full_hash": core.sha(full_path) == run["capture"]["full_sha256"],
        "role_hash": core.sha(role_path) == run["capture"]["role_sha256"],
        "behavior_typed": set(run["behavior"]["arm"]) == set(protocol["arms"]),
        "continuation": run["authorization"] == "analyze_C142_mobius_regardless_of_behavior",
    }
    report = {"phase": 1675, "campaign": "C141", "stage": "final", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_behavior_gate_passed": run["behavior"]["observation_gate_passed"], "authorization": "start_C142"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"contract", "final"}:
        raise SystemExit("contract|final")
    globals()[sys.argv[1]]()
