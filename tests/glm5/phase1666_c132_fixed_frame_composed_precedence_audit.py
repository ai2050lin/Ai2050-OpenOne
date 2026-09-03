#!/usr/bin/env python3
"""Independent C132 contract and behavior-failure audit; final audit is added only if capture is authorized."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1666_c132_fixed_frame_composed_precedence"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1666_c132_fixed_frame_composed_precedence as c132


def contract_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    checks = {
        "internal": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "counts": len(cases) == 256 and len(compiled) == 256,
        "balanced_truth": sum(row["truth_factor"] == 1 for row in cases) == 128,
        "query_fixed": all(row["query_left"] == row["values"][0] and row["query_right"] == row["values"][2] for row in cases),
        "single_link_nulls": protocol["zero_models"]["first_link_only"] == 0.75 and protocol["zero_models"]["second_link_only"] == 0.75,
        "fixed_main_frame": all("Route record:" in row["prompt"] and "Continuation:" in row["prompt"] and "Schedule note" not in row["prompt"] for row in cases),
        "roles": all(set(row["role_positions"]) == set(c132.ROLES) for row in compiled),
        "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "boundary": "not by itself a composition operator" in protocol["claim_boundary"],
    }
    report = {"phase": 1666, "campaign": "C132", "stage": "contract", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "run_c132_behavior" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def failure_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.load(OUT / "analysis/behavior_gate.json")
    closure = core.load(OUT / "analysis/closure.json")
    checks = {
        "contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"],
        "behavior_integrity": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"],
        "behavior_failed": not behavior["gate_passed"] and behavior["summary"]["global_accuracy"] < protocol["behavior_gate"]["global_accuracy_min"],
        "three_named_gates_failed": behavior["summary"]["global_accuracy"] < 0.95 and behavior["summary"]["by_truth"]["-1"] < 0.90 and behavior["summary"]["margin_over_best_single_link"] < 0.20,
        "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists(),
        "no_confirmation": not (OUT / "analysis/confirmation.json").exists(),
        "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "boundary": "no embeddings, HiddenStates" in closure["claim_boundary"] and "C129 transfer" in closure["claim_boundary"],
    }
    report = {"phase": 1666, "campaign": "C132", "stage": "behavior_failure_closure", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": False, "authorization": "close_composition_branch" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_behavior_failure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("contract", "failure"):
        raise SystemExit(f"usage: {Path(__file__).name} {{contract|failure}}")
    {"contract": contract_audit, "failure": failure_audit}[sys.argv[1]]()
