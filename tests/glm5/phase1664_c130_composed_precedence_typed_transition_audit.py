#!/usr/bin/env python3
"""Independent C130 contract, field, confirmation, and heatmap audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1664_c130_composed_precedence_typed_transition"
C129 = TESTS / "result/phase1663_c129_direct_precedence_typed_transition"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1661_c127_typed_transition_language_family as c127
import phase1664_c130_composed_precedence_typed_transition as c130


def contract_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    units = core.rows(OUT / "material/units.jsonl")
    cases = core.rows(OUT / "material/cases.jsonl")
    compiled = core.rows(OUT / "compiled/qwen3.jsonl")
    checks = {
        "internal": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "counts": len(units) == 32 and len(cases) == 256 and len(compiled) == 256,
        "balanced_truth": sum(row["truth_factor"] == 1 for row in cases) == 128,
        "partitions": sum(row["partition"] == "discovery" for row in cases) == 128 and sum(row["partition"] == "confirmation" for row in cases) == 128,
        "query_fixed": all(row["query_left"] == row["values"][0] and row["query_right"] == row["values"][2] for row in cases),
        "single_link_nulls": protocol["zero_models"]["first_link_only"] == 0.75 and protocol["zero_models"]["second_link_only"] == 0.75,
        "roles": all(set(row["role_positions"]) == set(c130.ROLES) for row in compiled),
        "typed_reference": protocol["cross_family_frozen_candidate"]["transition_index"] == 35 and protocol["cross_family_frozen_candidate"]["role"] == "boundary",
        "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "claim_boundary": "not by itself a composition operator" in protocol["claim_boundary"],
    }
    report = {"phase": 1664, "campaign": "C130", "stage": "contract", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": "run_c130_behavior" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_contract_audit.json", report)
    print(json.dumps(report, indent=2))


def final_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    freeze = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    confirmation = core.load(OUT / "analysis/confirmation.json")
    closure = core.load(OUT / "analysis/closure.json")
    raw = np.load(OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy", mmap_mode="r")
    fields = np.load(OUT / "analysis/unit_truth_role_typed_checkpoint.float32.npy", mmap_mode="r")
    rows = core.rows(OUT / "compiled/qwen3.jsonl")
    units = core.rows(OUT / "material/units.jsonl")
    lookup = {row["unit_id"]: index for index, row in enumerate(units)}
    recomputed = np.zeros_like(fields)
    for row_index, row in enumerate(rows):
        recomputed[lookup[row["unit_id"]]] += float(row["truth_factor"]) / 8.0 * c127.decode(raw[row_index])
    direct = np.load(OUT / "analysis/c129_direct_reference_increment.float32.npy")
    composed_discovery = np.load(OUT / "analysis/discovery_composed_fixed_increment.float32.npy")
    residual_discovery = np.load(OUT / "analysis/discovery_composition_residual.float32.npy")
    alpha = float(freeze["cross_family"]["alpha_discovery"])
    payload = core.load(PUBLIC)
    batch = payload["c130_composed_precedence_typed_transition_batch"]
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "behavior": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"],
        "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"],
        "discovery": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"],
        "confirmation": core.load(OUT / "audit/internal_confirmation_audit.json")["all_integrity_checks_passed"],
        "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "shapes": list(raw.shape) == [256, 7, 38, 2560] and list(fields.shape) == [32, 7, 38, 2560],
        "field_recompute": np.array_equal(recomputed, np.asarray(fields)),
        "typed_checkpoints": len(protocol["checkpoints"]) == 38 and protocol["checkpoints"][0] == "embedding" and protocol["checkpoints"][-1] == "post_final_norm",
        "direct_reference": np.array_equal(direct, np.load(C129 / "analysis/discovery_nominee_increment.float32.npy")),
        "residual_recompute": np.allclose(residual_discovery, composed_discovery - alpha * direct, rtol=0.0, atol=1e-6),
        "frozen_before_confirmation": freeze["confirmation_partition_unread"] and freeze["authorization"] == "validate_c130_confirmation",
        "visualization": len(batch["effect_rows"]) == 150 and len(batch["cross_family_and_residual_rows"]) == 5 and len(batch["representative_raw_rows"]) == 49 and all(len(row["values"]) == 2560 for row in [*batch["effect_rows"], *batch["cross_family_and_residual_rows"], *batch["representative_raw_rows"]]),
        "asset": core.sha(PUBLIC) == closure["heatmap"]["sha256"] == core.load(OUT / "audit/internal_closure_audit.json")["asset_sha256"],
        "boundary": "not by itself a composition operator" in protocol["claim_boundary"] and "attention, MLP" in protocol["observation_policy"],
    }
    report = {"phase": 1664, "campaign": "C130", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gates": {"behavior": core.load(OUT / "analysis/behavior_gate.json")["gate_passed"], "within_family": confirmation["within_family"]["all_gates_passed"], "cross_family_common_response": confirmation["cross_family_common_response"]["all_gates_passed"], "composition_residual": confirmation["composition_residual"]["all_gates_passed"]}, "authorization": "integrate_client_append_phase1664_and_consider_c131" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_closure_audit.json", report)
    print(json.dumps(report, indent=2))


def failure_audit() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    behavior = core.load(OUT / "analysis/behavior_gate.json")
    closure = core.load(OUT / "analysis/closure.json")
    checks = {
        "contract": core.load(OUT / "audit/independent_contract_audit.json")["all_checks_passed"],
        "behavior_integrity": core.load(OUT / "audit/internal_behavior_audit.json")["all_integrity_checks_passed"],
        "behavior_failed": not behavior["gate_passed"] and behavior["summary"]["global_accuracy"] < protocol["behavior_gate"]["global_accuracy_min"],
        "surface_localized": behavior["summary"]["by_surface"]["-1"] < behavior["summary"]["by_surface"]["1"],
        "null_margin_failed": behavior["summary"]["margin_over_best_single_link"] < protocol["behavior_gate"]["global_margin_over_best_single_link_min"],
        "no_hiddenstate": not (OUT / "raw/qwen3_uniform_role_checkpoints.bf16.npy").exists(),
        "no_confirmation": not (OUT / "analysis/confirmation.json").exists(),
        "closure": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
        "boundary": "did not capture embeddings or HiddenStates" in closure["claim_boundary"],
    }
    report = {"phase": 1664, "campaign": "C130", "stage": "behavior_failure_closure", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_gate_passed": False, "authorization": "start_c131_repaired_interface" if all(checks.values()) else "stop"}
    core.save(OUT / "audit/independent_behavior_failure_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("contract", "failure", "final"):
        raise SystemExit(f"usage: {Path(__file__).name} {{contract|failure|final}}")
    {"contract": contract_audit, "failure": failure_audit, "final": final_audit}[sys.argv[1]]()
