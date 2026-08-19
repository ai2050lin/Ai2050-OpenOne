#!/usr/bin/env python3
"""Phase1434: close C071 after frozen cross-surface classification."""
from __future__ import annotations

import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
PYTHON = ROOT / ".venv/Scripts/python.exe"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1434, "C071"
OUT = TESTS / "result/phase1434_c071_campaign_closure"
P1430 = TESTS / "result/phase1430_c071_cross_surface_role_contract"
P1431 = TESTS / "result/phase1431_c071_cross_surface_behavior"
P1432 = TESTS / "result/phase1432_c071_role_map_camera"
P1433 = TESTS / "result/phase1433_c071_cross_surface_mechanism"
PHASES = (
    (1430, "c071_cross_surface_role_contract"),
    (1431, "c071_cross_surface_behavior"),
    (1432, "c071_role_map_camera"),
    (1433, "c071_cross_surface_mechanism"),
)
DIRECTIONS = ("true_to_false", "false_to_true")
SPLITS = ("confirmation", "lockbox")


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1434 exists")
    protocol = core.load(P1430 / "protocol/preregistration.json")
    behavior = core.load(P1431 / "analysis/behavior_summary.json")
    camera = core.load(P1432 / "analysis/camera_summary.json")
    mechanism = core.load(P1433 / "analysis/mechanism_summary.json")
    scripts = []
    reruns = {}
    reran = True
    for phase, stem in PHASES:
        main_script = TESTS / f"phase{phase}_{stem}.py"
        audit_script = TESTS / f"phase{phase}_{stem}_audit.py"
        scripts.extend((main_script, audit_script))
        completed = subprocess.run(
            [str(PYTHON), str(audit_script)], cwd=str(ROOT),
            capture_output=True, text=True, check=False,
        )
        reran &= completed.returncode == 0
        reruns[str(phase)] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1200:],
            "stderr_tail": completed.stderr[-1200:],
        }
    compiled = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            compiled = False
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1430, P1431, P1432, P1433)]
    cells = mechanism["cell_results"]
    aggregates = mechanism["aggregate_metrics"]
    transfer_names = protocol["mechanism"]["surface_transfers"]
    flat_cells = [cells[transfer][direction] for transfer in transfer_names for direction in DIRECTIONS]
    checks = {
        "audits_reran": reran,
        "audits_pass": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": compiled,
        "contract_hash": mechanism["contract_sha256"] == protocol["contract_sha256"],
        "behavior": behavior["behavior_qualified"] and len(core.rows(P1431 / "raw/active_behavior.jsonl")) == 2880 and behavior["selected_count"] == 72,
        "camera": camera["camera_qualified"] and all(value == 0.0 for value in camera["max_errors"].values()),
        "mechanism_execution": mechanism["all_execution_checks_passed"] and mechanism["record_count"] == 960,
        "four_nonspecific_cells": len(flat_cells) == 4 and all(cell["classification"] == "cross_surface_nonspecific" for cell in flat_cells),
        "mapped_pass": all(cell["cross_surface_mapped_pass"] for cell in flat_cells),
        "same_pass": all(cell["same_surface_pass"] for cell in flat_cells),
        "wrong_pass": all(cell["wrong_donor_pass"] for cell in flat_cells),
        "selectivity_failed": all(not cell["selective_pass"] for cell in flat_cells),
        "six_family_breadth": all(
            len(cell["same_surface_qualified_families"]) == 6
            and len(cell["cross_surface_qualified_families"]) == 6
            and len(cell["wrong_donor_qualified_families"]) == 6
            for cell in flat_cells
        ),
        "all_mapped_sign": all(
            aggregates[transfer][direction][split]["desired_sign_fraction"]["cross_surface_role_mapped"] == 1.0
            for transfer in transfer_names for direction in DIRECTIONS for split in SPLITS
        ),
        "all_sign_gaps_below_gate": all(
            aggregates[transfer][direction][split]["mapped_vs_permuted_sign_gap"]
            < protocol["mechanism"]["mapped_vs_permuted_sign_gap_min"]
            for transfer in transfer_names for direction in DIRECTIONS for split in SPLITS
        ),
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed_after_cross_surface_nonspecific_quartet_transport",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "audit_rerun_outputs": reruns,
        "retained": {
            "behavior": "Qwen3 solved all 2880 controlled roster cases across six families and two order-different surfaces",
            "camera": "the semantic-role mapper and fixed derangement write exact full-dimensional state16 values while preserving recipient complement",
            "same_surface_transport": "same-surface quartet transport crossed the desired answer boundary in both directions, splits, transfers, and all six families",
            "cross_surface_mapped_transport": "role-mapped cross-surface quartet transport crossed the desired boundary in all registered cells with large oriented gains",
            "nonspecific_transport": "the frozen role-permuted donor also crossed the desired boundary frequently enough that role selectivity failed in all four cells",
            "classification": mechanism["overall_classification"],
        },
        "rejected": {
            "role_isomorphic_selectivity": "correct semantic-role correspondence did not beat the frozen derangement by the preregistered sign-fraction gap",
            "four_operands_claim": "the four physical states cannot yet be individually identified as record target/family and query target/family operands",
            "relative_encoding_proven": "the result is conditional transport in one controlled Qwen contract, not a general coding law",
        },
        "untested": [
            "whether the quartet behaves as an unordered multiset, a permutation subgroup, or a donor-global common component",
            "whether role-specific information exists but is masked by a stronger shared answer/identity component",
            "necessity, minimality, uniqueness, or natural-use formation of the quartet",
            "other layers, tasks, models, languages, tokenizations, or open natural language",
            "attention, MLP, parameters, gradients, dimensionality reduction, learned probes, or coordinate searches",
        ],
        "claim_boundary": {
            "allowed": "at fixed Qwen3 state16 under the controlled two-surface roster contract, a four-role whole-state bundle transports the registered answer across surfaces, but one frozen role derangement is also largely effective",
            "forbidden": [
                "cross-surface semantic-role isomorphism established",
                "semantic operands or neuron groups discovered",
                "relative encoding, natural-language manifold, necessity, or minimality proven",
                "cross-model or cross-task invariant established",
            ],
        },
        "theory_update": {
            "subject": "conditional permutation-nonspecific bundle transport",
            "statement": "a role-named state bundle can be output-sufficient across surfaces while the registered output remains unable to identify the internal role assignment",
            "formula": "F_{>16}(P_phi(Z_d), H_r^barR; c_r) approx Y_d and F_{>16}(P_{phi o pi}(Z_d), H_r^barR; c_r) approx Y_d",
            "mathematics": "typed state replacement and finite permutation responses remain sufficient; no new basic mathematics is licensed",
        },
        "next_question": {
            "campaign": "C072",
            "object": "exhaustive quartet-permutation response spectrum across surfaces",
            "reason": "one fixed derangement refutes the strong role-isomorphism claim but cannot distinguish ordered tuple, unordered multiset, permutation subgroup, or shared donor component",
            "preauthorized_classes": [
                "role_order_selective",
                "permutation_symmetric_multiset",
                "subgroup_structured",
                "heterogeneous_or_executor_failed",
            ],
            "constraints": [
                "fresh material and behavior-first qualification with semantic uniqueness and controlled naturalness audit",
                "freeze all 24 role permutations, partitions, models, state16, nulls, gates, and classification before model execution",
                "known-truth permutation camera before holdout reveal",
                "full-dimensional input embeddings, Hidden State, and logits only",
                "no attention, MLP, parameters, gradients, PCA, learned probes, layer search, role-subset search, or coordinate search",
                "every result closes C072; no post-reveal permutation or threshold selection",
            ],
        },
        "authorization": "preregister_c072_exhaustive_quartet_permutation_response_spectrum",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
