#!/usr/bin/env python3
"""Phase1439: close C072 after the frozen exhaustive permutation reveal."""
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

PHASE, CAMPAIGN = 1439, "C072"
OUT = TESTS / "result/phase1439_c072_campaign_closure"
P1435 = TESTS / "result/phase1435_c072_permutation_spectrum_contract"
P1436 = TESTS / "result/phase1436_c072_behavior"
P1437 = TESTS / "result/phase1437_c072_permutation_camera"
P1438 = TESTS / "result/phase1438_c072_permutation_spectrum"
PHASES = (
    (1435, "c072_permutation_spectrum_contract"),
    (1436, "c072_behavior"),
    (1437, "c072_permutation_camera"),
    (1438, "c072_permutation_spectrum"),
)
EXPECTED_SETS = (
    {"p00", "p01", "p02", "p03", "p06", "p07", "p08", "p09", "p12", "p13", "p14", "p15", "p16", "p22"},
    {"p00", "p01", "p02", "p04", "p06", "p07", "p08", "p10", "p12", "p14", "p18", "p20", "p21"},
    {"p00", "p01", "p02", "p03", "p06", "p07", "p08", "p09", "p12", "p13", "p14", "p15"},
    {"p00", "p01", "p06"},
)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1439 exists")
    protocol = core.load(P1435 / "protocol/preregistration.json")
    behavior = core.load(P1436 / "analysis/behavior_summary.json")
    camera = core.load(P1437 / "analysis/camera_summary.json")
    mechanism = core.load(P1438 / "analysis/permutation_spectrum_summary.json")

    scripts = []
    reruns = {}
    audits_reran = True
    for phase, stem in PHASES:
        main_script = TESTS / f"phase{phase}_{stem}.py"
        audit_script = TESTS / f"phase{phase}_{stem}_audit.py"
        scripts.extend((main_script, audit_script))
        completed = subprocess.run(
            [str(PYTHON), str(audit_script)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        audits_reran &= completed.returncode == 0
        reruns[str(phase)] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1200:],
            "stderr_tail": completed.stderr[-1200:],
        }

    scripts_compile = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            scripts_compile = False

    audits = [
        core.load(path / "audit/independent_final_audit.json")
        for path in (P1435, P1436, P1437, P1438)
    ]
    cells = [
        cell
        for directions in mechanism["cell_results"].values()
        for cell in directions.values()
    ]
    qualified_sets = [set(cell["qualified_permutations"]) for cell in cells]
    intersection = set.intersection(*qualified_sets)
    descriptive = mechanism["descriptive_strata"]
    axis_true_sign = []
    axis_false_sign = []
    axis_true_gain = []
    axis_false_gain = []
    for directions in descriptive.values():
        for split_data in directions.values():
            for split in ("confirmation", "lockbox"):
                axis = split_data[split]["preserves_record_query_axis"]
                axis_true_sign.append(axis["true"]["desired_sign_fraction"])
                axis_false_sign.append(axis["false"]["desired_sign_fraction"])
                axis_true_gain.append(axis["true"]["oriented_gain_median"])
                axis_false_gain.append(axis["false"]["oriented_gain_median"])

    checks = {
        "audits_reran": audits_reran,
        "audits_pass": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": scripts_compile,
        "contract_hash": mechanism["contract_sha256"] == protocol["contract_sha256"],
        "behavior": behavior["behavior_qualified"] and behavior["selected_count"] == 72
        and len(core.rows(P1436 / "raw/active_behavior.jsonl")) == 2880,
        "camera": camera["camera_qualified"] and camera["known_truth_count"] == 3072
        and camera["qwen_case_count"] == 1152 and all(value == 0.0 for value in camera["max_errors"].values()),
        "mechanism_execution": mechanism["all_execution_checks_passed"]
        and mechanism["record_count"] == 5184 and len(cells) == 4
        and all(cell["executor_pass"] for cell in cells),
        "formal_class": mechanism["overall_classification"] == "heterogeneous_or_executor_failed",
        "qualified_sets": all(actual == expected for actual, expected in zip(qualified_sets, EXPECTED_SETS)),
        "intersection": intersection == {"p00", "p01", "p06"},
        "not_subgroup": "p07" not in intersection,
        "not_role_selective": all(
            cell["identity_vs_best_nonidentity"][split]["sign"] < protocol["mechanism"]["identity_vs_best_nonidentity_sign_gap_min"]
            for cell in cells for split in ("confirmation", "lockbox")
        ),
        "not_symmetric": all(
            cell["symmetric_gain_range_ratio"][split] > protocol["mechanism"]["symmetric_gain_range_ratio_max"]
            for cell in cells for split in ("confirmation", "lockbox")
        ),
        "axis_candidate_descriptive": min(axis_true_sign) > max(axis_false_sign)
        and min(axis_true_gain) > max(axis_false_gain),
    }

    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed_after_heterogeneous_exhaustive_permutation_spectrum",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "audit_rerun_outputs": reruns,
        "retained": {
            "behavior": "Qwen3 solved all 2880 fresh controlled roster cases across six families and two order-different surfaces",
            "camera": "all 24 full-dimensional state16 role permutations were written exactly in known-truth and Qwen calibration systems",
            "executor": "same-surface, wrong-expected, self-output, finite-value, family-breadth, and contract controls passed",
            "spectrum": "the four qualified sets contain 14, 13, 12, and 3 permutations, with p00/p01/p06 as their intersection",
            "axis_candidate": "preserving the record/query side is a preregistered descriptive stratum with higher sign fractions and gains in every split, but has no frozen confirmation gate in C072",
            "classification": mechanism["overall_classification"],
        },
        "rejected": {
            "role_order_selective": "identity never obtained the frozen sign-fraction advantage over the best nonidentity permutation",
            "permutation_symmetric_multiset": "not all 24 permutations qualified and all split-wise gain-range ratios exceeded the frozen limit",
            "subgroup_structured": "the common qualified set p00/p01/p06 is not composition-closed because p01 composed with p06 is p07",
            "single_derangement_generalization": "C071's one effective derangement does not license arbitrary permutation equivalence",
            "relative_encoding_proven": "a conditional whole-state response spectrum in one controlled Qwen contract is not a general coding law",
        },
        "untested": [
            "independent confirmation that record/query-side preservation, rather than a correlated role feature, predicts the response spectrum",
            "necessity, minimality, uniqueness, natural-use formation, or natural-trajectory membership of the four-role state bundle",
            "why surface-transfer direction, answer direction, family, and within-side role swaps change qualification",
            "other layers, tasks, models, languages, tokenizations, or open natural language",
            "attention, MLP, parameters, gradients, dimensionality reduction, learned probes, or coordinate searches",
        ],
        "claim_boundary": {
            "allowed": "at Qwen3 state16 in the frozen C072 roster task, exact whole-state permutations have an executor-valid but heterogeneous response spectrum; the strongest descriptive partition preserves record versus query side",
            "forbidden": [
                "ordered semantic operands, an unordered multiset, or a stable permutation subgroup established",
                "record/query-side preservation independently confirmed as the causal invariant",
                "relative encoding, semantic neuron groups, or a language manifold discovered",
                "cross-model, cross-task, natural-language, necessity, or minimality claims",
            ],
        },
        "theory_update": {
            "subject": "conditional side-partitioned response spectrum",
            "statement": "full-state role-bundle efficacy is neither identity-specific nor permutation-invariant; a record/query side partition is the leading independent-confirmation candidate",
            "formula": "R(pi; c, y, f) is heterogeneous, while descriptively R(pi preserving record/query; c, y, f) exceeds R(pi crossing record/query; c, y, f)",
            "mathematics": "finite permutations, conditional intervention responses, and elementary group closure remain sufficient; no new mathematics or theory-renaming is licensed",
        },
        "next_question": {
            "campaign": "C073",
            "object": "independent record/query-side preservation response test",
            "reason": "C072 exhaustively rejects the stronger order, multiset, and subgroup classes while isolating a side-partition candidate that was descriptive rather than confirmatory",
            "requirements": [
                "fresh semantic materials, labels, members, surface families, and discovery/confirmation/lockbox partitions",
                "behavior-first qualification plus semantic uniqueness and controlled-naturalness audit",
                "freeze matched axis-preserving versus axis-crossing permutation contrasts before model execution",
                "balance or explicitly match fixed points, parity, cycle type, and entity/family-kind preservation so the side contrast is identifiable",
                "known-truth full-state camera before one holdout reveal",
                "full-dimensional input embeddings, Hidden State, and logits only",
                "no attention, MLP, parameters, gradients, dimensionality reduction, learned probes, layer search, role-subset search, or coordinate search",
                "C073 must close after its frozen branches; no C072 threshold or permutation reuse as confirmation",
            ],
        },
        "authorization": "preregister_c073_independent_record_query_side_preservation_test",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
