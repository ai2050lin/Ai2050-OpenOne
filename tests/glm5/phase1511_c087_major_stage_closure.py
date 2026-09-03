#!/usr/bin/env python3
"""Phase1511: close C087 and authorize the cross-root codebook factorial."""
from __future__ import annotations

import json
import py_compile
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1511_c087_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE_DIRS = {
    1504: "phase1504_c087_cross_root_semeval_contract",
    1505: "phase1505_c087_behavior_stratification",
    1506: "phase1506_c087_all_case_field_capture",
    1507: "phase1507_c087_descriptive_semantic_contrast_atlas",
    1508: "phase1508_c087_discovery_observation_freeze",
    1509: "phase1509_c087_dual_holdout_validation",
    1510: "phase1510_c087_stratum_and_c086_diagnostics",
}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1511 exists")
    audits, compiled = {}, []
    for phase, name in PHASE_DIRS.items():
        audit = core.load(RESULT / name / "audit/independent_final_audit.json")
        audits[str(phase)] = audit
        if not audit["all_checks_passed"]:
            raise RuntimeError((phase, audit))
        for suffix in (".py", "_audit.py"):
            script = TESTS / f"{name}{suffix}"
            py_compile.compile(str(script), doraise=True)
            compiled.append(str(script.relative_to(ROOT)).replace("\\", "/"))

    contract = core.load(RESULT / PHASE_DIRS[1504] / "protocol/preregistration.json")
    behavior = core.load(RESULT / PHASE_DIRS[1505] / "analysis/behavior_stratification_summary.json")
    capture = core.load(RESULT / PHASE_DIRS[1506] / "analysis/capture_metadata.json")
    atlas = core.load(RESULT / PHASE_DIRS[1507] / "analysis/semantic_contrast_atlas_summary.json")
    discovery = core.load(RESULT / PHASE_DIRS[1508] / "analysis/final.json")
    validation = core.load(RESULT / PHASE_DIRS[1509] / "analysis/dual_holdout_validation.json")
    diagnostics = core.load(RESULT / PHASE_DIRS[1510] / "analysis/stratum_and_c086_diagnostics.json")
    checks = {
        "contract": contract["contract_sha256"] == "81135260184b552ef9b197ae5a10cb292e43a5d5383a662e25169b61f9fda957",
        "all_audits": all(audit["all_checks_passed"] for audit in audits.values()),
        "all_scripts_compile": len(compiled) == 14,
        "behavior": behavior["global_accuracy"] == 0.8657407407407407 and behavior["stratum_counts"] == {"mixed": 88, "success": 128},
        "capture_failure_preserved": capture["acquisition_complete"] and not capture["execution_identity_gate_passed"],
        "atlas": atlas["state0_all_partition_mean_max_abs"] == 0.0,
        "discovery_frozen": bool(discovery["freeze_sha256"]),
        "dual_holdout_formal_failure": not validation["dual_holdout_primary_pass"],
        "diagnostics": all(diagnostics["checks"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    result = {
        "phase": 1511,
        "campaign": "C087",
        "status": "major_stage_complete_with_repeated_candidate_and_formal_gate_failure",
        "checks": checks,
        "audits": {phase: f"{audit['passed']}/{audit['total']}" for phase, audit in audits.items()},
        "compiled_scripts": compiled,
        "answer": {
            "behavior": {
                "accuracy": behavior["global_accuracy"],
                "balanced_accuracy": behavior["global_balanced_accuracy"],
                "truth_same_accuracy": behavior["truth"]["true"],
                "truth_different_accuracy": behavior["truth"]["false"],
                "strata": behavior["stratum_counts"],
            },
            "field": {
                "discovery": discovery["discovery"],
                "dual_holdout_primary_pass": validation["dual_holdout_primary_pass"],
                "cross_partition_state35_boundary_cosines": diagnostics["cross_partition_state35_boundary_cosines"],
                "c086_alignment_peak": diagnostics["c086_alignment_peak"],
                "lockbox_failure_anatomy": diagnostics["lockbox_failure_anatomy"],
            },
            "strict_conclusion": "Across 36 disjoint SemEval verb items, a late full-dimensional same-minus-different boundary response repeats strongly across prompt surfaces and lexical partitions. It remains a descriptive candidate because the capture execution-identity gate failed, the dual-holdout conjunction failed by a stronger-than-discovery lockbox effect, and semantic truth is not factorially separated from the same/different output direction.",
        },
        "core_puzzle": {
            "id": "K264",
            "evidence": "E2-QWEN-CROSS-ROOT-PROSPECTIVE-DESCRIPTIVE-CANDIDATE",
            "statement": "After lexical-root overlap is removed and candidate identity is exactly counterbalanced within each item-disjoint partition, a shared late boundary same-minus-different response emerges around state21-state22 and the three state35 partition centroids have pairwise cosine 0.997-0.998. The candidate is not confirmatory because acquisition execution identity and the frozen dual-holdout conjunction did not both pass, and output-code polarity remains confounded with semantic truth.",
        },
        "hard_boundaries": [
            "Phase1506 behavior-only versus hidden-state execution changed 4/864 predictions and 2/216 group strata; acquisition is complete but the execution identity gate failed",
            "the frozen dual-holdout conjunction failed because lockbox group-pairwise cosine was 0.151639 above discovery against a 0.15 absolute tolerance; the stronger effect does not repair the formal gate",
            "same/different semantic truth and same/different answer direction are not independently varied, so the observed contrast collapses semantic and output-code interaction terms",
            "success/mixed conditioning breaks candidate-identity counterbalance and is descriptive collider-conditioned evidence only",
            "the strongest cross-partition structure is at the answer boundary; it may be a decision/output variable rather than lexical-semantic relation identity",
            "C086 alignment peaks at state24 near 0.713 but falls near 0.20 by state35, excluding a single unchanged task-general vector account",
            "the source role is exactly zero under candidate replacement because it is causally upstream; no backward semantic transport is observed",
            "the study remains single-model, observational, English-only, and without a new independent human rating of the prompt wrapper",
        ],
        "theory": {
            "name": "条件化输出场闭合理论",
            "organizing_principle": "复用--差分--条件化（RDC）",
            "update": "add a cross-root late boundary response candidate and explicitly represent its non-identifiability under a fixed output code; retain a shared mid-late phase followed by task-specific late differentiation",
            "identifiability_formula": "Delta_H(P_fixed)=2*beta_R+2*P_fixed*beta_RP",
            "new_foundational_mathematics": False,
        },
        "authorization": "preregister_c088_cross_root_semantic_by_answer_code_factorial",
        "next_contract_requirements": [
            "use a fresh item-disjoint SemEval verb sample not used in C087",
            "factor semantic equivalence and answer-code polarity in the same cross-root natural-context contract",
            "retain a natural same/different anchor arm and add balanced symbolic standard/reversed code arms",
            "counterbalance candidate identity separately inside discovery, confirmation and lockbox",
            "stratify rather than stop on behavior; preserve all-case embeddings and all Hidden States",
            "freeze directional presence gates separately from effect-size equivalence gates so stronger replication is not mislabeled as structural absence",
            "continue without attention, MLP, parameter scans, PCA, TDA, or learned probes",
        ],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
