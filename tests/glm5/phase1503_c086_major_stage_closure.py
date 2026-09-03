#!/usr/bin/env python3
"""Phase1503: close C086 and freeze the next research authorization."""
from __future__ import annotations

import json
import py_compile
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1503_c086_major_stage_closure"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE_DIRS = {
    1496: "phase1496_c086_unlabeled_counterbalanced_contract",
    1497: "phase1497_c086_behavior_stratification",
    1498: "phase1498_c086_all_case_field_capture",
    1499: "phase1499_c086_four_factor_atlas",
    1500: "phase1500_c086_discovery_observation_freeze",
    1501: "phase1501_c086_dual_holdout_validation",
    1502: "phase1502_c086_stratum_and_c085_diagnostics",
}


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1503 exists")
    audits = {}
    compiled = []
    for phase, name in PHASE_DIRS.items():
        audit = core.load(RESULT / name / "audit/independent_final_audit.json")
        audits[str(phase)] = audit
        if not audit["all_checks_passed"]:
            raise RuntimeError((phase, audit))
        for suffix in (".py", "_audit.py"):
            script = TESTS / f"{name}{suffix}"
            py_compile.compile(str(script), doraise=True)
            compiled.append(str(script.relative_to(ROOT)).replace("\\", "/"))

    contract = core.load(RESULT / PHASE_DIRS[1496] / "protocol/preregistration.json")
    behavior = core.load(
        RESULT / PHASE_DIRS[1497] / "analysis/behavior_stratification_summary.json"
    )
    capture = core.load(RESULT / PHASE_DIRS[1498] / "analysis/capture_metadata.json")
    atlas = core.load(RESULT / PHASE_DIRS[1499] / "analysis/four_factor_atlas_summary.json")
    discovery = core.load(RESULT / PHASE_DIRS[1500] / "analysis/final.json")
    validation = core.load(RESULT / PHASE_DIRS[1501] / "analysis/final.json")
    diagnostics = core.load(RESULT / PHASE_DIRS[1502] / "analysis/diagnostic_summary.json")
    checks = {
        "contract": contract["contract_sha256"]
        == "fc99ec6920592602565f6bf314e8fe631ea0df597850cd5d31946b93658d0833",
        "all_audits": all(audit["all_checks_passed"] for audit in audits.values()),
        "all_scripts_compile": len(compiled) == 14,
        "behavior_mixed_only": behavior["stratum_counts"] == {"mixed": 216},
        "field_complete": capture["shape"] == [6912, 37, 7, 2560],
        "atlas_complete": len(atlas["effects"]) == 15,
        "discovery_frozen": bool(discovery["freeze_sha256"]),
        "dual_holdout": validation["status"] == "dual_holdout_confirmed",
        "diagnostics": all(diagnostics["checks"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    state35 = diagnostics["field_formation"]["state35"]
    result = {
        "phase": 1503,
        "campaign": "C086",
        "status": "major_stage_complete_with_mixed_behavior_field_confirmation",
        "checks": checks,
        "audits": {
            phase: f"{values['passed']}/{values['total']}"
            for phase, values in audits.items()
        },
        "compiled_scripts": compiled,
        "answer": {
            "behavior": {
                "accuracy": behavior["global_accuracy"],
                "strata": behavior["stratum_counts"],
                "predicted_no_rate": diagnostics["behavior"]["predicted_no_rate"],
            },
            "prospective_observation": {
                "dual_holdout_status": validation["status"],
                "state35_boundary": state35,
                "generic_relation_onset": diagnostics["field_formation"][
                    "generic_relation_onset_at_pairwise_0_8"
                ],
                "c085_alignment_onset": diagnostics["field_formation"][
                    "c085_alignment_onset_at_0_7"
                ],
            },
            "strict_conclusion": "Qwen3 contains a prospectively repeated, late, code-invariant same-versus-different response in this unlabeled controlled task, while failing to apply the arbitrary answer code reliably; because all composition sets are mixed and matched verbs share lexical roots, the observation is not yet a semantic-relation mechanism or a correct-behavior causal state.",
        },
        "core_puzzle": {
            "id": "K263",
            "evidence": "E3-QWEN-CONTROLLED-DIAGNOSTIC-OBSERVATION",
            "statement": "After explicit abstract relation labels are removed and answer polarity is counterbalanced, a late full-dimensional same/different relation-match response repeats across discovery, confirmation and lockbox, but it coexists with severe output-code failure and remains confounded by shared lexical roots.",
        },
        "hard_boundaries": [
            "all 216 composition sets are mixed; success and failed fields are M2 missing",
            "same-relation verb pairs are inflectional cognates, so semantic identity is not isolated from lexical-root identity",
            "the model predicts no on 92.96875 percent of cases and does not reliably execute the reversed answer code",
            "BF16 behavior-only and hidden-state forwards disagree on 131 of 6912 predictions although all group strata agree",
            "top one percent of coordinates carries only about ten percent of coefficient energy; no small neuron set is identified",
            "the study is single-model, controlled-English, noncausal, and lacks independent human naturalness review",
        ],
        "theory": {
            "name": "条件化输出场闭合理论",
            "organizing_principle": "复用--差分--条件化（RDC）",
            "update": "split the late response into a code-invariant match coefficient and a match-by-answer-code coefficient; retain both as task-scoped descriptive field terms rather than semantic causes",
            "new_foundational_mathematics": False,
        },
        "authorization": "preregister_c087_cross_root_paraphrase_layered_observation",
        "next_contract_requirements": [
            "orthogonalize semantic equivalence from lexical-root overlap using cross-root paraphrases and matched non-equivalents",
            "use a natural same/different output interface for the primary behavior panel; treat arbitrary code following as a separate nuisance panel",
            "split by relation expressions, not merely by entities, before discovery",
            "freeze zero models for token/root overlap, length, morphology, surface and candidate position",
            "obtain independent human semantic uniqueness and naturalness review before model execution",
            "continue observation-first full-dimensional embedding/all-Hidden-State analysis without attention, MLP, PCA or learned probes",
        ],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
