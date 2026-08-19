#!/usr/bin/env python3
"""Phase1424: close C069 with separate graded and discrete ledgers."""
from __future__ import annotations

import json, py_compile, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

OUT = TESTS / "result/phase1424_c069_campaign_closure"
P1420 = TESTS / "result/phase1420_c069_catalog_four_role_contract"
P1421 = TESTS / "result/phase1421_c069_catalog_behavior"
P1422 = TESTS / "result/phase1422_c069_quartet_camera"
P1423 = TESTS / "result/phase1423_c069_bidirectional_composition"


def main() -> None:
    if (OUT / "analysis/final.json").exists(): raise RuntimeError("Phase1424 exists")
    protocol = core.load(P1420 / "protocol/preregistration.json")
    behavior = core.load(P1421 / "analysis/behavior_summary.json")
    camera = core.load(P1422 / "analysis/camera_summary.json")
    composition = core.load(P1423 / "analysis/composition_summary.json")
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1420, P1421, P1422, P1423)]
    scripts = []
    for phase, stem in (
        (1420, "c069_catalog_four_role_contract"),
        (1421, "c069_catalog_behavior"),
        (1422, "c069_quartet_camera"),
        (1423, "c069_bidirectional_composition"),
    ):
        scripts.extend([TESTS / f"phase{phase}_{stem}.py", TESTS / f"phase{phase}_{stem}_audit.py"])
    compiled = True
    for script in scripts:
        try: py_compile.compile(str(script), doraise=True)
        except Exception: compiled = False
    aggregate = composition["split_direction_metrics"]
    checks = {
        "audits": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": compiled,
        "behavior_six_families": len(behavior["qualified_families"]) == 6,
        "behavior_72_sets": behavior["selected_count"] == 72,
        "camera_exact": camera["camera_qualified"] and camera["qwen_output_max_abs_diff"] == 0.0,
        "holdout_48": composition["holdout_set_count"] == 48 and composition["record_count"] == 864,
        "state16_catalog": protocol["mechanism"]["state_index"] == 16 and protocol["mechanism"]["surface"] == "catalog",
        "graded_confirmed": composition["graded_confirmed"] and len(composition["graded_qualified_families"]) == 6,
        "discrete_failed": not composition["discrete_confirmed"] and composition["discrete_qualified_families"] == [],
        "strong_failed": not composition["strong_confirmed"] and composition["strong_qualified_families"] == [],
        "true_mismatch_failed_both": all(not aggregate[split]["true_recipient"]["discrete_checks"]["mismatched_negative"] for split in ("confirmation", "lockbox")),
        "false_match_failed_both": all(not aggregate[split]["false_recipient"]["discrete_checks"]["matched_positive"] for split in ("confirmation", "lockbox")),
        "natural_false_passed": all(aggregate[split][direction]["discrete_checks"]["natural_false_negative"] for split in ("confirmation", "lockbox") for direction in ("true_recipient", "false_recipient")),
    }
    result = {
        "phase": 1424, "campaign": "C069",
        "status": "closed_after_graded_quartet_confirmation_and_failed_discrete_sufficiency",
        "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "retained": {
            "behavior": "six fresh families and 72 catalog-scoped composition sets qualified",
            "camera": "four-role self write is exact on both recipients and 256 known-truth systems",
            "graded_quartet": "all four aggregate split-direction cells and all six families passed controls, interaction, and false-recipient rescue gates",
            "interaction_medians": {split: {direction: aggregate[split][direction]["interaction_median"] for direction in aggregate[split]} for split in aggregate},
            "interaction_win_fractions": {split: {direction: aggregate[split][direction]["interaction_win_fraction"] for direction in aggregate[split]} for split in aggregate},
        },
        "rejected": {
            "strong_hypothesis": "catalog state16 record/query target-and-family quartet is a sufficient discrete relation state under the frozen whole-state write semantics",
            "failed_discrete_gates": {split: {direction: [key for key, value in aggregate[split][direction]["discrete_checks"].items() if not value] for direction in aggregate[split]} for split in aggregate},
            "discrete_qualified_families": composition["discrete_qualified_families"],
        },
        "untested": [
            "whether omitted causal support lies in the non-quartet state16 positions",
            "minimal or necessary distributed state bundles",
            "other states, models, languages, open text, and natural semantic families",
            "attention, MLP, parameters, gradients, dimensionality reduction, or learned probes",
        ],
        "claim_boundary": {
            "allowed": "Qwen controlled-registry state16 quartet has a replicated bidirectional graded interaction but fails frozen discrete sufficiency",
            "forbidden": ["relation manifold discovered", "semantic comparator localized", "relative encoding proven or refuted", "minimal/unique mechanism", "cross-model law"],
        },
        "next_question": {
            "campaign": "C070",
            "object": "state16 quartet-versus-complement causal support partition",
            "reason": "graded quartet effects are real but neither true-to-mismatch nor false-to-match is discretely sufficient; the direct missing-variable test is the predeclared non-quartet complement, not another layer search",
            "candidate_arms": ["self", "quartet_only", "complement_only", "full_state", "wrong_full_state"],
            "constraints": ["new material", "behavior first", "fixed state16", "same-shape prompts", "no layer/subset search", "no attention/MLP/parameters/gradients/PCA/probes"],
        },
        "authorization": "preregister_c070_quartet_complement_support_partition",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
