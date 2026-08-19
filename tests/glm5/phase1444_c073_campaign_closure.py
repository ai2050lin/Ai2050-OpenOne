#!/usr/bin/env python3
"""Phase1444: close C073 after the frozen side/phase competition."""
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

PHASE, CAMPAIGN = 1444, "C073"
OUT = TESTS / "result/phase1444_c073_campaign_closure"
P1440 = TESTS / "result/phase1440_c073_side_phase_contract"
P1441 = TESTS / "result/phase1441_c073_behavior"
P1442 = TESTS / "result/phase1442_c073_matched_camera"
P1443 = TESTS / "result/phase1443_c073_side_phase_competition"
PHASES = (
    (1440, "c073_side_phase_contract"),
    (1441, "c073_behavior"),
    (1442, "c073_matched_camera"),
    (1443, "c073_side_phase_competition"),
)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1444 exists")
    protocol = core.load(P1440 / "protocol/preregistration.json")
    preaudit = core.load(P1440 / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    behavior = core.load(P1441 / "analysis/behavior_summary.json")
    camera = core.load(P1442 / "analysis/camera_summary.json")
    mechanism = core.load(P1443 / "analysis/side_phase_summary.json")

    scripts = []
    reruns = {}
    audits_reran = True
    for phase, stem in PHASES:
        main_script = TESTS / f"phase{phase}_{stem}.py"
        audit_script = TESTS / f"phase{phase}_{stem}_audit.py"
        scripts.extend((main_script, audit_script))
        completed = subprocess.run([str(PYTHON), str(audit_script)], cwd=str(ROOT), capture_output=True, text=True, check=False)
        audits_reran &= completed.returncode == 0
        reruns[str(phase)] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1600:],
            "stderr_tail": completed.stderr[-1600:],
        }

    scripts_compile = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            scripts_compile = False

    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1440, P1441, P1442, P1443)]
    cells = [mechanism["cell_results"][route][direction][split] for route in protocol["mechanism"]["routes"] for direction in protocol["mechanism"]["directions"] for split in ("confirmation", "lockbox")]
    reversed_cells = [mechanism["cell_results"][route][direction][split] for route in protocol["mechanism"]["reversed_routes"] for direction in protocol["mechanism"]["directions"] for split in ("confirmation", "lockbox")]
    evidence_cells = [mechanism["cell_results"]["evidence_same"][direction][split] for direction in protocol["mechanism"]["directions"] for split in ("confirmation", "lockbox")]
    wrong_values = [cell["controls"]["wrong_identity_expected_sign_fraction"] for cell in cells]
    checks = {
        "audits_reran": audits_reran,
        "audits_pass": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": scripts_compile,
        "contract_hash": mechanism["contract_sha256"] == protocol["contract_sha256"] == "6c1c27dd36a7d08d3a76309a7bc52fd35e64cce4b4565b3894957d231bff87ad",
        "contract": preaudit["all_checks_passed"] and len(core.rows(P1440 / "material/active_cases.jsonl")) == 2880 and len(core.rows(P1440 / "material/composition_sets.jsonl")) == 72,
        "behavior": behavior["behavior_qualified"] and behavior["global_accuracy"] == 1.0 and behavior["global_balanced_accuracy"] == 1.0 and behavior["selected_count"] == 72,
        "camera": camera["camera_qualified"] and camera["known_truth_count"] == 1280 and camera["qwen_case_count"] == 480 and all(value == 0.0 for value in camera["max_errors"].values()),
        "mechanism_execution": mechanism["all_execution_checks_passed"] and mechanism["record_count"] == 2688 and len(cells) == 16,
        "formal_class": mechanism["overall_classification"] == "executor_failed",
        "executor_count": mechanism["classification_counts"]["total_executor_pass"] == sum(cell["executor_pass"] for cell in cells) == 8,
        "wrong_control": min(wrong_values) == 1.0,
        "reversed_unqualified": not any(cell["semantic_side_winner"] or cell["physical_phase_winner"] for cell in reversed_cells),
        "evidence_candidate": all(cell["executor_pass"] and cell["semantic_side_winner"] and not cell["physical_phase_winner"] for cell in evidence_cells),
        "one_shot": mechanism["reveal_manifest"]["one_shot"] and mechanism["reveal_manifest"]["holdout_count"] == 48,
    }
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed_at_executor_gate_after_matched_side_phase_reveal",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "audit_rerun_outputs": reruns,
        "retained": {
            "behavior": "Qwen3 solved all 2880 frozen cases across six fresh families and both order-reversed surfaces",
            "camera": "known-truth and Qwen full-state quartet writes were exact for p00/p01/p06/p07/p23 across all frozen routes and directions",
            "numeric_execution": "all 2688 holdout records were finite, balanced, BF16, nonquantized, and exactly written",
            "wrong_identity": "wrong-identity expected-sign fraction was 1.0 in all sixteen cells",
            "partial_transport": "only eight of sixteen correct-identity cells passed; whole-state quartet transport is a conditional partial operation rather than a globally admissible one in this contract",
            "same_surface_candidate": "all four evidence_same cells were executor-valid and favored p07 over p23 under the frozen winner gate",
            "classification": mechanism["overall_classification"],
        },
        "rejected": {
            "semantic_side_confirmed": "no reversed-order cell had a qualified semantic-side winner because the required executor domain was absent",
            "physical_phase_confirmed": "no reversed-order cell had a qualified physical-phase winner; descriptive p23 effects cannot override failed identity controls",
            "global_role_transport": "correct identity transfer varied from near zero to one across route and answer direction",
            "c072_candidate_as_law": "the C072 record/query-side stratum reproduced only in the executor-valid evidence_same subset, not as a cross-surface invariant",
            "relative_encoding_proven": "a conditional full-state response in one controlled Qwen task is not a general coding law",
        },
        "untested": [
            "semantic-side versus physical-phase competition on a cross-surface route where correct identity is independently qualified in both directions",
            "the minimal observable state needed to predict whether a source-to-target whole-state transfer is admissible",
            "necessity, minimality, uniqueness, natural use, or natural-trajectory membership of the role quartet",
            "other states, tasks, models, languages, tokenizations, or open natural language",
            "attention, MLP, parameters, gradients, dimensionality reduction, learned probes, hotspot searches, or coordinate searches",
        ],
        "claim_boundary": {
            "allowed": "in frozen C073 at Qwen3 state16, exact quartet transport is strongly route- and answer-direction-dependent; only half the cells qualify, preventing the matched semantic-side/physical-phase identification",
            "forbidden": [
                "semantic record/query side or physical early/late phase established as the invariant",
                "failed reversed cells treated as evidence that either candidate mechanism is absent",
                "global transport, semantic neurons, relative encoding, or a language manifold discovered",
                "cross-model, cross-task, necessity, minimality, or natural-language claims",
            ],
        },
        "theory_update": {
            "subject": "conditional domain of full-state transport",
            "statement": "a typed whole-state transport must be treated as a partial operator whose domain depends on source surface, target surface, answer direction, and downstream context",
            "formula": "T_pi^(s->t,y): Z_s ⇀ Z_t, and mechanism competition is identifiable only where E_identity(s->t,y)=1",
            "mathematics": "partial functions, conditional intervention responses, and finite matched comparisons remain sufficient; no new mathematics or theory renaming is licensed",
        },
        "next_question": {
            "campaign": "C074",
            "object": "independent directional transport-domain map before any new semantic permutation competition",
            "reason": "C073 failed at the identity executor in half the frozen cells, so admissible source-target-direction edges must be established before comparing candidate role maps",
            "requirements": [
                "fresh materials and surfaces with semantic uniqueness, controlled naturalness, and discovery/confirmation/lockbox partitions",
                "freeze an identity-only source-surface by target-surface by answer-direction transport matrix before model execution",
                "behavior-first qualification and known-truth full-state camera before one holdout reveal",
                "treat each directed edge as qualified or rejected without pooling opposite directions",
                "do not compare semantic permutations unless an edge passes identity, wrong-identity, self, numeric, and family-breadth gates independently",
                "full-dimensional input embeddings, Hidden State, and logits only",
                "no attention, MLP, parameters, gradients, dimensionality reduction, probes, layer search, role-subset search, coordinate search, or post-reveal threshold changes",
            ],
        },
        "authorization": "preregister_c074_directional_transport_domain_test",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
