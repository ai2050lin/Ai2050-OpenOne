#!/usr/bin/env python3
"""Phase1449: close C074 after the directional transport-domain reveal."""
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

PHASE, CAMPAIGN = 1449, "C074"
OUT = TESTS / "result/phase1449_c074_campaign_closure"
P1445 = TESTS / "result/phase1445_c074_directional_domain_contract"
P1446 = TESTS / "result/phase1446_c074_behavior"
P1447 = TESTS / "result/phase1447_c074_identity_camera"
P1448 = TESTS / "result/phase1448_c074_directional_domain_map"
PHASES = (
    (1445, "c074_directional_domain_contract"),
    (1446, "c074_behavior"),
    (1447, "c074_identity_camera"),
    (1448, "c074_directional_domain_map"),
)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1449 exists")
    protocol = core.load(P1445 / "protocol/preregistration.json")
    preaudit = core.load(P1445 / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    behavior = core.load(P1446 / "analysis/behavior_summary.json")
    camera = core.load(P1447 / "analysis/camera_summary.json")
    domain = core.load(P1448 / "analysis/directional_domain_summary.json")

    scripts, reruns, audits_reran = [], {}, True
    for phase, stem in PHASES:
        main_script = TESTS / f"phase{phase}_{stem}.py"
        audit_script = TESTS / f"phase{phase}_{stem}_audit.py"
        scripts.extend((main_script, audit_script))
        completed = subprocess.run([str(PYTHON), str(audit_script)], cwd=str(ROOT), capture_output=True, text=True, check=False)
        audits_reran &= completed.returncode == 0
        reruns[str(phase)] = {"returncode": completed.returncode, "stdout_tail": completed.stdout[-1600:], "stderr_tail": completed.stderr[-1600:]}
    scripts_compile = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            scripts_compile = False
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1445, P1446, P1447, P1448)]
    robust = core.rows(P1448 / "analysis/robust_edges.jsonl")
    evidence_robust = [row for row in robust if row["source"].endswith("evidence_first") and row["target"].endswith("evidence_first")]
    cross_order_robust = [row for row in robust if not row["same_order"]]
    question_robust = [row for row in robust if row["source"].endswith("question_first")]
    checks = {
        "audits_reran": audits_reran,
        "audits_pass": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": scripts_compile,
        "contract_hash": domain["contract_sha256"] == protocol["contract_sha256"] == "4545c5df1dc94bf5fd9138da1330d84c4d212e331352cf6fbabe1b82d2865ecb",
        "contract": preaudit["all_checks_passed"] and len(core.rows(P1445 / "material/active_cases.jsonl")) == 5760 and len(core.rows(P1445 / "material/composition_sets.jsonl")) == 72,
        "behavior": behavior["behavior_qualified"] and len(behavior["qualified_families"]) == 6 and behavior["selected_count"] == 72,
        "camera": camera["camera_qualified"] and camera["known_truth_count"] == 768 and camera["qwen_record_count"] == 1152 and all(value == 0.0 for value in camera["max_errors"].values()),
        "domain_execution": domain["all_execution_checks_passed"] and domain["record_count"] == 4608 and domain["cell_count"] == 64 and domain["edge_count"] == 32,
        "classes": domain["class_counts"] == {"robust": 10, "split_specific": 2, "rejected": 20},
        "robust_copy": len(robust) == domain["robust_edge_count"] == 10 and sorted(row["edge_id"] for row in robust) == sorted(domain["robust_edge_ids"]),
        "evidence_subgraph": len(evidence_robust) == 8,
        "no_cross_order_robust": not cross_order_robust,
        "question_direction": len(question_robust) == 2 and all(row["direction"] == "false_to_true" for row in question_robust),
        "one_shot": domain["reveal_manifest"]["one_shot"] and domain["reveal_manifest"]["holdout_count"] == 48,
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "status": "closed_with_sparse_directional_transport_domain",
        "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "audit_rerun_outputs": reruns,
        "retained": {
            "behavior": "Qwen3 qualified across six fresh families and all four frozen surfaces; three retained behavior errors did not invalidate any composition set",
            "camera": "known-truth and Qwen identity-only quartet writes were exact across all sixteen routes and both directions",
            "domain": "ten of thirty-two directed edges were robust, two split-specific, and twenty rejected",
            "evidence_subgraph": "the two evidence-first surfaces formed a complete bidirectional two-answer-direction robust subgraph",
            "question_subset": "two question-first source edges were robust only in false-to-true direction",
            "cross_order": "no edge changing evidence/question order was robust",
        },
        "rejected": {
            "global_transport": "identity quartet transport is not a globally admissible operation",
            "behavior_implies_transport": "behaviorally equivalent surfaces did not yield interchangeable state16 role bundles",
            "positive_gain_is_qualification": "many rejected edges moved the margin in the desired direction but did not cross the frozen output and breadth gates",
            "semantic_or_position_cause": "the map does not isolate whether order, prefix history, syntax, position, or downstream compatibility causes an edge to pass",
        },
        "untested": [
            "natural full-layer Hidden State trajectories associated with the ten robust edges",
            "whether relative responses predict robust versus rejected edges on new materials",
            "necessity, minimality, uniqueness, natural use, or coordinate-level structure",
            "other tasks, relations, models, languages, tokenizations, long contexts, or open generation",
            "attention, MLP, parameters, gradients, dimensionality reduction, learned probes, or hotspot searches",
        ],
        "claim_boundary": {
            "allowed": "in frozen C074 at Qwen3 state16, identity quartet transport has a sparse, asymmetric directed domain with ten robust edges and a complete evidence-first subgraph",
            "forbidden": ["semantic-side or physical-order mechanism identified", "semantic neuron group discovered", "necessity or natural implementation", "relative encoding proven", "cross-model or natural-language law"],
        },
        "theory_update": {
            "subject": "empirical domain of a typed partial transport operator",
            "statement": "whole-state role transport is defined only on a sparse subset of source-surface, target-surface, and answer-direction edges",
            "formula": "D_16^identity is a strict subset of S_source x S_target x Y_direction",
            "mathematics": "finite directed graphs, partial functions, and conditional intervention responses remain sufficient",
        },
        "next_question": {
            "campaign": "C075",
            "object": "observation-first full-layer embedding-HiddenState response atlas anchored to the ten frozen robust C074 edges",
            "requirements": [
                "freeze new relation/object materials and semantic-role maps before model execution",
                "collect raw full-dimensional embeddings and every-layer role-aligned Hidden States without attention, MLP, gradients, probes, or dimensionality reduction",
                "change one registered factor at a time and preserve continuous coordinate-level responses",
                "use discovery only for description; freeze any candidate pattern before confirmation and lockbox prediction",
                "failed routes close only their branch, not the observation campaign",
            ],
        },
        "authorization": "preregister_c075_full_hiddenstate_observation_atlas_on_c074_robust_edges",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
