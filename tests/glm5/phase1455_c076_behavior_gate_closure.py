#!/usr/bin/env python3
"""Phase1455: close C076 at the behavior gate."""
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

PHASE, CAMPAIGN = 1455, "C076"
OUT = TESTS / "result/phase1455_c076_behavior_gate_closure"
P1453 = TESTS / "result/phase1453_c076_relation_discrimination_contract"
P1454 = TESTS / "result/phase1454_c076_behavior"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1455 exists")
    protocol = core.load(P1453 / "protocol/preregistration.json")
    behavior = core.load(P1454 / "analysis/behavior_summary.json")
    final1454 = core.load(P1454 / "analysis/final.json")
    reruns, audits_reran, scripts = {}, True, []
    for phase, stem in ((1453, "c076_relation_discrimination_contract"), (1454, "c076_behavior")):
        main_script, audit_script = TESTS / f"phase{phase}_{stem}.py", TESTS / f"phase{phase}_{stem}_audit.py"
        scripts.extend((main_script, audit_script))
        completed = subprocess.run([str(PYTHON), str(audit_script)], cwd=str(ROOT), capture_output=True, text=True, check=False)
        audits_reran &= completed.returncode == 0
        reruns[str(phase)] = {"returncode": completed.returncode, "stdout_tail": completed.stdout[-1200:], "stderr_tail": completed.stderr[-1200:]}
    scripts_compile = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            scripts_compile = False
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1453, P1454)]
    checks = {
        "audits_reran": audits_reran, "audits_pass": all(a["all_checks_passed"] for a in audits), "scripts_compile": scripts_compile,
        "contract_hash": protocol["contract_sha256"] == "250e0dd68d401fe665a8ab0f369f2274c2bb10fb57efae4dae6c4685fccc4d3f",
        "behavior_failed": not behavior["behavior_qualified"] and final1454["authorization"] == "close_c076_at_behavior_gate",
        "metrics": abs(behavior["global_balanced_accuracy"] - 0.8391203703703703) < 1e-12,
        "no_relations": behavior["qualified_relations"] == [], "eligible": behavior["eligible_count"] == 57,
        "hidden_blocked": behavior["hidden_state_accessed"] is False and not (TESTS / "result/phase1455_c076_discovery_full_field_capture").exists(),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed_at_behavior_gate_after_morphology_nuisance_failure",
        "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "audit_rerun_outputs": reruns,
        "retained": {"numeric": "BF16 execution was finite and exactly repeatable", "behavior": "different-relation cases were often rejected but same-concept past/base pairs were not stable", "nuisance": "person and organization matching changed relation judgments despite explicit ignore instructions"},
        "rejected": {"qualified_object": "no relation passed the complete family-surface-cell gate", "semantic_absence": "behavior failure does not show that relation information is absent internally", "hidden_claim": "no Hidden State was read"},
        "untested": ["labeled relation calibration", "raw embedding-to-layer trajectories", "held-out response invariance", "causal use or natural relation semantics"],
        "next_question": {"campaign": "C077", "object": "known-truth labeled relation atlas retaining natural verbs and nuisance factors", "requirements": ["new names, organizations, surfaces, and partitions", "explicit canonical relation labels paired with natural verb clauses", "behavior-first, discovery-only raw all-layer capture, then frozen holdout prediction", "separate label-carrier calibration from unlabeled natural-semantic claims", "no attention, MLP, parameters, gradients, probes, or dimensionality reduction"]},
        "authorization": "preregister_c077_labeled_relation_full_field_calibration",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
