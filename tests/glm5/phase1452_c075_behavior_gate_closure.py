#!/usr/bin/env python3
"""Phase1452: close C075 at the preregistered behavior gate."""
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

PHASE, CAMPAIGN = 1452, "C075"
OUT = TESTS / "result/phase1452_c075_behavior_gate_closure"
P1450 = TESTS / "result/phase1450_c075_full_field_atlas_contract"
P1451 = TESTS / "result/phase1451_c075_behavior"


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1452 exists")
    protocol = core.load(P1450 / "protocol/preregistration.json")
    behavior = core.load(P1451 / "analysis/behavior_summary.json")
    final1451 = core.load(P1451 / "analysis/final.json")
    reruns, audits_reran = {}, True
    scripts = []
    for phase, stem in ((1450, "c075_full_field_atlas_contract"), (1451, "c075_behavior")):
        main_script = TESTS / f"phase{phase}_{stem}.py"
        audit_script = TESTS / f"phase{phase}_{stem}_audit.py"
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
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1450, P1451)]
    rows = core.rows(P1451 / "raw/active_behavior.jsonl")
    errors = [row for row in rows if not row["correct"]]
    checks = {
        "audits_reran": audits_reran,
        "audits_pass": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": scripts_compile,
        "contract_hash": protocol["contract_sha256"] == "e32fe3dd0bd0e98d85c8e9db41bd00f410457adec7b79854afaff1ba0a91e518",
        "behavior_failed": not behavior["behavior_qualified"] and final1451["authorization"] == "close_c075_at_behavior_gate",
        "global_metrics": behavior["global_balanced_accuracy"] > protocol["zero_model_gate"]["required_model_balanced_accuracy_min"],
        "single_relation": behavior["qualified_relations"] == ["supported"] and behavior["selected_count"] == 36,
        "error_count": len(errors) == 96,
        "error_cell": {row["cell"] for row in errors} == {"110"},
        "hidden_blocked": behavior["hidden_state_accessed"] is False and not (TESTS / "result/phase1452_c075_discovery_full_field_capture").exists(),
    }
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "closed_at_behavior_gate_before_hiddenstate_capture",
        "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()),
        "audit_rerun_outputs": reruns,
        "retained": {
            "global_behavior": "global balanced accuracy exceeded the frozen 0.97 gate on both surfaces",
            "error_localization": "all ninety-six errors occurred when person and organization matched but relation differed",
            "supported": "supported was the only relation with all thirty-six composition sets eligible",
            "numeric": "BF16 nonquantized execution was finite and exactly repeatable",
        },
        "rejected": {
            "six_relation_object": "five of six relations failed the fixed capture-shape behavior requirement",
            "supported_only_rescue": "post hoc restriction to supported is not an authorized six-relation atlas",
            "hidden_inference": "no Hidden State was read, so no relation representation or coding claim is available",
        },
        "untested": ["embedding-to-all-layer relation response fields", "held-out relation response prediction", "coordinate-level reuse or difference", "causal sufficiency, necessity, composition, or natural generation"],
        "next_question": {
            "campaign": "C076",
            "object": "new explicit relation-discrimination interface with the six relations and three matched factors",
            "requirements": [
                "new words, surfaces, and partitions; C075 cases remain closed",
                "make relation identity an explicit natural judgment rather than relying on an all-fields conjunction instruction",
                "retain entity-only, object-only, relation-only, and pairwise incomplete zero models",
                "behavior-first and raw full-layer discovery-only capture after qualification",
                "no attention, MLP, parameters, gradients, probes, or dimensionality reduction",
            ],
        },
        "authorization": "preregister_c076_explicit_relation_discrimination_atlas",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
