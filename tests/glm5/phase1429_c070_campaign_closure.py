#!/usr/bin/env python3
"""Phase1429: close C070 after the frozen support-partition classification."""
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

OUT = TESTS / "result/phase1429_c070_campaign_closure"
P1425 = TESTS / "result/phase1425_c070_quartet_complement_contract"
P1426 = TESTS / "result/phase1426_c070_roster_behavior"
P1427 = TESTS / "result/phase1427_c070_partition_camera"
P1428 = TESTS / "result/phase1428_c070_support_partition"
PHASES = (
    (1425, "c070_quartet_complement_contract"),
    (1426, "c070_roster_behavior"),
    (1427, "c070_partition_camera"),
    (1428, "c070_support_partition"),
)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1429 exists")
    protocol = core.load(P1425 / "protocol/preregistration.json")
    behavior = core.load(P1426 / "analysis/behavior_summary.json")
    camera = core.load(P1427 / "analysis/camera_summary.json")
    mechanism = core.load(P1428 / "analysis/support_partition_summary.json")
    scripts = []
    audits_reran = True
    audit_outputs = {}
    for phase, stem in PHASES:
        main_script = TESTS / f"phase{phase}_{stem}.py"
        audit_script = TESTS / f"phase{phase}_{stem}_audit.py"
        scripts.extend((main_script, audit_script))
        completed = subprocess.run(
            [str(PYTHON), str(audit_script)],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
        )
        audits_reran &= completed.returncode == 0
        audit_outputs[str(phase)] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
    compiled = True
    for script in scripts:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception:
            compiled = False
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (P1425, P1426, P1427, P1428)]
    directions = mechanism["direction_results"]
    aggregate = mechanism["aggregate_metrics"]
    checks = {
        "audits_reran": audits_reran,
        "audits_pass": all(audit["all_checks_passed"] for audit in audits),
        "scripts_compile": compiled,
        "contract_hash": mechanism["contract_sha256"] == protocol["contract_sha256"],
        "behavior": behavior["behavior_qualified"] and len(behavior["qualified_families"]) == 6 and behavior["selected_count"] == 72,
        "camera": camera["camera_qualified"] and camera["qwen_self_max_abs_diff"] == 0.0 and camera["qwen_donor_max_abs_diff"] == 0.0,
        "mechanism_execution": mechanism["all_execution_checks_passed"] and mechanism["record_count"] == 480,
        "bidirectional_quartet": all(directions[direction]["classification"] == "quartet_dominant" for direction in directions),
        "quartet_six_families": all(len(directions[direction]["quartet_qualified_families"]) == 6 for direction in directions),
        "full_six_families": all(len(directions[direction]["full_qualified_families"]) == 6 for direction in directions),
        "zero_complement_families": all(directions[direction]["complement_qualified_families"] == [] for direction in directions),
        "all_quartet_sign": all(aggregate[direction][split]["desired_sign_fraction"]["quartet_only"] == 1.0 for direction in directions for split in aggregate[direction]),
        "zero_complement_sign": all(aggregate[direction][split]["desired_sign_fraction"]["complement_only"] == 0.0 for direction in directions for split in aggregate[direction]),
    }
    result = {
        "phase": 1429,
        "campaign": "C070",
        "status": "closed_after_bidirectional_quartet_dominance_under_controlled_roster_contract",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "audit_rerun_outputs": audit_outputs,
        "retained": {
            "behavior": "six roster families and all 72 composition sets qualified",
            "camera": "state16 self and full-donor transports are exact on discovery",
            "conditional_sufficiency": "the four frozen role positions switch the answer in both directions on confirmation and lockbox across all six families",
            "classification": mechanism["overall_classification"],
            "quartet_desired_sign_fractions": {
                direction: {split: aggregate[direction][split]["desired_sign_fraction"]["quartet_only"] for split in aggregate[direction]}
                for direction in aggregate
            },
        },
        "rejected": {
            "complement_standalone_sufficiency": "the remaining 66 state16 positions alone do not cross the desired answer boundary under the frozen transport",
            "complement_desired_sign_fractions": {
                direction: {split: aggregate[direction][split]["desired_sign_fraction"]["complement_only"] for split in aggregate[direction]}
                for direction in aggregate
            },
            "joint_synergy_gate": "full-state did not show the frozen positive synergy advantage over the stronger partition",
        },
        "untested": [
            "necessity, minimality, uniqueness, or semantic purity of the quartet",
            "individual role contributions or interactions inside the quartet",
            "cross-surface role-isomorphic transport",
            "natural open-language relations, other models, languages, or tokenizations",
            "attention, MLP, parameters, gradients, dimensionality reduction, or learned probes",
        ],
        "claim_boundary": {
            "allowed": "Qwen controlled-roster state16 four-role whole-state bundle is conditionally sufficient for bidirectional answer switching while its physical-position complement is not independently sufficient",
            "forbidden": [
                "relation manifold or semantic neuron group discovered",
                "quartet is necessary, minimal, unique, or semantically pure",
                "relative encoding proven",
                "cross-surface, cross-model, or natural-language law",
            ],
        },
        "theory_update": {
            "subject": "conditionalized role-bundle support",
            "statement": "a functionally named state bundle may be sufficient only relative to the recipient complement and downstream computation; physical dominance is not semantic identity",
            "formula": "do(H16[r,P4] <- H16[d,P4]) changes the registered output in both directions under fixed recipient complement",
        },
        "next_question": {
            "campaign": "C071",
            "object": "cross-surface role-isomorphic external validity of the state16 quartet",
            "reason": "the direct unresolved alternative is that C070 transported fixed physical/surface identity rather than a surface-invariant functional role bundle",
            "constraints": [
                "fresh roster material and at least two order-different natural controlled surfaces",
                "explicit precompiled semantic-role map phi",
                "behavior first, camera first, discovery/confirmation/lockbox",
                "fixed Qwen state16 and frozen quartet; no layer, role-subset, or coordinate search",
                "Hidden State and logits only; no attention, MLP, parameters, gradients, PCA, or learned probes",
            ],
        },
        "authorization": "preregister_c071_cross_surface_role_isomorphic_quartet_transport",
    }
    core.save(OUT / "analysis/closure_summary.json", result)
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
