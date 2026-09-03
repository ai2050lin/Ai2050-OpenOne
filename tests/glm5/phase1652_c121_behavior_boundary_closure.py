#!/usr/bin/env python3
"""Close C121 and authorize a discovery-selected output-interface calibration."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1650_c121_structured_comparison_qualification"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

if __name__ == "__main__":
    behavior = core.load(OUT / "analysis/behavior_qualification.json")
    audit = core.load(OUT / "audit/independent_behavior_audit.json")
    if not audit["all_checks_passed"] or behavior["gate_passed"]: raise RuntimeError("C121 closure branch mismatch")
    closure = {
        "phase": 1652, "campaign": "C121", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "structured_comparison_behavior_failed_no_hidden_state_run",
        "headline": behavior["behavior"], "gate_checks": behavior["gate_checks"],
        "strict_conclusion": "Replacing number words with a fresh structured digit table did not qualify the direct true/false candidate interface. The failure is behavioral and output-interface conditioned; no embedding/HiddenState mechanism was tested.",
        "new_puzzles": {"K313-BOUNDARY": "a direct structured digit proposition retains a strong positive-answer asymmetry and fails every registered aggregate behavior gate"},
        "problems": ["one Qwen3", "controlled structured English", "candidate-logit interface", "strong truth-polarity asymmetry", "no free-generation audit", "no HiddenState archive"],
        "next_authorization": "execute_C122_discovery_selected_multi_output_interface_calibration; all interfaces and selection rule must be frozen before model execution",
    }
    core.save(OUT / "analysis/closure.json", closure)
    checks = {"behavior_audit": audit["all_checks_passed"], "failed": not behavior["gate_passed"], "all_gates": not any(behavior["gate_checks"].values()), "no_hidden": not (OUT / "raw/qwen3_role_subtoken_all_states.uint16.npy").exists(), "authorization": closure["next_authorization"].startswith("execute_C122")}
    report = {"phase": 1652, "campaign": "C121", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "authorization": closure["next_authorization"]}
    if not report["all_checks_passed"]: raise RuntimeError(report)
    core.save(OUT / "audit/internal_closure_audit.json", report)
    print(json.dumps({"closure": closure, "audit": report}, indent=2))
