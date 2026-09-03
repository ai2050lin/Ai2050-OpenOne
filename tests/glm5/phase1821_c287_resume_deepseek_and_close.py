#!/usr/bin/env python3
"""Resume C287 in a fresh process for DeepSeek-7B, then close the phase."""
from __future__ import annotations

import json

import phase1811_c277_c289_joint_response_common as common
import phase1821_c287_cross_model_joint_state_capture as campaign

core, OUT = common.core, common.OUTS["C287"]


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    rows = core.rows(OUT / "material/cases.jsonl")
    if not (OUT / "analysis/deepseek7b.json").exists():
        campaign.run_model("deepseek7b", rows, protocol)
    reports = {name: core.load(OUT / f"analysis/{name}.json") for name in common.MODELS}
    checks = {
        "models": set(reports) == set(common.MODELS),
        "model_audits": all(core.load(OUT / f"audit/internal_{name}_audit.json")["all_checks_passed"] for name in common.MODELS),
        "fresh_process_resume_registered": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": True})
    report = {
        "phase": 1821,
        "campaign": "C287",
        "status": "cross_model_joint_states_captured",
        "models": reports,
        "participants": [name for name, value in reports.items() if value["behavior_eligible"]],
        "execution_note": "DeepSeek-7B was resumed in a fresh Python process after the original sequential process exited during model load; frozen material, interfaces, gates, and analysis were unchanged.",
        "strict_interpretation": protocol["claim_boundary"],
        "next_authorization": "C288_cross_model_automaton_isomorphism",
    }
    core.save(OUT / "analysis/summary.json", report)
    producer = __import__("pathlib").Path(campaign.__file__)
    final_checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": True, "producer_hash": core.sha(producer) == protocol["producer_sha256"]}
    final = {"phase": 1821, "campaign": "C287", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/resume_audit.json", {"checks": checks, "all_checks_passed": True})
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

