#!/usr/bin/env python3
"""Independently audit the frozen Phase1112 behavior-stop result."""

from __future__ import annotations

import json
from pathlib import Path

import phase1112_one_shot_body_reader_protocol as protocol


def exists(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = protocol.read_json(root / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(root / "protocol" / "audit.json")
    behavior = protocol.read_json(root / "analysis" / "behavior_authorization.json")
    diagnostic = protocol.read_json(root / "analysis" / "behavior_failure_diagnostic.json")
    final = protocol.read_json(root / "analysis" / "final_summary.json")
    checks = {
        "phase_exact": final["phase"] == protocol.PHASE == 1112,
        "protocol_digest_link": final["protocol_digest"] == prereg["protocol_digest"],
        "protocol_digest_recomputes": protocol.digest({
            key: value for key, value in prereg.items() if key != "protocol_digest"
        }) == prereg["protocol_digest"],
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "protocol_audit_link": final["protocol_audit_digest"] == protocol_audit["audit_digest"],
        "behavior_link": final["behavior_authorization_digest"] == behavior["authorization_digest"],
        "behavior_not_authorized": not behavior["hidden_scan_authorized"],
        "no_cross_model_pairs": not behavior["cross_model_pairs"],
        "no_authorized_models": not behavior["authorized_models"],
        "glm4_four_pairs": len(behavior["models"]["glm4"]["passing_pairs"]) == 4,
        "qwen3_zero_pairs": len(behavior["models"]["qwen3"]["passing_pairs"]) == 0,
        "deepseek7b_zero_pairs": len(behavior["models"]["deepseek7b"]["passing_pairs"]) == 0,
        "prediction_vector": final["prospective_predictions"] == {
            "P1": True,
            "P2": False,
            "P3": False,
            "P4": False,
            "P5": False,
            "P6": False,
            "P7": False,
            "P8": True,
        },
        "hidden_not_tested": final["evidence"]["body_attention_reader"] == "not_tested_behavior_denied",
        "registry_closed": final["evidence"]["registry_status"] == "closed_to_further_hotspot_search",
        "causal_not_authorized": not final["causal_staircase_authorized"],
        "localization_not_authorized": not final["component_head_qkv_neuron_localization_authorized"],
        "automatic_next_false": not final["automatic_next_required"],
        "theory_name_stable": final["canonical_theory_name_unchanged"] == "条件化输出场闭合理论",
        "diagnostic_link": final["diagnostic_digest"] == diagnostic["diagnostic_digest"],
        "final_digest_recomputes": protocol.digest({
            key: value for key, value in final.items() if key != "final_summary_digest"
        }) == final["final_summary_digest"],
    }
    for model in protocol.MODELS:
        denial_path = root / "atlas" / model / "denial.json"
        summary_path = root / "behavior" / model / "summary.json"
        detail_path = root / "behavior" / model / "candidate_detail.jsonl"
        checks[f"{model}_behavior_files"] = exists(summary_path) and exists(detail_path)
        checks[f"{model}_denial_exists"] = exists(denial_path)
        denial = protocol.read_json(denial_path)
        checks[f"{model}_hidden_false"] = not denial["hidden_access"]
        checks[f"{model}_denial_link"] = (
            denial["denial_digest"] == final["denial_digests"][model]
        )
    result = {
        "schema_version": "phase1112_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(root / "audit" / "result_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
