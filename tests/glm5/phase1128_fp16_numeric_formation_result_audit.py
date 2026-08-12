#!/usr/bin/env python3
"""Deterministically audit all Phase1128 artifacts and evidence boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1128_fp16_numeric_formation_protocol as protocol


def expected_names(layer_count: int) -> list[str]:
    return [event["name"] for event in protocol.event_registry(layer_count)]


def audit_model(model_name: str, prereg: dict[str, Any]) -> dict[str, bool]:
    root = protocol.OUT_ROOT / "scan" / model_name
    summary = protocol.read_json(root / "summary.json")
    cases = protocol.read_jsonl(root / "cases.jsonl")
    events = protocol.read_jsonl(root / "events.jsonl")
    by_case: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_case.setdefault(int(event["case_index"]), []).append(event)
    names = expected_names(int(prereg["model_specs"][model_name]["layer_count"]))
    expected_count = len(names)
    registry_exact = all(
        [event["event_name"] for event in sorted(by_case.get(case_index, []), key=lambda row: row["event_order"])]
        == names
        for case_index in range(320)
    )
    precision = summary["precision"]
    return {
        "summary_protocol_link": summary["protocol_digest"] == prereg["protocol_digest"],
        "source_case_link": summary["source_case_digest"] == prereg["source"]["links"][model_name]["case_digest"],
        "source_score_link": summary["source_score_digest"] == prereg["source"]["links"][model_name]["score_detail_digest"],
        "case_digest": protocol.digest(cases) == summary["case_detail_digest"],
        "event_digest": protocol.digest(events) == summary["event_digest"],
        "case_count": len(cases) == 320,
        "case_order": [row["case_index"] for row in cases] == list(range(320)),
        "event_count": len(events) == 320 * expected_count,
        "event_registry_exact": registry_exact,
        "case_event_flags": all(row["event_count_expected"] and row["event_count"] == expected_count for row in cases),
        "fp16_unquantized": precision["has_fp16_parameters"] and not precision["has_bf16_parameters"]
        and not precision["has_quantized_modules"],
        "component_parity": all(
            row["candidate_finite_parity"] and row["suffix_finite_parity"] and row["total_finite_parity"]
            for row in cases
        ),
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    model_checks = {model: audit_model(model, prereg) for model in protocol.MODELS}
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".bin"}
    saved_files = [path for path in protocol.OUT_ROOT.rglob("*") if path.is_file()]
    checks = {
        "protocol_audit_passed": protocol_audit["passed"] is True,
        "protocol_digest_valid": protocol.digest({key: value for key, value in prereg.items() if key != "protocol_digest"})
        == prereg["protocol_digest"],
        "protocol_audit_link": protocol_audit["protocol_digest"] == prereg["protocol_digest"],
        "all_model_checks_passed": all(all(values.values()) for values in model_checks.values()),
        "final_protocol_link": final["protocol_digest"] == prereg["protocol_digest"],
        "final_protocol_audit_link": final["protocol_audit_digest"] == protocol_audit["audit_digest"],
        "final_digest_valid": protocol.digest({key: value for key, value in final.items() if key != "final_digest"})
        == final["final_digest"],
        "prediction_parity_consistent": final["predictions"]["P2_exact_source_finite_parity"] is True,
        "localization_coverage_consistent": final["predictions"]["P3_all_source_nonfinite_cases_localized"] is True,
        "qwen_reference_consistent": final["predictions"]["P4_qwen3_healthy_reference"] is True,
        "auto_continue_consistent": final["automatic_refinement"]["value"]
        == any(result["automatic_refinement_gate_passed"] for result in final["model_results"].values()),
        "no_raw_tensor_artifacts": not any(path.suffix.lower() in forbidden_suffixes for path in saved_files),
        "only_behavior_cases": all(
            set(row["partition"] for row in protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"))
            == set(("discovery", "independent_confirmation"))
            for model in protocol.MODELS
        ),
    }
    audit_core = {
        "schema_version": "phase1128_fp16_numeric_formation_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "model_checks": model_checks,
        "passed_count": sum(checks.values()) + sum(sum(values.values()) for values in model_checks.values()),
        "total_count": len(checks) + sum(len(values) for values in model_checks.values()),
        "passed": all(checks.values()) and all(all(values.values()) for values in model_checks.values()),
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    audit = dict(audit_core)
    audit["audit_digest"] = protocol.digest(audit_core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise RuntimeError("Phase1128 result audit failed")


if __name__ == "__main__":
    main()
