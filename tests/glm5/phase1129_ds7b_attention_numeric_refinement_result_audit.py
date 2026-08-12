#!/usr/bin/env python3
"""Audit the complete one-shot Phase1129 numerical refinement."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1129_ds7b_attention_numeric_refinement_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    scan = protocol.read_json(protocol.OUT_ROOT / "scan" / protocol.MODEL / "summary.json")
    cases = protocol.read_jsonl(protocol.OUT_ROOT / "scan" / protocol.MODEL / "cases.jsonl")
    events = protocol.read_jsonl(protocol.OUT_ROOT / "scan" / protocol.MODEL / "events.jsonl")
    final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    by_case: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        by_case[int(event["case_index"])].append(event)
    expected_names = [event["name"] for event in protocol.EVENT_REGISTRY]
    registry_exact = all(
        [event["event_name"] for event in sorted(by_case[index], key=lambda row: row["event_order"])]
        == expected_names
        for index in range(320)
    )
    forbidden_suffixes = {".npy", ".npz", ".pt", ".pth", ".safetensors", ".bin"}
    saved = [path for path in protocol.OUT_ROOT.rglob("*") if path.is_file()]
    checks = {
        "protocol_audit_passed": protocol_audit["passed"] is True,
        "protocol_digest_valid": protocol.digest({key: value for key, value in prereg.items() if key != "protocol_digest"})
        == prereg["protocol_digest"],
        "protocol_links": scan["protocol_digest"] == prereg["protocol_digest"]
        and final["protocol_digest"] == prereg["protocol_digest"],
        "case_digest": protocol.digest(cases) == scan["case_detail_digest"],
        "event_digest": protocol.digest(events) == scan["event_digest"],
        "case_count": len(cases) == 320 and [row["case_index"] for row in cases] == list(range(320)),
        "event_count": len(events) == 320 * len(protocol.EVENT_REGISTRY),
        "event_registry_exact": registry_exact,
        "event_count_flags": all(row["event_count_expected"] for row in cases),
        "fp16_unquantized": scan["precision"]["has_fp16_parameters"]
        and not scan["precision"]["has_bf16_parameters"] and not scan["precision"]["has_quantized_modules"],
        "finite_parity": all(row["candidate_finite_parity"] and row["suffix_finite_parity"]
                             and row["total_finite_parity"] for row in cases),
        "root_classification_complete": final["predictions"]["P3_all_source_nonfinite_cases_classified"] is True,
        "final_digest": protocol.digest({key: value for key, value in final.items() if key != "final_digest"})
        == final["final_digest"],
        "auto_continue_false": final["auto_continue"]["value"] is False,
        "no_raw_tensor_artifacts": not any(path.suffix.lower() in forbidden_suffixes for path in saved),
    }
    audit_core = {
        "schema_version": "phase1129_ds7b_attention_numeric_refinement_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    audit = dict(audit_core)
    audit["audit_digest"] = protocol.digest(audit_core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise RuntimeError("Phase1129 result audit failed")


if __name__ == "__main__":
    main()
