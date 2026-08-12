#!/usr/bin/env python3
"""Post-result resolution of the Phase1222 FP16 audit schema mismatch.

This supplement cannot alter behavior rows, gates, authorization, or the
frozen 13/14 result audit.  It only evaluates the precision payload using the
field names emitted by phase1023_fp16_utils.quantization_audit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1222_atomic_operation_independent_confirmation as p


OUT = p.OUT_ROOT / "audit/fp16_schema_resolution.json"


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    frozen = p.read_json(p.RESULT_AUDIT_PATH)
    summary = p.read_json(p.RUN_SUMMARY_PATH)
    raw = p.read_jsonl(p.RAW_PATH)
    final = p.read_json(p.FINAL_PATH)
    precision = summary["precision_audit"]
    failed_names = [item["name"] for item in frozen["checks"] if not item["passed"]]
    dtypes = precision.get("parameter_dtypes", {})
    checks = [
        check(
            "frozen_audit_failure_is_only_precision_schema",
            failed_names == ["fp16_nonquantized_execution"],
            failed_names,
        ),
        check(
            "all_parameter_elements_are_float16",
            set(dtypes) == {"float16"} and int(dtypes.get("float16", 0)) > 0,
            dtypes,
        ),
        check(
            "precision_flags_consistent",
            precision.get("has_fp16_parameters") is True
            and precision.get("has_bf16_parameters") is False,
            precision,
        ),
        check(
            "no_quantized_modules",
            precision.get("has_quantized_modules") is False
            and precision.get("suspicious_quantized_module_classes") == [],
            precision.get("suspicious_quantized_module_classes"),
        ),
        check(
            "raw_summary_unchanged",
            summary["raw_digest"] == p.digest(raw),
            summary["raw_digest"],
        ),
        check(
            "final_authorization_unchanged",
            final["final_digest"]
            == p.digest({key: value for key, value in final.items() if key != "final_digest"}),
            final["behavior"]["authorized_target_operation_tracks"],
        ),
        check(
            "supplement_scope_is_precision_only",
            final["claim_boundary"]["behavior_only"] is True
            and final["claim_boundary"]["hidden_state"] is False,
            final["claim_boundary"],
        ),
    ]
    report: dict[str, Any] = {
        "phase": p.PHASE,
        "audit_stage": "post_result_precision_schema_resolution",
        "does_not_upgrade_or_modify_behavior_gates": True,
        "frozen_result_audit_digest": frozen["audit_digest"],
        "check_count": len(checks),
        "passed_count": sum(item["passed"] for item in checks),
        "all_checks_passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    report["audit_digest"] = p.digest(report)
    p.write_json(OUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
