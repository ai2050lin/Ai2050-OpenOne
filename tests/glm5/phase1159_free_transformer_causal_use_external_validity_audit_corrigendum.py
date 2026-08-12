#!/usr/bin/env python3
"""Post-freeze corrigendum for one lexical false positive in the Phase1159 audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1159_free_transformer_causal_use_external_validity"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    original = read_json(OUT_ROOT / "audit/independent_audit.json")
    predictions = read_json(OUT_ROOT / "predictions/confirmation_predictions.json")
    failed = [row for row in original["checks"] if not row["passed"]]
    factors_only = canonical(predictions["factors"]).lower()
    checks = {
        "original_audit_digest_valid": digest({key: value for key, value in original.items() if key != "audit_digest"})
        == original["audit_digest"],
        "original_check_count_138": original["check_count"] == 138,
        "original_passed_count_137": original["passed_count"] == 137,
        "exactly_one_original_failure": len(failed) == 1,
        "failure_is_lexical_architecture_check": len(failed) == 1
        and failed[0]["name"] == "predictions.no_architecture_label",
        "declaration_explicitly_false": predictions.get("architecture_labels_used") is False,
        "compact_label_absent_from_factor_predictions": "compact" not in factors_only,
        "deep_label_absent_from_factor_predictions": "deep" not in factors_only,
        "no_architecture_key_inside_factor_predictions": "architecture" not in factors_only,
        "all_other_original_checks_passed": all(
            row["passed"] for row in original["checks"] if row["name"] != "predictions.no_architecture_label"
        ),
    }
    result = {
        "phase": 1159,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": "audit lexical false-positive corrigendum",
        "original_audit_digest": original["audit_digest"],
        "original_audit_status_preserved": "137/138; not overwritten",
        "correction_scope": (
            "The original substring test matched the boolean metadata key architecture_labels_used. "
            "It did not detect compact/deep labels or any architecture assignment in factor predictions."
        ),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "failed_count": len(checks) - sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "scientific_result_changed": False,
    }
    result["corrigendum_digest"] = digest(result)
    path = OUT_ROOT / "audit/independent_audit_corrigendum.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical(result))
    if not result["all_checks_passed"]:
        raise RuntimeError("Phase1159 audit corrigendum failed")


if __name__ == "__main__":
    main()
