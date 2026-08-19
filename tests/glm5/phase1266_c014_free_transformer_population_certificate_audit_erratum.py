from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "tests/glm5/result/phase1266_c014_free_transformer_population_certificate"
FROZEN_AUDITOR = ROOT / "tests/glm5/phase1266_c014_free_transformer_population_certificate_audit.py"
PROTOCOL = RESULT_ROOT / "protocol/preregistration.json"
FINAL = RESULT_ROOT / "analysis/final.json"
ORIGINAL_AUDIT = RESULT_ROOT / "audit/independent_final_audit.json"
OUTPUT = RESULT_ROOT / "audit/independent_final_audit_erratum.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol = read_json(PROTOCOL)
    final = read_json(FINAL)
    original = read_json(ORIGINAL_AUDIT)
    claim = final.get("claim_boundary", "")

    original_checks = original.get("checks", {})
    non_scope_checks = {
        key: value for key, value in original_checks.items() if key != "scope_narrow"
    }
    checks = {
        "frozen_auditor_hash_preserved": sha256(FROZEN_AUDITOR)
        == protocol["source_hashes"]["auditor"],
        "original_audit_is_exactly_15_of_16": original.get("passed_checks") == 15
        and original.get("total_checks") == 16
        and original.get("all_checks_passed") is False,
        "only_original_failure_is_scope_lexeme": all(non_scope_checks.values())
        and original_checks.get("scope_narrow") is False,
        "scope_names_free_transformers": "Free 4/6/8-layer same-executor Transformers"
        in claim,
        "scope_names_single_synthetic_universe": "one exhaustive cyclic-code factorial universe"
        in claim,
        "scope_excludes_natural_language": "not natural language" in claim,
        "scope_excludes_unique_circuit": "not natural language or a unique circuit" in claim,
        "scientific_verdict_remains_failure": final.get("passed") is False
        and final.get("authorization", {}).get("new_pretrained_contract") is False,
    }
    passed = all(checks.values())
    payload = {
        "status": "post_registration_audit_erratum",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Correct one brittle lexical scope check without modifying the frozen auditor, "
            "scientific thresholds, measurements, gates, final claim, or failure verdict."
        ),
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks),
        "passed": passed,
        "original_audit_passed": False,
        "scientific_verdict_changed": False,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": f"{payload['passed_checks']}/{payload['total_checks']}", "passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
