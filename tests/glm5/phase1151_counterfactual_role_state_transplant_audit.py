from __future__ import annotations

import json
from pathlib import Path

import phase1148_mandatory_mediation_calibration as p1148
import phase1151_counterfactual_role_state_transplant as p1151


def main() -> None:
    prereg = p1148.read_json(p1151.PREREG_PATH)
    p1151.verify_preregistration(prereg)
    stored = p1148.read_json(p1151.OUT_ROOT / "analysis" / "final.json")
    checks: dict[str, bool] = {}
    protocol_audit = p1148.read_json(p1151.OUT_ROOT / "protocol" / "audit.json")
    checks["protocol_audit"] = bool(protocol_audit["all_checks_passed"])
    protocol_audit_body = dict(protocol_audit)
    protocol_audit_digest = protocol_audit_body.pop("audit_digest")
    checks["protocol_audit_digest"] = (
        p1148.canonical_digest(protocol_audit_body) == protocol_audit_digest
    )

    recomputed_replicates = {}
    for replicate in prereg["replicates"]:
        result = p1151.evaluate_replicate(replicate, prereg)
        recomputed_replicates[replicate] = result
        checks[f"{replicate}.result_digest"] = p1148.canonical_digest(
            {key: value for key, value in result.items() if key != "result_digest"}
        ) == result["result_digest"]
        checks[f"{replicate}.exact_recomputation"] = (
            result == stored["per_replicate"][replicate]
        )

    sufficiency = all(item["sufficiency_passed"] for item in recomputed_replicates.values())
    specificity = all(item["specificity_passed"] for item in recomputed_replicates.values())
    checks["sufficiency_summary"] = (
        sufficiency == stored["counterfactual_sufficiency_confirmed"]
    )
    checks["specificity_summary"] = (
        specificity == stored["cross_role_position_specificity_confirmed"]
    )
    final_body = dict(stored)
    final_digest = final_body.pop("final_digest")
    checks["final_digest"] = p1148.canonical_digest(final_body) == final_digest
    checks["no_automatic_hotspot_search"] = not stored["auto_continue"]

    audit = {
        "phase": 1151,
        "protocol_digest": prereg["protocol_digest"],
        "audit_script_sha256": p1148.file_sha256(Path(__file__).resolve()),
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    audit["audit_digest"] = p1148.canonical_digest(audit)
    p1148.write_json(p1151.OUT_ROOT / "audit" / "independent_recomputation.json", audit)
    if not audit["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase1151 audit failed: {failed}")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
