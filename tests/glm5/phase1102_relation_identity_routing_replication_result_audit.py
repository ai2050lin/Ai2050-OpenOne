#!/usr/bin/env python3
"""Independently audit Phase1102 behavior-only stop artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import phase1102_relation_identity_routing_replication_protocol as protocol


TEST_ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_without(row: dict, key: str) -> str:
    copied = dict(row)
    copied.pop(key, None)
    return protocol.digest(copied)


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = protocol.read_json(root / "protocol" / "preregistration.json")
    common_audit = protocol.read_json(root / "protocol" / "audit.json")
    authorization = protocol.read_json(root / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(root / "analysis" / "final_summary.json")
    diagnostic = protocol.read_json(root / "analysis" / "failure_diagnostic.json")
    checks = {}
    checks["protocol_digest_recomputes"] = digest_without(prereg, "protocol_digest") == prereg["protocol_digest"]
    checks["protocol_audit_digest_recomputes"] = digest_without(common_audit, "audit_digest") == common_audit["audit_digest"]
    checks["protocol_audit_passed"] = common_audit["all_checks_passed"]
    checks["authorization_digest_recomputes"] = digest_without(authorization, "authorization_digest") == authorization["authorization_digest"]
    checks["final_digest_recomputes"] = digest_without(final, "final_digest") == final["final_digest"]
    checks["diagnostic_digest_recomputes"] = digest_without(diagnostic, "diagnostic_digest") == diagnostic["diagnostic_digest"]
    checks["source_phase1101_digest_matches"] = prereg["source_phase1101_authorization_digest"] == protocol.read_json(protocol.SOURCE_PHASE1101_AUTHORIZATION)["authorization_digest"]
    checks["hidden_scan_not_authorized"] = not authorization["hidden_scan_authorized"]
    checks["automatic_next_false"] = not final["automatic_next_required"]
    checks["no_atlas_artifacts"] = not any((root / "atlas" / model).exists() for model in protocol.MODELS)
    checks["diagnostic_non_upgrading"] = diagnostic["evidence_status"] == "non_upgrading_post_frozen_diagnostic"
    expected_cases = (
        len(protocol.RELATION_PAIRS) * len(protocol.SURFACES)
        * len(protocol.TEMPLATES) * protocol.ITEMS_PER_TEMPLATE
        * len(protocol.STATES)
    )
    for model in protocol.MODELS:
        cases = protocol.read_jsonl(root / "protocol" / f"cases.{model}.jsonl")
        model_audit = protocol.read_json(root / "protocol" / f"audit.{model}.json")
        summary = protocol.read_json(root / "behavior" / model / "summary.json")
        candidate = protocol.read_jsonl(root / "behavior" / model / "candidate_detail.jsonl")
        generation = protocol.read_jsonl(root / "behavior" / model / "generation_detail.jsonl")
        checks[f"{model}_case_count"] = len(cases) == expected_cases
        checks[f"{model}_case_digest"] = protocol.digest(cases) == prereg["model_case_digests"][model]
        checks[f"{model}_model_audit"] = model_audit["all_checks_passed"]
        checks[f"{model}_summary_digest"] = digest_without(summary, "summary_digest") == summary["summary_digest"]
        checks[f"{model}_candidate_count"] = len(candidate) == expected_cases
        checks[f"{model}_generation_count"] = len(generation) == summary["generation_count"]
        checks[f"{model}_fp16"] = summary["precision"]["has_fp16_parameters"]
        checks[f"{model}_not_bf16"] = not summary["precision"]["has_bf16_parameters"]
        checks[f"{model}_not_quantized"] = not summary["precision"]["has_quantized_modules"]
        checks[f"{model}_only_pair_coverage_failed"] = (
            set(key for key, value in authorization["models"][model]["gates"].items() if not value)
            == {"pair_coverage"}
        )
    script_paths = (
        TEST_ROOT / "phase1102_relation_identity_routing_replication_protocol.py",
        TEST_ROOT / "phase1102_relation_identity_routing_replication_behavior.py",
        TEST_ROOT / "phase1102_relation_identity_routing_replication_behavior_finalize.py",
        TEST_ROOT / "phase1102_relation_identity_routing_replication_finalize.py",
        TEST_ROOT / "phase1102_relation_identity_routing_replication_diagnostic.py",
        TEST_ROOT / "phase1102_relation_identity_routing_replication_result_audit.py",
        TEST_ROOT / "phase1102_run_sequential.py",
    )
    checks["all_scripts_exist"] = all(path.exists() for path in script_paths)
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema_version": "phase1102_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "failed_count": len(failed),
        "all_checks_passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "artifact_sha256": {
            "preregistration": file_sha256(root / "protocol" / "preregistration.json"),
            "behavior_authorization": file_sha256(root / "analysis" / "behavior_authorization.json"),
            "final_summary": file_sha256(root / "analysis" / "final_summary.json"),
            "failure_diagnostic": file_sha256(root / "analysis" / "failure_diagnostic.json"),
            **{
                f"script_{path.stem}": file_sha256(path)
                for path in script_paths if path.exists()
            },
        },
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(root / "audit" / "result_audit.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "check_count": len(checks),
        "failed_count": len(failed),
        "all_checks_passed": not failed,
        "failed_checks": failed,
        "audit_digest": result["audit_digest"],
    }, ensure_ascii=False), flush=True)
    if failed:
        raise RuntimeError(f"Phase1102 result audit failed: {failed}")


if __name__ == "__main__":
    main()
