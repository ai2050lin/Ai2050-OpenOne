#!/usr/bin/env python3
"""Audit Phase1092 artifacts, digests, frozen gates, and non-causal status."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1092_natural_bilingual_attribute_protocol as protocol


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_json(path: Path, digest_key: str) -> tuple[bool, str]:
    payload = protocol.read_json(path)
    recorded = str(payload[digest_key])
    body = dict(payload)
    body.pop(digest_key)
    return recorded == protocol.digest(body), recorded


def main() -> None:
    root = protocol.OUT_ROOT
    analysis = root / "analysis"
    required_json = [
        (root / "protocol" / "preregistration.json", "protocol_digest"),
        (root / "protocol" / "audit.json", "audit_digest"),
        (analysis / "behavior_authorization.json", "summary_digest"),
        (analysis / "within_language_identity.json", "within_digest"),
        (analysis / "cross_language_identity_geometry.json", "cross_language_digest"),
        (analysis / "heldout_world_geometry.json", "heldout_digest"),
        (analysis / "cross_model_geometry.json", "cross_model_digest"),
        (analysis / "shared_field.json", "shared_field_digest"),
        (analysis / "control_audit.json", "control_digest"),
        (analysis / "decomposition.json", "decomposition_digest"),
        (analysis / "physical_map.json", "physical_map_digest"),
        (analysis / "projection_audit.json", "projection_digest"),
        (analysis / "numeric_audit.json", "numeric_digest"),
        (analysis / "automatic_next.json", "automatic_next_digest"),
        (analysis / "final_summary.json", "summary_digest"),
        (analysis / "posthoc_template_robustness.json", "posthoc_digest"),
    ]
    for model_name in protocol.MODELS:
        required_json.extend([
            (root / "pilot" / f"{model_name}.json", "result_digest"),
            (root / "atlas" / model_name / "summary.json", "summary_digest"),
        ])

    digest_rows = []
    for path, digest_key in required_json:
        if not path.exists():
            digest_rows.append({
                "path": str(path.relative_to(ROOT)),
                "digest_key": digest_key,
                "exists": False,
                "passed": False,
            })
            continue
        passed, digest = verify_json(path, digest_key)
        digest_rows.append({
            "path": str(path.relative_to(ROOT)),
            "digest_key": digest_key,
            "exists": True,
            "digest": digest,
            "passed": passed,
        })

    prereg = protocol.read_json(root / "protocol" / "preregistration.json")
    static = protocol.read_json(root / "protocol" / "audit.json")
    behavior = protocol.read_json(analysis / "behavior_authorization.json")
    projection = protocol.read_json(analysis / "projection_audit.json")
    numeric = protocol.read_json(analysis / "numeric_audit.json")
    physical = protocol.read_json(analysis / "physical_map.json")
    automatic = protocol.read_json(analysis / "automatic_next.json")
    final = protocol.read_json(analysis / "final_summary.json")
    posthoc = protocol.read_json(analysis / "posthoc_template_robustness.json")

    model_rows = {}
    for model_name in protocol.MODELS:
        model_audit = protocol.read_json(
            root / "protocol" / f"audit.{model_name}.json"
        )
        case_rows = protocol.read_jsonl(
            root / "protocol" / f"cases.{model_name}.jsonl"
        )
        computed_case_digest = protocol.digest(case_rows)
        summary = protocol.read_json(
            root / "atlas" / model_name / "summary.json"
        )
        npz_path = root / "atlas" / model_name / "signed_fields.npz"
        npz_sha = sha256_file(npz_path)
        model_rows[model_name] = {
            "npz_exists": npz_path.exists(),
            "npz_sha256": npz_sha,
            "npz_matches_final": npz_sha == final["models"][model_name]["npz_sha256"],
            "computed_case_digest": computed_case_digest,
            "case_digest_matches": (
                computed_case_digest == model_audit["case_digest"]
                and computed_case_digest == prereg["model_case_digests"][model_name]
            ),
            "protocol_digest_matches": summary["protocol_digest"] == prereg["protocol_digest"],
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "hidden_finite_fraction": summary["hidden_finite_fraction_lower_bound"],
            "pre_query_global_max_abs": summary["pre_query_global_max_abs"],
            "projection_passed": projection["by_model"][model_name]["passed"],
            "numeric_passed": numeric["by_model"][model_name]["passed"],
            "passed": (
                npz_path.exists()
                and npz_sha == final["models"][model_name]["npz_sha256"]
                and computed_case_digest == model_audit["case_digest"]
                and computed_case_digest == prereg["model_case_digests"][model_name]
                and summary["protocol_digest"] == prereg["protocol_digest"]
                and projection["by_model"][model_name]["passed"]
                and numeric["by_model"][model_name]["passed"]
                and summary["pre_query_global_max_abs"] == 0.0
            ),
        }

    checks = {
        "all_required_digests_valid": all(row["passed"] for row in digest_rows),
        "static_protocol_passed": bool(static["all_checks_passed"]),
        "case_count_per_model": int(prereg["case_count_per_model"]) == 24576,
        "unit_count_per_model": int(prereg["unit_count_per_model"]) == 1536,
        "model_order_frozen": tuple(prereg["sequential_model_order"]) == tuple(protocol.MODELS),
        "fp16_no_quantization": prereg["precision"] == "fp16" and prereg["quantization"] == "none",
        "behavior_hidden_authorized": bool(behavior["hidden_scan_authorized"]),
        "two_formal_models": set(final["formal_models"]) == {"qwen3", "glm4"},
        "all_model_artifacts_passed": all(row["passed"] for row in model_rows.values()),
        "frozen_predictions_exact": (
            final["passed_predictions"] == ["P1", "P2", "P3", "P4"]
            and final["failed_predictions"] == ["P5", "P6", "P7", "P8", "P9"]
        ),
        "posthoc_does_not_upgrade": posthoc["evidence_upgrade_allowed"] is False,
        "causal_not_authorized": (
            behavior["causal_authorized"] is False
            and automatic["local_causal_authorized"] is False
            and physical["causal_selection_authorized"] is False
        ),
        "automatic_extension_not_authorized": (
            automatic["automatic_replication_authorized"] is False
            and automatic["automatic_hidden_extension_authorized"] is False
        ),
    }
    result = {
        "schema_version": "phase1092_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "digest_rows": digest_rows,
        "models": model_rows,
        "all_checks_passed": all(checks.values()),
    }
    result["result_audit_digest"] = protocol.digest(result)
    protocol.write_json(analysis / "result_audit.json", result)
    print({
        "phase": protocol.PHASE,
        "all_checks_passed": result["all_checks_passed"],
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "result_audit_digest": result["result_audit_digest"],
    })


if __name__ == "__main__":
    main()
