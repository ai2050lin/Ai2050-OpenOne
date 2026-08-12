#!/usr/bin/env python3
"""Independently audit all frozen Phase1113 artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1113_wordnet_semantic_quadrant_protocol as protocol


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def check_model(model_name: str, preregistration: dict[str, Any]) -> dict[str, Any]:
    cases = list(protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    ))
    details = list(protocol.read_jsonl(
        protocol.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl"
    ))
    summary = protocol.read_json(
        protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    case_by_index = {int(row["case_index"]): row for row in cases}
    detail_by_index = {int(row["case_index"]): row for row in details}
    checks = {
        "case_digest": protocol.digest(cases) == preregistration["case_digests"][model_name],
        "summary_digest": (
            protocol.digest({key: value for key, value in summary.items() if key != "summary_digest"})
            == summary["summary_digest"]
        ),
        "counts": len(cases) == len(details) == summary["candidate_count"] == 864,
        "case_indices_unique": len(case_by_index) == len(cases),
        "detail_indices_unique": len(detail_by_index) == len(details),
        "case_detail_indices_equal": set(case_by_index) == set(detail_by_index),
        "metadata_matches": all(
            detail_by_index[index]["record_id"] == case["record_id"]
            and detail_by_index[index]["concept_id"] == case["concept_id"]
            and detail_by_index[index]["split"] == case["split"]
            and detail_by_index[index]["quadrant"] == case["quadrant"]
            and detail_by_index[index]["expected_class"] == case["expected_class"]
            for index, case in case_by_index.items()
        ),
        "finite_fraction_recomputed": abs(
            sum(bool(row["finite"]) for row in details) / len(details)
            - summary["candidate_finite_fraction"]
        ) < 1e-12,
        "accuracy_recomputed": abs(
            sum(bool(row["hit"]) for row in details if row["finite"])
            / max(sum(bool(row["finite"]) for row in details), 1)
            - summary["candidate_accuracy"]
        ) < 1e-12,
        "precision_fp16": summary["precision"]["has_fp16_parameters"],
        "precision_not_bf16": not summary["precision"]["has_bf16_parameters"],
        "precision_not_quantized": not summary["precision"]["has_quantized_modules"],
    }
    return {
        "model": model_name,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_sha256": sha256_file(
            protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        ),
        "detail_sha256": sha256_file(
            protocol.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl"
        ),
        "summary_sha256": sha256_file(
            protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
        ),
    }


def main() -> None:
    preregistration = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    final = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json"
    )
    authorization = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    model_audits = {
        model_name: check_model(model_name, preregistration)
        for model_name in protocol.MODELS
    }
    result_files = sorted(
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file() and "audit" not in path.parts
    )
    forbidden_paths = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in result_files
        if any(
            marker in path.name.casefold()
            for marker in ("hidden_state", "activation", "attention_atlas", "causal_patch")
        )
    ]
    global_checks = {
        "wordnet_archive_sha256": sha256_file(protocol.WORDNET_ARCHIVE) == protocol.WORDNET_SHA256,
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "protocol_digest_matches_audit": (
            preregistration["protocol_digest"] == protocol_audit["protocol_digest"]
        ),
        "protocol_digest_recomputed": (
            protocol.digest({
                key: value for key, value in preregistration.items()
                if key != "protocol_digest"
            }) == preregistration["protocol_digest"]
        ),
        "final_digest_recomputed": (
            protocol.digest({key: value for key, value in final.items() if key != "final_digest"})
            == final["final_digest"]
        ),
        "authorization_matches_final": (
            authorization["qualified_models"] == final["qualified_models"]
            and authorization["cross_model_behavior_qualified"]
            == final["cross_model_behavior_qualified"]
        ),
        "hidden_state_not_authorized": not authorization["hidden_state_authorized"],
        "no_hidden_or_causal_artifacts": not forbidden_paths,
        "all_models_pass_artifact_audit": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
    }
    manifest = [
        {
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in result_files
    ]
    audit = {
        "schema_version": "phase1113_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "final_digest": final["final_digest"],
        "global_checks": global_checks,
        "model_audits": model_audits,
        "forbidden_paths": forbidden_paths,
        "artifact_count": len(manifest),
        "artifact_manifest": manifest,
        "all_checks_passed": all(global_checks.values()),
    }
    audit["audit_digest"] = protocol.digest(audit)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    print(json.dumps({
        "phase": protocol.PHASE,
        "all_checks_passed": audit["all_checks_passed"],
        "artifact_count": audit["artifact_count"],
        "forbidden_paths": forbidden_paths,
        "model_audits": {
            model: row["all_checks_passed"] for model, row in model_audits.items()
        },
        "audit_digest": audit["audit_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
