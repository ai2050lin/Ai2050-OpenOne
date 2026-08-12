#!/usr/bin/env python3
"""Audit frozen Phase1115 artifacts and the no-hidden-state boundary."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1115_wordnet_context_modulation_confirmation_protocol as protocol


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def audit_model(model_name: str, preregistration: dict[str, Any]) -> dict[str, Any]:
    cases = list(
        protocol.base.read_jsonl(
            protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        )
    )
    details = list(
        protocol.base.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl"
        )
    )
    summary = protocol.base.read_json(
        protocol.OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    case_by_index = {int(row["case_index"]): row for row in cases}
    detail_by_index = {int(row["case_index"]): row for row in details}
    finite = [row for row in details if row["finite"]]
    checks = {
        "case_digest": protocol.base.digest(cases)
        == preregistration["case_digests"][model_name],
        "summary_digest": protocol.base.digest(
            {key: value for key, value in summary.items() if key != "summary_digest"}
        )
        == summary["summary_digest"],
        "counts": len(cases) == len(details) == summary["candidate_count"] == 252,
        "indices_equal": set(case_by_index) == set(detail_by_index),
        "metadata_matches": all(
            detail_by_index[index]["record_id"] == case["record_id"]
            and detail_by_index[index]["pair_id"] == case["pair_id"]
            and detail_by_index[index]["sense"] == case["sense"]
            for index, case in case_by_index.items()
        ),
        "z_recomputed": all(
            math.isclose(
                row["scores"]["sense0"] - row["scores"]["sense1"],
                row["sense0_minus_sense1"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in finite
        ),
        "finite_fraction_recomputed": abs(
            len(finite) / len(details) - summary["candidate_finite_fraction"]
        )
        < 1e-12,
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
    preregistration = protocol.base.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.base.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    final = protocol.base.read_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json"
    )
    authorization = protocol.base.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    model_audits = {
        model: audit_model(model, preregistration) for model in protocol.MODELS
    }
    files = sorted(
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file() and "audit" not in path.parts
    )
    forbidden = [
        str(path.relative_to(protocol.OUT_ROOT))
        for path in files
        if any(
            marker in path.name.casefold()
            for marker in ("hidden_state", "activation", "attention", "causal_patch")
        )
    ]
    global_checks = {
        "wordnet_archive_sha256": sha256_file(protocol.WORDNET_ARCHIVE)
        == protocol.WORDNET_SHA256,
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "protocol_digest_recomputed": protocol.base.digest(
            {
                key: value
                for key, value in preregistration.items()
                if key != "protocol_digest"
            }
        )
        == preregistration["protocol_digest"],
        "final_digest_recomputed": protocol.base.digest(
            {key: value for key, value in final.items() if key != "final_digest"}
        )
        == final["final_digest"],
        "authorization_digest_recomputed": protocol.base.digest(
            {
                key: value
                for key, value in authorization.items()
                if key != "authorization_digest"
            }
        )
        == authorization["authorization_digest"],
        "hidden_state_not_authorized": not authorization["hidden_state_authorized"],
        "no_hidden_or_causal_artifacts": not forbidden,
        "all_models_pass": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
    }
    manifest = [
        {
            "path": str(path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    audit = {
        "schema_version": "phase1115_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": preregistration["protocol_digest"],
        "final_digest": final["final_digest"],
        "global_checks": global_checks,
        "model_audits": model_audits,
        "forbidden_paths": forbidden,
        "artifact_count": len(manifest),
        "artifact_manifest": manifest,
        "all_checks_passed": all(global_checks.values()),
    }
    audit["audit_digest"] = protocol.base.digest(audit)
    protocol.base.write_json(
        protocol.OUT_ROOT / "audit" / "result_audit.json", audit
    )
    print(
        json.dumps(
            {
                "phase": protocol.PHASE,
                "all_checks_passed": audit["all_checks_passed"],
                "artifact_count": audit["artifact_count"],
                "forbidden_paths": forbidden,
                "model_audits": {
                    model: row["all_checks_passed"]
                    for model, row in model_audits.items()
                },
                "audit_digest": audit["audit_digest"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
