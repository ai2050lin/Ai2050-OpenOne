#!/usr/bin/env python3
"""Audit Phase1076 protocol, model isolation, and result integrity."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1076_polarity_head_causal_protocol as protocol


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def valid_digest(payload: dict, key: str) -> bool:
    return protocol.digest({
        name: value
        for name, value in payload.items()
        if name != key
    }) == payload[key]


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    manifest = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "run_manifest.json"
    )
    decision = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_decision.json"
    )
    final = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    checks = {
        "protocol_digest_valid": valid_digest(
            prereg, "protocol_digest"
        ),
        "protocol_audit_passed": protocol_audit[
            "all_checks_passed"
        ],
        "behavior_decision_digest_valid": valid_digest(
            decision, "decision_digest"
        ),
        "final_summary_digest_valid": valid_digest(
            final, "summary_digest"
        ),
        "automatic_decision_digest_valid": valid_digest(
            automatic, "decision_digest"
        ),
        "manifest_completed": bool(manifest.get("completed")),
        "sequential_order_declared": (
            manifest["sequential_model_order"]
            == list(protocol.MODELS)
        ),
        "no_concurrent_model_processes_declared": (
            not manifest["concurrent_model_processes"]
        ),
        "precision_fp16": manifest["precision"] == "fp16",
        "quantization_none": manifest["quantization"] == "none",
        "automatic_stop_respected": (
            not automatic["should_continue_automatically"]
        ),
    }
    for model in protocol.MODELS:
        cases = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "protocol"
            / f"cases.{model}.jsonl"
        )
        audit = protocol.read_json(
            protocol.OUT_ROOT
            / "protocol"
            / f"audit.{model}.json"
        )
        behavior = protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "summary.json"
        )
        checks[f"{model}_protocol_case_count"] = (
            len(cases) == prereg["case_count_per_model"]
        )
        checks[f"{model}_protocol_audit"] = audit[
            "all_checks_passed"
        ]
        checks[f"{model}_behavior_count"] = (
            behavior["case_count"] == len(cases)
        )
        checks[f"{model}_behavior_fp16_no_quant"] = bool(
            behavior["precision"]["has_fp16_parameters"]
            and not behavior["precision"]["has_bf16_parameters"]
            and not behavior["precision"]["has_quantized_modules"]
        )
        if decision["should_run_causal_validation"]:
            causal = protocol.read_json(
                protocol.OUT_ROOT
                / "causal"
                / model
                / "summary.json"
            )
            records = protocol.read_jsonl(
                protocol.OUT_ROOT
                / "causal"
                / model
                / "causal_records.jsonl"
            )
            checks[f"{model}_causal_count"] = (
                causal["case_count"] == len(cases)
                and len(records) == len(cases)
            )
            checks[f"{model}_causal_fp16_no_quant"] = bool(
                causal["precision"]["has_fp16_parameters"]
                and not causal["precision"]["has_bf16_parameters"]
                and not causal["precision"][
                    "has_quantized_modules"
                ]
            )
            checks[f"{model}_causal_protocol_digest"] = (
                causal["protocol_digest"]
                == prereg["protocol_digest"]
            )
            checks[f"{model}_no_raw_tensor_dump"] = (
                not causal["raw_tensor_dumps"]
            )
            checks[f"{model}_all_interventions_present"] = all(
                set(row["margin_drops"])
                == set(protocol.INTERVENTIONS)
                for row in records
            )
    if not decision["should_run_causal_validation"]:
        checks["no_unauthorized_causal_directory"] = not (
            protocol.OUT_ROOT / "causal"
        ).exists()

    intervals = []
    for stage in manifest["model_stages"]:
        if stage.get("skipped_existing"):
            continue
        intervals.append((
            parse_time(stage["started_at_utc"]),
            parse_time(stage["completed_at_utc"]),
            stage["model"],
            stage["stage"],
        ))
    intervals.sort()
    checks["recorded_model_stages_do_not_overlap"] = all(
        intervals[index][1] <= intervals[index + 1][0]
        for index in range(len(intervals) - 1)
    )
    result_files = [
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
    ]
    checks["no_raw_tensor_files"] = not any(
        path.suffix.lower()
        in {".pt", ".pth", ".bin", ".npy", ".safetensors"}
        for path in result_files
    )
    payload = {
        "schema_version": "phase1076_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_integrity_checks_passed": all(checks.values()),
        "result_file_count": len(result_files),
        "result_bytes": sum(
            path.stat().st_size for path in result_files
        ),
        "behavior_authorized": decision[
            "should_run_causal_validation"
        ],
        "claim_status": final["claim_status"],
        "automatic_next": automatic,
    }
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "integrity_audit.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["all_integrity_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
