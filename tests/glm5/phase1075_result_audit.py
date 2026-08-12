#!/usr/bin/env python3
"""Audit Phase1075 protocol, sequential execution, and result integrity."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1075_relation_polarity_protocol as protocol


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    manifest = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "run_manifest.json"
    )
    decision = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_decision.json"
    )
    internal_prereg = protocol.read_json(
        protocol.OUT_ROOT
        / "analysis"
        / "internal_preregistration.json"
    )
    automatic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    checks = {
        "protocol_digest_valid": protocol.digest({
            key: value
            for key, value in prereg.items()
            if key != "protocol_digest"
        }) == prereg["protocol_digest"],
        "protocol_audit_passed": audit["all_checks_passed"],
        "all_model_protocol_audits_passed": all(
            value["all_checks_passed"]
            for value in audit["model_audits"].values()
        ),
        "behavior_decision_digest_valid": protocol.digest({
            key: value
            for key, value in decision.items()
            if key != "decision_digest"
        }) == decision["decision_digest"],
        "internal_preregistration_digest_valid": protocol.digest({
            key: value
            for key, value in internal_prereg.items()
            if key != "internal_preregistration_digest"
        }) == internal_prereg["internal_preregistration_digest"],
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
        "automatic_decision_present": (
            automatic["phase"] == protocol.PHASE
        ),
    }

    behavior_summaries = {}
    for model in protocol.MODELS:
        cases = protocol.read_jsonl(
            protocol.OUT_ROOT
            / "protocol"
            / f"cases.{model}.jsonl"
        )
        summary = protocol.read_json(
            protocol.OUT_ROOT
            / "behavior"
            / model
            / "summary.json"
        )
        behavior_summaries[model] = summary
        checks[f"{model}_case_count"] = (
            len(cases) == prereg["case_count_per_model"]
        )
        checks[f"{model}_behavior_count"] = (
            summary["case_count"] == len(cases)
        )
        precision = summary["precision"]
        checks[f"{model}_fp16_no_quant"] = bool(
            precision["has_fp16_parameters"]
            and not precision["has_bf16_parameters"]
            and not precision["has_quantized_modules"]
        )

    stage_intervals = []
    for stage in manifest["model_stages"]:
        if stage.get("skipped_existing"):
            continue
        stage_intervals.append((
            parse_time(stage["started_at_utc"]),
            parse_time(stage["completed_at_utc"]),
            stage["model"],
            stage["stage"],
        ))
    stage_intervals.sort()
    checks["recorded_model_stages_do_not_overlap"] = all(
        stage_intervals[index][1]
        <= stage_intervals[index + 1][0]
        for index in range(len(stage_intervals) - 1)
    )

    if decision["should_run_internal_mapping"]:
        expected_models = set(decision["selected_models"])
        present_models = set()
        for model in decision["selected_models"]:
            summary_path = (
                protocol.OUT_ROOT
                / "internal"
                / model
                / "summary.json"
            )
            if not summary_path.exists():
                continue
            present_models.add(model)
            summary = protocol.read_json(summary_path)
            checks[f"{model}_internal_fp16_no_quant"] = bool(
                summary["precision"]["has_fp16_parameters"]
                and not summary["precision"][
                    "has_bf16_parameters"
                ]
                and not summary["precision"][
                    "has_quantized_modules"
                ]
            )
            expected_relations = {
                relation
                for relation in decision["selected_relations"]
                if model in decision[
                    "authorized_models_by_relation"
                ][relation]
            }
            checks[f"{model}_authorized_internal_relations"] = (
                set(summary["authorized_relations"])
                == expected_relations
            )
            metrics = protocol.read_jsonl(
                protocol.OUT_ROOT
                / "internal"
                / model
                / "unit_metrics.jsonl"
            )
            checks[f"{model}_internal_metrics_nonempty"] = bool(metrics)
            checks[f"{model}_routing_aggregate_present"] = (
                protocol.OUT_ROOT
                / "internal"
                / model
                / "routing_aggregates.npz"
            ).exists()
        checks["all_selected_internal_models_present"] = (
            present_models == expected_models
        )
    else:
        checks["no_unauthorized_internal_directory"] = not (
            protocol.OUT_ROOT / "internal"
        ).exists()

    result_files = [
        path
        for path in protocol.OUT_ROOT.rglob("*")
        if path.is_file()
    ]
    checks["no_raw_tensor_dumps"] = not any(
        path.suffix.lower() in {".pt", ".pth", ".bin", ".npy"}
        for path in result_files
    )
    payload = {
        "schema_version": "phase1075_integrity_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_integrity_checks_passed": all(checks.values()),
        "result_file_count": len(result_files),
        "result_bytes": sum(path.stat().st_size for path in result_files),
        "behavior_relations": {
            model: summary["confirmed_relations"]
            for model, summary in behavior_summaries.items()
        },
        "selected_relations": decision["selected_relations"],
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
