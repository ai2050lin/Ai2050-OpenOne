#!/usr/bin/env python3
"""Independent recomputation audit for Phase1137."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1135_temporal_binding_intervention as source  # noqa: E402
import phase1137_qwen14b_temporal_binding_endpoint as phase  # noqa: E402


def check(rows: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    rows.append({"name": name, "passed": bool(passed), "detail": detail})


def close(a: float | None, b: float | None, tolerance: float = 1e-10) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    prereg = phase.read_json(phase.OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = phase.read_json(phase.OUT_ROOT / "protocol/audit.json")
    logical_cases = phase.read_jsonl(phase.OUT_ROOT / "protocol/logical_cases.jsonl")
    carrier = phase.read_jsonl(phase.OUT_ROOT / "protocol/cases.qwen3_14b.jsonl")
    summary = phase.read_json(phase.OUT_ROOT / "behavior/qwen3_14b/summary.json")
    scores = phase.read_jsonl(phase.OUT_ROOT / "behavior/qwen3_14b/scores.jsonl")
    stored_decisions = phase.read_jsonl(
        phase.OUT_ROOT / "analysis/behavior_decisions.qwen3_14b.jsonl"
    )
    final = phase.read_json(phase.OUT_ROOT / "analysis/final_summary.json")
    source_prereg = phase.read_json(phase.SOURCE_ROOT / "protocol/preregistration.json")
    source_authorization = phase.read_json(
        phase.SOURCE_ROOT / "analysis/behavior_authorization.json"
    )

    checks: list[dict[str, Any]] = []
    check(checks, "protocol_audit_passed", protocol_audit["all_checks_passed"], protocol_audit)
    prereg_core = {key: value for key, value in prereg.items() if key != "protocol_digest"}
    check(
        checks,
        "protocol_digest_recomputed",
        phase.digest(prereg_core) == prereg["protocol_digest"],
        prereg["protocol_digest"],
    )
    audit_core = {key: value for key, value in protocol_audit.items() if key != "audit_digest"}
    check(
        checks,
        "protocol_audit_digest_recomputed",
        phase.digest(audit_core) == protocol_audit["audit_digest"],
        protocol_audit["audit_digest"],
    )
    check(
        checks,
        "source_protocol_unchanged",
        source_prereg["protocol_digest"] == prereg["source"]["phase1135_protocol_digest"],
        source_prereg["protocol_digest"],
    )
    check(
        checks,
        "source_authorization_unchanged",
        source_authorization["authorization_digest"]
        == prereg["source"]["phase1135_authorization_digest"],
        source_authorization["authorization_digest"],
    )
    check(
        checks,
        "source_audit_file_unchanged",
        phase.sha256_file(phase.SOURCE_ROOT / "audit/independent_result_audit.json")
        == prereg["source"]["phase1135_audit_file_sha256"],
        prereg["source"]["phase1135_audit_file_sha256"],
    )
    check(
        checks,
        "source_qwen4_scores_file_unchanged",
        phase.sha256_file(phase.SOURCE_ROOT / "behavior/qwen3/scores.jsonl")
        == prereg["source"]["phase1135_qwen4_scores_file_sha256"],
        prereg["source"]["phase1135_qwen4_scores_file_sha256"],
    )
    check(
        checks,
        "source_gate_still_closed",
        source_authorization["hidden_scan_authorized"] is False,
        source_authorization["authorized_models"],
    )
    check(checks, "logical_case_count", len(logical_cases) == 2946, len(logical_cases))
    check(
        checks,
        "logical_case_digest",
        phase.digest(logical_cases) == prereg["source"]["logical_case_digest"],
        prereg["source"]["logical_case_digest"],
    )
    check(checks, "carrier_count", len(carrier) == 5892, len(carrier))
    check(
        checks,
        "carrier_digest",
        phase.digest(carrier) == prereg["carrier"]["candidate_case_digest"],
        prereg["carrier"]["candidate_case_digest"],
    )
    check(
        checks,
        "tokenizer_identity",
        phase.sha256_file(ROOT / "models/hf/qwen3-4b/tokenizer.json")
        == phase.sha256_file(phase.MODEL_ROOT / "tokenizer.json")
        == phase.EXPECTED_TOKENIZER_SHA256,
        phase.EXPECTED_TOKENIZER_SHA256,
    )
    check(checks, "score_count", len(scores) == 5892, len(scores))
    check(
        checks,
        "score_digest",
        phase.digest(scores) == summary["score_digest"],
        summary["score_digest"],
    )
    summary_core = {key: value for key, value in summary.items() if key != "summary_digest"}
    check(
        checks,
        "summary_digest",
        phase.digest(summary_core) == summary["summary_digest"],
        summary["summary_digest"],
    )
    check(
        checks,
        "finite_count_recomputed",
        sum(bool(row["finite"]) for row in scores) == summary["finite_count"],
        summary["finite_count"],
    )
    check(
        checks,
        "finite_fraction_recomputed",
        close(
            sum(bool(row["finite"]) for row in scores) / len(scores),
            summary["finite_fraction"],
        ),
        summary["finite_fraction"],
    )
    check(
        checks,
        "fp16_no_quantization",
        summary["precision"]["has_fp16_parameters"]
        and not summary["precision"]["has_bf16_parameters"]
        and not summary["precision"]["has_quantized_modules"],
        summary["precision"],
    )
    check(
        checks,
        "parameter_count",
        summary["parameter_count"] == phase.EXPECTED_PARAMETER_COUNT,
        summary["parameter_count"],
    )
    check(
        checks,
        "manifest_identity",
        summary["model_manifest_digest"] == phase.EXPECTED_MANIFEST_DIGEST,
        summary["model_manifest_digest"],
    )
    check(
        checks,
        "disk_offload_preserved",
        summary["placement"] == "cuda_disk_offload"
        and sum(value == "disk" for value in summary["device_map"].values()) >= 22,
        summary["placement"],
    )

    recomputed_decisions = phase.decisions_from_scores(scores)
    check(
        checks,
        "decision_digest",
        phase.digest(recomputed_decisions) == phase.digest(stored_decisions),
        len(stored_decisions),
    )
    recomputed_metrics = {
        split: source.behavior_metrics(recomputed_decisions, split)
        for split in ("discovery", "confirmation", "natural_use")
    }
    check(
        checks,
        "metric_digest",
        phase.digest(recomputed_metrics) == phase.digest(final["metrics"]),
        phase.digest(recomputed_metrics),
    )
    primary = all(
        recomputed_metrics[split]["passed"]
        for split in source.BEHAVIOR_THRESHOLDS["required_splits"]
    )
    interaction_split = {
        split: bool(
            recomputed_metrics[split]["posthoc_binding_interaction"]["median"] is not None
            and recomputed_metrics[split]["posthoc_binding_interaction"]["median"]
            > phase.SECONDARY_INTERACTION_THRESHOLDS["median_min"]
            and recomputed_metrics[split]["posthoc_binding_interaction"]["positive_fraction"]
            >= phase.SECONDARY_INTERACTION_THRESHOLDS["positive_fraction_min"]
        )
        for split in phase.SECONDARY_INTERACTION_THRESHOLDS["required_splits"]
    }
    interaction = all(interaction_split.values())
    check(checks, "primary_gate_recomputed", primary == final["primary_behavior_passed"], primary)
    check(
        checks,
        "interaction_gate_recomputed",
        interaction_split == final["interaction_split_pass"]
        and interaction == final["prospective_interaction_passed"],
        interaction_split,
    )
    check(
        checks,
        "same_family_decision_recomputed",
        bool(primary and interaction) == final["same_family_behavior_replication"],
        final["same_family_behavior_replication"],
    )
    check(
        checks,
        "phase1135_not_reopened",
        final["phase1135_gate_reopened"] is False
        and final["cross_architecture_replication"] is False,
        {
            "phase1135_gate_reopened": final["phase1135_gate_reopened"],
            "cross_architecture_replication": final["cross_architecture_replication"],
        },
    )
    check(checks, "no_hidden_scan", final["hidden_scanned"] is False, final["hidden_scanned"])
    final_core = {key: value for key, value in final.items() if key != "final_digest"}
    check(
        checks,
        "final_digest",
        phase.digest(final_core) == final["final_digest"],
        final["final_digest"],
    )
    check(
        checks,
        "auto_continue_consistent",
        final["auto_continue"] == bool(primary and interaction),
        final["auto_continue"],
    )

    audit_core = {
        "schema_version": "phase1137_qwen14b_temporal_result_audit.v1",
        "phase": phase.PHASE,
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(row["passed"]) for row in checks),
        "all_checks_passed": all(bool(row["passed"]) for row in checks),
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
    }
    audit = dict(audit_core)
    audit["audit_digest"] = phase.digest(audit_core)
    phase.write_json(phase.OUT_ROOT / "audit/independent_result_audit.json", audit)
    print(json.dumps({
        "phase": phase.PHASE,
        "checks": f"{audit['passed_count']}/{audit['check_count']}",
        "all_checks_passed": audit["all_checks_passed"],
        "audit_digest": audit["audit_digest"],
    }, ensure_ascii=False), flush=True)
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
