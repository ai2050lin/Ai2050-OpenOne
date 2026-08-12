#!/usr/bin/env python3
"""Audit Phase1103 artifacts, digests, scope, and sequential precision."""

from __future__ import annotations

import json
from pathlib import Path

import phase1103_natural_relation_route_protocol as protocol


def main() -> None:
    checks: dict[str, bool] = {}
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    protocol_audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    causal = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "causal_authorization.json"
    )
    final = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "final_summary.json"
    )
    diagnostic = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json"
    )
    prereg_without_digest = dict(prereg)
    protocol_digest = prereg_without_digest.pop("protocol_digest")
    checks["protocol_digest"] = (
        protocol.digest(prereg_without_digest) == protocol_digest
    )
    protocol_audit_without_digest = dict(protocol_audit)
    audit_digest = protocol_audit_without_digest.pop("audit_digest")
    checks["protocol_audit_digest"] = (
        protocol.digest(protocol_audit_without_digest) == audit_digest
    )
    checks["protocol_audit_passed"] = protocol_audit[
        "all_checks_passed"
    ]
    checks["protocol_revision2"] = prereg["protocol_revision"] == 2
    checks["case_count_frozen"] = prereg["case_count_per_model"] == 17280
    checks["precision_frozen"] = (
        prereg["precision"] == "fp16"
        and prereg["quantization"] == "none"
    )
    checks["split_names_disjoint"] = not (
        set(prereg["selected_names"][:12])
        & set(prereg["selected_names"][12:])
    )
    for model in protocol.MODELS:
        cases = protocol.read_jsonl(
            protocol.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"
        )
        model_audit = protocol.read_json(
            protocol.OUT_ROOT / "protocol" / f"audit.{model}.json"
        )
        summary = protocol.read_json(
            protocol.OUT_ROOT / "behavior" / model / "summary.json"
        )
        details = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "candidate_detail.jsonl"
        )
        generations = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "generation_detail.jsonl"
        )
        checks[f"{model}_case_digest"] = (
            protocol.digest(cases) == prereg["model_case_digests"][model]
            == model_audit["case_digest"]
        )
        checks[f"{model}_case_count"] = (
            len(cases) == prereg["case_count_per_model"]
            == summary["candidate_count"] == len(details)
        )
        checks[f"{model}_generation_count"] = (
            len(generations) == summary["generation_count"]
        )
        checks[f"{model}_behavior_digest"] = (
            protocol.digest({
                key: value for key, value in summary.items()
                if key != "summary_digest"
            }) == summary["summary_digest"]
        )
        checks[f"{model}_fp16_no_quant"] = (
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"]
        )
    behavior_without_digest = dict(behavior)
    behavior_digest = behavior_without_digest.pop("authorization_digest")
    checks["behavior_authorization_digest"] = (
        protocol.digest(behavior_without_digest) == behavior_digest
    )
    checks["shared_pair_rule"] = all(
        len(behavior["pair_models"][pair])
        >= protocol.THRESHOLDS["minimum_models_per_shared_pair"]
        for pair in behavior["shared_behavior_authorized_pairs"]
    )
    checks["causal_pair_scope"] = (
        set(behavior["causally_eligible_pairs"])
        <= set(behavior["shared_behavior_authorized_pairs"])
    )
    if behavior["causal_scan_authorized"]:
        for model in protocol.MODELS:
            summary = protocol.read_json(
                protocol.OUT_ROOT / "causal" / model / "summary.json"
            )
            checks[f"{model}_causal_digest"] = (
                protocol.digest({
                    key: value for key, value in summary.items()
                    if key != "summary_digest"
                }) == summary["summary_digest"]
            )
            checks[f"{model}_causal_scope"] = (
                set(summary.get("eligible_pairs", []))
                <= set(behavior["causally_eligible_pairs"])
            )
            if not summary.get("skipped"):
                detail = protocol.read_jsonl(
                    protocol.OUT_ROOT / "causal" / model
                    / "patch_detail.jsonl"
                )
                checks[f"{model}_causal_count"] = (
                    len(detail) == summary["patch_record_count"]
                )
                checks[f"{model}_causal_fp16_no_quant"] = (
                    summary["precision"]["has_fp16_parameters"]
                    and not summary["precision"]["has_bf16_parameters"]
                    and not summary["precision"]["has_quantized_modules"]
                )
    else:
        checks["no_causal_artifacts_after_behavior_stop"] = not (
            protocol.OUT_ROOT / "causal"
        ).exists()
    causal_without_digest = dict(causal)
    causal_digest = causal_without_digest.pop("causal_authorization_digest")
    checks["causal_authorization_digest"] = (
        protocol.digest(causal_without_digest) == causal_digest
    )
    checks["qualification_only_depth_selection"] = all(
        row["selection_used_only_qualification"]
        for row in causal.get("model_cells", {}).values()
    )
    checks["component_authorization_rule"] = (
        causal["component_scan_authorized"]
        == bool(causal["shared_confirmed_cells"])
    )
    diagnostic_without_digest = dict(diagnostic)
    diagnostic_digest = diagnostic_without_digest.pop("diagnostic_digest")
    checks["failure_diagnostic_digest"] = (
        protocol.digest(diagnostic_without_digest) == diagnostic_digest
    )
    checks["failure_diagnostic_did_not_reauthorize"] = (
        diagnostic["frozen_authorization_unchanged"]
        and diagnostic["authorization_digest"]
        == behavior["authorization_digest"]
    )
    final_without_digest = dict(final)
    final_digest = final_without_digest.pop("final_summary_digest")
    checks["final_summary_digest"] = (
        protocol.digest(final_without_digest) == final_digest
    )
    checks["final_references_failure_diagnostic"] = (
        final["failure_diagnostic_digest"] == diagnostic_digest
    )
    checks["automatic_next_rule"] = (
        final["automatic_next"] == causal["component_scan_authorized"]
    )
    result = {
        "schema_version": "phase1103_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "protocol_digest": prereg["protocol_digest"],
        "behavior_authorization_digest": behavior[
            "authorization_digest"
        ],
        "causal_authorization_digest": causal[
            "causal_authorization_digest"
        ],
        "final_summary_digest": final["final_summary_digest"],
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "audit" / "result_audit.json", result
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "checks": len(checks),
        "passed": sum(checks.values()),
        "all_checks_passed": result["all_checks_passed"],
        "failed_checks": [key for key, value in checks.items() if not value],
        "audit_digest": result["audit_digest"],
    }, ensure_ascii=False), flush=True)
    if not result["all_checks_passed"]:
        raise RuntimeError("Phase1103 result audit failed")


if __name__ == "__main__":
    main()
