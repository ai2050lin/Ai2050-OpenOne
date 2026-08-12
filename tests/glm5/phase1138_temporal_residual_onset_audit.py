#!/usr/bin/env python3
"""Independent result audit for Phase1138."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1138_temporal_residual_onset as phase  # noqa: E402


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> None:
    prereg = phase.read_json(phase.OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = phase.read_json(phase.OUT_ROOT / "protocol/audit.json")
    selection = phase.read_json(phase.OUT_ROOT / "analysis/discovery_selection.json")
    confirmation = phase.read_json(phase.OUT_ROOT / "analysis/causal_confirmation.json")
    checks: list[dict[str, Any]] = []

    prereg_core = {key: value for key, value in prereg.items() if key != "protocol_digest"}
    add(checks, "protocol_digest", phase.digest(prereg_core) == prereg["protocol_digest"], prereg["protocol_digest"])
    protocol_audit_core = {key: value for key, value in protocol_audit.items() if key != "audit_digest"}
    add(
        checks,
        "protocol_audit_digest",
        phase.digest(protocol_audit_core) == protocol_audit["audit_digest"],
        protocol_audit["audit_digest"],
    )
    add(checks, "protocol_checks", protocol_audit["all_checks_passed"], protocol_audit["checks"])
    add(
        checks,
        "source1135_unchanged",
        phase.sha256_file(phase.SOURCE1135 / "audit/independent_result_audit.json")
        == prereg["source"]["phase1135_audit_file_sha256"],
        prereg["source"]["phase1135_audit_file_sha256"],
    )
    add(
        checks,
        "qwen4_decisions_unchanged",
        phase.sha256_file(phase.SOURCE1135 / "analysis/behavior_decisions.qwen3.jsonl")
        == prereg["source"]["qwen4_decisions_sha256"],
        prereg["source"]["qwen4_decisions_sha256"],
    )
    add(
        checks,
        "qwen14_decisions_unchanged",
        phase.sha256_file(phase.SOURCE1137 / "analysis/behavior_decisions.qwen3_14b.jsonl")
        == prereg["source"]["qwen14_decisions_sha256"],
        prereg["source"]["qwen14_decisions_sha256"],
    )
    add(
        checks,
        "cohort_counts",
        all(len(prereg["behavior_conditioning"]["cohorts"][split]) == 13 for split in ("discovery", "confirmation")),
        {split: len(prereg["behavior_conditioning"]["cohorts"][split]) for split in ("discovery", "confirmation")},
    )
    add(
        checks,
        "cohort_disjoint",
        set(prereg["behavior_conditioning"]["cohorts"]["discovery"]).isdisjoint(
            prereg["behavior_conditioning"]["cohorts"]["confirmation"]
        ),
        True,
    )

    recomputed_model_metrics: dict[str, Any] = {}
    for model_name in phase.MODELS:
        summary = phase.read_json(phase.OUT_ROOT / f"causal/discovery/{model_name}/summary.json")
        rows = phase.read_jsonl(phase.OUT_ROOT / f"causal/discovery/{model_name}/patch_records.jsonl")
        add(checks, f"{model_name}_discovery_record_digest", phase.digest(rows) == summary["record_digest"], summary["record_digest"])
        summary_core = {key: value for key, value in summary.items() if key != "summary_digest"}
        add(checks, f"{model_name}_discovery_summary_digest", phase.digest(summary_core) == summary["summary_digest"], summary["summary_digest"])
        add(checks, f"{model_name}_discovery_record_count", len(rows) == 13 * 9 * len(phase.REQUESTED_FRACTIONS), len(rows))
        add(
            checks,
            f"{model_name}_fp16",
            summary["precision"]["has_fp16_parameters"]
            and not summary["precision"]["has_bf16_parameters"]
            and not summary["precision"]["has_quantized_modules"],
            summary["precision"],
        )
        metrics = [phase.depth_metrics(rows, value) for value in phase.REQUESTED_FRACTIONS]
        recomputed_model_metrics[model_name] = {
            "depth_metrics": metrics,
            "passing_requested_fractions": [row["requested_fraction"] for row in metrics if row["passed"]],
        }
        add(
            checks,
            f"{model_name}_discovery_metrics",
            phase.digest(metrics) == phase.digest(selection["models"][model_name]["depth_metrics"]),
            phase.digest(metrics),
        )

    shared = sorted(
        set(recomputed_model_metrics["qwen3_4b"]["passing_requested_fractions"])
        & set(recomputed_model_metrics["qwen3_14b"]["passing_requested_fractions"])
    )
    mechanistic = [value for value in shared if value <= phase.MAXIMUM_MECHANISTIC_FRACTION + 1e-12]
    runs = phase.contiguous_runs(mechanistic)
    qualifying = [run for run in runs if len(run) >= phase.MINIMUM_CONTIGUOUS_SHARED_DEPTHS]
    selected = qualifying[0][0] if qualifying else None
    add(checks, "shared_fractions_recomputed", shared == selection["shared_passing_requested_fractions"], shared)
    add(checks, "runs_recomputed", runs == selection["contiguous_runs"] and qualifying == selection["qualifying_runs"], runs)
    add(
        checks,
        "selection_recomputed",
        selected == selection["selected_requested_fraction"]
        and (selected is not None) == selection["confirmation_authorized"],
        selected,
    )
    selection_core = {key: value for key, value in selection.items() if key != "selection_digest"}
    add(checks, "selection_digest", phase.digest(selection_core) == selection["selection_digest"], selection["selection_digest"])

    if selection["confirmation_authorized"]:
        confirmation_models = {}
        for model_name in phase.MODELS:
            summary = phase.read_json(phase.OUT_ROOT / f"causal/confirmation/{model_name}/summary.json")
            rows = phase.read_jsonl(phase.OUT_ROOT / f"causal/confirmation/{model_name}/patch_records.jsonl")
            add(checks, f"{model_name}_confirmation_record_digest", phase.digest(rows) == summary["record_digest"], summary["record_digest"])
            add(checks, f"{model_name}_confirmation_record_count", len(rows) == 13 * 9, len(rows))
            metrics = phase.depth_metrics(rows, float(selection["selected_requested_fraction"]))
            confirmation_models[model_name] = bool(metrics["passed"])
            add(
                checks,
                f"{model_name}_confirmation_metrics",
                phase.digest(metrics) == phase.digest(confirmation["models"][model_name]["metrics"]),
                metrics["passed"],
            )
        confirmed = all(confirmation_models.values())
        add(
            checks,
            "confirmation_decision_recomputed",
            confirmed == confirmation["same_family_residual_event_confirmed"]
            and confirmed == confirmation["auto_continue"],
            confirmation_models,
        )
    else:
        add(
            checks,
            "confirmation_correctly_unrun",
            confirmation["confirmation_run"] is False
            and confirmation["same_family_residual_event_confirmed"] is False,
            confirmation,
        )

    add(
        checks,
        "claim_scope",
        confirmation.get("cross_architecture_conservation", False) is False
        and confirmation.get("human_annotation_eligible", False) is False,
        confirmation.get("claim_boundary"),
    )
    confirmation_core = {key: value for key, value in confirmation.items() if key != "confirmation_digest"}
    add(
        checks,
        "confirmation_digest",
        phase.digest(confirmation_core) == confirmation["confirmation_digest"],
        confirmation["confirmation_digest"],
    )

    audit_core = {
        "schema_version": "phase1138_temporal_residual_result_audit.v1",
        "phase": phase.PHASE,
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(row["passed"]) for row in checks),
        "all_checks_passed": all(bool(row["passed"]) for row in checks),
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "confirmation_digest": confirmation["confirmation_digest"],
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
