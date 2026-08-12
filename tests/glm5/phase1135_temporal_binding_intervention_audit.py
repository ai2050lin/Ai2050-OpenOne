#!/usr/bin/env python3
"""Independent result audit for Phase1135."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/glm5/result/phase1134_external_api_temporal_annotation"
    / "analysis/external_machine_consensus_package.jsonl"
)
RESULT = ROOT / "tests/glm5/result/phase1135_temporal_binding_intervention"
MODELS = ("qwen3", "glm4", "deepseek7b")
STATES = ("original_pre", "original_post", "swapped_pre", "swapped_post", "prior_pre", "prior_post")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    protocol = read_json(RESULT / "protocol/preregistration.json")
    protocol_audit = read_json(RESULT / "protocol/audit.json")
    cases = read_jsonl(RESULT / "protocol/logical_cases.jsonl")
    behavior = read_json(RESULT / "analysis/behavior_authorization.json")
    causal = read_json(RESULT / "analysis/causal_confirmation.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("protocol_audit_passed", protocol_audit.get("all_checks_passed") is True, protocol_audit.get("passed_count"))
    check("source_hash_frozen", protocol.get("source_sha256") == sha256(SOURCE), protocol.get("source_sha256"))
    check("source_count_491", protocol.get("source_count") == 491, protocol.get("source_count"))
    check("logical_case_count_2946", len(cases) == 2946, len(cases))
    check("logical_case_ids_unique", len({row["case_id"] for row in cases}) == len(cases), len(cases))
    check("six_states_exact", {row["state"] for row in cases} == set(STATES), sorted({row["state"] for row in cases}))
    check("machine_scope_frozen", all(row.get("machine_validation_only") is True and row.get("human_annotation_eligible") is False for row in cases), len(cases))
    check("behavior_scope_nonhuman", behavior.get("human_annotation_eligible") is False, behavior.get("evidence_scope"))

    recomputed_authorized = []
    for model in MODELS:
        summary_path = RESULT / "behavior" / model / "summary.json"
        score_path = RESULT / "behavior" / model / "scores.jsonl"
        decision_path = RESULT / "analysis" / f"behavior_decisions.{model}.jsonl"
        check(f"{model}_behavior_summary_exists", summary_path.exists(), str(summary_path))
        check(f"{model}_behavior_scores_exists", score_path.exists(), str(score_path))
        check(f"{model}_behavior_decisions_exists", decision_path.exists(), str(decision_path))
        if not (summary_path.exists() and score_path.exists() and decision_path.exists()):
            continue
        summary = read_json(summary_path)
        scores = read_jsonl(score_path)
        decisions = read_jsonl(decision_path)
        precision = summary.get("precision", {})
        check(f"{model}_fp16_present", precision.get("has_fp16_parameters") is True, precision.get("parameter_dtypes"))
        check(f"{model}_no_bf16", precision.get("has_bf16_parameters") is False, precision.get("parameter_dtypes"))
        check(f"{model}_no_quantization", precision.get("has_quantized_modules") is False, precision.get("suspicious_quantized_module_classes"))
        check(f"{model}_score_count_5892", len(scores) == 5892, len(scores))
        check(f"{model}_decision_count_2946", len(decisions) == 2946, len(decisions))
        check(f"{model}_score_keys_complete", len({(row["case_id"], row["candidate_key"]) for row in scores}) == 5892, len(scores))
        check(f"{model}_finite_matches", abs(summary["finite_fraction"] - sum(row["finite"] for row in scores) / len(scores)) < 1e-12, summary["finite_fraction"])
        model_result = behavior["models"][model]
        independent_auth = all(model_result["splits"][split]["passed"] for split in ("discovery", "confirmation"))
        check(f"{model}_authorization_recomputes", independent_auth == model_result["authorized_for_hidden_scan"], independent_auth)
        if independent_auth:
            recomputed_authorized.append(model)
    check("authorized_model_list_exact", recomputed_authorized == behavior["authorized_models"], recomputed_authorized)
    check("two_model_behavior_gate_recomputes", behavior["hidden_scan_authorized"] == (len(recomputed_authorized) >= 2), behavior["hidden_scan_authorized"])

    if behavior["hidden_scan_authorized"]:
        discovery_path = RESULT / "analysis/causal_discovery_selection.json"
        check("discovery_selection_exists", discovery_path.exists(), str(discovery_path))
        if discovery_path.exists():
            discovery = read_json(discovery_path)
            for model in discovery.get("models_authorized_for_confirmation", []):
                summary_path = RESULT / "causal/confirmation" / model / "summary.json"
                records_path = RESULT / "causal/confirmation" / model / "patch_records.jsonl"
                check(f"{model}_confirmation_summary_exists", summary_path.exists(), str(summary_path))
                check(f"{model}_confirmation_records_exist", records_path.exists(), str(records_path))
                if records_path.exists():
                    rows = read_jsonl(records_path)
                    check(f"{model}_confirmation_machine_only", all(row.get("machine_validation_only") is True for row in rows), len(rows))
                    self_rows = [row for row in rows if row["patch_kind"] == "self_patch_audit"]
                    check(f"{model}_self_patch_present", bool(self_rows), len(self_rows))
                    check(f"{model}_causal_finite_recomputes", abs(read_json(summary_path)["finite_fraction"] - sum(row["finite"] for row in rows) / len(rows)) < 1e-12, len(rows))
    else:
        check("hidden_hard_stop_recorded", causal.get("component_search_authorized") is False and not causal.get("confirmed_models"), causal.get("next_action"))

    confirmed = [name for name in MODELS if causal.get("models", {}).get(name, {}).get("confirmed")]
    check("confirmed_models_exact", confirmed == causal.get("confirmed_models"), confirmed)
    check("component_gate_recomputes", causal.get("component_search_authorized") == (len(confirmed) >= 2), causal.get("component_search_authorized"))
    check("causal_scope_nonhuman", causal.get("human_annotation_eligible") is False, causal.get("evidence_scope"))
    check("claim_boundary_present", bool(causal.get("claim_boundary")), causal.get("claim_boundary"))

    result = {
        "schema_version": "phase1135_independent_result_audit.v1",
        "phase": 1135,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_count": sum(row["passed"] for row in checks),
        "check_count": len(checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "failed_checks": [row for row in checks if not row["passed"]],
    }
    output = RESULT / "audit/independent_result_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "phase": 1135,
        "audit": f"{result['passed_count']}/{result['check_count']}",
        "all_checks_passed": result["all_checks_passed"],
        "failed_checks": [row["name"] for row in result["failed_checks"]],
    }, ensure_ascii=False), flush=True)
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
