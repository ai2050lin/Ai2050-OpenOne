#!/usr/bin/env python3
"""Independent audit for Phase1136."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/glm5/result/phase1136_fp16_sdpa_behavior_qualification"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    protocol = read_json(RESULT / "protocol/preregistration.json")
    protocol_audit = read_json(RESULT / "protocol/audit.json")
    final = read_json(RESULT / "analysis/final.json")
    checks = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("protocol_passed", protocol_audit["all_checks_passed"] is True, protocol_audit["passed_count"])
    check("glm4_only", protocol["authorized_model"] == "glm4" and protocol["excluded_model"] == "deepseek7b", protocol["authorized_model"])
    check("sdpa_math_requested", protocol["attention_backend"] == "sdpa" and protocol["sdpa_kernel_policy"] == "math_only", [protocol["attention_backend"], protocol["sdpa_kernel_policy"]])
    if final.get("engineering_failure"):
        check("failure_before_scores", final["failure_stage"] == "before_load_complete" and final["smoke_score_count"] == 0, [final["failure_stage"], final["smoke_score_count"]])
        check("two_native_crashes_recorded", len(final["attempts"]) == 2 and all(row["scores_observed"] == 0 for row in final["attempts"]), final["attempts"])
        check("no_smoke_artifact", not (RESULT / "smoke/scores.jsonl").exists(), str(RESULT / "smoke/scores.jsonl"))
        check("repair_hard_stop", final["new_causal_phase_authorized"] is False and final["authorized_models_for_new_causal_phase"] == ["qwen3"], final["next_action"])
    else:
        smoke_scores = read_jsonl(RESULT / "smoke/scores.jsonl")
        smoke_decisions = read_jsonl(RESULT / "smoke/decisions.jsonl")
        check("fp16_only", final["precision"]["has_fp16_parameters"] and not final["precision"]["has_bf16_parameters"], final["precision"]["parameter_dtypes"])
        check("no_quantization", not final["precision"]["has_quantized_modules"], final["precision"]["suspicious_quantized_module_classes"])
        check("sdpa_effective", final["placement"]["effective_attention_backend"] == "sdpa", final["placement"]["effective_attention_backend"])
        check("sdpa_math_effective", final["placement"]["sdpa_kernel_policy"] == "math_only", final["placement"]["sdpa_kernel_policy"])
        check("smoke_count_256", len(smoke_scores) == 256 and len(smoke_decisions) == 128, [len(smoke_scores), len(smoke_decisions)])
        recomputed_smoke_finite = sum(row["finite"] for row in smoke_decisions) / len(smoke_decisions)
        reported_smoke_finite = sum(final["smoke_metrics"][split]["finite_fraction"] * final["smoke_metrics"][split]["count"] for split in ("discovery", "confirmation")) / sum(final["smoke_metrics"][split]["count"] for split in ("discovery", "confirmation"))
        check("smoke_finite_recomputes", abs(recomputed_smoke_finite - reported_smoke_finite) < 1e-12, recomputed_smoke_finite)
        smoke_pass = all(final["smoke_metrics"][split]["passed"] for split in ("discovery", "confirmation"))
        check("smoke_gate_recomputes", smoke_pass == final["smoke_passed"], smoke_pass)
        if final["smoke_passed"]:
            full_scores = read_jsonl(RESULT / "full/scores.jsonl")
            full_decisions = read_jsonl(RESULT / "full/decisions.jsonl")
            check("full_counts", len(full_scores) == 3928 and len(full_decisions) == 1964, [len(full_scores), len(full_decisions)])
            full_pass = all(final["full_metrics"][split]["passed"] for split in ("discovery", "confirmation"))
            check("full_gate_recomputes", full_pass == final["full_passed"], full_pass)
        else:
            check("full_hard_stop", final["full_score_count"] == 0 and not (RESULT / "full/scores.jsonl").exists(), final["full_score_count"])
    check("new_causal_gate_recomputes", final["new_causal_phase_authorized"] == (len(final["authorized_models_for_new_causal_phase"]) >= 2), final["authorized_models_for_new_causal_phase"])
    check("nonhuman_scope", final["human_annotation_eligible"] is False, final["claim_boundary"])
    output = {
        "schema_version": "phase1136_independent_audit.v1",
        "phase": 1136,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_count": sum(row["passed"] for row in checks),
        "check_count": len(checks),
        "all_checks_passed": all(row["passed"] for row in checks),
        "failed_checks": [row for row in checks if not row["passed"]],
    }
    path = RESULT / "audit/independent_result_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": 1136, "audit": f"{output['passed_count']}/{output['check_count']}", "all_checks_passed": output["all_checks_passed"], "failed": [row["name"] for row in output["failed_checks"]]}), flush=True)
    return 0 if output["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
