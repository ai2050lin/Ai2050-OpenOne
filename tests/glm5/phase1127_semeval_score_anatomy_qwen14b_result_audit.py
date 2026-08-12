#!/usr/bin/env python3
"""Independent result audit for Phase1127."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1126_semeval_lexsub_natural_cloze_protocol as source_protocol
import phase1127_semeval_score_anatomy_qwen14b_protocol as protocol


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def main() -> None:
    prereg = strict_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = strict_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    final = strict_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    units = protocol.read_jsonl(protocol.OUT_ROOT / "analysis" / "interaction_units.jsonl")
    qwen14_summary = strict_json(protocol.OUT_ROOT / "behavior" / "qwen3_14b" / "summary.json")
    qwen14_scores = protocol.read_jsonl(protocol.OUT_ROOT / "behavior" / "qwen3_14b" / "scores.jsonl")
    source_final = strict_json(source_protocol.OUT_ROOT / "analysis" / "final_summary.json")

    final_core = {key: value for key, value in final.items() if key != "final_digest"}
    qwen14_core = {key: value for key, value in qwen14_summary.items() if key != "summary_digest"}
    additivity_errors = []
    for row in units:
        for route in ("active", "matched"):
            total = row[f"total_{route}_z"]
            candidate = row[f"candidate_{route}_z"]
            suffix = row[f"suffix_{route}_z"]
            if total is not None and candidate is not None and suffix is not None:
                additivity_errors.append(abs(float(total) - float(candidate) - float(suffix)))
    independent_max_error = max(additivity_errors) if additivity_errors else None

    source_detail_checks = {}
    for model in protocol.SOURCE_MODELS:
        summary = strict_json(source_protocol.OUT_ROOT / "behavior" / model / "summary.json")
        details = source_protocol.read_jsonl(source_protocol.OUT_ROOT / "behavior" / model / "scores.jsonl")
        source_detail_checks[model] = (
            source_protocol.digest(details) == summary["detail_digest"]
            and summary["detail_digest"] == prereg["source"]["source_score_digests"][model]
        )

    qwen14_partitions = final["model_results"]["qwen3_14b"]["partitions"]
    recomputed_qwen14_pass = all(
        qwen14_partitions[partition]["components"]["total"]["passed"]
        for partition in protocol.PARTITIONS
    )
    recomputed_same_family = (
        final["model_results"]["qwen3_4b"]["total_passed_both_partitions"]
        and recomputed_qwen14_pass
    )
    hidden_names = ("hidden", "hidden_states", "components", "causal", "patching", "sae")
    hidden_artifacts = [name for name in hidden_names if (protocol.OUT_ROOT / name).exists()]

    checks = {
        "protocol_audit_passed": protocol_audit["passed"] is True,
        "protocol_digest_valid": protocol.digest({
            key: value for key, value in prereg.items() if key != "protocol_digest"
        }) == prereg["protocol_digest"],
        "protocol_links_match": final["protocol_digest"] == prereg["protocol_digest"],
        "source_final_unchanged": source_final["final_digest"] == prereg["source"]["phase1126_final_digest"],
        "source_details_unchanged": all(source_detail_checks.values()),
        "qwen14_detail_digest": protocol.digest(qwen14_scores) == qwen14_summary["detail_digest"],
        "qwen14_summary_digest": protocol.digest(qwen14_core) == qwen14_summary["summary_digest"],
        "qwen14_case_count": len(qwen14_scores) == qwen14_summary["case_count"] == 320,
        "qwen14_fp16": (
            qwen14_summary["precision"]["has_fp16_parameters"] is True
            and qwen14_summary["precision"]["has_bf16_parameters"] is False
        ),
        "qwen14_not_quantized": qwen14_summary["precision"]["has_quantized_modules"] is False,
        "qwen14_parameter_count": qwen14_summary["parameter_count"] == prereg["model"]["expected_parameter_count"],
        "qwen14_disk_offload": qwen14_summary["placement"] == "cuda_disk_offload",
        "unit_count": len(units) == 4 * 2 * 10 * 2,
        "unit_digest": protocol.digest(units) == final["unit_digest"],
        "score_additivity": (
            independent_max_error is not None
            and independent_max_error <= prereg["score_identity"]["additivity_tolerance"]
            and math.isclose(independent_max_error, final["score_additivity_max_abs_error"], abs_tol=1e-15)
        ),
        "final_digest": protocol.digest(final_core) == final["final_digest"],
        "qwen14_prediction_consistent": final["predictions"]["P5_qwen14_total_behavior"] == recomputed_qwen14_pass,
        "same_family_prediction_consistent": final["predictions"]["P6_same_family_replication"] == recomputed_same_family,
        "phase1126_gate_not_reopened": final["predictions"]["P7_phase1126_cross_model_gate_reopened"] is False,
        "hidden_unauthorized": final["predictions"]["P8_hidden_authorized"] is False,
        "auto_continue_false": final["auto_continue"]["value"] is False,
        "no_hidden_artifacts": not hidden_artifacts,
        "holdout_absent_from_qwen14": all(row["partition"] != "hidden_holdout" for row in qwen14_scores),
    }
    audit_core = {
        "schema_version": "phase1127_semeval_score_anatomy_qwen14b_result_audit.v1",
        "phase": protocol.PHASE,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
        "independent_additivity_max_abs_error": independent_max_error,
        "hidden_artifacts": hidden_artifacts,
    }
    audit = dict(audit_core)
    audit["audit_digest"] = protocol.digest(audit_core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase1127 result audit failed: {failed}")


if __name__ == "__main__":
    main()
