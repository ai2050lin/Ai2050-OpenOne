#!/usr/bin/env python3
"""Independently audit Phase1121 protocol, raw outputs, and frozen final decision."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1121_wordnet_adjective_double_orthogonal_finalize as finalize
import phase1121_wordnet_adjective_double_orthogonal_protocol as protocol


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    stored_final = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    recomputed = finalize.compute_final()
    checks: dict[str, bool] = {
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "protocol_digest_matches": stored_final["protocol_digest"] == prereg["protocol_digest"],
        "final_recomputation_exact": stored_final == recomputed,
        "final_digest_recomputed": protocol.digest({key: value for key, value in stored_final.items() if key != "final_digest"}) == stored_final["final_digest"],
        "model_set_complete": set(stored_final["models"]) == set(protocol.MODELS),
        "authorization_logic": stored_final["hidden_trajectory_authorized"] == (
            stored_final["pythia_qualified"]
            and len(stored_final["qualified_reference_models"]) >= prereg["thresholds"]["minimum_qualified_reference_models"]
        ),
    }
    for model in protocol.MODELS:
        rows = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model}.jsonl"))
        detail = list(protocol.read_jsonl(protocol.OUT_ROOT / "behavior" / model / "candidate_detail.jsonl"))
        summary = protocol.read_json(protocol.OUT_ROOT / "behavior" / model / "summary.json")
        metrics = stored_final["models"][model]
        prefix = f"{model}_"
        checks[prefix + "case_digest"] = protocol.digest(rows) == prereg["case_digests"][model]
        checks[prefix + "case_count"] = len(rows) == prereg["case_count_per_model"]
        checks[prefix + "detail_count"] = len(detail) == len(rows) == metrics["case_count"]
        checks[prefix + "detail_digest"] = protocol.digest(detail) == summary["detail_digest"] == metrics["detail_digest"]
        checks[prefix + "summary_digest"] = protocol.digest({key: value for key, value in summary.items() if key != "summary_digest"}) == summary["summary_digest"]
        checks[prefix + "fp16"] = summary["precision"]["has_fp16_parameters"] and not summary["precision"]["has_bf16_parameters"]
        checks[prefix + "no_quantization"] = not summary["precision"]["has_quantized_modules"]
        checks[prefix + "interaction_count"] = metrics["interaction_count"] == protocol.SELECTED_ITEM_COUNT * len(protocol.TEMPLATES) * len(protocol.SURFACES)
        checks[prefix + "surface_pair_count"] = metrics["cross_surface_pair_count"] == protocol.SELECTED_ITEM_COUNT * len(protocol.TEMPLATES)
        checks[prefix + "qualification_is_gate_conjunction"] = metrics["qualified"] == all(metrics["gates"].values())

    audit_core = {
        "schema_version": "phase1121_adjective_double_orthogonal_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": stored_final["final_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = protocol.digest(audit_core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1121 result audit failed")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
