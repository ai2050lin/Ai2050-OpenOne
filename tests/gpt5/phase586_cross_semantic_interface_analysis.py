#!/usr/bin/env python3
"""Record the Phase586 cross-judge interface stop without semantic consensus."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase586_cross_semantic_audit as audit
import phase586_cross_semantic_audit_protocol as protocol


V1_DIR = protocol.OUT_DIR / "v1_interface_calibration"
OUTPUT = protocol.OUT_DIR / "phase586_cross_semantic_interface_decision.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_name = path.name.replace("_summary.json", "_rows.jsonl.gz")
    rows_path = path.with_name(rows_name)
    if payload["rows_sha256"] != protocol.sha256_file(rows_path):
        raise RuntimeError(f"Phase586 interface row drift: {rows_path}")
    return payload


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "judge_model": summary["judge_model"],
        "parse_rate": summary["parse_rate"],
        "repeat_exact_rate": summary["repeat_exact_rate"],
        "judge_quality_gate_passes": summary["judge_quality_gate_passes"],
        "judgment_counts": summary["judgment_counts"],
        "rows_sha256": summary["rows_sha256"],
        "sealed_split_read": summary["sealed_split_read"],
    }


def main() -> None:
    v1 = {
        model: compact(
            read_summary(V1_DIR / f"phase586_{model}_cross_semantic_audit_summary.json")
        )
        for model in protocol.MODELS
    }
    v2 = {
        model: compact(read_summary(audit.paths(model)["summary"]))
        for model in ("qwen3", "glm4")
    }
    payload = {
        "schema_version": "phase586_cross_semantic_interface_decision.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "status": "stopped_common_judge_interface_not_portable",
        "protocol_v1_results": v1,
        "protocol_v2_partial_results": v2,
        "protocol_v2_deepseek7b_status": (
            "not_run_after_glm4_failed_the_common_interface_quality_gate"
        ),
        "common_interface_passed_all_three_judges": False,
        "semantic_consensus_formed": False,
        "sealed_behavior_authorized_model_relations": {},
        "internal_trace_authorized_model_relations": {},
        "causal_intervention_authorized": False,
        "sealed_validation_authorized": False,
        "sealed_split_read": False,
        "evidence_classification": {
            "judge_interface_calibration": True,
            "natural_behavior_result": False,
            "internal_structure_result": False,
            "mechanism_result": False,
        },
        "stop_reason": (
            "No single frozen label-generation interface met the parse and repeat gate "
            "for all three judge models; model-specific prompt patching was prohibited."
        ),
    }
    protocol.write_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "semantic_consensus_formed": False,
                "sealed_split_read": False,
                "internal_trace_authorized_model_relations": {},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
