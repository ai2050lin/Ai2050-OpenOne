#!/usr/bin/env python3
"""Audit Phase399 multi-position event conservation before discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase399_dynamic_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    freeze = read_json(OUT / "phase399_behavior_freeze_summary.json")
    expected_groups = len(freeze["eligible_surfaces"])
    summaries = []
    audits = []
    for model in MODELS:
        root = OUT / "dynamic_trace/instrument/private/models" / model
        summary = read_json(root / "complete.json")
        rows = read_jsonl(root / "group_audit_rows.jsonl")
        if summary["group_count"] != expected_groups or len(rows) != expected_groups:
            raise RuntimeError(f"Phase399 instrument denominator mismatch for {model}")
        summaries.append(summary)
        audits.extend(rows)
    expected_cases = expected_groups * 16 * len(MODELS)
    valid = (
        all(summary["valid"] for summary in summaries)
        and all(row["quality_gate_pass"] for row in audits)
        and sum(summary["case_count"] for summary in summaries) == expected_cases
    )
    result = {
        "schema_version": "73.5.0",
        "phase_id": "Phase399-InstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "eligible_surface_count": expected_groups,
            "model_count": len(MODELS),
            "case_count": sum(summary["case_count"] for summary in summaries),
            "group_model_cell_count": len(audits),
        },
        "results": {
            "quality_group_model_cell_count": sum(
                row["quality_gate_pass"] for row in audits
            ),
            "first_answer_replay_match_count": sum(
                row["first_answer_replay_match_count"] for row in audits
            ),
            "target_completion_replay_match_count": sum(
                row["target_completion_replay_match_count"] for row in audits
            ),
            "post_target_replay_match_count": sum(
                row["post_target_replay_match_count"] for row in audits
            ),
            "max_block_relative_error": max(
                row["max_block_relative_error"] for row in audits
            ),
            "max_attention_replay_relative_error": max(
                row["max_attention_replay_relative_error"] for row in audits
            ),
            "max_probability_sum_error": max(
                row["max_probability_sum_error"] for row in audits
            ),
        },
        "valid": valid,
        "authorization": {
            "run_discovery_trace": valid,
            "run_calibration_trace": False,
            "open_physical_holdout": False,
            "run_joint_causal_intervention": False,
            "head_channel_or_neuron_scan": False,
        },
        "claim_boundary": {
            "instrument_conservation_is_a_language_mechanism": False,
            "role_partition_is_a_causal_binding_graph": False,
        },
    }
    write_json(OUT / "phase399_instrument_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
