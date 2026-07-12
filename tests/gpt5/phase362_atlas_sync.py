#!/usr/bin/env python3
"""Aggregate and publish Phase362 evidence without sealed tensors or labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
TARGETS = (
    ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
    ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    cases = read_json(OUT / "phase362_case_summary.json")
    anchors = read_json(OUT / "phase362_anchor_replay_summary.json")
    frozen = read_json(OUT / "phase362_frozen_candidate_summary.json")
    posthoc = read_json(OUT / "phase362_posthoc_survivor_summary.json")
    completions = [read_json(OUT / "models" / model / "complete.json") for model in MODELS]
    summary = {
        "schema_version": "39.4.0", "phase_id": "Phase362", "created_at": now(),
        "denominator": {
            **cases["denominator"],
            "generation_trace_case_count": sum(row["case_count"] for row in completions),
            "generation_time_count": 3,
            "generation_ledger_row_count": sum(row["ledger_row_count"] for row in completions),
            "generation_trace_sealed_byte_count": sum(row["sealed_byte_count"] for row in completions),
            "anchor_layer_file_count": anchors["denominator"]["layer_file_count"],
            "anchor_sealed_byte_count": anchors["denominator"]["total_byte_count"],
        },
        "quality": {
            "phase361_candidate_hash_unchanged": cases["frozen_phase361_candidates"]["sha256"],
            "phase361_case_overlap_count": cases["quality"]["phase361_case_overlap_count"],
            "all_generation_model_completions_valid": all(row["valid"] for row in completions),
            "all_generation_gates_pass": all(row["all_gates_pass"] for row in completions),
            "all_anchor_replay_gates_pass": anchors["quality"]["all_offline_gates_pass"],
            "anchor_max_errors": anchors["quality"]["max_errors"],
        },
        "results": {
            "phase361_frozen_candidate_count": frozen["denominator"]["frozen_candidate_count"],
            "independent_strong_baseline_survivor_count": frozen["results"]["b3_independently_best_all_models_count"],
            "selective_next_layer_association_count": posthoc["results"]["selective_next_layer_association_count"],
            "temporal_predictive_survivor_count": 0,
            "competition_predictive_survivor_count": 0,
            "operation_specific_mechanism_count": 0,
        },
        "identifiability_audit": frozen["identifiability_audit"],
        "claim_boundary": {
            "attention_source_edge_format_replayable": True,
            "three_generation_times_recorded": True,
            "phase361_next_layer_candidates_independently_tested": True,
            "phase361_candidates_can_test_temporal_prediction": False,
            "physical_confirmation_opened": False,
            "causal_intervention_executed": False,
            "single_unit_causal_count": 0,
            "language_encoding_closed": False,
        },
        "next_decision": "freeze_new_temporal_hypotheses_before_using_physical_confirmation",
    }
    write_json(OUT / "phase362_global_summary.json", summary)
    public = summary
    updated_at = now()
    for target in TARGETS:
        write_json(target / "phase362_global_summary.json", public)
        write_json(target / "phase362_anchor_replay_summary.json", anchors)
        write_json(target / "phase362_frozen_candidate_summary.json", frozen)
        write_json(target / "phase362_posthoc_survivor_summary.json", posthoc)
        manifest_path = target / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["updated_at"] = updated_at
        manifest["phase362"] = {
            "status": "independent_next_layer_survivors_temporal_formula_missing",
            "case_count": summary["denominator"]["case_count"],
            "independent_calibration_case_count": summary["denominator"]["independent_calibration_case_count"],
            "physical_confirmation_case_count": summary["denominator"]["physical_confirmation_case_count"],
            "generation_ledger_row_count": summary["denominator"]["generation_ledger_row_count"],
            "anchor_count": summary["denominator"]["anchor_count"],
            "next_layer_survivor_count": summary["results"]["independent_strong_baseline_survivor_count"],
            "temporal_predictive_survivor_count": 0,
            "operation_specific_mechanism_count": 0,
            "physical_confirmation_opened": False,
            "raw_tensors_frontend_exported": False,
            "files": [
                "phase362_global_summary.json", "phase362_anchor_replay_summary.json",
                "phase362_frozen_candidate_summary.json", "phase362_posthoc_survivor_summary.json",
            ],
        }
        write_json(manifest_path, manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
