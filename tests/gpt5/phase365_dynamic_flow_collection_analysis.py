#!/usr/bin/env python3
"""Aggregate the three-model Phase365-B engineering collection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    manifests = [read_json(OUT / "models" / model / "manifest.json") for model in MODELS]
    summary = {
        "schema_version": "42.6.0", "phase_id": "Phase365-B", "created_at": now(),
        "denominator": {
            "model_count": 3, "case_count": sum(row["case_count"] for row in manifests),
            "generation_time_count": 3, "layer_file_count": sum(
                item["kind"] == "layer" for row in manifests for item in row["files"]
            ),
            "time_meta_file_count": sum(
                item["kind"] == "time_meta" for row in manifests for item in row["files"]
            ),
            "total_byte_count": sum(row["total_byte_count"] for row in manifests),
        },
        "results": {
            "valid_model_count": sum(row["valid"] for row in manifests),
            "all_case_gate_model_count": sum(row["all_case_gates_pass"] for row in manifests),
            "max_errors": {
                key: max(row["gate_maxima"][key] for row in manifests)
                for key in manifests[0]["gate_maxima"]
            },
            "single_neuron_writes_offline_recoverable": all(row["valid"] for row in manifests),
            "dynamic_bundle_extraction_executed": False,
            "language_path_candidate_count": 0,
        },
        "quality": {
            "models_executed_sequentially": True,
            "semantic_labels_used_by_collection": False,
            "target_specific_competition_used_by_collection": False,
            "physical_confirmation_opened": False,
            "causal_intervention_executed": False,
            "scope_is_four_registered_roles_not_all_token_positions": True,
        },
        "authorization": {
            "offline_dynamic_bundle_extraction_authorized": all(row["valid"] for row in manifests),
            "physical_confirmation_authorized": False,
            "causal_intervention_authorized": False,
        },
        "next_decision": "derive_label_blind_typed_events_and_freeze_path_extraction" if all(row["valid"] for row in manifests) else "repair_collection_before_path_extraction",
    }
    write_json(OUT / "phase365_engineering_collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
