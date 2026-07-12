#!/usr/bin/env python3
"""Audit the final 96-case-per-model manifests after any instrumentation repair."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    rows = [read_json(COLLECTION / "models" / model / "manifest.json") for model in MODELS]
    if any(row["case_count"] != 96 for row in rows):
        raise RuntimeError("Final model denominator is not 96 each")
    summary = {
        "schema_version": "43.3.0", "phase_id": "Phase366", "created_at": now(),
        "denominator": {
            "model_count": 3, "case_count": 288, "case_count_per_model": 96,
            "generation_time_count": 3,
            "layer_file_count": sum(item["kind"] == "layer" for row in rows for item in row["files"]),
            "time_meta_file_count": sum(item["kind"] == "time_meta" for row in rows for item in row["files"]),
            "total_byte_count": sum(row["total_byte_count"] for row in rows),
        },
        "results": {
            "valid_model_count": sum(row["valid"] for row in rows),
            "max_errors": {key: max(row["gate_maxima"][key] for row in rows) for key in rows[0]["gate_maxima"]},
            "original_deepseek_product_failure_repaired_by_full_sequence_hooks": rows[2]["gate_maxima"]["mlp_product"] <= 0.01,
        },
        "quality": {
            "gates_lowered_after_failure": False,
            "physical_confirmation_opened": False,
            "causal_intervention_executed": False,
            "semantic_labels_used_by_collector": False,
        },
        "authorization": {
            "full_dynamic_bundle_extraction_authorized": all(row["valid"] for row in rows),
            "scientific_motif_scoring_authorized_before_bundle_extraction": False,
            "physical_confirmation_authorized": False,
        },
        "next_decision": "rerun_dynamic_bundle_extraction_on_all_288_cases" if all(row["valid"] for row in rows) else "stop_and_repair_instrumentation",
    }
    write_json(COLLECTION / "phase366_full_collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
