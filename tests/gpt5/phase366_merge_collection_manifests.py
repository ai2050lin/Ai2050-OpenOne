#!/usr/bin/env python3
"""Merge Phase365 pilot and Phase366 remaining manifests into the frozen 288-case ledger."""

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
    merged_rows = []
    for model in MODELS:
        root = COLLECTION / "models" / model
        pilot = read_json(root / "manifest_phase365_pilot.json")
        remaining = read_json(root / "manifest.json")
        if pilot["case_count"] != 32 or remaining["case_count"] != 64:
            raise RuntimeError(f"Unexpected merge denominators for {model}")
        case_ids = [row["blind_case_id"] for row in [*pilot["case_rows"], *remaining["case_rows"]]]
        if len(case_ids) != len(set(case_ids)):
            raise RuntimeError(f"Duplicate case ids while merging {model}")
        merged = {
            "schema_version": "43.2.0", "phase_id": "Phase366", "created_at": now(),
            "model": model, "case_count": 96, "generation_time_count": 3,
            "layer_count": pilot["layer_count"],
            "file_count": pilot["file_count"] + remaining["file_count"],
            "total_byte_count": pilot["total_byte_count"] + remaining["total_byte_count"],
            "all_case_gates_pass": pilot["all_case_gates_pass"] and remaining["all_case_gates_pass"],
            "gate_maxima": {
                key: max(pilot["gate_maxima"][key], remaining["gate_maxima"][key])
                for key in pilot["gate_maxima"]
            },
            "case_rows": [*pilot["case_rows"], *remaining["case_rows"]],
            "files": [*pilot["files"], *remaining["files"]],
            "source_manifests": ["manifest_phase365_pilot.json", "manifest_phase366_remaining.json"],
            "valid": pilot["valid"] and remaining["valid"],
        }
        write_json(root / "manifest_phase366_remaining.json", remaining)
        write_json(root / "manifest.json", merged)
        merged_rows.append(merged)
    summary = {
        "schema_version": "43.2.0", "phase_id": "Phase366", "created_at": now(),
        "denominator": {
            "model_count": 3, "case_count": sum(row["case_count"] for row in merged_rows),
            "generation_time_count": 3,
            "layer_file_count": sum(item["kind"] == "layer" for row in merged_rows for item in row["files"]),
            "time_meta_file_count": sum(item["kind"] == "time_meta" for row in merged_rows for item in row["files"]),
            "total_byte_count": sum(row["total_byte_count"] for row in merged_rows),
        },
        "results": {
            "valid_model_count": sum(row["valid"] for row in merged_rows),
            "max_errors": {key: max(row["gate_maxima"][key] for row in merged_rows) for key in merged_rows[0]["gate_maxima"]},
            "physical_confirmation_opened": False,
            "dynamic_bundle_extraction_complete_for_all_cases": False,
        },
        "next_decision": (
            "rerun_dynamic_bundle_extraction_on_merged_288_case_manifest"
            if all(row["valid"] for row in merged_rows)
            else "repair_failed_model_instrumentation_without_lowering_gates"
        ),
    }
    write_json(COLLECTION / "phase366_full_collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
