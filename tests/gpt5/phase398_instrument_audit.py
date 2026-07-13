#!/usr/bin/env python3
"""Audit Phase398 compact query-trace instrumentation before discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")
EXPECTED_LAYERS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    rows = []
    for model in MODELS:
        complete = read_json(OUT / f"query_trace/instrument/private/models/{model}/complete.json")
        expected_rows = 3 * EXPECTED_LAYERS[model] * 2 * 4
        valid = bool(
            complete["valid"]
            and complete["case_count"] == 48
            and complete["group_count"] == 3
            and complete["layer_count"] == EXPECTED_LAYERS[model]
            and complete["factorial_effect_row_count"] == expected_rows
            and complete["prefix_transition_match_count"] == 48
            and complete["target_completion_argmax_match_count"] == 48
            and complete["all_block_conservation_pass"]
            and complete["max_block_relative_error"] <= 0.01
        )
        rows.append({
            "model": model,
            "case_count": complete["case_count"],
            "group_count": complete["group_count"],
            "layer_count": complete["layer_count"],
            "max_block_relative_error": complete["max_block_relative_error"],
            "exact_replay_count": complete["target_completion_argmax_match_count"],
            "valid": valid,
        })
    authorized = all(row["valid"] for row in rows)
    report = {
        "schema_version": "72.5.0",
        "phase_id": "Phase398-InstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instrument_case_count": sum(row["case_count"] for row in rows),
        "instrument_group_count": sum(row["group_count"] for row in rows),
        "model_audits": rows,
        "gates": {
            "three_models_complete": len(rows) == 3,
            "all_144_incremental_replays_exact": sum(row["exact_replay_count"] for row in rows) == 144,
            "all_component_conservation_errors_le_0_01": all(row["max_block_relative_error"] <= 0.01 for row in rows),
        },
        "authorization": {
            "run_discovery_trace": authorized,
            "run_calibration_trace": False,
            "open_physical_holdout": False,
            "run_causal_intervention": False,
            "single_neuron_scan": False,
        },
    }
    write_json(OUT / "phase398_instrument_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not authorized:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
