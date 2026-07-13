#!/usr/bin/env python3
"""Combine Phase401 instrument ledgers and authorize or block discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase401_local_edge_graph"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    models = {
        model: read_json(OUT / "instrument" / model / "complete.json")
        for model in MODELS
    }
    valid = all(row["valid"] for row in models.values())
    payload = {
        "schema_version": "75.7.0",
        "phase_id": "Phase401-InstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "joint_gate": {
            "all_three_models_complete": set(models) == set(MODELS),
            "all_cases_all_layers_ledger_pass": all(
                row["ledger_pass_case_count"] == row["case_count"]
                for row in models.values()
            ),
            "all_same_shape_generated_replays_pass": all(
                row["exact_replay_pass_case_count"] == row["case_count"]
                for row in models.values()
            ),
            "pass": valid,
        },
        "authorization": {
            "run_discovery_local_edges": valid,
            "run_calibration": False,
            "open_physical_holdout": False,
            "head_channel_or_neuron_scan": False,
        },
        "stopping_decision": (
            None if valid else "repair_instrument_and_stop_local_edge_analysis"
        ),
        "claim_boundary": {
            "instrument_pass_is_a_language_edge": False,
            "instrument_failure_disproves_internal_language_processing": False,
        },
    }
    write_json(OUT / "phase401_instrument_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
