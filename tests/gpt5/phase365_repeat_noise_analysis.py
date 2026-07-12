#!/usr/bin/env python3
"""Aggregate the sequential six-forward repeat-noise format gate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/repeat_noise_format_gate"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    rows = [read_json(OUT / "models" / model / "complete.json") for model in MODELS]
    summary = {
        "schema_version": "42.3.0", "phase_id": "Phase365-A", "created_at": now(),
        "denominator": {
            "model_count": 3,
            "fixed_prompt_count_per_model": 1,
            "repeat_count_per_model": 2,
            "total_forward_run_count": 6,
            "selected_layer_count_per_run": 3,
            "model_order": list(MODELS),
        },
        "results": {
            "valid_model_count": sum(row["valid"] for row in rows),
            "repeat_exact_model_count": sum(row["results"]["repeat_exact_equal"] for row in rows),
            "mlp_replay_gate_model_count": sum(row["results"]["all_layer_replay_gates_pass"] for row in rows),
            "max_direct_relative_error": max(row["results"]["max_direct_relative_error"] for row in rows),
            "max_neuron_write_relative_error": max(row["results"]["max_neuron_write_relative_error"] for row in rows),
            "max_repeat_relative_error": max(row["results"]["max_repeat_relative_error"] for row in rows),
            "observed_fixed_execution_repeat_noise_floor": max(
                row["results"]["max_repeat_relative_error"] for row in rows
            ),
        },
        "quality": {
            "models_executed_sequentially": True,
            "causal_intervention_executed": False,
            "physical_confirmation_opened": False,
            "semantic_target_used": False,
            "repeat_noise_is_template_noise": False,
            "repeat_noise_is_natural_condition_noise": False,
        },
        "authorization": {
            "phase365_96_case_engineering_run_authorized": all(row["valid"] for row in rows),
            "language_mechanism_claim_authorized": False,
            "physical_confirmation_authorized": False,
        },
        "next_decision": "freeze_96_case_engineering_collection_schema" if all(row["valid"] for row in rows) else "repair_instrumentation_before_any_expansion",
    }
    write_json(OUT / "phase365_repeat_noise_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
