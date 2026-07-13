#!/usr/bin/env python3
"""Audit Phase395 patch locality before the full calibration intervention."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    failures: list[str] = []
    direction_count = 0
    scenario_count = 0
    generation_count = 0
    maximum_patch_error = 0.0
    maximum_outside_error = 0.0
    maximum_identity_effect = 0.0
    for model in MODELS:
        root = OUT / "causal/instrument_audit" / model
        complete = read_json(root / "complete.json")
        rows = read_jsonl(root / "direction_rows.jsonl")
        if not complete["valid"] or len(rows) != 8:
            failures.append(f"{model}:denominator")
        direction_count += len(rows)
        for row in rows:
            if len(row["scenario_rows"]) != 9 or len(row["generation_rows"]) != 7:
                failures.append(f"{model}:{row['direction_id']}:scenario_denominator")
            scenario_count += len(row["scenario_rows"])
            generation_count += len(row["generation_rows"])
            for scenario in row["scenario_rows"]:
                audit = scenario["patch_audit"]
                expected_calls = 0 if scenario["scenario"] == "no_intervention" else 1
                if audit["patch_call_count"] != expected_calls:
                    failures.append(f"{model}:{row['direction_id']}:{scenario['scenario']}:calls")
                maximum_patch_error = max(maximum_patch_error, audit["max_patch_error"])
                maximum_outside_error = max(maximum_outside_error, audit["max_outside_error"])
                if scenario["scenario"] == "identity_same_literal_candidate":
                    maximum_identity_effect = max(
                        maximum_identity_effect,
                        abs(scenario["normalized_margin_mediation"]),
                    )
            for generation in row["generation_rows"]:
                audit = generation["patch_audit"]
                if audit["patch_call_count"] != 1:
                    failures.append(f"{model}:{row['direction_id']}:{generation['scenario']}:generation_calls")
                maximum_patch_error = max(maximum_patch_error, audit["max_patch_error"])
                maximum_outside_error = max(maximum_outside_error, audit["max_outside_error"])
    if maximum_patch_error > 0.01 or maximum_outside_error > 0.01:
        failures.append("patch_locality")
    if maximum_identity_effect > 0.01:
        failures.append("identity_effect")
    valid = not failures and direction_count == 24 and scenario_count == 216 and generation_count == 168
    payload = {
        "schema_version": "69.9.0",
        "phase_id": "Phase395-CausalInstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": len(MODELS),
            "direction_count": direction_count,
            "scenario_count": scenario_count,
            "generation_count": generation_count,
        },
        "audit": {
            "maximum_patch_error": maximum_patch_error,
            "maximum_outside_error": maximum_outside_error,
            "maximum_identity_normalized_effect": maximum_identity_effect,
            "failure_count": len(failures),
            "failures": failures,
            "valid": valid,
        },
        "authorization": {
            "full_calibration": valid,
            "physical_holdout": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "instrument_validity_is_causal_result": False,
            "instrument_validity_is_binding_mechanism": False,
        },
    }
    path = OUT / "phase395_causal_instrument_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
