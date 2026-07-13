#!/usr/bin/env python3
"""Audit Phase397 compact factor-trace instrumentation across three models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    model_rows = []
    all_instruments: list[dict[str, Any]] = []
    for model in MODELS:
        root = OUT / "factor_trace/instrument" / model
        complete = read_json(root / "complete.json")
        rows = read_jsonl(root / "instrument_rows.jsonl")
        factors = read_jsonl(root / "factor_rows.jsonl")
        valid = (
            complete["valid"]
            and complete["group_count"] == 3
            and complete["natural_case_count"] == 30
            and len(rows) == 3
            and len(factors) == 27
            and all(row["valid"] for row in rows)
        )
        model_rows.append(
            {
                "model": model,
                "natural_case_count": complete["natural_case_count"],
                "factor_pair_count": len(factors),
                "identity_patch_logit_max_error": max(row["identity_patch_logit_max_error"] for row in rows),
                "maximum_query_source_relative_delta": max(row["maximum_query_source_relative_delta"] for row in rows),
                "maximum_patch_error": max(row["identity_patch_audit"]["max_patch_error"] for row in rows),
                "maximum_outside_error": max(row["identity_patch_audit"]["max_outside_error"] for row in rows),
                "valid": valid,
            }
        )
        all_instruments.extend(rows)
    valid = all(row["valid"] for row in model_rows)
    payload = {
        "schema_version": "71.5.0",
        "phase_id": "Phase397-FactorTraceInstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "task_surfaces": ["possession_relation", "role_filling", "coreference_resolution"],
            "group_count": len(all_instruments),
            "natural_case_count": sum(row["natural_case_count"] for row in model_rows),
            "factor_pair_count": sum(row["factor_pair_count"] for row in model_rows),
        },
        "models": model_rows,
        "results": {
            "all_three_model_instruments_valid": valid,
            "causal_source_invariance_pass": all(row["maximum_query_source_relative_delta"] <= 1e-6 for row in model_rows),
            "identity_patch_locality_pass": all(row["maximum_patch_error"] <= 0.01 and row["maximum_outside_error"] <= 0.01 for row in model_rows),
            "language_binding_discovered": False,
        },
        "authorization": {
            "discovery_trace": valid,
            "calibration_trace": False,
            "physical_holdout_trace": False,
            "causal_intervention": False,
            "single_neuron_scan": False,
        },
        "claim_boundary": {
            "instrument_validity_is_binding_mechanism": False,
            "query_source_invariance_is_query_integration_localization": False,
        },
    }
    path = OUT / "phase397_factor_trace_instrument_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
